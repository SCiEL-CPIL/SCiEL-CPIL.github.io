# EVOLVE-BLOCK-START
import numpy as np
import time

def _precompute_pooling_matrices(n=10):
    """Precompute pooling matrices for slopes -1, 0, 1, 2.
    Each matrix A_r has shape (k_r, n*n) where k_r is number of unique values.
    A_r[v, idx] = 1 if cell idx maps to value v under slope r.
    """
    y_idx, x_idx = np.indices((n, n))
    flat_idx = np.arange(n * n)
    
    matrices = {}
    for r in [-1, 0, 1, 2]:
        vals = (x_idx + r * y_idx).flatten()
        unique_vals, inverse = np.unique(vals, return_inverse=True)
        k = len(unique_vals)
        A = np.zeros((k, n * n), dtype=np.float64)
        A[inverse, flat_idx] = 1.0
        matrices[r] = (A, unique_vals, inverse)
    
    return matrices

def _compute_all_entropies(P_flat, matrices):
    """Compute H_{-1}, H_0, H_1, H_2 from flattened P."""
    entropies = {}
    for r in [-1, 0, 1, 2]:
        A, _, _ = matrices[r]
        probs = A @ P_flat
        probs = np.maximum(probs, 0)
        mask = probs > 1e-15
        if mask.sum() == 0:
            entropies[r] = 0.0
        else:
            p = probs[mask]
            entropies[r] = -np.sum(p * np.log2(p))
    return entropies

def _compute_ratio(P_flat, matrices):
    """Compute D/M ratio using true max for evaluation."""
    ent = _compute_all_entropies(P_flat, matrices)
    D = ent[-1]
    M = max(ent[0], ent[1], ent[2])
    if M < 1e-9:
        return 0.0, D, M
    return D / M, D, M

def _gradient_soft_log_ratio(P_flat, matrices, beta=10.0):
    """
    Compute gradient of log(D) - log(soft_max(H_0, H_1, H_2)) w.r.t. P_flat.
    soft_max = (1/beta) * log(sum(exp(beta * H_r)))
    """
    ent = _compute_all_entropies(P_flat, matrices)
    D = ent[-1]
    
    if D < 1e-15:
        return np.zeros_like(P_flat)
    
    # Gradient of log(D)
    A_num, _, inv_num = matrices[-1]
    q_num = A_num @ P_flat
    q_num = np.maximum(q_num, 1e-15)
    log_q_num = np.log2(q_num)
    # dD/dP_i = -(log2(q_v) + 1/ln(2)) where v is the bin for i
    dD_dP = -(log_q_num[inv_num] + 1.0 / np.log(2))
    dlogD_dP = dD_dP / D
    
    # Gradient of log(soft_max)
    H_vals = np.array([ent[0], ent[1], ent[2]])
    max_H = np.max(H_vals)
    # Numerical stability for log-sum-exp
    exp_vals = np.exp(beta * (H_vals - max_H))
    sum_exp = np.sum(exp_vals)
    soft_max = max_H + (1.0 / beta) * np.log(sum_exp)
    
    if soft_max < 1e-15:
        return dlogD_dP
    
    # Weights for each H_r in the soft max
    weights = exp_vals / sum_exp
    
    # d(soft_max)/dP = sum_r w_r * dH_r/dP
    dsoftmax_dP = np.zeros_like(P_flat)
    for idx, r in enumerate([0, 1, 2]):
        if weights[idx] < 1e-12:
            continue
        A_r, _, inv_r = matrices[r]
        q_r = A_r @ P_flat
        q_r = np.maximum(q_r, 1e-15)
        log_q_r = np.log2(q_r)
        dH_r_dP = -(log_q_r[inv_r] + 1.0 / np.log(2))
        dsoftmax_dP += weights[idx] * dH_r_dP
    
    dlogM_dP = dsoftmax_dP / soft_max
    
    return dlogD_dP - dlogM_dP

def _project_simplex(P, n=10):
    """Project P onto simplex: non-negative, sum to 1."""
    P = np.clip(P, 0, None)
    s = P.sum()
    if s > 1e-15:
        P = P / s
    else:
        P = np.ones((n, n)) / (n * n)
    return P

def _gradient_ascent(P, matrices, n, max_iters=2000, time_limit=300):
    """
    Run gradient ascent on soft log-ratio objective.
    Uses annealing of beta from soft to hard.
    """
    start_time = time.time()
    P_flat = P.flatten().copy()
    
    best_P_flat = P_flat.copy()
    best_ratio, _, _ = _compute_ratio(P_flat, matrices)
    
    step_size = 0.02
    patience = 0
    no_improve_count = 0
    
    # Annealing parameters
    beta_start = 2.0
    beta_end = 100.0  # Very high to approximate hard max at the end
    anneal_fraction = 0.7  # Anneal over first 70% of time/iters
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Linear annealing of beta based on iteration count
        # We use iteration count for smoothness, but check time for hard stop
        frac = min(it / max(1, int(max_iters * anneal_fraction)), 1.0)
        beta = beta_start + (beta_end - beta_start) * frac
        
        # Compute gradient
        grad = _gradient_soft_log_ratio(P_flat, matrices, beta)
        
        # Gradient step
        P_new_flat = P_flat + step_size * grad
        P_new = _project_simplex(P_new_flat.reshape(n, n), n)
        P_new_flat = P_new.flatten()
        
        # Evaluate true ratio
        new_ratio, _, _ = _compute_ratio(P_new_flat, matrices)
        
        if new_ratio > best_ratio + 1e-10:
            best_ratio = new_ratio
            best_P_flat = P_new_flat.copy()
            patience = 0
            no_improve_count = 0
        else:
            patience += 1
            no_improve_count += 1
        
        # Adaptive step size reduction
        if patience > 30:
            step_size *= 0.9
            patience = 0
            
        # Early stopping if no improvement for a long time
        if no_improve_count > 300:
            break
            
        P_flat = P_new_flat
        
    return best_P_flat.reshape(n, n), best_ratio

def _structured_init(n, rng):
    """Generate structured initial distributions."""
    inits = []
    
    # Uniform
    inits.append(np.ones((n, n)) / (n * n))
    
    # Diagonal emphasis (x - y = 0)
    for w in [0.1, 0.3, 0.5, 0.7]:
        P = np.ones((n, n)) * 0.01
        for i in range(n):
            P[i, i] = w
        P = P / P.sum()
        inits.append(P)
        
    # Anti-diagonal emphasis (x + y = constant)
    for w in [0.1, 0.3, 0.5, 0.7]:
        P = np.ones((n, n)) * 0.01
        for i in range(n):
            P[i, n - 1 - i] = w
        P = P / P.sum()
        inits.append(P)
        
    # Row emphasis (x varies, y fixed) - affects H_0, H_1, H_2 differently than H_-1
    for w in [0.2, 0.5]:
        P = np.ones((n, n)) * 0.01
        for j in range(n):
            P[j, :] = w
        P = P / P.sum()
        inits.append(P)
        
    # Col emphasis
    for w in [0.2, 0.5]:
        P = np.ones((n, n)) * 0.01
        for i in range(n):
            P[:, i] = w
        P = P / P.sum()
        inits.append(P)

    # Random sparse
    for _ in range(10):
        P = np.zeros((n, n))
        num_nonzero = rng.integers(10, 50)
        idx = rng.choice(n * n, size=num_nonzero, replace=False)
        P.flat[idx] = rng.dirichlet(np.ones(num_nonzero))
        P = P / P.sum()
        inits.append(P)
        
    return inits

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Optimize P to maximize H_{-1} / max(H_0, H_1, H_2) using soft-max gradient ascent.
    """
    n = 10
    start_time = time.time()
    # 630s is the kill limit. Use 620s as internal deadline to be safe.
    deadline = start_time + 620
    
    matrices = _precompute_pooling_matrices(n)
    rng = np.random.default_rng(42)
    
    # Generate initial candidates
    candidates = _structured_init(n, rng)
    
    # Add more random inits for diversity
    for _ in range(15):
        P = rng.dirichlet(np.ones(n * n))
        candidates.append(P.reshape(n, n))
    
    best_P = None
    best_ratio = -np.inf
    
    # 1. Evaluate all initial candidates
    for P_init in candidates:
        if time.time() > deadline - 120:
            break
        P_flat = P_init.flatten()
        ratio, _, _ = _compute_ratio(P_flat, matrices)
        if ratio > best_ratio:
            best_ratio = ratio
            best_P = P_init.copy()
            
    # 2. Sort candidates by initial ratio
    candidate_ratios = []
    for P_init in candidates:
        P_flat = P_init.flatten()
        ratio, _, _ = _compute_ratio(P_flat, matrices)
        candidate_ratios.append((ratio, P_init))
    candidate_ratios.sort(key=lambda x: x[0], reverse=True)
    
    # 3. Run gradient ascent from top candidates
    # We have ~500s left. Let's try top 15 candidates.
    # Each run gets limited time.
    top_k = 15
    for i, (ratio, P_init) in enumerate(candidate_ratios[:top_k]):
        if time.time() > deadline - 30:
            break
            
        # Allocate time dynamically
        remaining = deadline - time.time()
        # Reserve 20s for final polish and safety
        if remaining < 50:
            break
            
        # Give more time to better initial candidates
        # First few get more time
        if i < 3:
            time_limit = min(remaining * 0.4, 150)
        elif i < 10:
            time_limit = min(remaining * 0.2, 80)
        else:
            time_limit = min(remaining * 0.1, 40)
            
        if time_limit < 10:
            continue
            
        P_opt, ratio_opt = _gradient_ascent(
            P_init, matrices, n,
            max_iters=5000,
            time_limit=time_limit
        )
        
        if ratio_opt > best_ratio:
            best_ratio = ratio_opt
            best_P = P_opt.copy()
            
    # 4. Final Polish: Run a short, high-beta optimization from the best found
    if best_P is not None and time.time() < deadline - 20:
        remaining = deadline - time.time() - 5
        if remaining > 10:
            # Use a very high beta effectively (hard max) but smooth enough for gradient
            # The _gradient_ascent function handles beta annealing internally.
            # To force "hard" behavior, we could modify the function or just rely on the annealing.
            # Since annealing ends at beta=100, it's already quite hard.
            # Let's just run a short refinement.
            P_final, ratio_final = _gradient_ascent(
                best_P, matrices, n,
                max_iters=2000,
                time_limit=remaining
            )
            if ratio_final > best_ratio:
                best_ratio = ratio_final
                best_P = P_final.copy()
                
    # Ensure validity
    if best_P is None:
        best_P = np.ones((n, n)) / (n * n)
        
    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s > 1e-15:
        best_P = best_P / s
    else:
        best_P = np.ones((n, n)) / (n * n)
        
    return best_P

# EVOLVE-BLOCK-END
