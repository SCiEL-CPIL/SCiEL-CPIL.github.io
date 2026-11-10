# EVOLVE-BLOCK-START
import numpy as np
import time


def _precompute_pooling_matrices(n=10):
    """Precompute pooling matrices for slopes -1, 0, 1, 2."""
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
    """Compute D/M ratio."""
    ent = _compute_all_entropies(P_flat, matrices)
    D = ent[-1]
    M = max(ent[0], ent[1], ent[2])
    if M < 1e-9:
        return 0.0, D, M
    return D / M, D, M


def _gradient_entropy(P_flat, A, inverse):
    """Compute gradient of entropy H = -sum p log2 p where p = A @ P_flat."""
    q = A @ P_flat
    q = np.maximum(q, 1e-15)
    log_q = np.log2(q)
    # dH/dP_i = sum_v (dH/dq_v) * A[v,i]
    # dH/dq_v = -(log2(q_v) + 1/ln(2))
    dH_dq = -(log_q + 1.0 / np.log(2))
    # Gradient w.r.t. P_flat: A.T @ dH_dq
    return A.T @ dH_dq


def _soft_max_gradient(P_flat, matrices, beta):
    """
    Compute soft-max of H_0, H_1, H_2 and its gradient.
    Returns (soft_max_value, gradient_wrt_P_flat, H_values, weights)
    """
    ent = _compute_all_entropies(P_flat, matrices)
    H_vals = np.array([ent[0], ent[1], ent[2]])
    max_H = np.max(H_vals)
    exp_vals = np.exp(beta * (H_vals - max_H))
    weights = exp_vals / np.sum(exp_vals)
    soft_max = max_H + (1.0 / beta) * np.log(np.sum(exp_vals))
    
    # Gradient of soft_max w.r.t. P_flat
    grad = np.zeros_like(P_flat)
    for idx, r in enumerate([0, 1, 2]):
        if weights[idx] < 1e-12:
            continue
        A_r, _, inv_r = matrices[r]
        grad += weights[idx] * _gradient_entropy(P_flat, A_r, inv_r)
    
    return soft_max, grad, H_vals, weights


def _gradient_soft_log_ratio(P_flat, matrices, beta=10.0):
    """Compute gradient of log(D) - log(soft_max(H_0, H_1, H_2)) w.r.t. P_flat."""
    ent = _compute_all_entropies(P_flat, matrices)
    D = ent[-1]
    
    if D < 1e-15:
        return np.zeros_like(P_flat)
    
    # d(log D)/dP
    A_num, _, _ = matrices[-1]
    dlogD_dP = _gradient_entropy(P_flat, A_num, None) / D
    
    # Soft max gradient
    soft_max, dsoftmax_dP, _, _ = _soft_max_gradient(P_flat, matrices, beta)
    
    if soft_max < 1e-15:
        return dlogD_dP
    
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


def _line_search(P_flat, grad, matrices, step_size, max_backtrack=10, c=0.01):
    """Simple backtracking line search for the log-ratio objective."""
    P_current = _project_simplex(P_flat.reshape(10, 10), 10).flatten()
    _, current_ratio, _ = _compute_ratio(P_current, matrices)
    
    # Use log-ratio as objective for line search
    ent = _compute_all_entropies(P_current, matrices)
    D = ent[-1]
    M = max(ent[0], ent[1], ent[2])
    if D < 1e-15 or M < 1e-15:
        return P_current, current_ratio
    
    current_obj = np.log(D) - np.log(M)
    
    alpha = step_size
    for _ in range(max_backtrack):
        P_new = _project_simplex((P_flat + alpha * grad).reshape(10, 10), 10).flatten()
        new_ratio, _, _ = _compute_ratio(P_new, matrices)
        
        ent_new = _compute_all_entropies(P_new, matrices)
        D_new = ent_new[-1]
        M_new = max(ent_new[0], ent_new[1], ent_new[2])
        
        if D_new < 1e-15 or M_new < 1e-15:
            alpha *= 0.5
            continue
        
        new_obj = np.log(D_new) - np.log(M_new)
        
        if new_obj >= current_obj + c * alpha * np.dot(grad, P_new - P_current):
            return P_new, new_ratio
        alpha *= 0.5
    
    return P_current, current_ratio


def _gradient_ascent(P, matrices, n, max_iters=2000, time_limit=300):
    """Run gradient ascent on soft log-ratio objective with annealed beta and adaptive step size."""
    start_time = time.time()
    P_flat = P.flatten().copy()
    
    best_P_flat = P_flat.copy()
    best_ratio, _, _ = _compute_ratio(P_flat, matrices)
    
    step_size = 0.05
    patience = 0
    no_improve_count = 0
    
    # Anneal beta from soft (small) to hard (large)
    beta_start = 1.0
    beta_end = 200.0
    anneal_fraction = 0.7
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Anneal beta
        frac = min(it / max(1, int(max_iters * anneal_fraction)), 1.0)
        beta = beta_start + (beta_end - beta_start) * frac
        
        # Compute gradient using soft max
        grad = _gradient_soft_log_ratio(P_flat, matrices, beta)
        
        # Line search
        P_new_flat, new_ratio = _line_search(P_flat, grad, matrices, step_size)
        
        if new_ratio > best_ratio + 1e-10:
            best_ratio = new_ratio
            best_P_flat = P_new_flat.copy()
            patience = 0
            no_improve_count = 0
        else:
            patience += 1
            no_improve_count += 1
        
        # Adaptive step size
        if patience > 30:
            step_size *= 0.7
            patience = 0
        
        if no_improve_count > 300:
            break
        
        P_flat = P_new_flat
    
    return best_P_flat.reshape(n, n), best_ratio


def _structured_init(n, rng):
    """Generate structured initial distributions."""
    inits = []
    
    # Uniform
    inits.append(np.ones((n, n)) / (n * n))
    
    # Diagonal emphasis
    for diag_weight in [0.1, 0.3, 0.5, 0.7]:
        P = np.ones((n, n)) * 0.01
        for i in range(n):
            P[i, i] = diag_weight
        P = P / P.sum()
        inits.append(P)
    
    # Anti-diagonal emphasis
    for anti_weight in [0.1, 0.3, 0.5, 0.7]:
        P = np.ones((n, n)) * 0.01
        for i in range(n):
            P[i, n - 1 - i] = anti_weight
        P = P / P.sum()
        inits.append(P)
    
    # Corner emphasis
    for corner_weight in [0.2, 0.4, 0.6]:
        P = np.ones((n, n)) * 0.01
        P[0, 0] = corner_weight
        P[n-1, n-1] = corner_weight
        P[0, n-1] = corner_weight
        P[n-1, 0] = corner_weight
        P = P / P.sum()
        inits.append(P)
    
    # Row emphasis
    for row_weight in [0.3, 0.5]:
        P = np.ones((n, n)) * 0.01
        P[0, :] = row_weight
        P[n-1, :] = row_weight
        P = P / P.sum()
        inits.append(P)
    
    # Column emphasis
    for col_weight in [0.3, 0.5]:
        P = np.ones((n, n)) * 0.01
        P[:, 0] = col_weight
        P[:, n-1] = col_weight
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


def _random_init(n, rng):
    """Generate a random initial distribution on the simplex."""
    P = rng.dirichlet(np.ones(n * n))
    return P.reshape(n, n)


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Optimize P to maximize H_{-1} / max(H_0, H_1, H_2) using gradient ascent
    on the log-ratio objective with soft-max surrogate for the denominator.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 620  # safety margin
    
    # Precompute pooling matrices
    matrices = _precompute_pooling_matrices(n)
    
    rng = np.random.default_rng(42)
    
    # Generate initial candidates
    candidates = _structured_init(n, rng)
    
    # Add random inits
    for _ in range(15):
        candidates.append(_random_init(n, rng))
    
    best_P = None
    best_ratio = -np.inf
    
    # Evaluate all initial candidates
    candidate_ratios = []
    for P_init in candidates:
        if time.time() > deadline - 100:
            break
        P_flat = P_init.flatten()
        ratio, _, _ = _compute_ratio(P_flat, matrices)
        candidate_ratios.append((ratio, P_init))
        if ratio > best_ratio:
            best_ratio = ratio
            best_P = P_init.copy()
    
    # Sort candidates by initial ratio
    candidate_ratios.sort(key=lambda x: x[0], reverse=True)
    
    # Run gradient ascent from top initial points
    num_attempts = min(15, len(candidate_ratios))
    for i in range(num_attempts):
        if time.time() > deadline - 30:
            break
        
        remaining_time = deadline - time.time()
        if remaining_time < 20:
            break
        
        # Allocate time: more time for earlier (better) initial points
        if i < 3:
            attempt_time = min(remaining_time / (num_attempts - i), 150)
        elif i < 8:
            attempt_time = min(remaining_time / (num_attempts - i), 80)
        else:
            attempt_time = min(remaining_time / (num_attempts - i), 40)
        
        _, P_init = candidate_ratios[i]
        
        P_opt, ratio_opt = _gradient_ascent(
            P_init, matrices, n,
            max_iters=10000,
            time_limit=attempt_time
        )
        
        if ratio_opt > best_ratio:
            best_ratio = ratio_opt
            best_P = P_opt.copy()
    
    # Final polish with very high beta and small steps
    if best_P is not None and time.time() < deadline - 15:
        remaining_time = deadline - time.time() - 5
        if remaining_time > 10:
            P_flat = best_P.flatten().copy()
            step_size = 0.002
            for _ in range(20000):
                if time.time() > deadline - 5:
                    break
                grad = _gradient_soft_log_ratio(P_flat, matrices, beta=500.0)
                P_new_flat, new_ratio = _line_search(P_flat, grad, matrices, step_size)
                if new_ratio > best_ratio:
                    best_ratio = new_ratio
                    best_P = P_new_flat.reshape(n, n).copy()
                    P_flat = P_new_flat
                else:
                    step_size *= 0.9
                    if step_size < 1e-6:
                        break
    
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
