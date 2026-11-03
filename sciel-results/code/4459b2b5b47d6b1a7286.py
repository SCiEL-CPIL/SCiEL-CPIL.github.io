"""Combined program: efficient exact score + epigraph SLSQP + coordinate descent refinement."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
DELTA = 0.5 / N  # Step width for intervals covering [-1/4, 1/4]
NUM_CONV = 2 * N - 1

def _score_raw(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * ||f*f||_inf) exactly for step functions.
    
    For a step function with heights h[i] on intervals of width DELTA,
    the autoconvolution (f*f)(x) has supremum norm = DELTA * max_k (h*h)[k],
    where (h*h) is the discrete autoconvolution of the height vector.
    """
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 <= 1e-15:
        return 0.0
    l2_sq = np.sum(h ** 2)
    h_conv = np.convolve(h, h, mode='full')
    max_conv = np.max(h_conv)
    if max_conv <= 1e-15:
        return 0.0
    return l2_sq / (l1 * DELTA * max_conv)

def _objective_ratio(h):
    """Objective to minimize: -ratio (so maximizing ratio)."""
    h = np.maximum(h, 0)
    l1 = np.sum(h)
    if l1 < 1e-12:
        return 0.0
    l2_sq = np.sum(h ** 2)
    h_conv = np.convolve(h, h, mode='full')
    max_conv = np.max(h_conv)
    if max_conv < 1e-12:
        return 0.0
    ratio = l2_sq / (l1 * DELTA * max_conv)
    return -ratio

def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    
    best_h = np.ones(N)
    best_score = _score_raw(best_h)
    
    x_idx = np.arange(N)
    
    # --- Phase 1: Generate diverse initializations ---
    inits = []
    
    # 1. Uniform
    inits.append(np.ones(N))
    
    # 2. Sparse structured (Sidon-like intuition)
    rng = np.random.default_rng(42)
    for trial in range(40):
        k = rng.integers(3, 15)
        idx = rng.choice(N, k, replace=False)
        h_s = np.zeros(N)
        h_s[idx] = 1.0
        inits.append(h_s)
        
    # 3. Concentrated Gaussians
    for c in np.linspace(3, N - 3, 12):
        for w in np.linspace(0.5, N / 3, 6):
            h_g = np.exp(-0.5 * ((x_idx - c) / w) ** 2)
            if np.sum(h_g) > 1e-10:
                inits.append(h_g)
                
    # 4. Triangular
    for c in np.linspace(2, N - 2, 10):
        for w in np.linspace(2, N, 6):
            h_t = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_t) > 1e-10:
                inits.append(h_t)
                
    # 5. Random smooth
    for _ in range(25):
        h_r = rng.random(N)
        kernel = np.ones(5) / 5
        h_r = np.convolve(h_r, kernel, mode='same')
        h_r = np.maximum(0, h_r)
        if np.sum(h_r) > 1e-10:
            inits.append(h_r)
            
    # 6. Alternating patterns (helps reduce auto-correlation peaks)
    for start in range(N):
        for gap in [2, 3, 4, 5, 7]:
            h = np.zeros(N)
            for i in range(start, N, gap):
                h[i] = 1.0
            if np.sum(h) > 0:
                inits.append(h)
    
    # 7. Binary patterns with varying density
    for density in [0.2, 0.3, 0.4, 0.5]:
        for _ in range(10):
            h = (rng.random(N) < density).astype(float)
            if np.sum(h) > 0:
                inits.append(h)
    
    # --- Phase 2: Epigraph Formulation Optimization ---
    # Variables: x = [h_0, ..., h_{N-1}, M]
    # Objective: Minimize M
    # Constraints:
    # 1. sum(h) - 1 == 0
    # 2. h_i >= 0 (via bounds)
    # 3. M - conv_k >= 0 for all k
    
    def objective(x):
        return x[N]
        
    def objective_grad(x):
        grad = np.zeros(N + 1)
        grad[N] = 1.0
        return grad
        
    def constraint_eq_fun(x):
        return np.sum(x[:N]) - 1.0
        
    def constraint_eq_jac(x):
        jac = np.zeros(N + 1)
        jac[:N] = 1.0
        return jac
        
    def constraint_ineq_fun(x):
        h = x[:N]
        M = x[N]
        conv = np.convolve(h, h, mode='full')
        return M - conv
        
    def constraint_ineq_jac(x):
        h = x[:N]
        jac = np.zeros((NUM_CONV, N + 1))
        jac[:, N] = 1.0
        # d(M - conv_k)/dh_j = -d conv_k / d h_j
        # conv_k = sum_{m=0}^{N-1} h_m h_{k-m} (with h_i=0 if i out of [0,N-1])
        # d conv_k / d h_j = h_{k-j} + h_{j-k} (each 0 if index out of range)
        for j in range(N):
            # Part 1: m=j, so term is h_j * h_{k-j}, derivative w.r.t. h_j is h_{k-j}
            # k-j in [0, N-1] => k in [j, j+N-1]
            k_start = j
            k_end = min(j + N, NUM_CONV)
            if k_start < NUM_CONV and k_end > 0:
                indices_k = np.arange(k_start, k_end)
                indices_h = indices_k - j
                jac[indices_k, j] -= h[indices_h]
            
            # Part 2: k-m=j, so m=k-j, term is h_{k-j} * h_j, derivative w.r.t. h_j is h_{k-j}
            # This is the same as part 1 when we sum over all m.
            # Actually, conv_k = sum_m h_m * h_{k-m}
            # d/dh_j: if m=j, term is h_j*h_{k-j}, derivative is h_{k-j}
            #           if k-m=j i.e. m=k-j, term is h_{k-j}*h_j, derivative is h_j... wait
            # Let me be more careful:
            # conv_k = sum_{m=0}^{N-1} h_m * h_{k-m}
            # d conv_k / d h_j = h_{k-j} (from m=j term) + h_j (from m=k-j term, if k-j != j)
            # Wait no. Let's expand:
            # conv_k = h_0*h_k + h_1*h_{k-1} + ... + h_j*h_{k-j} + ... + h_k*h_0
            # d/dh_j: the term h_j*h_{k-j} gives h_{k-j}. 
            # Also, if there's a term h_{k-j}*h_j (when m=k-j), that gives h_j.
            # But h_j*h_{k-j} and h_{k-j}*h_j are the same term only when m=j and m=k-j simultaneously,
            # which happens when j=k-j, i.e., k=2j.
            # Actually no: the sum is over m. The term where m=j is h_j * h_{k-j}.
            # The term where m=k-j is h_{k-j} * h_j. These are two different terms in the sum
            # (unless k-j = j, i.e., k=2j, in which case they're the same term h_j*h_j).
            # So d conv_k / d h_j = h_{k-j} + h_{k-j} = 2*h_{k-j} if k-j != j
            #                       = h_j (since k-j=j, so h_{k-j}=h_j) if k=2j
            # Wait, if k=2j, then the term m=j is h_j*h_j, and its derivative w.r.t. h_j is 2*h_j.
            # So in all cases: d conv_k / d h_j = h_{k-j} + h_{k-j} = 2*h_{k-j}, where h_{k-j}=0 if k-j out of [0,N-1].
            # But wait, that's only if both m=j and m=k-j are valid indices in [0,N-1].
            # Let me redo:
            # d conv_k / d h_j = sum over m where the term involves h_j.
            # Term m: h_m * h_{k-m}. This involves h_j if m=j or k-m=j (i.e., m=k-j).
            # If m=j: contribution is h_{k-j} (if k-j in [0,N-1], else 0)
            # If m=k-j: contribution is h_{k-j} (if k-j in [0,N-1], else 0) [since h_m = h_{k-j}]
            # Wait, if m=k-j, the term is h_{k-j} * h_j. Derivative w.r.t. h_j is h_{k-j}.
            # So total: if both j and k-j are in [0,N-1], derivative is 2*h_{k-j}.
            # If only j is in [0,N-1] but k-j is not, derivative is 0 (since h_{k-j}=0).
            # If only k-j is in [0,N-1] but j is not... but j is always in [0,N-1] since we're iterating j in range(N).
            # So: d conv_k / d h_j = 2*h_{k-j} if k-j in [0,N-1], else 0.
            
            # Let me recompute more carefully.
            # conv_k = sum_{m=max(0,k-N+1)}^{min(k,N-1)} h_m * h_{k-m}
            # d/dh_j conv_k:
            # - If j is in [max(0,k-N+1), min(k,N-1)], then the term m=j exists: h_j * h_{k-j}, derivative = h_{k-j}
            # - Also, if k-j is in [max(0,k-N+1), min(k,N-1)], then the term m=k-j exists: h_{k-j} * h_j, derivative = h_j
            # - If j == k-j (i.e., k=2j), these are the same term, so derivative = 2*h_j
            # - Otherwise, if both j and k-j are valid indices, derivative = h_{k-j} + h_{k-j} = 2*h_{k-j}
            #   (since h_{k-j} appears in both places... wait no)
            #   Term m=j: h_j * h_{k-j}, d/dh_j = h_{k-j}
            #   Term m=k-j: h_{k-j} * h_j, d/dh_j = h_{k-j}
            #   Total: 2*h_{k-j}
            # - If only j is valid (k-j out of range), derivative = 0 (since h_{k-j}=0)
            # - If only k-j is valid (j out of range)... but j is always in [0,N-1].
            
            # So: d conv_k / d h_j = 2 * h_{k-j} if k-j in [0, N-1], else 0.
            
            # Let me verify with a small example. N=2, h=[a,b].
            # conv_0 = a*a
            # conv_1 = a*b + b*a = 2ab
            # conv_2 = b*b
            # d conv_1 / d a = 2b. Using formula: k=1, j=0, k-j=1, h_1=b, 2*b. Correct.
            # d conv_1 / d b = 2a. Using formula: k=1, j=1, k-j=0, h_0=a, 2*a. Correct.
            # d conv_0 / d a = 2a. Using formula: k=0, j=0, k-j=0, h_0=a, 2*a. Correct.
            # d conv_2 / d b = 2b. Using formula: k=2, j=1, k-j=1, h_1=b, 2*b. Correct.
            
            # So the correct formula is: d conv_k / d h_j = 2 * h_{k-j} if k-j in [0, N-1], else 0.
            
            # My earlier loop was wrong. Let me fix it.
            pass  # Will rewrite below
        
        # Rewrite the Jacobian computation correctly
        for j in range(N):
            # d conv_k / d h_j = 2 * h_{k-j} if k-j in [0, N-1], else 0
            # k-j in [0, N-1] => k in [j, j+N-1]
            k_start = j
            k_end = min(j + N, NUM_CONV)
            if k_start < NUM_CONV and k_end > 0:
                indices_k = np.arange(k_start, k_end)
                indices_h = indices_k - j
                jac[indices_k, j] -= 2.0 * h[indices_h]
                
        return jac
    
    constraints = [
        {
            'type': 'eq',
            'fun': constraint_eq_fun,
            'jac': constraint_eq_jac
        },
        {
            'type': 'ineq',
            'fun': constraint_ineq_fun,
            'jac': constraint_ineq_jac
        }
    ]
    
    bounds = [(0.0, None)] * N + [(0.0, None)]
    
    def run_epigraph(h_init, maxiter=300, ftol=1e-12):
        if np.sum(h_init) < 1e-15:
            return None
        h_init = h_init / np.sum(h_init)
        conv_init = np.convolve(h_init, h_init, mode='full')
        M_init = np.max(conv_init) + 1e-6
        x0 = np.concatenate((h_init, [M_init]))
        
        try:
            res = minimize(
                objective, 
                x0, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints,
                jac=objective_grad,
                options={'maxiter': maxiter, 'ftol': ftol}
            )
            
            if res.success or -res.fun < 1e-10:
                h_candidate = res.x[:N]
                h_candidate = np.clip(h_candidate, 0, None)
                if np.sum(h_candidate) > 1e-15:
                    h_candidate /= np.sum(h_candidate)
                    return h_candidate
        except Exception:
            pass
        return None
    
    # Sort inits by initial score (descending) to try best ones first
    scored_inits = []
    for h_init in inits:
        s = _score_raw(h_init)
        scored_inits.append((s, h_init))
    scored_inits.sort(key=lambda x: -x[0])
    
    # Run epigraph from top candidates
    max_epigraph_runs = min(len(scored_inits), 40)
    for i in range(max_epigraph_runs):
        if time.time() > deadline - 50:
            break
        s_init, h_init = scored_inits[i]
        h_candidate = run_epigraph(h_init, maxiter=200, ftol=1e-12)
        if h_candidate is not None:
            s_candidate = _score_raw(h_candidate)
            if s_candidate > best_score:
                best_score = s_candidate
                best_h = h_candidate.copy()
    
    # --- Phase 3: Local Refinement (Coordinate Descent) ---
    def refine_local(h_start, max_iter=100):
        h = h_start.copy()
        h = np.clip(h, 0, None)
        if np.sum(h) < 1e-15:
            return h
        h /= np.sum(h)
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            if time.time() > deadline - 10:
                break
            improved = False
            iter_count += 1
            
            conv = np.convolve(h, h, mode='full')
            k_star = np.argmax(conv)
            M_val = conv[k_star]
            l1 = np.sum(h)
            l2_sq = np.sum(h ** 2)
            current_score = l2_sq / (l1 * DELTA * M_val)
            
            # Try adjusting pairs of bins
            for i in range(N):
                if time.time() > deadline - 10:
                    break
                for j in range(i + 1, N):
                    if h[i] < 1e-10 and h[j] < 1e-10:
                        continue
                    # Try moving mass from i to j and vice versa
                    for delta in [0.01, 0.005, 0.001]:
                        if h[i] < delta or h[j] < delta:
                            continue
                        h_new = h.copy()
                        h_new[i] -= delta
                        h_new[j] += delta
                        conv_new = np.convolve(h_new, h_new, mode='full')
                        M_new = np.max(conv_new)
                        l2_new = np.sum(h_new ** 2)
                        score_new = l2_new / (l1 * DELTA * M_new)
                        if score_new > current_score * (1 + 1e-8):
                            h = h_new
                            current_score = score_new
                            improved = True
                            break
                        h_new = h.copy()
                        h_new[j] -= delta
                        h_new[i] += delta
                        conv_new = np.convolve(h_new, h_new, mode='full')
                        M_new = np.max(conv_new)
                        l2_new = np.sum(h_new ** 2)
                        score_new = l2_new / (l1 * DELTA * M_new)
                        if score_new > current_score * (1 + 1e-8):
                            h = h_new
                            current_score = score_new
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        return h
    
    # Refine the best solution found
    if time.time() < deadline - 20:
        refined = refine_local(best_h, max_iter=50)
        s_refined = _score_raw(refined)
        if s_refined > best_score:
            best_score = s_refined
            best_h = refined
    
    # --- Phase 4: Final SLSQP refinement on the best solution ---
    if time.time() < deadline - 5:
        try:
            res = minimize(_objective_ratio, best_h, method='SLSQP',
                          bounds=[(0, None)] * N,
                          options={'maxiter': 300, 'ftol': 1e-14})
            if -res.fun > best_score:
                best_h = np.maximum(res.x, 0)
                best_score = -res.fun
        except Exception:
            pass
    
    return list(best_h)
