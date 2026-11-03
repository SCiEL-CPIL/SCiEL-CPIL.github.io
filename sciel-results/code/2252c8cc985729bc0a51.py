"""
Second autocorrelation inequality: maximize ratio of squared L2 norm to L1 norm times sup norm of autoconvolution.

Strategy: 
1. Normalize L1=1 (ratio is scale-invariant)
2. Formulate as: minimize M subject to sum(h)=1, h>=0, M >= (h*h)_k for all k
3. Use SLSQP with the auxiliary variable M (smooth formulation)
4. Add spectral refinement using eigendecomposition of autoconvolution Toeplitz matrix
5. Multi-start from diverse initializations (sparse peaks, uniform, structured)
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DELTA = 1.0 / N  # Interval width for [-1/4, 1/4] divided into N steps

def _autoconv(h):
    """Linear autoconvolution of h with itself."""
    return np.convolve(h, h, mode='full')

def _score_raw(h):
    """Raw score: L2^2 / (delta * L1 * max(autoconv))."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0
    l2_sq = np.sum(h**2)
    c = _autoconv(h)
    max_c = np.max(c)
    if max_c <= 0:
        return 0.0
    return l2_sq / (DELTA * l1 * max_c)

def _score_normalized(h):
    """Score with L1 normalized to 1: L2^2 / (delta * max(autoconv))."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0
    h_norm = h / l1
    l2_sq = np.sum(h_norm**2)
    c = _autoconv(h_norm)
    max_c = np.max(c)
    if max_c <= 0:
        return 0.0
    return l2_sq / (DELTA * max_c)

def _solve_slsqp(h_init, maxiter=500, ftol=1e-10):
    """
    Solve: minimize M subject to:
      - sum(h) = 1
      - M - (h*h)_k >= 0 for all k
      - h >= 0, M >= 0
    Returns optimized h (normalized to L1=1) or None on failure.
    """
    h_init = np.asarray(h_init, dtype=np.float64)
    s = np.sum(h_init)
    if s <= 0:
        return None
    h_init = h_init / s
    
    c_init = _autoconv(h_init)
    M_init = np.max(c_init) + 1e-6
    
    x0 = np.concatenate([h_init, [M_init]])
    
    def obj(x):
        return x[N]  # Minimize M
    
    def obj_grad(x):
        g = np.zeros(N + 1)
        g[N] = 1.0
        return g
    
    def cons_eq(x):
        return np.sum(x[:N]) - 1.0
    
    def cons_eq_grad(x):
        g = np.zeros(N + 1)
        g[:N] = 1.0
        return g
    
    def cons_ineq(x):
        h = x[:N]
        M = x[N]
        c = _autoconv(h)
        return M - c  # M - c_k >= 0
    
    def cons_ineq_grad(x):
        h = x[:N]
        # d(M - c_k)/dh_i = -d(c_k)/dh_i = -(h_{k-i} + h_{k-i}) if valid
        # c_k = sum_j h_j * h_{k-j}
        # d(c_k)/dh_i = h_{k-i} + h_{k-i} = 2*h_{k-i} (with out-of-bounds = 0)
        # So d(M-c_k)/dh_i = -2*h_{k-i}
        # d(M-c_k)/dM = 1
        
        grad = np.zeros((2*N - 1, N + 1))
        for k in range(2*N - 1):
            for i in range(N):
                j = k - i
                if 0 <= j < N:
                    grad[k, i] = -2.0 * h[j]
            grad[k, N] = 1.0
        return grad
    
    bounds = [(0.0, None)] * N + [(0.0, None)]
    
    constraints = [
        {'type': 'eq', 'fun': cons_eq, 'jac': cons_eq_grad},
        {'type': 'ineq', 'fun': cons_ineq, 'jac': cons_ineq_grad}
    ]
    
    try:
        res = minimize(
            obj, x0, method='SLSQP', jac=obj_grad,
            bounds=bounds, constraints=constraints,
            options={'maxiter': maxiter, 'ftol': ftol, 'disp': False}
        )
        h_res = np.maximum(0, res.x[:N])
        s_res = np.sum(h_res)
        if s_res > 1e-12:
            h_res = h_res / s_res
            return h_res
        return None
    except Exception:
        return None

def _spectral_refine(h, iterations=200):
    """
    Refine h by finding the direction in the null space of the 
    active autoconvolution constraints that increases L2^2.
    
    Uses the fact that for fixed L1 and fixed peak autoconv,
    we want to maximize L2^2. The gradient of L2^2 wrt h is 2h.
    We project this onto the feasible set.
    """
    h = np.asarray(h, dtype=np.float64).copy()
    s = np.sum(h)
    if s <= 0:
        return h
    h = h / s
    
    for _ in range(iterations):
        c = _autoconv(h)
        max_c = np.max(c)
        
        # Active constraints: positions where c_k is close to max_c
        active = np.where(c >= max_c - 1e-8)[0]
        
        # Build the constraint Jacobian for active constraints
        # d(c_k)/dh_i = 2*h_{k-i}
        A = np.zeros((len(active), N))
        for idx, k in enumerate(active):
            for i in range(N):
                j = k - i
                if 0 <= j < N:
                    A[idx, i] = 2.0 * h[j]
        
        # Gradient of L2^2 is 2*h
        grad = 2.0 * h
        
        # Project gradient onto null space of A (active constraints)
        # and onto the hyperplane sum(h)=1
        # Use: find d that maximizes grad^T d subject to A*d = 0, sum(d) = 0
        # This is equivalent to projecting grad onto the orthogonal complement
        
        # Simple approach: use SVD to find null space
        try:
            U, S_vals, Vt = np.linalg.svd(A, full_matrices=True)
            # Null space of A is spanned by columns of Vt.T corresponding to zero singular values
            # For A (m x N), null space dimension is N - rank(A)
            rank = np.sum(S_vals > 1e-10)
            if rank < N:
                null_space = Vt[rank:].T  # (N, N-rank)
                
                # Also need sum(d) = 0
                # Project null_space onto sum=0 hyperplane
                ones = np.ones(N)
                # For each null space vector, subtract its mean
                null_space_centered = null_space - null_space.mean(axis=0)
                
                # Find direction in null space that maximizes grad^T d
                # d = null_space_centered @ c, maximize grad^T @ null_space_centered @ c
                # c = (null_space_centered^T @ grad) / ||...||
                proj = null_space_centered.T @ grad
                norm_proj = np.linalg.norm(proj)
                if norm_proj > 1e-12:
                    c_vec = proj / norm_proj
                    d = null_space_centered @ c_vec
                    # Normalize d
                    d_norm = np.linalg.norm(d)
                    if d_norm > 1e-12:
                        d = d / d_norm
                        # Line search
                        best_alpha = 0
                        best_l2 = np.sum(h**2)
                        for alpha in np.linspace(0, 0.5, 20):
                            h_try = h + alpha * d
                            h_try = np.maximum(0, h_try)
                            s_try = np.sum(h_try)
                            if s_try > 0:
                                h_try = h_try / s_try
                                l2_try = np.sum(h_try**2)
                                # Check if peak autoconv didn't increase too much
                                c_try = _autoconv(h_try)
                                max_c_try = np.max(c_try)
                                if max_c_try <= max_c * 1.01:
                                    if l2_try > best_l2:
                                        best_l2 = l2_try
                                        best_alpha = alpha
                        if best_alpha > 0:
                            h = h + best_alpha * d
                            h = np.maximum(0, h)
                            s = np.sum(h)
                            if s > 0:
                                h = h / s
                            continue
        except Exception:
            pass
        
        # Fallback: simple coordinate ascent
        improved = False
        for i in range(N):
            orig = h[i]
            for factor in [0.8, 0.9, 1.1, 1.2]:
                h_try = h.copy()
                h_try[i] = orig * factor
                h_try = np.maximum(0, h_try)
                s_try = np.sum(h_try)
                if s_try > 0:
                    h_try = h_try / s_try
                    c_try = _autoconv(h_try)
                    max_c_try = np.max(c_try)
                    if max_c_try <= max_c * 1.005:
                        l2_try = np.sum(h_try**2)
                        l2_orig = np.sum(h**2)
                        if l2_try > l2_orig:
                            h = h_try
                            improved = True
                            break
        if not improved:
            break
    
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    def update_best(h):
        nonlocal best_h, best_score
        h = np.asarray(h, dtype=np.float64)
        h = np.maximum(0, h)
        s = np.sum(h)
        if s <= 0:
            return
        h = h / s
        sc = _score_raw(h)
        if sc > best_score:
            best_score = sc
            best_h = h.copy()
    
    # Phase 1: Generate diverse initializations
    initializations = []
    
    # Uniform
    initializations.append(np.ones(N) / N)
    
    # Sparse peaks with various gaps
    for gap in [2, 3, 4, 5]:
        for start in range(gap):
            h = np.zeros(N)
            count = 0
            for i in range(start, N, gap):
                h[i] = 1.0
                count += 1
            if count > 0:
                initializations.append(h / np.sum(h))
    
    # Random sparse configurations
    np.random.seed(42)
    for trial in range(200):
        if time.time() > deadline - 60:
            break
        num_peaks = np.random.randint(3, 20)
        indices = np.random.choice(N, num_peaks, replace=False)
        h = np.zeros(N)
        h[indices] = np.random.uniform(0.5, 2.0, size=num_peaks)
        initializations.append(h / np.sum(h))
    
    # Structured: alternating high/low
    for period in [2, 3, 4]:
        for offset in range(period):
            h = np.zeros(N)
            for i in range(N):
                if i % period == offset:
                    h[i] = 1.0
            if np.sum(h) > 0:
                initializations.append(h / np.sum(h))
    
    # Phase 2: SLSQP optimization from each initialization
    for h_init in initializations:
        if time.time() > deadline - 20:
            break
        
        h_opt = _solve_slsqp(h_init, maxiter=300, ftol=1e-10)
        if h_opt is not None:
            update_best(h_opt)
    
    # Phase 3: Spectral refinement on best solutions
    if best_h is not None and time.time() < deadline - 10:
        h_refined = _spectral_refine(best_h, iterations=100)
        update_best(h_refined)
    
    # Phase 4: More SLSQP restarts with perturbation of best
    if time.time() < deadline - 10:
        for restart in range(30):
            if time.time() > deadline - 5:
                break
            
            h_pert = best_h.copy()
            # Random perturbation
            mask = np.random.random(N) < 0.3
            h_pert[mask] *= np.random.uniform(0.5, 2.0, size=np.sum(mask))
            h_pert = np.maximum(0, h_pert)
            s = np.sum(h_pert)
            if s > 0:
                h_pert = h_pert / s
            
            h_opt = _solve_slsqp(h_pert, maxiter=200, ftol=1e-10)
            if h_opt is not None:
                update_best(h_opt)
    
    # Phase 5: Final spectral refinement
    if best_h is not None and time.time() < deadline - 2:
        h_refined = _spectral_refine(best_h, iterations=50)
        update_best(h_refined)
    
    # Ensure valid output
    best_h = np.maximum(0, best_h)
    s = np.sum(best_h)
    if s > 0:
        best_h = best_h / s
    else:
        best_h = np.zeros(N)
        best_h[0] = 1.0
    
    return list(best_h)
