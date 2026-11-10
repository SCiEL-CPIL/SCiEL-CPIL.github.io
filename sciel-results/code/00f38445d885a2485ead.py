import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _compute_entropies(P):
    """
    Computes D = H(x-y) and M = max(H(x), H(x+y), H(x+2y)) for a 10x10 grid P[y, x].
    Returns (D, M).
    """
    n = 10
    ys = np.arange(n)
    xs = np.arange(n)
    Y, X = np.meshgrid(ys, xs, indexing='ij')
    
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    P_flat = P.flatten()
    
    # D: distribution of x - y
    val_d = X_flat - Y_flat
    # Values in [-9, 9]
    counts_d = np.zeros(19)
    for v in range(-9, 10):
        mask = (val_d == v)
        counts_d[v + 9] = np.sum(P_flat[mask])
        
    # H0: distribution of x
    val_h0 = X_flat
    counts_h0 = np.zeros(n)
    for v in range(n):
        mask = (val_h0 == v)
        counts_h0[v] = np.sum(P_flat[mask])
        
    # H1: distribution of x + y
    val_h1 = X_flat + Y_flat
    # Values in [0, 18]
    counts_h1 = np.zeros(19)
    for v in range(19):
        mask = (val_h1 == v)
        counts_h1[v] = np.sum(P_flat[mask])
        
    # H2: distribution of x + 2y
    val_h2 = X_flat + 2*Y_flat
    # Values in [0, 27]
    counts_h2 = np.zeros(28)
    for v in range(28):
        mask = (val_h2 == v)
        counts_h2[v] = np.sum(P_flat[mask])

    def shannon_entropy(counts):
        p = counts[counts > 1e-15]
        if len(p) == 0:
            return 0.0
        p = p / np.sum(p)
        return -np.sum(p * np.log2(p))
    
    D = shannon_entropy(counts_d)
    H0 = shannon_entropy(counts_h0)
    H1 = shannon_entropy(counts_h1)
    H2 = shannon_entropy(counts_h2)
    
    M = max(H0, H1, H2)
    
    return D, M

def _get_P_from_params(params):
    """
    Construct P from quadratic parameters.
    params: [A11, A12, A22, b1, b2, c]
    W[y, x] = A11*x^2 + 2*A12*x*y + A22*y^2 + b1*x + b2*y + c
    """
    A11, A12, A22, b1, b2, c = params
    n = 10
    xs = np.arange(n)
    ys = np.arange(n)
    Y, X = np.meshgrid(ys, xs, indexing='ij')
    
    W = A11 * X**2 + 2 * A12 * X * Y + A22 * Y**2 + b1 * X + b2 * Y + c
    
    # Softmax
    W_shifted = W - np.max(W)
    P_flat = np.exp(W_shifted)
    P_flat /= np.sum(P_flat)
    P = P_flat.reshape(10, 10)
    return P

def _objective(params):
    P = _get_P_from_params(params)
    D, M = _compute_entropies(P)
    if M < 1e-9 or D < 1e-9:
        return 1e10
    return -D / M

def _optimize_sparse_support(support_indices, time_limit):
    """
    Optimize weights on a given set of support indices.
    support_indices: list of (y, x) tuples.
    Returns best P and score.
    """
    n_support = len(support_indices)
    if n_support == 0:
        return None, 0.0
    
    # Initialize weights uniformly
    w_init = np.ones(n_support) / n_support
    
    def obj_sparse(w):
        P = np.zeros((10, 10))
        # Normalize w
        w_norm = w / np.sum(w)
        for i, (y, x) in enumerate(support_indices):
            P[y, x] = w_norm[i]
        D, M = _compute_entropies(P)
        if M < 1e-9 or D < 1e-9:
            return 1e10
        return -D / M
    
    bounds = [(1e-6, 1.0)] * n_support
    
    try:
        res = minimize(obj_sparse, w_init, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 100, 'ftol': 1e-12})
        if res.fun < 1e9:
            w_opt = res.x
            w_opt = w_opt / np.sum(w_opt)
            P = np.zeros((10, 10))
            for i, (y, x) in enumerate(support_indices):
                P[y, x] = w_opt[i]
            D, M = _compute_entropies(P)
            if M > 1e-9 and D > 1e-9:
                score = D / M
                return P, score
    except Exception:
        pass
    return None, 0.0

def arithmetic_kakeya_distribution() -> np.ndarray:
    start_time = time.time()
    timeout = 630.0
    safety_margin = 10.0
    deadline = start_time + timeout - safety_margin
    
    best_P = np.ones((10, 10)) / 100.0
    best_score = 0.0
    
    # 1. Structured Starts for L-BFGS-B
    bounds = [
        (-5.0, 5.0),   # A11
        (-5.0, 5.0),   # A12
        (-5.0, 5.0),   # A22
        (-5.0, 5.0),   # b1
        (-5.0, 5.0),   # b2
        (0.0, 0.0)     # c
    ]
    
    init_points = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # Uniform
        [2.0, -2.0, 2.0, 0.0, 0.0, 0.0], # Spread x-y
        [2.0, -2.0, 2.0, 1.0, -1.0, 0.0], # Shifted
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],   # Spread x+y
        [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],   # Mild spread
        [3.0, -3.0, 3.0, 0.0, 0.0, 0.0],  # Strong spread x-y
        [1.0, -0.5, 1.0, 0.0, 0.0, 0.0],  # Moderate
        [-2.0, 2.0, -2.0, 0.0, 0.0, 0.0], # Concentrate x-y
        [2.0, 2.0, 2.0, 0.0, 0.0, 0.0],   # Concentrate x+y
    ]
    
    rng = np.random.default_rng(42)
    for _ in range(30):
        p = rng.uniform([-5, -5, -5, -5, -5, 0], [5, 5, 5, 5, 5, 0])
        init_points.append(p)
    
    for p_init in init_points:
        if time.time() > deadline:
            break
        try:
            res = minimize(_objective, p_init, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.fun < 1e9:
                P = _get_P_from_params(res.x)
                D, M = _compute_entropies(P)
                if M > 1e-9 and D > 1e-9:
                    score = D / M
                    if score > best_score:
                        best_score = score
                        best_P = P.copy()
        except Exception:
            continue

    # 2. Differential Evolution
    if time.time() < deadline - 60:
        try:
            res_de = differential_evolution(_objective, bounds, seed=42, maxiter=100, popsize=15, tol=1e-6, polish=True)
            if res_de.fun < 1e9:
                P = _get_P_from_params(res_de.x)
                D, M = _compute_entropies(P)
                if M > 1e-9 and D > 1e-9:
                    score = D / M
                    if score > best_score:
                        best_score = score
                        best_P = P.copy()
        except Exception:
            pass

    # 3. Sparse Support Optimization
    # Try various structured supports
    supports = []
    
    # Diagonals x-y = k
    for k in range(-9, 10):
        supp = [(y, x) for y in range(10) for x in range(10) if x - y == k]
        if supp:
            supports.append(supp)
            
    # Anti-diagonals x+y = k
    for k in range(0, 19):
        supp = [(y, x) for y in range(10) for x in range(10) if x + y == k]
        if supp:
            supports.append(supp)
            
    # Rows and Columns
    for i in range(10):
        supports.append([(i, x) for x in range(10)])
        supports.append([(y, i) for y in range(10)])
        
    # Random supports of size 10, 15, 20
    for size in [10, 15, 20]:
        for _ in range(5):
            indices = rng.choice(100, size=size, replace=False)
            supp = [(i // 10, i % 10) for i in indices]
            supports.append(supp)
    
    # Also try subsets of the best continuous support? 
    # For now, just iterate through these.
    
    for supp in supports:
        if time.time() > deadline:
            break
        if not supp:
            continue
        # Limit support size to avoid slow optimization
        if len(supp) > 30:
            # Sample from large supports
            indices = rng.choice(len(supp), size=20, replace=False)
            supp = [supp[i] for i in indices]
            
        P_cand, score_cand = _optimize_sparse_support(supp, deadline - time.time())
        if P_cand is not None and score_cand > best_score:
            best_score = score_cand
            best_P = P_cand

    # Final Check and Cleanup
    best_P = np.asarray(best_P, dtype=float)
    if best_P.shape != (10, 10):
        best_P = np.ones((10, 10)) / 100.0
        
    best_P = np.clip(best_P, 0, None)
    s = np.sum(best_P)
    if s <= 1e-9:
        best_P = np.ones((10, 10)) / 100.0
    else:
        best_P = best_P / s
        
    if not np.all(np.isfinite(best_P)):
        best_P = np.ones((10, 10)) / 100.0

    return best_P
