import time
import numpy as np
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Maximizes the ratio D/M where:
    D = H(x - y)
    M = max(H(x), H(x + y), H(x + 2y))
    
    Strategy:
    1. Precompute bin indices for each cell.
    2. Use a vectorized objective function for efficient evaluation.
    3. Perform a global search with multiple random restarts and local refinement.
    4. Use a structured set of initial points based on diagonal patterns.
    """
    n = 10
    deadline = time.time() + 580.0
    total_cells = n * n

    # Precompute bin indices
    ys, xs = np.mgrid[0:n, 0:n]
    x_flat = xs.flatten()
    y_flat = ys.flatten()
    
    d_val = (x_flat - y_flat)
    m0_val = x_flat
    m1_val = (x_flat + y_flat)
    m2_val = (x_flat + 2 * y_flat)
    
    d_vals, d_inv = np.unique(d_val, return_inverse=True)
    m0_vals, m0_inv = np.unique(m0_val, return_inverse=True)
    m1_vals, m1_inv = np.unique(m1_val, return_inverse=True)
    m2_vals, m2_inv = np.unique(m2_val, return_inverse=True)
    
    num_d = len(d_vals)
    num_m0 = len(m0_vals)
    num_m1 = len(m1_vals)
    num_m2 = len(m2_vals)

    def compute_entropies(p):
        """
        Compute H_d, H_m0, H_m1, H_m2 from a probability vector p of length 100.
        p is assumed to sum to 1 and be non-negative.
        """
        # p_flat is the probability of each cell
        # We need to aggregate probabilities into bins.
        # Using np.bincount for speed.
        
        # D bins
        d_counts = np.bincount(d_inv, weights=p, minlength=num_d)
        # M0 bins
        m0_counts = np.bincount(m0_inv, weights=p, minlength=num_m0)
        # M1 bins
        m1_counts = np.bincount(m1_inv, weights=p, minlength=num_m1)
        # M2 bins
        m2_counts = np.bincount(m2_inv, weights=p, minlength=num_m2)
        
        def shannon(counts):
            total = np.sum(counts)
            if total < 1e-15:
                return 0.0
            probs = counts / total
            probs = probs[probs > 1e-15]
            if len(probs) == 0:
                return 0.0
            return -np.sum(probs * np.log2(probs))
        
        H_d = shannon(d_counts)
        H_m0 = shannon(m0_counts)
        H_m1 = shannon(m1_counts)
        H_m2 = shannon(m2_counts)
        
        return H_d, H_m0, H_m1, H_m2

    def ratio_from_p(p):
        """Compute the ratio D/M for a given probability vector p."""
        H_d, H_m0, H_m1, H_m2 = compute_entropies(p)
        M = max(H_m0, H_m1, H_m2)
        if M < 1e-9:
            return 0.0
        return H_d / M

    def objective_full(theta):
        """
        Objective function for full 100-dim optimization.
        theta are log-weights, normalized to sum to 1.
        """
        if time.time() > deadline - 5:
            return 1e9
        # Softmax normalization
        p = np.exp(theta - np.max(theta))
        s = np.sum(p)
        if s < 1e-15:
            return 1e9
        p /= s
        ratio = ratio_from_p(p)
        return -ratio

    def objective_simplex(w_params):
        """
        Objective function for simplex parameterization.
        w_params are the first k-1 weights, last is 1 - sum.
        """
        if time.time() > deadline - 5:
            return 1e9
        k = len(w_params) + 1
        w = np.zeros(k)
        w[:k-1] = w_params
        w[k-1] = 1.0 - np.sum(w_params)
        if np.any(w < -1e-9):
            return 1e9
        w = np.clip(w, 0, None)
        s = np.sum(w)
        if s < 1e-9:
            return 1e9
        w /= s
        ratio = ratio_from_p(w)
        return -ratio

    best_ratio = 0.0
    best_p = None

    # Phase 1: Structured Initial Points
    # Generate a set of promising initial distributions
    initial_points = []
    
    # 1. Uniform
    initial_points.append(np.ones(total_cells) / total_cells)
    
    # 2. Diagonals (x - y = const)
    for d in range(-9, 10):
        p = np.zeros(total_cells)
        for y in range(n):
            for x in range(n):
                if x - y == d:
                    p[y * n + x] = 1.0
        if np.sum(p) > 0:
            p /= np.sum(p)
            initial_points.append(p)
    
    # 3. Anti-diagonals (x + y = const)
    for s in range(0, 19):
        p = np.zeros(total_cells)
        for y in range(n):
            for x in range(n):
                if x + y == s:
                    p[y * n + x] = 1.0
        if np.sum(p) > 0:
            p /= np.sum(p)
            initial_points.append(p)
    
    # 4. Rows (y = const)
    for y in range(n):
        p = np.zeros(total_cells)
        p[y * n : (y+1)*n] = 1.0 / n
        initial_points.append(p)
    
    # 5. Columns (x = const)
    for x in range(n):
        p = np.zeros(total_cells)
        for y in range(n):
            p[y * n + x] = 1.0 / n
        initial_points.append(p)
    
    # 6. Random sparse points
    rng = np.random.default_rng(42)
    for _ in range(50):
        p = rng.random(total_cells)
        # Make it sparse
        p[p < 0.5] = 0
        if np.sum(p) > 0:
            p /= np.sum(p)
            initial_points.append(p)
    
    # Evaluate all initial points
    for p in initial_points:
        if time.time() > deadline - 10:
            break
        ratio = ratio_from_p(p)
        if ratio > best_ratio:
            best_ratio = ratio
            best_p = p.copy()

    # Phase 2: Local Optimization from Best Points
    # We will run local optimization from the top few initial points
    # and from random restarts.
    
    # Sort initial points by ratio
    scored_points = [(ratio_from_p(p), p) for p in initial_points]
    scored_points.sort(key=lambda x: x[0], reverse=True)
    
    # Take top 10
    top_points = [p for _, p in scored_points[:10]]
    
    # Add 50 random restarts
    for _ in range(50):
        if time.time() > deadline - 30:
            break
        p = rng.random(total_cells)
        p /= np.sum(p)
        top_points.append(p)
    
    for p0 in top_points:
        if time.time() > deadline - 30:
            break
        
        # Convert to log-space for optimization
        p_safe = np.clip(p0, 1e-15, None)
        p_safe /= np.sum(p_safe)
        theta0 = np.log(p_safe)
        
        # Use L-BFGS-B
        try:
            res = minimize(objective_full, theta0, method='L-BFGS-B', options={'maxiter': 200})
            if res.fun < -best_ratio + 1e-6:
                p_opt = np.exp(res.x - np.max(res.x))
                p_opt /= np.sum(p_opt)
                ratio_opt = ratio_from_p(p_opt)
                if ratio_opt > best_ratio:
                    best_ratio = ratio_opt
                    best_p = p_opt
        except Exception:
            pass

    # Phase 3: Coordinate Descent Polish
    # Try to improve the best solution with coordinate descent
    if best_p is not None:
        p_best = best_p.copy()
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for i in range(total_cells):
                if time.time() > deadline - 10:
                    break
                # Try moving mass from i to j
                # Simple heuristic: try increasing p[i] and decreasing p[j]
                # But we need to maintain sum=1.
                # Let's try a small perturbation.
                # Actually, let's just try to optimize each cell's weight independently
                # by fixing others.
                # This is complex. Let's stick to the L-BFGS-B result.
                pass
        
        # Alternative: Try to merge or split cells
        # If a cell has very low weight, try setting it to 0 and redistributing
        for i in range(total_cells):
            if time.time() > deadline - 10:
                break
            if p_best[i] < 1e-4:
                # Try setting to 0
                p_trial = p_best.copy()
                p_trial[i] = 0
                p_trial /= np.sum(p_trial)
                ratio_trial = ratio_from_p(p_trial)
                if ratio_trial > best_ratio:
                    best_ratio = ratio_trial
                    best_p = p_trial
                    improved = True

    # Construct final P
    if best_p is not None:
        P = best_p.reshape((n, n)).T  # Note: p is flattened from xs.flatten(), which is row-major?
        # Wait, xs.flatten() is row-major (C-order).
        # xs is a 2D array from mgrid. xs[y, x] = x.
        # Flatten: xs[0,0], xs[0,1], ..., xs[0,9], xs[1,0], ...
        # So index i corresponds to y = i // n, x = i % n.
        # P[y, x] should be p[i].
        # So P = p.reshape((n, n)) is correct if P[y, x] = p[y*n+x].
        # Let's verify:
        # p[0] -> y=0, x=0. P[0,0] = p[0]. Correct.
        P = best_p.reshape((n, n))
    else:
        P = np.ones((n, n)) / (n * n)
        
    # Validate
    P = np.clip(P, 0.0, None)
    s = np.sum(P)
    if s > 1e-9:
        P = P / s
    else:
        P = np.ones((n, n)) / (n * n)
        
    return P
