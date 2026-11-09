# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Maximize the arithmetic Kakeya entropy ratio D/M on a 10x10 grid.
    D = H_{-1} (entropy of x - y), M = max(H_0, H_1, H_2).
    Uses a combinatorial greedy approach to select and weight grid cells.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin for 630s limit

    # Precompute grid indices
    X, Y = np.meshgrid(np.arange(n), np.arange(n), indexing='xy')
    # X[y, x] = x, Y[y, x] = y

    # Precompute value arrays for each slope
    val_minus1 = X - Y      # x - y, range -9..9
    val_0 = X               # x, range 0..9
    val_1 = X + Y           # x + y, range 0..18
    val_2 = X + 2 * Y       # x + 2y, range 0..27

    def compute_entropy_from_values(values, weights):
        """Compute Shannon entropy (bits) of the distribution defined by pooling weights by value."""
        unique_vals, inverse = np.unique(values, return_inverse=True)
        counts = np.zeros(len(unique_vals))
        np.add.at(counts, inverse, weights)
        total = counts.sum()
        if total <= 1e-12:
            return 0.0
        probs = counts / total
        probs = probs[probs > 1e-15]
        if len(probs) == 0:
            return 0.0
        return -np.sum(probs * np.log2(probs))

    def compute_ratio(P):
        """Compute D/M given probability matrix P."""
        D = compute_entropy_from_values(val_minus1, P)
        H0 = compute_entropy_from_values(val_0, P)
        H1 = compute_entropy_from_values(val_1, P)
        H2 = compute_entropy_from_values(val_2, P)
        M = max(H0, H1, H2)
        if M < 1e-9:
            return -1.0
        return D / M

    # Strategy: Greedy set-cover style approach
    # We want to maximize D - M (or D/M), which is submodular-like in the support set.
    # Approach: Start with empty support, iteratively add cells that maximize marginal gain in (D - M).
    # Then optimize weights on the selected support.
    
    best_P = None
    best_score = -1.0

    def optimize_weights_on_support(support_mask, max_iter=100):
        """Optimize weights on cells where support_mask is True using coordinate descent."""
        # Get indices of supported cells
        idx = np.argwhere(support_mask)
        if len(idx) == 0:
            return None, -1.0
        
        k = len(idx)
        # Initial weights: uniform
        w = np.ones(k) / k
        
        # Precompute which value each cell contributes to for each slope
        # For each cell i in support, get its (y, x)
        ys = idx[:, 0]
        xs = idx[:, 1]
        
        v_m1 = xs - ys
        v_0 = xs
        v_1 = xs + ys
        v_2 = xs + 2 * ys
        
        # Unique values and their counts
        # For efficiency, we'll use bincount
        max_v_m1 = 9
        max_v_0 = 9
        max_v_1 = 18
        max_v_2 = 27
        
        # Offset to make indices non-negative
        off_m1 = 9  # v_m1 in -9..9 -> 0..18
        off_0 = 0
        off_1 = 0
        off_2 = 0
        
        def get_entropies(w):
            """Compute D and M given weights w."""
            # Normalize
            w_norm = w / w.sum()
            
            # Count for each value
            cnt_m1 = np.bincount(v_m1 + off_m1, weights=w_norm, minlength=19)
            cnt_0 = np.bincount(v_0 + off_0, weights=w_norm, minlength=10)
            cnt_1 = np.bincount(v_1 + off_1, weights=w_norm, minlength=19)
            cnt_2 = np.bincount(v_2 + off_2, weights=w_norm, minlength=28)
            
            def entropy_from_counts(cnts):
                probs = cnts[cnts > 1e-15]
                if len(probs) == 0:
                    return 0.0
                return -np.sum(probs * np.log2(probs))
            
            D = entropy_from_counts(cnt_m1)
            H0 = entropy_from_counts(cnt_0)
            H1 = entropy_from_counts(cnt_1)
            H2 = entropy_from_counts(cnt_2)
            M = max(H0, H1, H2)
            return D, M, D / M if M > 1e-9 else -1.0
        
        # Coordinate descent
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(k):
                # Try adjusting w[i]
                # Simple approach: try a few multipliers
                best_w_i = w[i]
                best_ratio = compute_ratio(P_from_weights(w, support_mask))[1] if False else None
                # Actually, let's just do a simple local search
                # Try scaling w[i] by factors
                current_D, current_M, current_ratio = get_entropies(w)
                for factor in [0.5, 0.7, 1.3, 2.0, 0.3, 3.0]:
                    new_w = w.copy()
                    new_w[i] = w[i] * factor
                    new_w = np.maximum(new_w, 1e-10)
                    _, _, new_ratio = get_entropies(new_w)
                    if new_ratio > current_ratio + 1e-8:
                        best_w_i = new_w[i]
                        current_ratio = new_ratio
                        improved = True
                w[i] = best_w_i
            
            if not improved:
                break
        
        # Build P
        P = np.zeros((n, n))
        P[ys, xs] = w / w.sum()
        score = compute_ratio(P)
        return P, score

    # Phase 1: Greedy support selection
    # Start with all cells, then iteratively remove cells that decrease the score least
    # Or start empty and add cells that increase the score most
    
    # Let's try: start with a good initial support (e.g., all cells), then prune
    # Actually, let's try multiple strategies:
    
    # Strategy A: Start with uniform, then do local search on weights
    P_uniform = np.ones((n, n)) / (n * n)
    score_uniform = compute_ratio(P_uniform)
    if score_uniform > best_score:
        best_score = score_uniform
        best_P = P_uniform.copy()
    
    # Strategy B: Greedy add from empty
    # This might be slow, let's try a smarter initialization
    # Based on the insight that we want to boost corners for D, let's try:
    # P[y, x] proportional to some function that's high at corners
    
    # Try a parametric family: P[y, x] ∝ (1 + a*(x-4.5)^2 + b*(y-4.5)^2 + c*x*y)
    # Optimize a, b, c
    
    Xc = X - 4.5
    Yc = Y - 4.5
    
    def parametric_P(a, b, c):
        """Generate P from parameters a, b, c."""
        logP = np.log(np.maximum(1 + a * Xc**2 + b * Yc**2 + c * Xc * Yc, 1e-10))
        logP -= np.max(logP)
        P = np.exp(logP)
        P /= P.sum()
        return P
    
    def parametric_score(params):
        a, b, c = params
        P = parametric_P(a, b, c)
        return -compute_ratio(P)
    
    # Optimize parameters
    inits = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [1.0, 1.0, 0.0],
        [-0.5, -0.5, 0.0],
        [0.0, 0.0, 0.5],
        [0.0, 0.0, -0.5],
        [0.5, 0.5, 0.5],
        [0.5, 0.5, -0.5],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    
    for init in inits:
        if time.time() > deadline - 10:
            break
        try:
            res = minimize(parametric_score, init, method='Nelder-Mead', 
                          options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < 1e5:
                a, b, c = res.x
                P = parametric_P(a, b, c)
                score = compute_ratio(P)
                if score > best_score:
                    best_score = score
                    best_P = P.copy()
        except Exception:
            pass
    
    # Strategy C: Direct cell weight optimization
    # Use a more flexible ansatz: P[y, x] ∝ exp(sum_i theta_i * phi_i(x, y))
    # where phi_i are basis functions
    
    # Basis: 1, x, y, x^2, y^2, xy, x^3, y^3, x^2y, xy^2, x^4, y^4
    # This is getting complex. Let's try a simpler approach:
    # Optimize each cell's weight independently with coordinate descent
    
    # Start from best_P found so far
    if best_P is not None:
        P_opt = best_P.copy()
        # Coordinate descent on each cell
        for iteration in range(20):
            if time.time() > deadline - 10:
                break
            improved = False
            for y in range(n):
                for x in range(n):
                    if time.time() > deadline - 10:
                        break
                    old_val = P_opt[y, x]
                    if old_val < 1e-10:
                        continue
                    # Try different values for P[y, x]
                    best_val = old_val
                    best_score_local = compute_ratio(P_opt)
                    for factor in [0.5, 0.7, 1.3, 2.0, 0.3, 3.0, 0.1, 5.0]:
                        new_P = P_opt.copy()
                        new_P[y, x] = old_val * factor
                        # Renormalize
                        new_P /= new_P.sum()
                        score = compute_ratio(new_P)
                        if score > best_score_local + 1e-8:
                            best_score_local = score
                            best_val = old_val * factor
                            improved = True
                    P_opt[y, x] = best_val
                    P_opt /= P_opt.sum()
            if not improved:
                break
        
        final_score = compute_ratio(P_opt)
        if final_score > best_score:
            best_score = final_score
            best_P = P_opt.copy()
    
    # Final validation
    if best_P is None:
        best_P = np.ones((n, n)) / (n * n)
    
    best_P = np.clip(best_P, 0, None)
    total = best_P.sum()
    if total <= 1e-9:
        best_P = np.ones((n, n)) / (n * n)
    else:
        best_P /= total
    
    return best_P

# EVOLVE-BLOCK-END
