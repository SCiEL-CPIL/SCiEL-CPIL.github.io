# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def _entropy_from_counts(counts):
    """Shannon entropy in bits from a count array (zeros ignored)."""
    total = counts.sum()
    if total <= 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    if len(probs) == 0:
        return 0.0
    return -np.sum(probs * np.log2(probs))

def _fast_ratio(P):
    """
    Fast computation of D/M using vectorized bincount-style approach.
    P[y, x] is the joint probability for x, y in 0..9.
    """
    if P.shape != (10, 10):
        return 0.0
    if not np.all(np.isfinite(P)):
        return 0.0
    if np.any(P < -1e-9):
        return 0.0
    total = P.sum()
    if total <= 1e-9:
        return 0.0

    P = P.copy()
    P[P < 0] = 0.0
    P = P / P.sum()

    y_idx, x_idx = np.indices((10, 10))
    flat_P = P.flatten()

    # x - y: range -9..9, offset by 9 to get 0..18
    v_minus1 = (x_idx - y_idx + 9).flatten()
    counts_minus1 = np.bincount(v_minus1, weights=flat_P, minlength=19)
    H_minus1 = _entropy_from_counts(counts_minus1)

    # x: range 0..9
    v_x = x_idx.flatten()
    counts_x = np.bincount(v_x, weights=flat_P, minlength=10)
    H_0 = _entropy_from_counts(counts_x)

    # x + y: range 0..18
    v_plus1 = (x_idx + y_idx).flatten()
    counts_plus1 = np.bincount(v_plus1, weights=flat_P, minlength=19)
    H_1 = _entropy_from_counts(counts_plus1)

    # x + 2y: range 0..27
    v_plus2 = (x_idx + 2 * y_idx).flatten()
    counts_plus2 = np.bincount(v_plus2, weights=flat_P, minlength=28)
    H_2 = _entropy_from_counts(counts_plus2)

    M = max(H_0, H_1, H_2)
    if M <= 1e-9:
        return 0.0

    return H_minus1 / M

def _nelder_mead_minimize(objective, x0, max_iter=500, tol=1e-10):
    """
    Nelder-Mead simplex method for minimization.
    Returns (x_best, f_best).
    """
    n = len(x0)
    # Initial simplex
    simplex = [np.array(x0, dtype=np.float64)]
    f_simplex = [objective(simplex[0])]
    
    # Create other vertices
    for i in range(n):
        xi = x0.copy()
        step = 0.1 if abs(x0[i]) < 1e-6 else 0.1 * abs(x0[i])
        xi[i] += step
        simplex.append(xi)
        f_simplex.append(objective(xi))
    
    alpha = 1.0   # reflection
    gamma = 2.0   # expansion
    rho = 0.5     # contraction
    sigma = 0.5   # shrink
    
    for iteration in range(max_iter):
        # Sort by function value
        order = np.argsort(f_simplex)
        simplex = [simplex[i] for i in order]
        f_simplex = [f_simplex[i] for i in order]
        
        # Check convergence
        if abs(f_simplex[-1] - f_simplex[0]) < tol:
            break
        
        # Centroid of all but worst point
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflect
        x_r = centroid + alpha * (centroid - simplex[-1])
        f_r = objective(x_r)
        
        if f_r < f_simplex[0]:
            # Try expansion
            x_e = centroid + gamma * (x_r - centroid)
            f_e = objective(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                f_simplex[-1] = f_e
            else:
                simplex[-1] = x_r
                f_simplex[-1] = f_r
        elif f_r < f_simplex[-2]:
            simplex[-1] = x_r
            f_simplex[-1] = f_r
        else:
            # Contraction
            x_c = centroid + rho * (simplex[-1] - centroid)
            f_c = objective(x_c)
            if f_c < f_simplex[-1]:
                simplex[-1] = x_c
                f_simplex[-1] = f_c
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    f_simplex[i] = objective(simplex[i])
    
    # Return best point
    best_idx = np.argmin(f_simplex)
    return simplex[best_idx], f_simplex[best_idx]

def _structured_polish(P, deadline, max_iters=1000):
    """
    Coordinate descent that moves mass between cells sharing the same
    values for x, x+y, and x+2y to preserve H0, H1, H2 while maximizing H-1.
    """
    n = 10
    y_idx, x_idx = np.indices((n, n))
    
    # Precompute keys for the constraints
    # Key = (x, x+y, x+2y)
    # If two cells have the same key, moving mass between them doesn't change H0, H1, H2.
    keys = []
    for y in range(n):
        for x in range(n):
            keys.append((x, x+y, x+2*y))
    keys = np.array(keys)
    
    # Group indices by key
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    groups = {}
    for i, k in enumerate(inverse):
        if k not in groups:
            groups[k] = []
        groups[k].append(i)
    
    # Only groups with size > 1 are useful for moving mass
    movable_groups = [idxs for idxs in groups.values() if len(idxs) > 1]
    
    if not movable_groups:
        return P

    P = P.copy()
    P[P < 0] = 0.0
    P = P / P.sum()
    
    # Precompute x-y values for each index
    xy_diffs = (x_idx - y_idx).flatten()
    
    best_score = _fast_ratio(P)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        improved = False
        # Shuffle groups to avoid bias
        np.random.shuffle(movable_groups)
        
        for idxs in movable_groups:
            if time.time() > deadline:
                break
            
            # Get current probabilities for these indices
            p_vals = P.flatten()[idxs]
            if p_vals.sum() <= 1e-15:
                continue
                
            # Try moving mass from the cell with lowest p to the cell with highest p?
            # No, we want to maximize H(x-y).
            # H(x-y) is maximized when mass is spread evenly across x-y values.
            # Within a group (same x, x+y, x+2y), the x-y values might differ.
            # Let's check the x-y values for this group.
            diffs = xy_diffs[idxs]
            
            # If all diffs are the same, moving mass doesn't change H(x-y).
            if len(np.unique(diffs)) == 1:
                continue
                
            # We want to equalize the marginal distribution of x-y.
            # This is a global objective, so local moves are tricky.
            # However, we can use a gradient-free approach:
            # Try a small perturbation that moves mass from a cell with 'rare' x-y diff
            # to a cell with 'common' x-y diff? No, that would decrease entropy.
            # We want to move mass FROM common x-y diff TO rare x-y diff.
            
            # Let's compute the current counts of x-y diffs globally.
            # This is expensive inside the loop. Let's approximate or just do a simple hill climb.
            # Objective: Maximize H(x-y).
            # Let's just try to make P more uniform within the group? No.
            
            # Alternative: Just use a random perturbation within the group and accept if score improves.
            # Since the group is small (max 10), we can do a 1D search for the optimal distribution
            # within the group that maximizes the global H(x-y).
            
            # Actually, let's just do a simple coordinate ascent on the log-probs of the group.
            # Or simpler: try moving a small epsilon from the index with the smallest contribution to H(x-y)
            # to the one with the largest potential.
            
            # Let's just do a random swap within the group.
            i, j = np.random.choice(len(idxs), 2, replace=False)
            idx_i, idx_j = idxs[i], idxs[j]
            
            p_i = P.flatten()[idx_i]
            p_j = P.flatten()[idx_j]
            
            if p_i < 1e-12 and p_j < 1e-12:
                continue
                
            # Try moving mass from i to j
            # We want to see if this increases H(x-y).
            # dH/dp is complex. Let's just evaluate.
            
            # To save time, we can estimate the change in H(x-y).
            # H = - sum p_v log p_v
            # If we move epsilon from u to v:
            # Change in H approx -epsilon * (log(p_v+eps) - log(p_v)) + epsilon * (log(p_u-eps) - log(p_u))
            # This is getting complicated. Let's just do a small step and check.
            
            # Step size
            step = 0.01 * max(p_i, p_j)
            if step > 1e-6:
                new_P = P.flatten().copy()
                new_P[idx_i] -= step
                new_P[idx_j] += step
                if new_P[idx_i] < 0:
                    new_P[idx_i] = 0
                    new_P[idx_j] += new_P[idx_i] # renormalize roughly? No, just skip if invalid
                else:
                    new_P = new_P / new_P.sum()
                    new_P = new_P.reshape((n, n))
                    new_score = _fast_ratio(new_P)
                    if new_score > best_score:
                        P = new_P
                        best_score = new_score
                        improved = True
                        
        if not improved:
            # Try smaller steps or break
            # If no improvement in a full pass, we might be at a local optimum
            # Let's break to save time, or reduce step size?
            # For now, break.
            break
            
    return P

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Returns a 10x10 array P of finite weights, P[y, x] for integers x, y in 0..9.
    Optimizes the ratio D/M where D = H_{-1} and M = max(H_0, H_1, H_2).
    Uses a parametric family P(x,y) ~ exp(alpha*x + beta*y) optimized by Nelder-Mead,
    followed by structured coordinate descent polish.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # safety margin

    n = 10
    rng = np.random.RandomState(42)
    y_idx, x_idx = np.indices((n, n))
    x_flat = x_idx.flatten()
    y_flat = y_idx.flatten()

    # Parametric family: P(x,y) proportional to exp(alpha * x + beta * y)
    # This is a 2D parameter space (alpha, beta).
    # We can also add a gamma * (x-y) term? No, that's what we're optimizing.
    # Let's stick to alpha, beta.
    
    def param_to_P(params):
        alpha, beta = params
        # log P(x,y) = alpha * x + beta * y
        log_p = alpha * x_flat + beta * y_flat
        log_p -= np.max(log_p)
        p = np.exp(log_p)
        p /= p.sum()
        return p.reshape((n, n))

    def objective(params):
        P = param_to_P(params)
        return -_fast_ratio(P)

    # Initial guesses for (alpha, beta)
    # Uniform is (0,0).
    # We want to maximize H(x-y).
    # x-y is maximized entropy when x and y are independent and uniform?
    # If x,y independent uniform, P(x,y) = 1/100. alpha=0, beta=0.
    # But maybe tilting helps?
    
    best_P = np.ones((n, n)) / (n*n)
    best_score = _fast_ratio(best_P)
    
    # Grid search over alpha, beta to find good starting points
    alphas = np.linspace(-1.5, 1.5, 20)
    betas = np.linspace(-1.5, 1.5, 20)
    
    for a in alphas:
        for b in betas:
            if time.time() > deadline - 10:
                break
            P = param_to_P([a, b])
            score = _fast_ratio(P)
            if score > best_score:
                best_score = score
                best_P = P.copy()
                # Refine around this point with Nelder-Mead
                x0 = np.array([a, b])
                try:
                    x_opt, f_opt = _nelder_mead_minimize(objective, x0, max_iter=200, tol=1e-12)
                    P_opt = param_to_P(x_opt)
                    score_opt = _fast_ratio(P_opt)
                    if score_opt > best_score:
                        best_score = score_opt
                        best_P = P_opt.copy()
                except Exception:
                    pass

    # Apply structured polish
    if time.time() < deadline - 5:
        best_P = _structured_polish(best_P, deadline)
        best_score = _fast_ratio(best_P)
        
    # Final cleanup
    best_P[best_P < 0] = 0
    total = best_P.sum()
    if total > 1e-9:
        best_P = best_P / total
    else:
        best_P = np.ones((n, n)) / (n*n)

    return best_P

# EVOLVE-BLOCK-END
