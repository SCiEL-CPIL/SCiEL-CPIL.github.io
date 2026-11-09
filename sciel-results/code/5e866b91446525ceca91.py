import numpy as np
import time
import random

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya entropy ratio on a 10x10 grid.
    
    Maximize D / M where:
    D = H(x - y)  (entropy of x - y distribution)
    M = max(H(x), H(x+y), H(x+2y))
    
    Returns:
        P: 10x10 array where P[y, x] is the joint probability.
    """
    n = 10
    start_time = time.time()
    timeout = 580  # Safety margin for 630s limit
    
    # Precompute projection indices
    y_grid, x_grid = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    
    # x - y + 9: range 0..18
    idx_neg1 = (x_grid - y_grid + 9).flatten().astype(int)
    # x: range 0..9
    idx_0 = x_grid.flatten().astype(int)
    # x + y: range 0..18
    idx_1 = (x_grid + y_grid).flatten().astype(int)
    # x + 2*y: range 0..27
    idx_2 = (x_grid + 2 * y_grid).flatten().astype(int)
    
    size_neg1 = 19
    size_0 = 10
    size_1 = 19
    size_2 = 28

    def compute_ratio(P):
        """Compute H(x-y) / max(H(x), H(x+y), H(x+2y))"""
        p_flat = P.flatten()
        p_flat = np.clip(p_flat, 0, None)
        total = np.sum(p_flat)
        if total < 1e-9:
            return 0.0
        p_flat = p_flat / total
        
        def entropy_from_counts(counts):
            nonzero = counts[counts > 1e-15]
            if len(nonzero) == 0:
                return 0.0
            return -np.sum(nonzero * np.log2(nonzero))
        
        c_neg1 = np.bincount(idx_neg1, weights=p_flat, minlength=size_neg1)
        c_0 = np.bincount(idx_0, weights=p_flat, minlength=size_0)
        c_1 = np.bincount(idx_1, weights=p_flat, minlength=size_1)
        c_2 = np.bincount(idx_2, weights=p_flat, minlength=size_2)
        
        H_neg1 = entropy_from_counts(c_neg1)
        H_0 = entropy_from_counts(c_0)
        H_1 = entropy_from_counts(c_1)
        H_2 = entropy_from_counts(c_2)
        
        M = max(H_0, H_1, H_2)
        if M < 1e-9:
            return 0.0
        
        return H_neg1 / M

    # Baseline: uniform distribution
    P_best = np.ones((n, n)) / (n * n)
    best_score = compute_ratio(P_best)
    
    # Strategy 1: Try sparse permutation-based initializations
    # These naturally make H(x-y) high (diagonal-like) while keeping H(x) low
    
    def optimize_coordinate_descent(P_init, max_iters=5000):
        """Coordinate descent optimization on the simplex."""
        P = P_init.copy()
        P = np.clip(P, 0, None)
        s = np.sum(P)
        if s < 1e-9:
            return P, 0.0
        P = P / s
        
        current_score = compute_ratio(P)
        
        for i in range(max_iters):
            if time.time() - start_time > timeout - 10:
                break
            
            # Pick two cells to transfer probability mass
            y1, x1 = random.randint(0, n-1), random.randint(0, n-1)
            y2, x2 = random.randint(0, n-1), random.randint(0, n-1)
            if y1 == y2 and x1 == x2:
                continue
            
            # Determine transfer amount
            # Try several step sizes
            base_delta = random.uniform(0.005, 0.05)
            
            # Direction 1: move mass from (y2,x2) to (y1,x1)
            delta = base_delta * min(P[y2, x2], 0.1)
            if P[y2, x2] < delta * 0.5:
                delta = P[y2, x2] * 0.4
            if delta < 1e-8:
                continue
            
            P_new = P.copy()
            P_new[y1, x1] += delta
            P_new[y2, x2] -= delta
            
            new_score = compute_ratio(P_new)
            if new_score > current_score:
                P = P_new
                current_score = new_score
            else:
                # Try reverse direction
                P_new = P.copy()
                P_new[y1, x1] -= delta
                P_new[y2, x2] += delta
                if P_new[y1, x1] < 0:
                    continue
                new_score = compute_ratio(P_new)
                if new_score > current_score:
                    P = P_new
                    current_score = new_score
        
        return P, current_score

    # Generate diverse initializations
    initializations = []
    
    # 1. Uniform
    initializations.append(P_best.copy())
    
    # 2. Random permutations (diagonal-like structures)
    for _ in range(30):
        perm = np.random.permutation(n)
        P_init = np.zeros((n, n))
        for x in range(n):
            y = perm[x]
            P_init[y, x] = 1.0 / n
        initializations.append(P_init)
    
    # 3. Anti-diagonal (maximizes x-y spread)
    P_anti = np.zeros((n, n))
    for x in range(n):
        y = n - 1 - x
        P_anti[y, x] = 1.0 / n
    initializations.append(P_anti)
    
    # 4. Random sparse distributions
    for _ in range(20):
        P_sparse = np.zeros((n, n))
        num_nonzero = random.randint(5, 15)
        indices = np.random.choice(n * n, size=num_nonzero, replace=False)
        weights = np.random.dirichlet(np.ones(num_nonzero))
        for idx, w in zip(indices, weights):
            y, x = divmod(idx, n)
            P_sparse[y, x] = w
        initializations.append(P_sparse)
    
    # 5. Structured patterns: bands around diagonals
    for offset in range(-3, 4):
        P_band = np.zeros((n, n))
        count = 0
        for x in range(n):
            y = x + offset
            if 0 <= y < n:
                P_band[y, x] = 1.0
                count += 1
        if count > 0:
            P_band /= count
            initializations.append(P_band)
    
    # 6. Concentrated distributions (few cells with high weight)
    for _ in range(15):
        P_conc = np.zeros((n, n))
        num_cells = random.randint(3, 8)
        indices = np.random.choice(n * n, size=num_cells, replace=False)
        weights = np.random.dirichlet(np.ones(num_cells) * 0.5)  # More concentrated
        for idx, w in zip(indices, weights):
            y, x = divmod(idx, n)
            P_conc[y, x] = w
        initializations.append(P_conc)

    # Optimize each initialization
    for init_idx, P_init in enumerate(initializations):
        if time.time() - start_time > timeout - 30:
            break
        
        P_opt, score = optimize_coordinate_descent(P_init, max_iters=3000)
        if score > best_score:
            best_score = score
            P_best = P_opt
    
    # Strategy 2: Local refinement with adaptive step sizes
    P_current = P_best.copy()
    current_score = best_score
    
    # Multiple passes with decreasing step sizes
    for pass_num, step_scale in enumerate([0.1, 0.05, 0.02, 0.01, 0.005]):
        if time.time() - start_time > timeout - 20:
            break
        
        for i in range(2000):
            if time.time() - start_time > timeout - 10:
                break
            
            y1, x1 = random.randint(0, n-1), random.randint(0, n-1)
            y2, x2 = random.randint(0, n-1), random.randint(0, n-1)
            if y1 == y2 and x1 == x2:
                continue
            
            delta = step_scale * random.uniform(0.5, 1.5) * max(P_current[y2, x2], 0.01)
            if P_current[y2, x2] < delta * 0.5:
                delta = P_current[y2, x2] * 0.4
            if delta < 1e-9:
                continue
            
            P_new = P_current.copy()
            P_new[y1, x1] += delta
            P_new[y2, x2] -= delta
            
            new_score = compute_ratio(P_new)
            if new_score > current_score:
                P_current = P_new
                current_score = new_score
            else:
                P_new = P_current.copy()
                P_new[y1, x1] -= delta
                P_new[y2, x2] += delta
                if P_new[y1, x1] < 0:
                    continue
                new_score = compute_ratio(P_new)
                if new_score > current_score:
                    P_current = P_new
                    current_score = new_score
        
        if current_score > best_score:
            best_score = current_score
            P_best = P_current

    # Strategy 3: Try some structured constructions
    # The key insight: we want H(x-y) high but H(x), H(x+y), H(x+2y) low
    # x-y has range -9..9 (19 values), x has range 0..9 (10 values)
    # So even uniform gives H(x-y) up to ~4.25 bits, H(x) up to 3.32 bits
    # Ratio for uniform is about 4.25/3.32 ≈ 1.28
    
    # To improve: concentrate x-y values while spreading x values less
    # Or: make x-y as uniform as possible while making x, x+y, x+2y concentrated
    
    # Try: P[y,x] proportional to something that makes x-y uniform
    # If P[y,x] = 1/100 for all, x-y is roughly uniform over -9..9
    # But x is uniform over 0..9, so H(x) = log2(10) ≈ 3.32
    # H(x-y) for uniform is slightly less than log2(19) ≈ 4.25 due to edge effects
    
    # Better: try to make x-y more uniform than x
    # One approach: P[y,x] = c * f(x-y) where f is chosen to make x-y uniform
    # But this couples x and y in a specific way
    
    # Try diagonal bands with varying weights
    for trial in range(10):
        if time.time() - start_time > timeout - 15:
            break
        
        P_trial = np.zeros((n, n))
        # Assign weights based on x-y value
        for y in range(n):
            for x in range(n):
                d = x - y
                # Try different weighting schemes
                if trial % 3 == 0:
                    # Uniform over x-y
                    P_trial[y, x] = 1.0
                elif trial % 3 == 1:
                    # Emphasize middle x-y values
                    P_trial[y, x] = 1.0 - abs(d) / 10.0
                else:
                    # Random weights
                    P_trial[y, x] = random.uniform(0.5, 1.5)
        
        P_trial = np.clip(P_trial, 0, None)
        s = np.sum(P_trial)
        if s < 1e-9:
            continue
        P_trial /= s
        
        # Optimize this
        P_opt, score = optimize_coordinate_descent(P_trial, max_iters=2000)
        if score > best_score:
            best_score = score
            P_best = P_opt

    # Final cleanup
    P_best = np.clip(P_best, 0, None)
    total = np.sum(P_best)
    if total < 1e-9:
        return np.ones((n, n)) / (n * n)
    P_best = P_best / total
    
    return P_best
