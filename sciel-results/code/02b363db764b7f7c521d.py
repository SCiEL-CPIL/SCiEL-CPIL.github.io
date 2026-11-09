"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize, Bounds

N = 50

# EVOLVE-BLOCK-START

def _compute_radius(c):
    """
    Compute the exact maximum radius for a given set of centers c (N, 3).
    """
    # Wall distances
    wall_dists = 0.5 - np.abs(c)
    r_wall = np.min(wall_dists)
    
    # Pairwise distances
    # c shape (N, 3)
    # diffs shape (N, N, 3)
    diffs = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dists, np.inf)
    r_pair = np.min(dists) / 2.0
    
    return min(r_wall, r_pair)

def _soft_min_radius_and_grad(c, beta):
    """
    Compute the soft minimum radius and its gradient.
    x_k are the constraints: wall distances and half-pairwise distances.
    r_soft = -1/beta * ln(sum(exp(-beta * x_k)))
    grad = sum(p_k * grad_x_k) where p_k = exp(-beta * x_k) / sum(exp(-beta * x_k))
    """
    # 1. Compute all x_k values and store indices/structure for gradient
    # Wall distances: N*3 values
    # Pairwise distances: N*(N-1)/2 values
    
    # To avoid huge memory allocation for N=50 (1225 pairs + 150 walls = ~1375 values),
    # we can compute in chunks or just compute all at once since N is small.
    
    # Wall distances
    # c: (N, 3)
    abs_c = np.abs(c)
    w = 0.5 - abs_c # (N, 3)
    
    # Pairwise distances
    # diffs: (N, N, 3)
    diffs = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2) # (N, N)
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    # We only need upper triangle to avoid double counting, but for soft min sum, 
    # symmetry doesn't matter if we are consistent. 
    # However, for gradient, each pair contributes to two points.
    # Let's sum over all i < j.
    
    # For efficiency in gradient, let's compute the terms explicitly.
    
    # Log-sum-exp stability
    # We have a set of values X. 
    # Let's flatten them.
    # Walls: w.flatten() -> 150 values
    # Pairs: dists[mask] / 2.0 -> 1225 values
    
    x_vals = np.concatenate([w.flatten(), dists[mask] / 2.0])
    
    # Compute softmax weights p_k
    # p_k = exp(-beta * x_k) / sum(exp(-beta * x_k))
    # To prevent underflow, subtract max:
    # p_k = exp(-beta * (x_k - min_x)) / sum(exp(-beta * (x_k - min_x)))
    # Note: r_soft = min_x - 1/beta * ln(sum(exp(-beta*(x_k - min_x))))
    
    min_x = np.min(x_vals)
    shifted = x_vals - min_x
    exp_terms = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_terms)
    
    # If sum_exp is 0 (overflow in exp? No, exp is <= 1 here). 
    # If beta is huge and min_x is very negative? No, x are positive.
    # If beta is huge, exp(-beta * shifted) is 0 for all except min_x.
    # sum_exp might be small if beta is large.
    # r_soft = min_x - (1/beta) * ln(sum_exp)
    
    r_soft = min_x - (1.0 / beta) * np.log(sum_exp)
    
    # Gradient
    # grad_r = sum(p_k * grad_x_k)
    # p_k = exp_terms / sum_exp
    
    # We need to distribute the gradient back to c.
    # grad_c = np.zeros_like(c)
    
    # 1. Walls
    # w = 0.5 - |c|
    # grad w / grad c = -sign(c)
    # p_w are the first N*3 weights
    p_w = exp_terms[:N*3] / sum_exp
    p_w = p_w.reshape(N, 3)
    
    # sign(c) is 1 if c>0, -1 if c<0. 
    # If c=0, sign is 0, but we can use 1 or -1 arbitrarily.
    # np.sign(0) is 0. Let's replace 0 with 1.
    signs = np.sign(c)
    signs[signs == 0] = 1.0
    
    grad_wall = -p_w * signs
    
    # 2. Pairs
    # d_ij / 2
    # grad d_ij / grad c_i = (c_i - c_j) / (2 * ||c_i - c_j||)
    # grad d_ij / grad c_j = (c_j - c_i) / (2 * ||c_i - c_j||)
    
    p_pairs = exp_terms[N*3:] / sum_exp
    # p_pairs corresponds to the flattened upper triangle of dists/2
    
    # We need to map these back to pairs (i, j) with i < j.
    # np.triu_indices(N, k=1) gives the indices.
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    # Calculate unit vectors
    # diffs_ij = c[i] - c[j]
    # dist_ij = ||diffs_ij||
    # Note: dists matrix has dists[i,j] = ||c_i - c_j||.
    # We used dists[mask] which is dists[i,j] for i<j.
    # So dists_pairs = dists[i_idx, j_idx]
    
    dists_pairs = dists[i_idx, j_idx]
    
    # Avoid division by zero
    # If dist is 0, gradient is undefined. 
    # In practice, this shouldn't happen if we start with distinct points.
    # If it does, we can skip or use 0.
    safe_dists = np.where(dists_pairs > 1e-12, dists_pairs, 1.0)
    
    # Unit vectors from j to i
    # u = (c_i - c_j) / dist
    # grad for c_i is p * u / 2
    # grad for c_j is p * -u / 2
    
    # Vectorized update
    # u shape (num_pairs, 3)
    u = (c[i_idx] - c[j_idx]) / safe_dists[:, np.newaxis]
    
    # Weighted unit vectors
    # p_pairs shape (num_pairs,)
    # u_weighted = p_pairs[:, np.newaxis] * u * 0.5
    
    u_weighted = 0.5 * p_pairs[:, np.newaxis] * u
    
    # Accumulate to grad_c
    # grad_c[i_idx] += u_weighted
    # grad_c[j_idx] -= u_weighted
    
    grad_pair = np.zeros_like(c)
    np.add.at(grad_pair, i_idx, u_weighted)
    np.add.at(grad_pair, j_idx, -u_weighted)
    
    grad_total = grad_wall + grad_pair
    
    return r_soft, grad_total

def _make_fibonacci_lattice(n, shrink=0.9):
    """
    Generate points on a Fibonacci lattice scaled to fit in [-shrink/2, shrink/2]^3?
    Actually, Fibonacci lattice is usually for sphere surface or 2D.
    For 3D box, we can use a 3D Halton or just a scaled Fibonacci 3D spiral?
    A common 3D quasi-random sequence is the 3D Halton sequence.
    Let's implement a simple 3D Halton sequence.
    """
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index:
            f /= base
            r += f * (index % base)
            index //= base
        return r

    points = []
    for i in range(1, n + 1):
        p = np.array([halton(i, 2), halton(i, 3), halton(i, 5)])
        # Map to [-0.5, 0.5]
        p = p * shrink - shrink / 2.0
        points.append(p)
    return np.array(points)

def _make_grid_packing(n):
    """
    Create a grid packing.
    """
    # 4x4x4 grid has 64 points.
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    # Shuffle and take n
    idx = np.random.choice(len(grid), n, replace=False)
    return grid[idx].copy()

def _make_random_packing(n, shrink=0.8):
    """
    Random points in a shrunken box.
    """
    return np.random.uniform(-shrink/2, shrink/2, size=(n, 3))

def _optimize_lbgfsb(c_init, deadline, beta_start=10, beta_end=1000, steps=5):
    """
    Optimize using L-BFGS-B with increasing beta.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    # Bounds
    bounds = Bounds(-0.5, 0.5)
    
    # Beta schedule
    betas = np.geomspace(beta_start, beta_end, steps)
    
    for beta in betas:
        if time.time() > deadline:
            break
            
        # Objective: minimize -r_soft
        def func(c_flat):
            c = c_flat.reshape(N, 3)
            r, _ = _soft_min_radius_and_grad(c, beta)
            return -r
            
        def grad(c_flat):
            c = c_flat.reshape(N, 3)
            _, g = _soft_min_radius_and_grad(c, beta)
            return -g.flatten()
            
        # Optimize
        try:
            res = minimize(
                func, 
                c.flatten(), 
                jac=grad, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            c = res.x.reshape(N, 3)
            # Clip to be safe
            c = np.clip(c, -0.5, 0.5)
            
            r_exact = _compute_radius(c)
            if r_exact > best_r:
                best_r = r_exact
                best_c = c.copy()
                
        except Exception:
            # If optimization fails, keep current
            pass
            
    return best_c, best_r

def _polish_nelder_mead(c, deadline):
    """
    Polish using Nelder-Mead on the exact radius (negative).
    """
    c = c.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    def func(c_flat):
        c = c_flat.reshape(N, 3)
        # Penalize if out of bounds
        if np.any(c < -0.5) or np.any(c > 0.5):
            return 1.0 # Large penalty
        r = _compute_radius(c)
        return -r
        
    # NM is slow for 150 variables. 
    # Only do this if we have plenty of time left.
    if time.time() > deadline - 5:
        return best_c, best_r
        
    try:
        # NM might get stuck. Limit iterations.
        # maxiter 1000 might be too slow.
        # Let's try a small number of iterations.
        res = minimize(
            func, 
            c.flatten(), 
            method='Nelder-Mead', 
            options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8}
        )
        c_new = np.clip(res.x.reshape(N, 3), -0.5, 0.5)
        r_new = _compute_radius(c_new)
        if r_new > best_r:
            best_r = r_new
            best_c = c_new
    except Exception:
        pass
        
    return best_c, best_r

def construct():
    start_time = time.time()
    # 150s limit. Use 140s for safety.
    deadline = start_time + 140.0
    
    best_c = None
    best_r = 0.0
    
    # List of initializers
    inits = []
    
    # 1. Halton
    inits.append(_make_fibonacci_lattice(N, 0.9))
    
    # 2. Grid
    inits.append(_make_grid_packing(N))
    
    # 3. Random
    inits.append(_make_random_packing(N, 0.8))
    
    # 4. Another Random with different seed
    inits.append(_make_random_packing(N, 0.85))
    
    # 5. FCC-like? 
    # Let's just use more Halton/Grid variations
    inits.append(_make_fibonacci_lattice(N, 0.8))
    inits.append(_make_grid_packing(N))
    
    # Shuffle inits to vary strategy per run? No, deterministic is better for reproducibility if we want.
    # But we have time for multiple runs.
    
    # We will run each initializer and keep the best.
    # If time allows, we can re-run the best one with higher beta.
    
    for c_init in inits:
        if time.time() > deadline - 10:
            break
            
        # Allocate time for this run
        # Roughly (deadline - now) / remaining_inits
        # But we want to finish each run reasonably.
        # Let's just run until deadline - 5s, but break if we finish early?
        # L-BFGS-B with maxiter 50 is fast.
        
        c_opt, r_opt = _optimize_lbgfsb(c_init, deadline - 5.0)
        
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt
            
    # Final Polish
    if best_c is not None:
        # Use remaining time for NM polish
        c_final, r_final = _polish_nelder_mead(best_c, deadline)
        if r_final > best_r:
            best_r = r_final
            best_c = c_final
    else:
        # Fallback
        best_c = _make_grid_packing(N)
        
    # Ensure validity
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c

# EVOLVE-BLOCK-END
