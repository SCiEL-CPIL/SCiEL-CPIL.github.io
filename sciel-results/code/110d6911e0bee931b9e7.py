"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _score(c):
    """Compute the packing radius for centres c (N,3)."""
    c = np.asarray(c)
    # Distance to walls
    wall = 0.5 - np.max(np.abs(c), axis=1)
    r_wall = np.min(wall)
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    r_pair = np.min(dist) / 2.0
    return min(r_wall, r_pair)

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill

    # Initial guess: simple cubic grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    c0 = grid[:N].astype(float).copy()
    
    best_c = c0.copy()
    best_r = _score(best_c)

    # Objective: maximize min(r_wall, r_pair) => minimize -min(...)
    # We'll use a smooth approximation: -log-sum-exp of negative distances
    # Actually, let's just do gradient ascent on the raw min, or use SLSQP with constraints.
    # Since N=50, pairwise is 1225 pairs. SLSQP might be slow.
    # Let's try L-BFGS-B on a smooth surrogate: -log(sum(exp(-beta * d_ij))) - log(sum(exp(-beta * w_i)))
    # Or simpler: just use the exact min with subgradient? No, let's use a soft-min.

    def objective(x):
        c = x.reshape(-1, 3)
        # Wall distances
        w = 0.5 - np.max(np.abs(c), axis=1)
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(d, np.inf)
        
        # Soft min for walls and pairs
        beta = 50.0
        # To avoid overflow/underflow in exp, we can use log-sum-exp trick or just clip
        # Since we want to maximize the minimum, we minimize -min.
        # Soft min: -log(sum(exp(-beta * val))) / beta
        # Let's compute soft min of all constraints (walls and half-pairwise dists)
        
        # Combine all constraints into one vector for soft-min?
        # Or separate.
        # Let's just use the raw min for now, it's often good enough for local optimization if we start well.
        # But L-BFGS needs gradient.
        # Let's use SLSQP which can handle non-smooth? No.
        
        # Let's try a different approach: 
        # Start with a good initial guess (e.g. from a known packing or random restarts with local search).
        
        # For now, let's just do a few local optimizations using SLSQP on the raw radius.
        # SLSQP can handle the non-smoothness reasonably well in practice for this size.
        
        return -_score(c)

    def grad(x):
        # Numerical gradient is too slow.
        # Let's use analytic gradient for the soft-min surrogate.
        c = x.reshape(-1, 3)
        w = 0.5 - np.max(np.abs(c), axis=1)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(d, np.inf)
        
        beta = 50.0
        
        # Soft min of walls
        exp_w = np.exp(-beta * w)
        sum_w = np.sum(exp_w)
        grad_w = np.zeros_like(c)
        for i in range(N):
            # d/dx_i of -log(sum(exp(-beta*w_j)))/beta
            # = (1/beta) * sum_j (exp(-beta*w_j) * (-beta * dw_j/dx_i)) / sum(exp)
            # = - sum_j (exp(-beta*w_j) * dw_j/dx_i) / sum(exp)
            # Only j=i contributes to dw_j/dx_i
            # dw_i/dx_i = -sign(x_ik) for the max coordinate k
            # This is complex. Let's stick to SLSQP on the raw function.
            pass
            
        # Let's just use SLSQP.
        return None

    # Strategy: Multiple restarts with SLSQP local optimization.
    # Initial guesses: 
    # 1. Grid
    # 2. Random points
    # 3. Perturbed grid
    
    rng = np.random.default_rng(42)
    
    # Try a few different initializations
    inits = [c0]
    # Add a few random inits
    for _ in range(5):
        c_rand = (rng.random((N, 3)) - 0.5) * 0.9
        inits.append(c_rand)
        
    # Also try a "body-centered" like grid if possible? 
    # For N=50, 3x4x4 = 48, plus 2? 
    # Let's stick to grid and random.
    
    for c_init in inits:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(objective, c_init.flatten(), method='SLSQP', 
                           options={'maxiter': 500, 'ftol': 1e-9})
            if res.success:
                c_opt = res.x.reshape(-1, 3)
                r_opt = _score(c_opt)
                if r_opt > best_r:
                    best_r = r_opt
                    best_c = c_opt
        except Exception:
            pass

    # Final polish: if we have time, do a finer local search on the best solution
    if time.time() < deadline - 5:
        try:
            res = minimize(objective, best_c.flatten(), method='SLSQP', 
                           options={'maxiter': 1000, 'ftol': 1e-12})
            if res.success:
                c_opt = res.x.reshape(-1, 3)
                r_opt = _score(c_opt)
                if r_opt > best_r:
                    best_r = r_opt
                    best_c = c_opt
        except Exception:
            pass

    return best_c
# EVOLVE-BLOCK-END
