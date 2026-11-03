"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _radius(c):
    """Compute the maximum common radius for centers c of shape (N,2)."""
    n = c.shape[0]
    if n == 0:
        return 0.0
    iu = np.triu_indices(n, k=1)
    diff = c[iu[0]] - c[iu[1]]
    d = np.sqrt((diff ** 2).sum(axis=1))
    rp = d.min() / 2.0 if d.size > 0 else 0.5
    # Wall distances
    # min(0.5 - x, x + 0.5, 0.5 - y, y + 0.5)
    dx = np.minimum(0.5 - c[:, 0], c[:, 0] + 0.5)
    dy = np.minimum(0.5 - c[:, 1], c[:, 1] + 0.5)
    rw = np.minimum(dx, dy).min()
    return min(rp, rw)

def _energy(c, r):
    """
    Compute the energy for a configuration c with radius r.
    Energy is the sum of squared overlaps (pairwise and wall).
    We want to minimize this.
    """
    n = c.shape[0]
    # Pairwise overlaps
    iu = np.triu_indices(n, k=1)
    diff = c[iu[0]] - c[iu[1]]
    d = np.sqrt((diff ** 2).sum(axis=1))
    overlaps = np.maximum(0, 2 * r - d)
    e_pairs = np.sum(overlaps ** 2)
    
    # Wall overlaps
    # Distance to walls: min(0.5 - x, x + 0.5, ...)
    # Overlap is max(0, r - dist)
    dx = np.minimum(0.5 - c[:, 0], c[:, 0] + 0.5)
    dy = np.minimum(0.5 - c[:, 1], c[:, 1] + 0.5)
    dist_walls = np.minimum(dx, dy)
    wall_overlaps = np.maximum(0, r - dist_walls)
    e_walls = np.sum(wall_overlaps ** 2)
    
    return e_pairs + e_walls

def _init_hexagonal():
    """Hexagonal lattice initialization."""
    pts = []
    s = 0.14
    xs = np.arange(-0.5 + s / 2, 0.5, s)
    for row, y in enumerate(np.arange(-0.5 + s / 2, 0.5, s * np.sqrt(3) / 2)):
        offset = (s / 2) if row % 2 else 0.0
        for x in xs:
            xx = x + offset
            if -0.5 <= xx <= 0.5 and -0.5 <= y <= 0.5:
                pts.append([xx, y])
    pts = np.array(pts)
    if len(pts) >= N:
        dists = np.hypot(pts[:, 0], pts[:, 1])
        idx = np.argsort(dists)[:N]
        return pts[idx].copy()
    else:
        ticks = (np.arange(7) + 0.5) / 7 - 0.5
        g = np.array([(x, y) for y in ticks for x in ticks])
        return g[:N].copy()

def _init_grid_7x7():
    ticks = (np.arange(7) + 0.5) / 7 - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    return g[:N].copy()

def _init_random():
    return np.random.uniform(-0.45, 0.45, (N, 2))

def _simulated_annealing(c0, t0, deadline, max_iter=5000, T_start=0.01, T_end=1e-6, alpha=0.9995):
    """
    Simulated Annealing to maximize radius.
    We fix r to a target value and try to find a configuration that fits.
    Actually, it's better to optimize r as well or iterate r.
    Let's use a strategy: 
    1. Start with a small r.
    2. Use SA to find a valid configuration for that r (minimize energy).
    3. Increase r slightly.
    4. Repeat.
    
    However, SA on 100 dimensions (50 points * 2) is slow.
    Alternative: Optimize the energy function E(c, r) directly where r is a variable?
    No, standard SA is for fixed parameters.
    
    Let's use a hybrid: 
    - We will run SA on the positions c, with a fixed r.
    - We will increase r incrementally.
    - This is similar to the current program's _expand_radius but using SA for the "relaxation" step instead of simple push-apart.
    """
    c = c0.copy()
    best_c = c.copy()
    best_r = _radius(c)
    
    # We will try to grow r from current best_r up to a limit
    r_current = max(0.01, best_r)
    step_r = 0.0005
    
    # Precompute indices for speed
    iu = np.triu_indices(N, k=1)
    
    while r_current < 0.075 and (time.time() - t0) < deadline - 10:
        # Run SA for this r_current
        T = T_start
        c_sa = c.copy()
        best_e = _energy(c_sa, r_current)
        
        # If current config is already valid for this r, we can try to jump higher
        # But usually we need to relax.
        
        for it in range(max_iter):
            if (time.time() - t0) > deadline - 5:
                break
            
            # Propose a move: perturb one point
            i = np.random.randint(N)
            delta = np.random.normal(0, T * 0.5, 2)
            c_new = c_sa.copy()
            c_new[i] += delta
            
            # Keep inside bounds? No, energy handles it. But huge moves are bad.
            # Clip to square to avoid extreme energies
            c_new[i, 0] = np.clip(c_new[i, 0], -0.5, 0.5)
            c_new[i, 1] = np.clip(c_new[i, 1], -0.5, 0.5)
            
            e_new = _energy(c_new, r_current)
            dE = e_new - best_e
            
            if dE < 0 or np.random.rand() < np.exp(-dE / T):
                c_sa = c_new
                best_e = e_new
                
            T *= alpha
            if T < T_end:
                break
        
        # After SA, check if we achieved a valid config for r_current
        # If energy is very low, we accept r_current as achievable and move to next c
        # If energy is high, we might need to retry or lower r? 
        # For simplicity, if best_e < 1e-4, we assume it's close enough to valid.
        if best_e < 1e-4:
            c = c_sa
            r_current += step_r
        else:
            # If we failed, try a smaller step or break
            # In this strategy, we stick with the last valid c
            # But we should update c to the best found c_sa to maintain continuity
            c = c_sa
            # If we can't grow, break
            if best_e > 1.0:
                break
                
    # Update best
    r_final = _radius(c)
    if r_final > best_r:
        best_r = r_final
        best_c = c.copy()
        
    return best_c

def _nelder_mead_polish(c, t0, deadline):
    """
    Polish the configuration using Nelder-Mead on the negative radius.
    Since _radius is non-smooth, Nelder-Mead is a good derivative-free choice.
    """
    c = c.copy()
    
    # Define objective: minimize -radius
    def obj(x):
        c_x = x.reshape(N, 2)
        # Ensure validity (clip)
        c_x = np.clip(c_x, -0.5, 0.5)
        return -_radius(c_x)
    
    # Options
    options = {
        'maxiter': 200,
        'xatol': 1e-8,
        'fatol': 1e-10,
        'adaptive': True
    }
    
    try:
        res = minimize(obj, c.flatten(), method='Nelder-Mead', options=options)
        if (time.time() - t0) < deadline - 2:
            c_new = res.x.reshape(N, 2)
            c_new = np.clip(c_new, -0.5, 0.5)
            r_new = _radius(c_new)
            r_old = _radius(c)
            if r_new > r_old:
                return c_new
    except Exception:
        pass
    return c

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin for 150s limit
    
    # Strategy: Run multiple restarts with different initializations.
    # 1. Hexagonal
    # 2. Grid
    # 3. Random
    
    best_c = None
    best_r = -1.0
    
    # Restart 1: Hexagonal
    c1 = _init_hexagonal()
    # Add small jitter to break symmetry
    c1 += np.random.normal(0, 0.001, c1.shape)
    c1 = np.clip(c1, -0.5, 0.5)
    
    # Allocate time: 40s for SA on hex
    c1 = _simulated_annealing(c1, t0, t0 + 40)
    r1 = _radius(c1)
    if r1 > best_r:
        best_r = r1
        best_c = c1.copy()
        
    # Restart 2: Grid
    if (time.time() - t0) < deadline - 40:
        c2 = _init_grid_7x7()
        c2 += np.random.normal(0, 0.001, c2.shape)
        c2 = np.clip(c2, -0.5, 0.5)
        
        c2 = _simulated_annealing(c2, t0, t0 + 80) # Use remaining time roughly
        r2 = _radius(c2)
        if r2 > best_r:
            best_r = r2
            best_c = c2.copy()

    # Restart 3: Random (if time permits)
    if (time.time() - t0) < deadline - 30:
        c3 = _init_random()
        # Random start is usually bad, but SA might find something.
        # To make it better, do a quick relaxation first?
        # Let's just run SA with a tighter time budget
        c3 = _simulated_annealing(c3, t0, t0 + 110)
        r3 = _radius(c3)
        if r3 > best_r:
            best_r = r3
            best_c = c3.copy()

    # Final Polish with Nelder-Mead on the best candidate
    if best_c is not None and (time.time() - t0) < deadline - 10:
        best_c = _nelder_mead_polish(best_c, t0, deadline)
        best_r = _radius(best_c)
        
    # Fallback if something went wrong
    if best_c is None or not np.all(np.isfinite(best_c)) or best_r <= 0:
        ticks = (np.arange(7) + 0.5) / 7 - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_c = grid[:N]
        
    # Ensure strict validity
    best_c = np.clip(best_c, -0.5 + 1e-15, 0.5 - 1e-15)
    
    return best_c
# EVOLVE-BLOCK-END
