"""50-sphere packing in a unit cube using smooth surrogate optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START

def _compute_radius(c):
    """Compute the common radius for centres c (N,3) in cube [-0.5,0.5]^3."""
    if len(c) == 0:
        return 0.0
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    return min(min_pair / 2.0, min_wall)

def _neg_radius(c):
    """Negative radius for minimization."""
    return -_compute_radius(c)

def _riesz_energy_and_grad(c, s, eps):
    """
    Compute Riesz kernel energy and gradient.
    E = sum_{i<j} 1/(|c_i - c_j|^s + eps^s) + 0.5 * sum_i |c_i|^2
    The first term penalizes close pairs (maximizes min distance for large s).
    The second term is harmonic confinement to keep spheres inside the cube.
    """
    c = np.asarray(c).reshape(-1, 3)
    n = len(c)
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, np.inf)
    
    # Riesz kernel: 1/(d^s + eps^s)
    dists_s = np.power(dists_sq, s / 2.0)
    kernel = 1.0 / (dists_s + eps**s)
    np.fill_diagonal(kernel, 0.0)
    
    # Energy from pairs (each pair counted once)
    pair_energy = 0.5 * np.sum(kernel)
    
    # Harmonic confinement
    confinement = 0.5 * np.sum(c**2)
    
    total_energy = pair_energy + confinement
    
    # Gradient
    dists_sq_safe = np.where(dists_sq > 1e-30, dists_sq, 1.0)
    dists_sq_neg = np.power(dists_sq_safe, (s - 2.0) / 2.0)
    
    f = dists_s + eps**s
    grad_kernel = -s * dists_sq_neg / (f * f)
    np.fill_diagonal(grad_kernel, 0.0)
    
    grad_pair = np.sum(grad_kernel[:, :, np.newaxis] * diff, axis=1)
    grad_conf = c
    
    total_grad = grad_pair + grad_conf
    
    return total_energy, total_grad

def _refine_riesz(c0, time_limit, seed=0):
    """
    Refine using Riesz kernel electrostatics model.
    Anneal the exponent s from small to large to sharpen the repulsion.
    """
    c = c0.copy()
    n = len(c)
    x0 = c.flatten()
    
    start_time = time.time()
    
    s_schedule = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0])
    eps_schedule = np.array([0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.002])
    
    for idx, (s_val, eps_val) in enumerate(zip(s_schedule, eps_schedule)):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 5:
            break
        
        remaining = time_limit - elapsed - 5
        n_remaining = len(s_schedule) - idx
        time_per_phase = remaining / n_remaining
        if time_per_phase < 1.0:
            break
        
        def obj(x):
            c_x = x.reshape(n, 3)
            c_x = np.clip(c_x, -0.499, 0.499)
            e, g = _riesz_energy_and_grad(c_x, s_val, eps_val)
            return e, g.flatten()
        
        bounds = [(-0.499, 0.499)] * (n * 3)
        
        try:
            result = minimize(
                obj, x0, method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'maxfun': 15000,
                         'ftol': 1e-14, 'gtol': 1e-10}
            )
            if result.success or result.fun < obj(x0)[0]:
                c = result.x.reshape(n, 3)
                x0 = result.x
        except Exception:
            pass
    
    return np.clip(c, -0.499, 0.499)

def _init_grid():
    """5x5x2 grid initialization."""
    m = 5
    ticks_xy = (np.arange(m) + 0.5) / m - 0.5
    ticks_z = np.array([-0.25, 0.25])
    
    grid = []
    for z in ticks_z:
        for y in ticks_xy:
            for x in ticks_xy:
                grid.append([x, y, z])
    
    return np.array(grid)

def _init_fcc():
    """FCC-like initialization."""
    # FCC lattice points in a cube
    # Generate points on FCC lattice and select 50 closest to center
    points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                # FCC: (i,j,k) where i+j+k is even, scaled
                if (i + j + k) % 2 == 0:
                    points.append([i * 0.12, j * 0.12, k * 0.12])
    
    points = np.array(points)
    # Select 50 closest to origin
    dists = np.sum(points**2, axis=1)
    idx = np.argsort(dists)[:50]
    c = points[idx].copy()
    
    # Center and scale to fit in cube
    c -= np.mean(c, axis=0)
    max_ext = np.max(np.abs(c))
    if max_ext > 0.45:
        c *= 0.45 / max_ext
    
    return c

def _init_bcc():
    """BCC-like initialization."""
    points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                if (i + j + k) % 2 == 1:
                    points.append([i * 0.12, j * 0.12, k * 0.12])
    
    points = np.array(points)
    dists = np.sum(points**2, axis=1)
    idx = np.argsort(dists)[:50]
    c = points[idx].copy()
    
    c -= np.mean(c, axis=0)
    max_ext = np.max(np.abs(c))
    if max_ext > 0.45:
        c *= 0.45 / max_ext
    
    return c

def _init_random(seed=0):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    c = rng.uniform(-0.45, 0.45, (N, 3))
    return c

def _repulsion_anneal(c0, time_limit, seed=0):
    """
    Vectorized repulsion annealing: push apart closest pairs with decaying steps.
    """
    c = c0.copy()
    n = len(c)
    start_time = time.time()
    
    step = 0.01
    decay = 0.9995
    min_step = 1e-10
    rng = np.random.RandomState(seed)
    
    for it in range(200000):
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=-1)
        np.fill_diagonal(dists_sq, np.inf)
        
        r_current = _compute_radius(c)
        threshold = (r_current * 1.1) ** 2
        
        i_idx, j_idx = np.where(dists_sq < threshold)
        mask = i_idx < j_idx
        i_idx = i_idx[mask]
        j_idx = j_idx[mask]
        
        if len(i_idx) == 0:
            wall_dists = 0.5 - np.abs(c)
            min_wall = np.min(wall_dists)
            if min_wall < r_current:
                k, axis = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
                if c[k, axis] > 0:
                    c[k, axis] -= step
                else:
                    c[k, axis] += step
            break
        
        for i, j in zip(i_idx, j_idx):
            d = np.sqrt(dists_sq[i, j])
            if d < 1e-15:
                direction = rng.randn(3)
                norm = np.linalg.norm(direction)
                direction /= norm
            else:
                direction = c[i] - c[j]
                norm = np.linalg.norm(direction)
                direction /= norm
            
            max_step = np.inf
            for coord in range(3):
                if direction[coord] > 0:
                    max_step = min(max_step, (0.499 - c[i][coord]) / direction[coord])
                    max_step = min(max_step, (c[j][coord] + 0.499) / direction[coord])
                elif direction[coord] < 0:
                    max_step = min(max_step, (c[i][coord] + 0.499) / (-direction[coord]))
                    max_step = min(max_step, (0.499 - c[j][coord]) / (-direction[coord]))
            
            actual_step = min(step, max_step)
            if actual_step > 1e-12:
                c[i] += direction * actual_step
                c[j] -= direction * actual_step
        
        step = max(min_step, step * decay)
        
        if it % 1000 == 0 and step > 1e-6:
            noise = rng.normal(0, step * 0.1, c.shape)
            c = np.clip(c + noise, -0.499, 0.499)
    
    return c

def _refine_exact(c0, time_limit, seed=0):
    """
    Exact refinement: push apart closest pairs, pull in wall-violating spheres.
    """
    c = c0.copy()
    n = len(c)
    start_time = time.time()
    
    step = 0.005
    decay = 0.999
    min_step = 1e-8
    rng = np.random.RandomState(seed)
    
    for it in range(100000):
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        # Compute pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        # Find closest pair
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_idx
        if i > j:
            i, j = j, i
        d_pair = dists[i, j]
        
        # Wall distances
        wall_dists = 0.5 - np.abs(c)
        min_wall_idx = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
        k, axis = min_wall_idx
        d_wall = wall_dists[k, axis]
        
        if d_pair / 2.0 < d_wall:
            # Push i and j apart
            direction = c[i] - c[j]
            norm = np.linalg.norm(direction)
            if norm < 1e-15:
                direction = rng.randn(3)
                norm = np.linalg.norm(direction)
            direction /= norm
            
            # Max step to stay in bounds
            max_step = np.inf
            for coord in range(3):
                if direction[coord] > 0:
                    space_i = 0.499 - c[i][coord]
                    if space_i < 0: space_i = 0
                    max_step = min(max_step, space_i / direction[coord])
                    space_j = c[j][coord] + 0.499
                    if space_j < 0: space_j = 0
                    max_step = min(max_step, space_j / direction[coord])
                elif direction[coord] < 0:
                    space_i = c[i][coord] + 0.499
                    if space_i < 0: space_i = 0
                    max_step = min(max_step, space_i / (-direction[coord]))
                    space_j = 0.499 - c[j][coord]
                    if space_j < 0: space_j = 0
                    max_step = min(max_step, space_j / (-direction[coord]))
            
            actual_step = min(step, max_step)
            if actual_step > 1e-12:
                c[i] += direction * actual_step
                c[j] -= direction * actual_step
        else:
            # Pull k inward
            if c[k, axis] > 0:
                dir_val = -1.0
            else:
                dir_val = 1.0
            actual_step = min(step, 0.499 - abs(c[k, axis]))
            if actual_step > 1e-12:
                c[k, axis] += dir_val * actual_step
        
        step = max(min_step, step * decay)
        
        # Small noise
        if it % 500 == 0 and step > 1e-5:
            noise = rng.normal(0, step * 0.05, c.shape)
            c = np.clip(c + noise, -0.499, 0.499)
    
    return c

def _final_polish(c, time_limit):
    """Final exact polish with very small steps."""
    c = c.copy()
    start_time = time.time()
    
    step = 0.001
    decay = 0.999
    min_step = 1e-10
    
    for it in range(50000):
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_idx
        if i > j:
            i, j = j, i
        d_pair = dists[i, j]
        
        wall_dists = 0.5 - np.abs(c)
        min_wall_idx = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
        k, axis = min_wall_idx
        d_wall = wall_dists[k, axis]
        
        if d_pair / 2.0 < d_wall:
            direction = c[i] - c[j]
            norm = np.linalg.norm(direction)
            if norm < 1e-15:
                continue
            direction /= norm
            
            max_step = np.inf
            for coord in range(3):
                if direction[coord] > 0:
                    max_step = min(max_step, (0.499 - c[i][coord]) / direction[coord])
                    max_step = min(max_step, (c[j][coord] + 0.499) / direction[coord])
                elif direction[coord] < 0:
                    max_step = min(max_step, (c[i][coord] + 0.499) / (-direction[coord]))
                    max_step = min(max_step, (0.499 - c[j][coord]) / (-direction[coord]))
            
            actual_step = min(step, max_step)
            if actual_step > 1e-12:
                c[i] += direction * actual_step
                c[j] -= direction * actual_step
        else:
            if c[k, axis] > 0:
                dir_val = -1.0
            else:
                dir_val = 1.0
            actual_step = min(step, 0.499 - abs(c[k, axis]))
            if actual_step > 1e-12:
                c[k, axis] += dir_val * actual_step
        
        step = max(min_step, step * decay)
    
    return c

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin from 150s limit
    
    # Multiple initializations
    inits = [
        (_init_grid, "grid"),
        (_init_fcc, "fcc"),
        (_init_bcc, "bcc"),
        (_init_random, "rand"),
    ]
    
    best_c = None
    best_r = -np.inf
    
    # Time allocation
    # Phase 1: Riesz kernel refinement on each init (70s total)
    # Phase 2: Repulsion annealing on best (50s)
    # Phase 3: Final polish (20s)
    
    phase1_time = 70.0
    phase2_time = 50.0
    phase3_time = 20.0
    
    # Phase 1: Riesz kernel refinement
    t_per_init = phase1_time / len(inits)
    
    for init_func, name in inits:
        elapsed = time.time() - start_time
        if elapsed > deadline - phase2_time - phase3_time - 5:
            break
        
        t_limit = min(t_per_init, deadline - elapsed - phase2_time - phase3_time - 5)
        if t_limit < 2:
            break
        
        c0 = init_func()
        c_refined = _refine_riesz(c0, time_limit=t_limit, seed=42)
        r = _compute_radius(c_refined)
        
        if r > best_r:
            best_r = r
            best_c = c_refined.copy()
    
    # Phase 2: Repulsion annealing on best
    if best_c is not None:
        elapsed = time.time() - start_time
        t_limit = min(phase2_time, deadline - elapsed - phase3_time - 2)
        if t_limit > 2:
            c_refined = _repulsion_anneal(best_c, t_limit, seed=123)
            r = _compute_radius(c_refined)
            if r > best_r:
                best_r = r
                best_c = c_refined.copy()
    
    # Phase 3: Final polish
    if best_c is not None:
        elapsed = time.time() - start_time
        t_limit = min(phase3_time, deadline - elapsed - 1)
        if t_limit > 1:
            c_polished = _final_polish(best_c, t_limit)
            r = _compute_radius(c_polished)
            if r > best_r:
                best_r = r
                best_c = c_polished.copy()
    
    if best_c is None:
        best_c = _init_grid()
    
    # Ensure validity
    if not np.all(np.isfinite(best_c)):
        best_c = _init_grid()
    
    best_c = np.clip(best_c, -0.49999, 0.49999)
    
    return best_c

# EVOLVE-BLOCK-END
