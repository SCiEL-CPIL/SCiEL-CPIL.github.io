import numpy as np
from scipy.optimize import minimize
import time

N = 16
D = 2

def pairwise_distances_sq(pts):
    """Compute all pairwise squared distances (faster, no sqrt)."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    iu = np.triu_indices(N, k=1)
    return dist_sq[iu]

def objective(x):
    """(max_dist / min_dist)^2"""
    pts = x.reshape(N, D)
    d_sq = pairwise_distances_sq(pts)
    # Handle degenerate cases
    if np.any(d_sq < 1e-24):
        return 1e10
    d = np.sqrt(d_sq)
    ratio = d.max() / d.min()
    return ratio**2

def objective_smooth(x, beta=200.0):
    """Smooth approximation using log-sum-exp on squared distances."""
    pts = x.reshape(N, D)
    d_sq = pairwise_distances_sq(pts)
    if np.any(d_sq < 1e-24):
        return 1e10
    d = np.sqrt(d_sq)
    # log-sum-exp for max
    max_d = np.log(np.sum(np.exp(beta * d))) / beta
    # For min, use negative log-sum-exp: min = -logsumexp(-beta*d)/beta
    min_d = -np.log(np.sum(np.exp(-beta * d))) / beta
    if min_d < 1e-12:
        return 1e10
    ratio = max_d / min_d
    return ratio**2

def simulated_annealing(seed, time_budget=10.0):
    """Run Simulated Annealing to find a good basin."""
    start_time = time.time()
    x = seed.copy()
    current_val = objective(x)
    best_x = x.copy()
    best_val = current_val
    
    # Temperature schedule
    T_start = 1.0
    T_end = 1e-6
    # Estimate number of steps based on time budget
    target_steps = int(time_budget * 10000)
    
    if target_steps < 100:
        return best_x, best_val
        
    alpha = (T_end / T_start) ** (1.0 / target_steps)
    T = T_start
    
    step_size = 0.1
    accept_rate = 0.5
    
    for i in range(target_steps):
        if time.time() - start_time > time_budget:
            break
            
        # Adaptive step size
        if accept_rate > 0.7:
            step_size *= 1.1
        elif accept_rate < 0.3:
            step_size *= 0.9
        step_size = np.clip(step_size, 1e-4, 10.0)
        
        # Perturb
        noise = np.random.normal(0, step_size, x.shape)
        x_new = x + noise
        
        val_new = objective(x_new)
        
        if val_new < current_val:
            x = x_new
            current_val = val_new
            if current_val < best_val:
                best_val = current_val
                best_x = x.copy()
            accepted = True
        else:
            delta = val_new - current_val
            prob = np.exp(-delta / T)
            accepted = np.random.rand() < prob
            if accepted:
                x = x_new
                current_val = val_new
        
        # Track acceptance rate
        accept_rate = 0.9 * accept_rate + 0.1 * (1.0 if accepted else 0.0)
        
        # Cool down
        T *= alpha
        
    return best_x, best_val

def make_seeds():
    """Generate multiple diverse initial configurations."""
    seeds = []
    
    # 1. 4x4 grid
    grid = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
    seeds.append(grid.flatten())
    
    # 2. Hexagonal lattice subset
    # Hex lattice: rows offset
    hex_pts = []
    for row in range(4):
        x_offset = (row % 2) * 0.5
        for col in range(4):
            hex_pts.append([col + x_offset, row * np.sqrt(3)/2])
    hex_pts = np.array(hex_pts[:16])
    seeds.append(hex_pts.flatten())
    
    # 3. Concentric circles: 1 center + 5 + 10
    pts = [np.array([0.0, 0.0])]
    # Inner ring: 5 points
    for i in range(5):
        angle = 2 * np.pi * i / 5
        pts.append(np.array([1.0 * np.cos(angle), 1.0 * np.sin(angle)]))
    # Outer ring: 10 points
    for i in range(10):
        angle = 2 * np.pi * i / 10 + np.pi/10
        pts.append(np.array([2.2 * np.cos(angle), 2.2 * np.sin(angle)]))
    seeds.append(np.array(pts).flatten())
    
    # 4. Two concentric octagons
    pts = []
    for i in range(8):
        angle = 2 * np.pi * i / 8
        pts.append(np.array([1.0 * np.cos(angle), 1.0 * np.sin(angle)]))
    for i in range(8):
        angle = 2 * np.pi * i / 8 + np.pi/8
        pts.append(np.array([2.0 * np.cos(angle), 2.0 * np.sin(angle)]))
    seeds.append(np.array(pts).flatten())
    
    # 5. 1 center + 6 + 9 (hexagon + outer)
    pts = [np.array([0.0, 0.0])]
    for i in range(6):
        angle = 2 * np.pi * i / 6
        pts.append(np.array([1.0 * np.cos(angle), 1.0 * np.sin(angle)]))
    for i in range(9):
        angle = 2 * np.pi * i / 9 + np.pi/9
        pts.append(np.array([2.0 * np.cos(angle), 2.0 * np.sin(angle)]))
    seeds.append(np.array(pts).flatten())
    
    # 6. 2 center + 6 + 8
    pts = [np.array([-0.5, 0.0]), np.array([0.5, 0.0])]
    for i in range(6):
        angle = 2 * np.pi * i / 6
        pts.append(np.array([1.5 * np.cos(angle), 1.5 * np.sin(angle)]))
    for i in range(8):
        angle = 2 * np.pi * i / 8 + np.pi/8
        pts.append(np.array([2.8 * np.cos(angle), 2.8 * np.sin(angle)]))
    seeds.append(np.array(pts).flatten())
    
    # 7. Regular 16-gon
    pts = []
    for i in range(16):
        angle = 2 * np.pi * i / 16
        pts.append(np.array([np.cos(angle), np.sin(angle)]))
    seeds.append(np.array(pts).flatten())
    
    # 8. Random seeds
    for _ in range(5):
        pts = np.random.rand(N, D) * 4 - 2
        seeds.append(pts.flatten())
        
    return seeds

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    seeds = make_seeds()
    best_x = None
    best_val = np.inf
    
    # Phase 1: Quick local optimization on seeds
    for seed in seeds:
        if time.time() > deadline - 30.0: # Leave time for SA and final polish
            break
        try:
            res = minimize(
                objective_smooth,
                seed,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            val = objective(res.x)
            if val < best_val:
                best_val = val
                best_x = res.x.copy()
        except Exception:
            pass

    # Phase 2: Simulated Annealing from best local min and random seeds
    sa_starts = [best_x] if best_x is not None else []
    sa_starts.extend(seeds[:3]) # Add some random/grid starts
    
    for seed in sa_starts:
        if time.time() > deadline - 20.0:
            break
        try:
            sa_x, sa_val = simulated_annealing(seed, time_budget=10.0)
            if sa_val < best_val:
                best_val = sa_val
                best_x = sa_x.copy()
        except Exception:
            pass

    # Phase 3: High-precision polish
    if best_x is not None and time.time() < deadline - 5.0:
        # L-BFGS-B with tight tolerance
        try:
            res = minimize(
                objective_smooth,
                best_x,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            val = objective(res.x)
            if val < best_val:
                best_val = val
                best_x = res.x.copy()
        except Exception:
            pass
        
        # Nelder-Mead for final exact objective polish
        if time.time() < deadline - 2.0:
            try:
                res = minimize(
                    objective,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                )
                val = objective(res.x)
                if val < best_val:
                    best_val = val
                    best_x = res.x.copy()
            except Exception:
                pass

    if best_x is None:
        # Fallback
        grid = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
        return grid

    return best_x.reshape(N, D)
