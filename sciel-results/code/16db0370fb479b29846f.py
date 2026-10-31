# EVOLVE-BLOCK-START

import numpy as np
import time
from scipy.optimize import minimize

N = 26

def _is_valid(pos, radii, tol=1e-9):
    """Strict validity check."""
    n = len(radii)
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
        
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    if np.any(dist[mask] < (min_dist[mask] - tol)):
        return False
    return True

def _post_process(pos, radii, iterations=200):
    """Shrink radii to ensure strict validity."""
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(iterations):
        valid = True
        
        # Boundary constraints
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r:
                radii[i] = max(max_r * 0.999999, 1e-8)
                valid = False
        
        # Overlap constraints
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    if sum_r > 1e-12:
                        k = dist / sum_r * 0.999999
                        radii[i] = max(radii[i] * k, 1e-8)
                        radii[j] = max(radii[j] * k, 1e-8)
                        valid = False
                    else:
                        radii[i] = max(radii[i] * 0.99, 1e-8)
                        radii[j] = max(radii[j] * 0.99, 1e-8)
                        valid = False
                        
        if valid:
            break
            
    return pos, radii

def _solve_socp(pos_init, radii_init, maxiter=500, ftol=1e-15):
    """
    Solve using SLSQP with proper SOCP formulation.
    Variables: x_i, y_i for each circle, and r_i for each circle.
    Objective: maximize sum(r_i)
    Constraints:
      - Boundary: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
      - Non-overlap: ||p_i - p_j|| >= r_i + r_j for all i < j
    """
    n = len(radii_init)
    # Variable vector: [x_0, y_0, ..., x_{n-1}, y_{n-1}, r_0, ..., r_{n-1}]
    x0 = np.concatenate([pos_init.flatten(), radii_init])
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((1e-6, 1.0 - 1e-6))  # x_i
        bounds.append((1e-6, 1.0 - 1e-6))  # y_i
    for i in range(n):
        bounds.append((1e-6, 0.5))  # r_i
    
    def objective(x):
        r = x[2*n:]
        return -np.sum(r)
    
    # Inequality constraints: each returns array >= 0
    def constraint_func(x):
        p = x[:2*n].reshape(n, 2)
        r = x[2*n:]
        
        # Boundary constraints: x_i - r_i >= 0, 1 - x_i - r_i >= 0, etc.
        c_bound = np.concatenate([
            p[:, 0] - r,
            1.0 - p[:, 0] - r,
            p[:, 1] - r,
            1.0 - p[:, 1] - r
        ])
        
        # Non-overlap constraints: ||p_i - p_j|| - (r_i + r_j) >= 0
        c_overlap = []
        for i in range(n):
            for j in range(i + 1, n):
                dx = p[i, 0] - p[j, 0]
                dy = p[i, 1] - p[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                c_overlap.append(dist - (r[i] + r[j]))
        c_overlap = np.array(c_overlap)
        
        return np.concatenate([c_bound, c_overlap])
    
    try:
        res = minimize(
            objective, x0, method='SLSQP',
            bounds=bounds,
            constraints=[{'type': 'ineq', 'fun': constraint_func}],
            options={'maxiter': maxiter, 'ftol': ftol}
        )
        
        if res.fun < -0.5:  # Check if we got a reasonable solution
            p = res.x[:2*n].reshape(n, 2)
            r = res.x[2*n:]
            r = np.maximum(r, 1e-8)
            p, r = _post_process(p, r)
            if _is_valid(p, r):
                return p, r
    except Exception:
        pass
    return None

def _init_hexagonal():
    """Hexagonal packing initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    r = 0.075
    dx = 2 * r
    dy = 2 * r * np.sqrt(3) / 2
    
    idx = 0
    for row in range(6):
        for col in range(6):
            if idx >= N:
                break
            x = 0.5 + col * dx + (row % 2) * dx / 2
            y = 0.5 + row * dy
            x = x - (5 * dx + dx/2) / 2 + 0.5
            y = y - (5 * dy) / 2 + 0.5
            x = np.clip(x, r + 0.005, 1 - r - 0.005)
            y = np.clip(y, r + 0.005, 1 - r - 0.005)
            pos[idx] = [x, y]
            radii[idx] = r
            idx += 1
    
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _init_grid():
    """Grid initialization with staggered rows."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    r = 0.08
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            if idx >= N:
                break
            x = (j + 0.5) / 5.0 + (0.04 if i % 2 == 0 else -0.04)
            y = (i + 0.5) / 5.0
            x = np.clip(x, r + 0.005, 1 - r - 0.005)
            y = np.clip(y, r + 0.005, 1 - r - 0.005)
            pos[idx] = [x, y]
            radii[idx] = r
    
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _init_random(seed):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.08, 0.92, (N, 2))
    radii = rng.uniform(0.06, 0.09, N)
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _init_low_discrepancy(seed=42):
    """Initialize using low-discrepancy sequence."""
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    golden_ratio = (1 + np.sqrt(5)) / 2
    for i in range(N):
        t = (i + 0.5) / N
        x = (t * golden_ratio) % 1.0
        y = (t * golden_ratio * golden_ratio) % 1.0
        x = 0.08 + 0.84 * x
        y = 0.08 + 0.84 * y
        pos[i] = [x, y]
        radii[i] = 0.075
    
    pos += rng.normal(0, 0.015, (N, 2))
    pos = np.clip(pos, 0.05, 0.95)
    
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _local_refine(pos, radii, deadline, rng_seed=456):
    """Iterative local refinement with decreasing perturbation."""
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    rng = np.random.RandomState(rng_seed)
    
    scales = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
    
    for scale in scales:
        while time.time() < deadline - 5:
            perturbed_pos = best_pos + rng.normal(0, scale, (N, 2))
            perturbed_pos = np.clip(perturbed_pos, 0.01, 0.99)
            perturbed_radii = best_radii * (1 + rng.normal(0, 0.015, N))
            perturbed_radii = np.clip(perturbed_radii, 0.015, 0.25)
            
            perturbed_pos, perturbed_radii = _post_process(perturbed_pos, perturbed_radii)
            
            res = _solve_socp(perturbed_pos, perturbed_radii, maxiter=300, ftol=1e-14)
            if res:
                p, r = res
                s = np.sum(r)
                if s > best_score:
                    best_score = s
                    best_pos = p.copy()
                    best_radii = r.copy()
    
    return best_pos, best_radii

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    best_pos = None
    best_radii = None
    best_score = -1

    # 1. Try multiple initializations with SOCP solver
    inits = []
    inits.append(_init_hexagonal())
    inits.append(_init_grid())
    inits.append(_init_low_discrepancy(42))
    inits.append(_init_low_discrepancy(123))
    inits.append(_init_random(1))
    inits.append(_init_random(2))
    inits.append(_init_random(3))
    inits.append(_init_random(4))
    
    for pos, radii in inits:
        if time.time() > deadline - 20:
            break
        res = _solve_socp(pos, radii, maxiter=400, ftol=1e-14)
        if res:
            p, r = res
            s = np.sum(r)
            if s > best_score:
                best_score = s
                best_pos = p
                best_radii = r

    # 2. Local refinement on best solution
    if best_pos is not None and time.time() < deadline - 8:
        refined_pos, refined_radii = _local_refine(best_pos, best_radii, deadline, rng_seed=456)
        s = np.sum(refined_radii)
        if s > best_score:
            best_score = s
            best_pos = refined_pos
            best_radii = refined_radii

    # 3. Additional refinement passes with different seeds
    if best_pos is not None and time.time() < deadline - 3:
        for seed in [789, 1024, 2048, 3072]:
            if time.time() > deadline - 2:
                break
            refined_pos, refined_radii = _local_refine(best_pos, best_radii, deadline, rng_seed=seed)
            s = np.sum(refined_radii)
            if s > best_score:
                best_score = s
                best_pos = refined_pos
                best_radii = refined_radii

    if best_pos is None:
        pos, radii = _init_grid()
        pos, radii = _post_process(pos, radii)
        return np.column_stack([pos, radii])

    return np.column_stack([best_pos, best_radii])

# EVOLVE-BLOCK-END
