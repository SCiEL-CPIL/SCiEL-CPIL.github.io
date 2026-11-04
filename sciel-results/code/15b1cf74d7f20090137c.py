"""Rewritten program: multi-objective evolutionary algorithm for circle packing."""
import time
import numpy as np
from scipy.optimize import minimize

N = 21
DEADLINE = 115.0  # safety margin before 120s kill
START_TIME = None

def _elapsed():
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()
    return time.time() - START_TIME

def _perimeter_ok(w, h):
    return 2 * (w + h) <= 4.0 + 1e-9

def _compute_objectives(x, w, h):
    """
    x: array of shape (N, 3) = [cx, cy, r]
    Returns (sum_r, penalty) where penalty is total constraint violation.
    """
    cx, cy, r = x[:, 0], x[:, 1], x[:, 2]
    sum_r = np.sum(r)
    
    # Boundary violations
    # Left: cx - r >= 0
    viol_left = np.maximum(0, -(cx - r))
    # Right: cx + r <= w
    viol_right = np.maximum(0, (cx + r) - w)
    # Bottom: cy - r >= 0
    viol_bottom = np.maximum(0, -(cy - r))
    # Top: cy + r <= h
    viol_top = np.maximum(0, (cy + r) - h)
    boundary_pen = np.sum(viol_left) + np.sum(viol_right) + np.sum(viol_bottom) + np.sum(viol_top)
    
    # Pairwise overlap violations
    # dist(i,j) >= r_i + r_j
    # Use vectorized computation
    n = len(x)
    cx_diff = cx[:, None] - cx[None, :]
    cy_diff = cy[:, None] - cy[None, :]
    dist = np.sqrt(cx_diff**2 + cy_diff**2)
    r_sum = r[:, None] + r[None, :]
    overlap = np.maximum(0, r_sum - dist)
    # Only count upper triangle to avoid double counting
    iu = np.triu_indices(n, k=1)
    pair_pen = np.sum(overlap[iu])
    
    total_pen = boundary_pen + 2.0 * pair_pen  # weight overlaps more heavily
    return sum_r, total_pen

def _local_optimize(x0, w, h):
    """
    Run a local gradient-based optimization to refine a configuration.
    Returns (best_x, best_sum_r, best_pen).
    """
    def objective(x):
        s, p = _compute_objectives(x, w, h)
        return p - 10.0 * s  # Minimize penalty, maximize sum (via negative)
    
    def jac(x):
        # Numerical gradient
        eps = 1e-7
        n = len(x) * 3
        g = np.zeros(n)
        x_flat = x.flatten()
        f0 = objective(x)
        for i in range(n):
            x2 = x.copy()
            x2_flat = x2.flatten()
            x2_flat[i] += eps
            x2 = x2_flat.reshape(n, 3)
            g[i] = (objective(x2) - f0) / eps
        return g
    
    best_x = x0.copy()
    best_obj = objective(x0)
    best_s, best_p = _compute_objectives(x0, w, h)
    
    for _ in range(3):
        res = minimize(objective, x0, method='L-BFGS-B', jac=jac,
                       options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8})
        if res.fun < best_obj:
            best_obj = res.fun
            best_x = res.x.reshape(N, 3)
            best_s, best_p = _compute_objectives(best_x, w, h)
        x0 = res.x.reshape(N, 3)
        if _elapsed() > DEADLINE:
            break
    
    return best_x, best_s, best_p

def _crossover(parent1, parent2, w, h):
    """Simple blend crossover with mutation."""
    child = 0.5 * parent1 + 0.5 * parent2
    # Add noise
    noise = np.random.normal(0, 0.02, size=child.shape)
    child += noise
    # Clamp radii to be positive
    child[:, 2] = np.maximum(child[:, 2], 0.01)
    # Clamp positions to be within rectangle (with some slack)
    child[:, 0] = np.clip(child[:, 0], 0, w)
    child[:, 1] = np.clip(child[:, 1], 0, h)
    return child

def _mutate(x, w, h, rate=0.1):
    """Gaussian mutation."""
    x_new = x.copy()
    mask = np.random.random(x_new.shape) < rate
    x_new[mask] += np.random.normal(0, 0.05, size=x_new.shape)[mask]
    x_new[:, 2] = np.maximum(x_new[:, 2], 0.01)
    x_new[:, 0] = np.clip(x_new[:, 0], 0, w)
    x_new[:, 1] = np.clip(x_new[:, 1], 0, h)
    return x_new

def construct():
    global START_TIME
    START_TIME = time.time()
    
    # We'll try a few aspect ratios
    # Perimeter 4: 2(w+h) = 4 => w+h = 2
    # Try aspect ratios: w/h = 1 (square 1x1), 2 (2x0.5), 0.5 (0.5x2)
    aspect_ratios = [1.0, 1.5, 0.75, 2.0, 0.5]
    
    best_overall = None
    best_score = -np.inf
    
    pop_size = 20
    max_generations = 50
    
    for ar in aspect_ratios:
        if _elapsed() > DEADLINE - 5:
            break
        
        if ar >= 1:
            w = 2.0 * ar / (1 + ar)
            h = 2.0 / (1 + ar)
        else:
            w = 2.0 * ar / (1 + ar)
            h = 2.0 / (1 + ar)
        
        # Initialize population
        pop = []
        for i in range(pop_size):
            # Random initialization
            x = np.zeros((N, 3))
            x[:, 0] = np.random.uniform(0, w, size=N)
            x[:, 1] = np.random.uniform(0, h, size=N)
            x[:, 2] = np.random.uniform(0.05, 0.2, size=N)
            pop.append(x)
        
        # Run NSGA-II-like evolution
        for gen in range(max_generations):
            if _elapsed() > DEADLINE - 5:
                break
            
            # Evaluate population
            evals = []
            for x in pop:
                s, p = _compute_objectives(x, w, h)
                evals.append((s, p))
            
            # Sort by non-dominated sorting
            # Simple approach: sort by (penalty, -sum_r)
            sorted_indices = np.argsort([evals[i][1] - 10*evals[i][0] for i in range(pop_size)])
            
            # Keep best half, generate new half
            new_pop = []
            for idx in sorted_indices[:pop_size//2]:
                new_pop.append(pop[idx].copy())
            
            # Generate offspring
            while len(new_pop) < pop_size:
                if _elapsed() > DEADLINE - 5:
                    break
                i, j = np.random.choice(pop_size, 2, replace=False)
                child = _crossover(pop[i], pop[j], w, h)
                if np.random.random() < 0.3:
                    child = _mutate(child, w, h)
                new_pop.append(child)
            
            # Local optimization on best individuals
            for idx in range(min(3, pop_size)):
                if _elapsed() > DEADLINE - 5:
                    break
                refined, s, p = _local_optimize(new_pop[idx], w, h)
                new_pop[idx] = refined
            
            pop = new_pop
        
        # After evolution, do final local optimization on best
        if pop:
            evals = []
            for x in pop:
                s, p = _compute_objectives(x, w, h)
                evals.append((s, p))
            
            # Find best by (penalty, -sum_r)
            best_idx = np.argmin([evals[i][1] - 10*evals[i][0] for i in range(len(pop))])
            best_x, best_s, best_p = _local_optimize(pop[best_idx], w, h)
            
            if best_p < 1e-6:  # Valid solution
                if best_s > best_score:
                    best_score = best_s
                    best_overall = best_x.copy()
    
    # If no valid solution found, return a simple valid one
    if best_overall is None:
        # Fallback: grid packing in 1x1 square
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_overall = np.array(cells[:N])
    
    return best_overall
