"""Circle packing in perimeter-4 rectangle using DE + local refinement."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21
DEADLINE = 115.0  # Safety margin before 120s limit


def _max_feasible_radii(centers, w, h):
    """
    For given centers in a box w x h, compute the maximum feasible radius
    for each circle such that all circles are disjoint and inside the box.
    Returns array of radii and the sum.
    """
    n = len(centers)
    radii = np.zeros(n)
    for i in range(n):
        xi, yi = centers[i]
        # Distance to walls
        r = min(xi, w - xi, yi, h - yi)
        if r <= 0:
            radii[i] = 0
            continue
        # Distance to other circles
        for j in range(n):
            if i != j:
                dx = xi - centers[j][0]
                dy = yi - centers[j][1]
                d = np.sqrt(dx * dx + dy * dy)
                r = min(r, d / 2.0)
                if r <= 0:
                    break
        radii[i] = max(0.0, r)
    return radii, np.sum(radii)


def _objective_de(x):
    """
    Objective for differential evolution.
    x = [w, x1, y1, x2, y2, ..., xN, yN]
    w is the width, h = 2 - w.
    Returns negative sum of max feasible radii.
    """
    w = x[0]
    h = 2.0 - w
    if w < 0.15 or h < 0.15:
        return 1e6
    
    centers = x[1:].reshape(N, 2)
    
    # Quick bounds check
    if np.any(centers[:, 0] < 0) or np.any(centers[:, 0] > w) or \
       np.any(centers[:, 1] < 0) or np.any(centers[:, 1] > h):
        return 1e6
    
    radii, total_r = _max_feasible_radii(centers, w, h)
    if np.any(radii <= 0):
        return 1e6
    return -total_r


def _differential_evolution(objective, bounds, pop_size=35, max_iter=300,
                            mutation=(0.5, 1.0), crossover_prob=0.7,
                            time_limit=60.0):
    """
    Simple differential evolution implementation.
    """
    n_dims = len(bounds)
    start_time = time.time()
    
    # Initialize population
    pop = np.zeros((pop_size, n_dims))
    for i in range(pop_size):
        for j in range(n_dims):
            pop[i, j] = random.uniform(bounds[j][0], bounds[j][1])
    
    # Evaluate initial population
    fitness = np.array([objective(ind) for ind in pop])
    
    # Track best
    best_idx = np.argmin(fitness)
    best_x = pop[best_idx].copy()
    best_f = fitness[best_idx]
    
    for gen in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        new_pop = []
        new_fitness = []
        
        for i in range(pop_size):
            r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
            
            F = random.uniform(*mutation)
            mutant = pop[r1] + F * (pop[r2] - pop[r3])
            
            for j in range(n_dims):
                mutant[j] = np.clip(mutant[j], bounds[j][0], bounds[j][1])
            
            trial = pop[i].copy()
            for j in range(n_dims):
                if random.random() < crossover_prob:
                    trial[j] = mutant[j]
            
            f_trial = objective(trial)
            
            if f_trial <= fitness[i]:
                new_pop.append(trial)
                new_fitness.append(f_trial)
            else:
                new_pop.append(pop[i])
                new_fitness.append(fitness[i])
        
        pop = np.array(new_pop)
        fitness = np.array(new_fitness)
        
        gen_best_idx = np.argmin(fitness)
        if fitness[gen_best_idx] < best_f:
            best_f = fitness[gen_best_idx]
            best_x = pop[gen_best_idx].copy()
    
    return best_x, best_f


def _local_refine_lbfgs(centers, w, h, maxiter=300):
    """
    Local refinement using L-BFGS-B to optimize sum of max feasible radii.
    This is a differentiable approximation: we optimize a smooth objective
    based on distances.
    """
    n = len(centers)
    params = centers.flatten().copy()
    
    def obj(p):
        c = p.reshape(n, 2)
        # Clip to box
        c[:, 0] = np.clip(c[:, 0], 0, w)
        c[:, 1] = np.clip(c[:, 1], 0, h)
        
        total_r = 0.0
        for i in range(n):
            xi, yi = c[i]
            r = min(xi, w - xi, yi, h - yi)
            if r <= 0:
                return 1e6
            for j in range(n):
                if i != j:
                    dx = xi - c[j][0]
                    dy = yi - c[j][1]
                    d = np.sqrt(dx * dx + dy * dy)
                    r = min(r, d / 2.0)
                    if r <= 0:
                        return 1e6
            total_r += r
        return -total_r
    
    bounds = []
    for _ in range(n):
        bounds.append((0, w))
        bounds.append((0, h))
    
    try:
        result = minimize(
            obj, params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if result.fun < obj(params):
            new_centers = result.x.reshape(n, 2)
            new_centers[:, 0] = np.clip(new_centers[:, 0], 0, w)
            new_centers[:, 1] = np.clip(new_centers[:, 1], 0, h)
            return new_centers
    except Exception:
        pass
    
    return centers


def _local_refine_coord(centers, w, h, max_iters=100, step_size=0.01):
    """
    Coordinate descent local refinement.
    """
    n = len(centers)
    centers = centers.copy()
    
    def compute_score(c):
        radii, total = _max_feasible_radii(c, w, h)
        return total
    
    current_score = compute_score(centers)
    
    for it in range(max_iters):
        improved = False
        for i in range(n):
            xi, yi = centers[i]
            
            for dx in [-step_size, 0, step_size]:
                nx = xi + dx
                if 0 < nx < w:
                    test = centers.copy()
                    test[i][0] = nx
                    test_score = compute_score(test)
                    if test_score > current_score + 1e-10:
                        centers[i][0] = nx
                        current_score = test_score
                        improved = True
            
            for dy in [-step_size, 0, step_size]:
                ny = yi + dy
                if 0 < ny < h:
                    test = centers.copy()
                    test[i][1] = ny
                    test_score = compute_score(test)
                    if test_score > current_score + 1e-10:
                        centers[i][1] = ny
                        current_score = test_score
                        improved = True
        
        if not improved:
            step_size *= 0.5
            if step_size < 1e-6:
                break
    
    return centers, current_score


def _validate(w, h, x, y, r, tol=1e-9):
    """Check if the packing is valid."""
    if w < 0.1 or h < 0.1:
        return False
    if np.any(r < -tol):
        return False
    if np.any(x - r < -tol) or np.any(w - x - r < -tol):
        return False
    if np.any(y - r < -tol) or np.any(h - y - r < -tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            if dist_sq < (r[i] + r[j] - tol) ** 2:
                return False
    return True


def _grid_init(w):
    """Create a grid initialization for N circles in rectangle w x (2-w)."""
    h = 2.0 - w
    aspect = w / h
    if aspect >= 1.0:
        n_cols = int(np.round(np.sqrt(N * aspect)))
        n_rows = int(np.ceil(N / n_cols))
    else:
        n_rows = int(np.round(np.sqrt(N / aspect)))
        n_cols = int(np.ceil(N / n_rows))
    
    while n_cols * n_rows < N:
        if aspect >= 1.0:
            n_cols += 1
        else:
            n_rows += 1
    
    x = np.zeros(N)
    y = np.zeros(N)
    
    cell_w = w / n_cols
    cell_h = h / n_rows
    
    idx = 0
    for i in range(n_rows):
        for j in range(n_cols):
            if idx >= N:
                break
            x[idx] = (j + 0.5) * cell_w
            y[idx] = (i + 0.5) * cell_h
            idx += 1
    return x, y


def construct():
    start_time = time.time()
    
    best_score = -np.inf
    best_result = None
    
    def update_best(w, h, centers):
        nonlocal best_score, best_result
        x = centers[:, 0]
        y = centers[:, 1]
        radii, total_r = _max_feasible_radii(centers, w, h)
        if np.any(radii <= 0):
            return
        if _validate(w, h, x, y, radii):
            if total_r > best_score:
                best_score = total_r
                best_result = np.column_stack([x, y, radii])
    
    # Bounds for DE
    # w in [0.3, 1.7], centers in [0, 2]
    bounds = [(0.3, 1.7)]
    for _ in range(N * 2):
        bounds.append((0.01, 1.99))
    
    # Run DE with time limit
    de_time_limit = min(50.0, DEADLINE - time.time() - 30)
    if de_time_limit > 15:
        best_x, best_f = _differential_evolution(
            _objective_de, bounds,
            pop_size=35,
            max_iter=300,
            mutation=(0.5, 1.0),
            crossover_prob=0.7,
            time_limit=de_time_limit
        )
        
        w = best_x[0]
        h = 2.0 - w
        centers = best_x[1:].reshape(N, 2)
        centers[:, 0] = np.clip(centers[:, 0], 0, w)
        centers[:, 1] = np.clip(centers[:, 1], 0, h)
        
        update_best(w, h, centers)
        
        # Local refinement with L-BFGS-B
        if time.time() < DEADLINE - 15:
            refined = _local_refine_lbfgs(centers, w, h, maxiter=200)
            update_best(w, h, refined)
        
        # Local refinement with coordinate descent
        if time.time() < DEADLINE - 5:
            coord_centers, coord_score = _local_refine_coord(refined, w, h, max_iters=80, step_size=0.01)
            update_best(w, h, coord_centers)
    
    # Try a few fixed aspect ratios with grid init + L-BFGS-B refinement
    aspect_ratios = [0.7, 0.8, 0.9, 1.0, 1.1, 1.3]
    for ar in aspect_ratios:
        if time.time() > DEADLINE - 10:
            break
        
        w = 2.0 * ar / (1.0 + ar)
        h = 2.0 - w
        
        x, y = _grid_init(w)
        centers = np.column_stack([x, y])
        
        update_best(w, h, centers)
        
        if time.time() < DEADLINE - 10:
            refined = _local_refine_lbfgs(centers, w, h, maxiter=150)
            update_best(w, h, refined)
    
    # Final fallback
    if best_result is None:
        x, y = _grid_init(1.0)
        centers = np.column_stack([x, y])
        radii, _ = _max_feasible_radii(centers, 1.0, 1.0)
        best_result = np.column_stack([x, y, radii])
    
    return best_result
