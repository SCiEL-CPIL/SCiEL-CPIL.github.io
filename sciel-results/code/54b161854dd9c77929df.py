"""
Circle packing in a perimeter-4 rectangle.
Uses differential evolution to optimize circle centers and box aspect ratio,
maximizing the sum of maximum feasible radii.
"""

import numpy as np
import time
import random

N = 21
DEADLINE = 115.0  # Safety margin before 120s limit


def _objective(x):
    """
    Objective function: returns negative sum of max feasible radii.
    x = [aspect_ratio, x1, y1, x2, y2, ..., xN, yN]
    aspect_ratio determines w and h with w + h = 2.
    """
    aspect = x[0]
    w = aspect
    h = 2.0 - w
    
    if w < 0.15 or h < 0.15:
        return 1e6  # Penalize invalid box
    
    centers = x[1:].reshape(N, 2)
    
    # Compute max feasible radius for each circle
    total_r = 0.0
    for i in range(N):
        xi, yi = centers[i]
        # Distance to walls
        r = min(xi, w - xi, yi, h - yi)
        if r <= 0:
            return 1e6  # Circle outside box
        
        # Distance to other circles
        for j in range(N):
            if i != j:
                dx = xi - centers[j][0]
                dy = yi - centers[j][1]
                d = np.sqrt(dx * dx + dy * dy)
                r = min(r, d / 2.0)
                if r <= 0:
                    break
        
        if r <= 0:
            return 1e6
        
        total_r += r
    
    return -total_r


def _differential_evolution(objective, bounds, pop_size=40, max_iter=300, 
                            mutation=(0.5, 1.0), crossover_prob=0.7,
                            time_limit=100.0):
    """
    Simple differential evolution implementation.
    bounds: list of (min, max) for each dimension.
    Returns best_x, best_f.
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
        # Check time
        if time.time() - start_time > time_limit:
            break
        
        # Update population
        new_pop = []
        new_fitness = []
        
        for i in range(pop_size):
            # Select three distinct random individuals
            r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
            
            # Mutation
            F = random.uniform(*mutation)
            mutant = pop[r1] + F * (pop[r2] - pop[r3])
            
            # Clamp to bounds
            for j in range(n_dims):
                mutant[j] = np.clip(mutant[j], bounds[j][0], bounds[j][1])
            
            # Crossover
            trial = pop[i].copy()
            for j in range(n_dims):
                if random.random() < crossover_prob:
                    trial[j] = mutant[j]
            
            # Evaluate
            f_trial = objective(trial)
            
            # Selection
            if f_trial <= fitness[i]:
                new_pop.append(trial)
                new_fitness.append(f_trial)
            else:
                new_pop.append(pop[i])
                new_fitness.append(fitness[i])
        
        pop = np.array(new_pop)
        fitness = np.array(new_fitness)
        
        # Update best
        gen_best_idx = np.argmin(fitness)
        if fitness[gen_best_idx] < best_f:
            best_f = fitness[gen_best_idx]
            best_x = pop[gen_best_idx].copy()
        
        # Early stopping: no improvement for 50 generations
        if gen % 50 == 49:
            # Check if we've made significant progress recently
            pass  # Continue for now
    
    return best_x, best_f


def _local_refine(centers, w, h, max_iters=200, step_size=0.01):
    """
    Local refinement: coordinate descent to increase sum of max radii.
    """
    n = len(centers)
    centers = centers.copy()
    
    # Compute initial max radii
    def compute_max_radii(c):
        max_r = np.zeros(n)
        for i in range(n):
            xi, yi = c[i]
            r = min(xi, w - xi, yi, h - yi)
            for j in range(n):
                if i != j:
                    dx = xi - c[j][0]
                    dy = yi - c[j][1]
                    d = np.sqrt(dx * dx + dy * dy)
                    r = min(r, d / 2.0)
            max_r[i] = max(0.0, r)
        return max_r
    
    current_radii = compute_max_radii(centers)
    current_score = np.sum(current_radii)
    
    for it in range(max_iters):
        improved = False
        
        for i in range(n):
            xi, yi = centers[i]
            
            # Try small perturbations in x
            for dx in [-step_size, 0, step_size]:
                nx = xi + dx
                if 0 < nx < w:
                    test_centers = centers.copy()
                    test_centers[i][0] = nx
                    test_radii = compute_max_radii(test_centers)
                    if np.sum(test_radii) > current_score + 1e-10:
                        centers[i][0] = nx
                        current_radii = test_radii
                        current_score = np.sum(test_radii)
                        improved = True
            
            # Try small perturbations in y
            for dy in [-step_size, 0, step_size]:
                ny = yi + dy
                if 0 < ny < h:
                    test_centers = centers.copy()
                    test_centers[i][1] = ny
                    test_radii = compute_max_radii(test_centers)
                    if np.sum(test_radii) > current_score + 1e-10:
                        centers[i][1] = ny
                        current_radii = test_radii
                        current_score = np.sum(test_radii)
                        improved = True
        
        if not improved:
            # Reduce step size
            step_size *= 0.5
            if step_size < 1e-6:
                break
    
    return centers, current_radii, current_score


def construct():
    start_time = time.time()
    
    best_score = 0.0
    best_result = None
    
    # Bounds for differential evolution
    # x[0] = aspect ratio (w), x[1:43] = center coordinates
    # w is in [0.3, 1.7], h = 2 - w
    bounds = [(0.3, 1.7)]
    
    # For each circle, center is in [0, 2] x [0, 2] (loose bounds)
    for _ in range(N * 2):
        bounds.append((0.01, 1.99))
    
    # Run differential evolution
    pop_size = 30
    max_iter = 400
    time_limit = DEADLINE - time.time() - 10  # Leave 10s for local refinement
    
    if time_limit > 20:
        best_x, best_f = _differential_evolution(
            _objective, bounds, 
            pop_size=pop_size, 
            max_iter=max_iter,
            mutation=(0.5, 1.0),
            crossover_prob=0.7,
            time_limit=time_limit
        )
        
        # Extract best solution
        aspect = best_x[0]
        w = aspect
        h = 2.0 - w
        centers = best_x[1:].reshape(N, 2)
        
        # Clamp centers to be inside box
        for i in range(N):
            centers[i][0] = np.clip(centers[i][0], 0.01, w - 0.01)
            centers[i][1] = np.clip(centers[i][1], 0.01, h - 0.01)
        
        # Local refinement
        if time.time() < DEADLINE - 5:
            centers, radii, score = _local_refine(centers, w, h, max_iters=150)
        else:
            # Compute radii directly
            radii = np.zeros(N)
            for i in range(N):
                xi, yi = centers[i]
                r = min(xi, w - xi, yi, h - yi)
                for j in range(N):
                    if i != j:
                        dx = xi - centers[j][0]
                        dy = yi - centers[j][1]
                        d = np.sqrt(dx * dx + dy * dy)
                        r = min(r, d / 2.0)
                radii[i] = max(0.0, r)
            score = np.sum(radii)
        
        if score > best_score:
            best_score = score
            best_result = (centers, radii, w)
    
    # If DE didn't run or failed, try a simpler approach
    if best_result is None or best_score < 0.5:
        # Fallback: grid layout in square
        w = 1.0
        h = 1.0
        side = 5
        centers = []
        for i in range(side):
            for j in range(side):
                if len(centers) < N:
                    x = (j + 0.5) / side
                    y = (i + 0.5) / side
                    centers.append([x, y])
        centers = np.array(centers[:N])
        
        # Compute radii
        radii = np.zeros(N)
        for i in range(N):
            xi, yi = centers[i]
            r = min(xi, w - xi, yi, h - yi)
            for j in range(N):
                if i != j:
                    dx = xi - centers[j][0]
                    dy = yi - centers[j][1]
                    d = np.sqrt(dx * dx + dy * dy)
                    r = min(r, d / 2.0)
            radii[i] = max(0.0, r)
        
        score = np.sum(radii)
        if score > best_score:
            best_score = score
            best_result = (centers, radii, w)
    
    # Return result
    if best_result is not None and best_score > 0:
        centers, radii, w = best_result
        result = np.column_stack((centers, radii))
        return result
    
    # Absolute fallback
    side = 5
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.001
                cells.append([x, y, r])
    return np.array(cells[:N])
