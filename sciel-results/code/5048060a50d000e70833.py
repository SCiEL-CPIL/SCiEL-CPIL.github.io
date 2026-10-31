import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 115.0

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

def bary_to_cart(bary):
    """Convert barycentric coords (u, v) to Cartesian (x, y)."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def calc_areas_bary(bary):
    """Calculate areas of all triangles formed by N points in barycentric coords."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    ui = u[tri_idx[:, 0]]
    vi = v[tri_idx[:, 0]]
    wi = w[tri_idx[:, 0]]
    
    uj = u[tri_idx[:, 1]]
    vj = v[tri_idx[:, 1]]
    wj = w[tri_idx[:, 1]]
    
    uk = u[tri_idx[:, 2]]
    vk = v[tri_idx[:, 2]]
    wk = w[tri_idx[:, 2]]
    
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    
    return np.abs(det)

def clamp_bary(bary, eps=1e-10):
    """Clamp barycentric coordinates to the valid triangle region."""
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    # Ensure u >= 0, v >= 0, u + v <= 1
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    sums = u + v
    invalid = sums > (1 - eps)
    if np.any(invalid):
        scale = (1 - eps) / np.maximum(sums, eps)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    return np.column_stack([u, v])

def hard_min_value(bary):
    """Compute the hard minimum area value."""
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return np.min(areas)

def local_refine(bary_init, time_limit=1.0):
    """
    Refine a single solution using Nelder-Mead.
    """
    start_time = time.time()
    if time.time() - start_time > time_limit:
        return bary_init

    def hard_obj(x):
        return -hard_min_value(x.reshape(N, 2))
    
    try:
        result = minimize(
            hard_obj,
            bary_init.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-16, 'adaptive': True}
        )
        return clamp_bary(result.x.reshape(N, 2))
    except Exception:
        return bary_init

def genetic_algorithm(time_limit=100.0, seed=42):
    """
    Genetic Algorithm for maximizing the minimum triangle area.
    """
    rng = np.random.default_rng(seed)
    start_time = time.time()
    
    pop_size = 60
    elite_size = 5
    mutation_rate = 0.1
    crossover_rate = 0.8
    
    # Initialize population
    # Mix of random, hexagonal, and clustered starts
    population = []
    
    # 1. Random starts
    for _ in range(pop_size // 2):
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        population.append(np.column_stack([u, v]))
        
    # 2. Hexagonal-ish starts
    for scale in [0.2, 0.25, 0.3, 0.4]:
        pts = []
        max_k = int(1.0 / scale) + 1
        for k in range(max_k + 1):
            w_val = k * scale
            if w_val > 1.0:
                break
            num_pts_in_row = int((1 - w_val) / scale) + 1
            for j in range(num_pts_in_row):
                u_val = j * scale
                v_val = (1 - w_val) - u_val
                if u_val >= -1e-9 and v_val >= -1e-9 and u_val + v_val <= 1 + 1e-9:
                    pts.append([u_val, v_val])
        if len(pts) >= N:
            idx = rng.choice(len(pts), N, replace=False)
            population.append(np.array(pts)[idx])
            
    # 3. Centered starts
    center = np.array([1/3, 1/3])
    for r in [0.1, 0.2, 0.3]:
        angles = rng.uniform(0, 2*np.pi, N)
        radii = rng.uniform(0, r, N)
        u = center[0] + radii * np.cos(angles)
        v = center[1] + radii * np.sin(angles)
        population.append(clamp_bary(np.column_stack([u, v])))
        
    # Pad population if needed
    while len(population) < pop_size:
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        population.append(np.column_stack([u, v]))
        
    # Evaluate initial fitness
    fitness = [hard_min_value(ind) for ind in population]
    
    best_val = max(fitness)
    best_bary = population[fitness.index(best_val)].copy()
    
    generation = 0
    while time.time() - start_time < time_limit:
        generation += 1
        
        # Sort by fitness
        sorted_indices = np.argsort(fitness)[::-1]
        population = [population[i] for i in sorted_indices]
        fitness = [fitness[i] for i in sorted_indices]
        
        # Update best
        if fitness[0] > best_val:
            best_val = fitness[0]
            best_bary = population[0].copy()
            
        # Create next generation
        new_population = []
        
        # Elitism
        for i in range(elite_size):
            new_population.append(population[i].copy())
            
        # Fill rest with offspring
        while len(new_population) < pop_size:
            # Tournament Selection
            def select():
                idx1 = rng.integers(0, pop_size)
                idx2 = rng.integers(0, pop_size)
                if fitness[idx1] > fitness[idx2]:
                    return population[idx1]
                else:
                    return population[idx2]
            
            parent1 = select()
            parent2 = select()
            
            if rng.random() < crossover_rate:
                # Arithmetic Crossover
                alpha = rng.random()
                child = alpha * parent1 + (1 - alpha) * parent2
            else:
                # No crossover, just copy
                child = parent1.copy()
                
            # Mutation
            if rng.random() < mutation_rate:
                # Gaussian mutation on all points
                noise = rng.normal(0, 0.05, child.shape)
                child = clamp_bary(child + noise)
            else:
                # Smaller mutation always
                noise = rng.normal(0, 0.01, child.shape)
                child = clamp_bary(child + noise)
                
            new_population.append(child)
            
        # Evaluate new population
        fitness = [hard_min_value(ind) for ind in new_population]
        population = new_population
        
        # Periodic local refinement of top few
        if generation % 10 == 0:
            for i in range(min(3, len(population))):
                # Check time
                if time.time() - start_time > time_limit - 10:
                    break
                refined = local_refine(population[i], time_limit=1.0)
                val = hard_min_value(refined)
                if val > fitness[i]:
                    population[i] = refined
                    fitness[i] = val
                    if val > best_val:
                        best_val = val
                        best_bary = refined.copy()

    return best_bary

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Run GA
    best_bary = genetic_algorithm(time_limit=TIME_LIMIT - 10, seed=42)
    
    # Final polish on the best found
    if time.time() < deadline - 2:
        best_bary = local_refine(best_bary, time_limit=5.0)
        
    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart
