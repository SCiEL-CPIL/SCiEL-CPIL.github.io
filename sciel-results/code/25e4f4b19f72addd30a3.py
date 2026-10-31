import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 145.0  # Safety margin for 150s limit

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2
AREA_EQUI = 0.5 * H  # Area of the unit equilateral triangle

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

def clamp_bary(bary, eps=1e-9):
    """Clamp barycentric coordinates to the strictly valid triangle region."""
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    sums = u + v
    invalid = sums > (1 - eps)
    if np.any(invalid):
        scale = (1 - eps) / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    return np.column_stack([u, v])

def hard_min_obj(x):
    """
    Objective function for minimization.
    We want to maximize the minimum area, so we minimize the negative of the minimum area.
    """
    bary = x.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def soft_min_obj(x, beta=50.0):
    """
    Soft minimum objective using log-sum-exp approximation.
    This is smoother than hard min, allowing better gradient flow for swarm updates.
    Returns negative soft-min to minimize.
    """
    bary = x.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    # We want to maximize min(areas), so minimize -soft_min(areas)
    # soft_min(areas) approx min(areas) as beta -> inf
    # To avoid overflow, use log-sum-exp trick
    max_a = np.max(areas)
    log_sum_exp = max_a + np.log(np.sum(np.exp(-beta * (areas - max_a))))
    soft_min_val = -1.0 / beta * log_sum_exp
    
    return -soft_min_val

def generate_hexagonal_init():
    """
    Generate a hexagonal grid initialization.
    """
    pts = []
    for i in range(4):
        for j in range(4 - i):
            pts.append([i/3.0, j/3.0])
    pts.append([1/3.0, 1/3.0])
    bary = np.array(pts)
    return clamp_bary(bary)

def generate_random_init(rng):
    """Generate a random valid configuration."""
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return clamp_bary(np.column_stack([u, v]))

def generate_corner_init():
    """Generate an initialization biased towards corners/edges."""
    corner_bary = np.array([
        [0.01, 0.01], 
        [0.98, 0.01], 
        [0.01, 0.98], 
        [0.33, 0.33], 
        [0.5, 0.0],   
        [0.0, 0.5],   
        [0.5, 0.5],   
        [0.25, 0.25],
        [0.75, 0.25],
        [0.25, 0.75],
        [0.5, 0.25]
    ])
    return clamp_bary(corner_bary)

def refine_local(bary, time_limit=5.0, beta=100.0):
    """
    Perform a local refinement using Nelder-Mead on the soft-min objective.
    """
    start_time = time.time()
    
    try:
        # Define objective that switches from soft to hard if time allows?
        # For now, use soft-min with high beta for smoothness, then polish with hard min
        def obj(x):
            return soft_min_obj(x, beta=beta)
            
        res = minimize(
            obj,
            bary.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        
        candidate_bary = clamp_bary(res.x.reshape(N, 2))
        
        # If we have time, do a final hard-min polish
        if time.time() < start_time + time_limit - 0.5:
            def hard_obj(x):
                return hard_min_obj(x)
                
            res2 = minimize(
                hard_obj,
                candidate_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            candidate_bary = clamp_bary(res2.x.reshape(N, 2))
            
        return candidate_bary
    except Exception:
        return bary

def particle_swarm_optimization(initial_bary, time_limit, n_particles=50):
    """
    Particle Swarm Optimization (PSO) in barycentric space.
    """
    start_time = time.time()
    
    # Initialize particles
    # p_best: best position for each particle
    # g_best: global best position
    # v: velocity
    
    # Create initial swarm around the initial_bary
    rng = np.random.default_rng(42)
    
    # Particle 0 is the initial best
    # Other particles are random perturbations
    positions = np.zeros((n_particles, N, 2))
    velocities = np.zeros((n_particles, N, 2))
    
    # First particle is the initial solution
    positions[0] = initial_bary
    
    # Other particles are random or perturbed
    for i in range(1, n_particles):
        # Mix of random and perturbed
        if i % 2 == 0:
            positions[i] = generate_random_init(rng)
        else:
            positions[i] = clamp_bary(initial_bary + rng.normal(0, 0.1, (N, 2)))
        
    # Evaluate initial fitness
    # We minimize -min_area. Lower is better.
    fitness = np.zeros(n_particles)
    p_best = np.zeros_like(positions)
    p_best_fit = np.zeros(n_particles)
    
    g_best = positions[0].copy()
    g_best_fit = np.inf
    
    for i in range(n_particles):
        fit = soft_min_obj(positions[i].flatten(), beta=20.0)
        fitness[i] = fit
        p_best[i] = positions[i]
        p_best_fit[i] = fit
        
        if fit < g_best_fit:
            g_best_fit = fit
            g_best = positions[i].copy()
            
    # PSO Parameters
    c1 = 1.5  # Cognitive coefficient
    c2 = 1.5  # Social coefficient
    w = 0.9   # Inertia weight
    
    while time.time() < start_time + time_limit:
        # Update velocities and positions
        for i in range(n_particles):
            r1 = rng.random(N * 2)
            r2 = rng.random(N * 2)
            
            # Velocity update
            # v = w*v + c1*r1*(p_best - x) + c2*r2*(g_best - x)
            v_update = w * velocities[i] + c1 * r1 * (p_best[i] - positions[i]) + c2 * r2 * (g_best - positions[i])
            
            # Limit velocity to prevent divergence
            max_vel = 0.1
            v_update = np.clip(v_update, -max_vel, max_vel)
            
            velocities[i] = v_update
            positions[i] = positions[i] + velocities[i]
            
            # Clamp positions
            positions[i] = clamp_bary(positions[i])
            
            # Evaluate new position
            fit = soft_min_obj(positions[i].flatten(), beta=20.0)
            
            # Update personal best
            if fit < p_best_fit[i]:
                p_best_fit[i] = fit
                p_best[i] = positions[i]
                
                # Update global best
                if fit < g_best_fit:
                    g_best_fit = fit
                    g_best = positions[i].copy()
                    
        # Decay inertia
        w = max(0.4, w * 0.99)
        
    return g_best, -g_best_fit

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_val = -np.inf
    best_bary = None
    
    rng = np.random.default_rng(42)
    
    # 1. Generate diverse initializations
    inits = [
        generate_hexagonal_init(),
        generate_corner_init()
    ]
    
    # Add some random inits
    for _ in range(3):
        inits.append(generate_random_init(rng))
        
    # 2. PSO Phase
    # Allocate most time to PSO from the best initializations
    
    # Sort inits by quality
    scored_inits = []
    for bary in inits:
        val = -hard_min_obj(bary.flatten())
        scored_inits.append((val, bary))
    scored_inits.sort(key=lambda x: x[0], reverse=True)
    
    # Run PSO from the top 2 initializations
    pso_time_budget = (deadline - time.time()) * 0.7
    
    for i in range(min(2, len(scored_inits))):
        val, bary = scored_inits[i]
        
        if time.time() > deadline - 10.0:
            break
            
        time_allocated = min(pso_time_budget, deadline - time.time() - 10.0)
        if time_allocated < 5.0:
            break
            
        pso_bary, pso_val = particle_swarm_optimization(bary, time_allocated, n_particles=40)
        
        if pso_val > best_val:
            best_val = pso_val
            best_bary = pso_bary
            
    # 3. Local Refinement Phase
    # Polish the best result found by PSO
    if best_bary is not None:
        # Use the best initial if PSO failed or didn't run
        if best_val < scored_inits[0][0]:
            best_bary = scored_inits[0][1]
            best_val = scored_inits[0][0]
            
        local_time = deadline - time.time() - 2.0
        if local_time > 2.0:
            polished_bary = refine_local(best_bary, time_limit=local_time, beta=50.0)
            polished_val = -hard_min_obj(polished_bary.flatten())
            
            if polished_val > best_val:
                best_val = polished_val
                best_bary = polished_bary
                
    # Fallback
    if best_bary is None:
        best_bary = inits[0]
        
    cart = bary_to_cart(best_bary)
    return cart

# EVOLVE-BLOCK-END
