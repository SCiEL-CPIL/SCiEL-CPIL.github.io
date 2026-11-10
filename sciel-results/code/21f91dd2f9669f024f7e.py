"""
Strategy: Physics-Based Force-Directed Annealing + Differential Evolution Multi-Start.

Instead of smooth surrogates (log-sum-exp) and gradient-based solvers (SLSQP/L-BFGS-B),
this approach uses:
1. Force-directed repulsion: Circles repel each other with inverse-square forces and
   are attracted to the center/repelled by walls. This naturally finds local maxima
   of the minimum distance without needing differentiability of the min function.
2. Simulated annealing: Temperature is lowered to escape local minima.
3. Differential Evolution (DE): A population-based global optimizer to explore
   different basins of attraction.
4. Nelder-Mead Polish: Final refinement on the exact nonsmooth objective.

This is a fundamentally different optimization landscape traversal: it uses
second-derivative-free, physics-inspired dynamics rather than gradient information.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time
import random

N = 50

def compute_radius(centres):
    """Compute the radius for a given set of centres."""
    if centres.shape != (N, 2):
        return 0.0
    if np.any(~np.isfinite(centres)):
        return 0.0
    # Check boundaries: all coords must be in [-0.5, 0.5]
    if np.any(np.abs(centres) > 0.5 + 1e-9):
        return 0.0
    # Compute min distance between pairs
    dists = pdist(centres, metric='euclidean')
    min_pair_dist = np.min(dists)
    # Compute min distance to boundary
    dist_to_walls = np.array([
        0.5 - np.abs(centres[:, 0]),  # left/right
        0.5 - np.abs(centres[:, 1])   # top/bottom
    ])
    min_wall_dist = np.min(dist_to_walls)
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return max(0.0, r)

def _force_directed_step(centres, temp, k_rep=1.0, k_wall=5.0, dt=0.01):
    """
    One step of force-directed dynamics.
    Forces:
    - Repulsion between circles: F_ij = k_rep * (1/d_ij^2) * direction (if d < 2r_target)
      Actually, for packing, we want to maximize min distance.
      Use a force that pushes apart when close.
      F_ij = k_rep * (1/d_ij^3) * (c_i - c_j) for all pairs, but scaled to be strong when close.
      Better: Use a soft repulsion that becomes strong as d -> 0.
      Let's use F_ij = k_rep * (1/d_ij^2) * (c_i - c_j) / d_ij = k_rep * (c_i - c_j) / d_ij^3
    - Wall repulsion: Push away from walls if too close.
      F_wall = k_wall * (1/d_wall^2) * normal
    """
    forces = np.zeros_like(centres)
    
    # Pairwise repulsion
    # Vectorized computation of pairwise differences
    # For N=50, a double loop is acceptable if optimized, but let's try vectorized
    # diff[i,j] = c_i - c_j
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]  # (N, N, 2)
    dists_sq = np.sum(diff**2, axis=2)  # (N, N)
    
    # Avoid division by zero
    dists_sq[np.eye(N, dtype=bool)] = 1e-10
    
    # Force magnitude: k_rep / dists_sq^1.5 (since F ~ 1/d^2, direction is diff/d, so F_vec = k*diff/d^3)
    # F_vec_ij = k_rep * diff_ij / (dists_sq_ij * sqrt(dists_sq_ij))
    # dists_sq * sqrt(dists_sq) = dists^3
    inv_d3 = 1.0 / (dists_sq * np.sqrt(dists_sq))
    inv_d3[np.eye(N, dtype=bool)] = 0.0
    
    # Force on i from j: k_rep * (c_i - c_j) / d^3
    # Sum over j for each i
    forces += k_rep * np.sum(diff * inv_d3[:, :, np.newaxis], axis=1)
    
    # Wall repulsion
    # Distance to walls
    d_left = centres[:, 0] + 0.5
    d_right = 0.5 - centres[:, 0]
    d_bottom = centres[:, 1] + 0.5
    d_top = 0.5 - centres[:, 1]
    
    # Force components
    # If close to left wall (d_left small), push right (+x)
    # Force = k_wall / d^2 * normal
    # Only apply if d < some threshold, or always apply with 1/d^2
    # To avoid singularity, use max(d, epsilon)
    eps = 1e-6
    
    f_left = k_wall / np.maximum(d_left, eps)**2
    f_right = k_wall / np.maximum(d_right, eps)**2
    f_bottom = k_wall / np.maximum(d_bottom, eps)**2
    f_top = k_wall / np.maximum(d_top, eps)**2
    
    forces[:, 0] += f_left - f_right  # Left wall pushes +x, right wall pushes -x
    forces[:, 1] += f_bottom - f_top  # Bottom wall pushes +y, top wall pushes -y
    
    # Add noise for simulated annealing
    noise = np.random.normal(0, temp, size=centres.shape)
    
    # Update positions
    new_centres = centres + dt * (forces + noise)
    
    # Clip to square
    np.clip(new_centres, -0.5, 0.5, out=new_centres)
    
    return new_centres

def _force_directed_anneal(centres, steps=500, t_start=0.1, t_end=0.001, k_rep=10.0, k_wall=50.0, dt=0.005):
    """
    Run force-directed annealing.
    """
    c = centres.copy()
    for i in range(steps):
        # Linear cooling
        t = t_start * (t_end / t_start) ** (i / steps)
        c = _force_directed_step(c, t, k_rep=k_rep, k_wall=k_wall, dt=dt)
        
    return c

def _de_global_search(pop_size=20, max_iter=50, F=0.8, CR=0.9):
    """
    Differential Evolution for global search.
    Objective: Maximize radius (minimize -radius).
    """
    def objective(x):
        c = x.reshape(N, 2)
        return -compute_radius(c)
    
    # Initialize population
    pop = np.random.uniform(-0.5, 0.5, size=(pop_size, 2 * N))
    # Ensure valid initial population (clip)
    np.clip(pop, -0.5, 0.5, out=pop)
    
    # Evaluate initial population
    fitness = np.array([objective(ind) for ind in pop])
    best_idx = np.argmin(fitness)
    best_x = pop[best_idx].copy()
    best_f = fitness[best_idx]
    
    for gen in range(max_iter):
        for i in range(pop_size):
            # Select three distinct random indices
            r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
            
            # Mutation
            mutant = pop[r1] + F * (pop[r2] - pop[r3])
            
            # Crossover
            trial = np.random.random(2 * N) < CR
            if not np.any(trial):
                trial[np.random.randint(2 * N)] = True
            trial_x = np.where(trial, mutant, pop[i])
            
            # Clip to bounds
            np.clip(trial_x, -0.5, 0.5, out=trial_x)
            
            # Evaluate
            trial_f = objective(trial_x)
            
            # Selection
            if trial_f < fitness[i]:
                pop[i] = trial_x
                fitness[i] = trial_f
                if trial_f < best_f:
                    best_f = trial_f
                    best_x = trial_x.copy()
    
    return best_x.reshape(N, 2), -best_f

def _polish(centres, maxiter=1000):
    """
    High-precision polish using Nelder-Mead on the exact nonsmooth objective.
    """
    def objective(x):
        c = x.reshape(N, 2)
        r = compute_radius(c)
        return -r
    
    res = minimize(objective, centres.flatten(), method='Nelder-Mead',
                   options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
    
    new_c = res.x.reshape(N, 2)
    np.clip(new_c, -0.5, 0.5, out=new_c)
    return new_c

def _grid_init():
    """Generate a grid initialization."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()

def _random_init():
    """Generate a random initialization within the square."""
    return np.random.uniform(-0.5, 0.5, size=(N, 2))

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    best_centres = _grid_init()
    best_r = compute_radius(best_centres)
    
    # Phase 1: Force-Directed Annealing from multiple starts
    # Start 1: Grid
    if time.time() < deadline - 40:
        c_fd = _force_directed_anneal(best_centres, steps=300, t_start=0.05, t_end=0.001, k_rep=10.0, k_wall=50.0, dt=0.005)
        r_fd = compute_radius(c_fd)
        if r_fd > best_r:
            best_r = r_fd
            best_centres = c_fd
    
    # Start 2: Random
    if time.time() < deadline - 40:
        c_rand = _random_init()
        c_fd = _force_directed_anneal(c_rand, steps=300, t_start=0.1, t_end=0.001, k_rep=10.0, k_wall=50.0, dt=0.005)
        r_fd = compute_radius(c_fd)
        if r_fd > best_r:
            best_r = r_fd
            best_centres = c_fd
    
    # Phase 2: Differential Evolution Global Search
    # DE is slower but explores the space more broadly.
    # We run it for a limited number of generations.
    if time.time() < deadline - 30:
        # Use a smaller population and fewer iterations to fit in time
        de_centres, de_r = _de_global_search(pop_size=10, max_iter=30, F=0.8, CR=0.9)
        if de_r > best_r:
            best_r = de_r
            best_centres = de_centres
    
    # Phase 3: Polish the best result found
    if time.time() < deadline - 10:
        c_polished = _polish(best_centres, maxiter=500)
        r_polished = compute_radius(c_polished)
        
        if r_polished > best_r:
            best_r = r_polished
            best_centres = c_polished
    
    # Phase 4: Final tiny local search (Nelder-Mead again with smaller steps or random nudges)
    if time.time() < deadline - 2:
        # A few random nudges to escape any remaining flat spots
        for _ in range(20):
            if time.time() > deadline - 2:
                break
            c_try = best_centres + np.random.normal(0, 0.001, size=(N, 2))
            np.clip(c_try, -0.5, 0.5, out=c_try)
            r_try = compute_radius(c_try)
            if r_try > best_r:
                best_r = r_try
                best_centres = c_try
    
    # Final validation
    best_centres = np.clip(best_centres, -0.5, 0.5)
    if np.any(~np.isfinite(best_centres)):
        best_centres = _grid_init()
    
    return best_centres
