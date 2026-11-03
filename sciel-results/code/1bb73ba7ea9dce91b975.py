import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    """
    Particle Swarm Optimization for packing N equal circles in a unit circle.
    """
    
    def compute_radius(points):
        """Compute the maximum radius for given center positions."""
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pairwise = np.min(dists)
        
        radii = np.sqrt(np.sum(points * points, axis=1))
        min_boundary = 1.0 - np.max(radii)
        
        return min(min_pairwise / 2.0, min_boundary)
    
    def initialize_population(pop_size):
        """Initialize population with diverse configurations."""
        population = np.zeros((pop_size, N, 2))
        for i in range(pop_size):
            if i % 3 == 0:
                r_init = np.sqrt(np.random.rand(N))
                theta_init = np.random.rand(N) * 2 * np.pi
                population[i] = np.column_stack((r_init * np.cos(theta_init), r_init * np.sin(theta_init)))
            elif i % 3 == 1:
                angles = np.linspace(0, 2*np.pi, N, endpoint=False) + np.random.uniform(-0.1, 0.1, N)
                radii = np.random.uniform(0.1, 0.8, N)
                population[i] = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
            else:
                r_init = np.sqrt(np.random.rand(N)) * 0.9
                theta_init = np.random.rand(N) * 2 * np.pi
                population[i] = np.column_stack((r_init * np.cos(theta_init), r_init * np.sin(theta_init)))
        return population
    
    pop_size = 40
    w_init = 0.8
    w_final = 0.4
    c1 = 2.0
    c2 = 2.0
    max_vel_init = 0.3
    max_vel_final = 0.02
    
    start_time = time.time()
    time_limit = 110.0
    
    population = initialize_population(pop_size)
    
    fitness = np.zeros(pop_size)
    for i in range(pop_size):
        fitness[i] = compute_radius(population[i])
    
    pbest = population.copy()
    pbest_fitness = fitness.copy()
    
    global_best_idx = np.argmax(fitness)
    gbest = population[global_best_idx].copy()
    gbest_fitness = fitness[global_best_idx]
    
    velocities = np.random.uniform(-0.05, 0.05, (pop_size, N, 2))
    
    iteration = 0
    while time.time() - start_time < time_limit:
        elapsed = time.time() - start_time
        progress = min(elapsed / time_limit, 1.0)
        
        w = w_init + (w_final - w_init) * progress
        max_vel = max_vel_init + (max_vel_final - max_vel_init) * progress
        
        for i in range(pop_size):
            r1 = np.random.rand(N, 2)
            r2 = np.random.rand(N, 2)
            
            velocities[i] = (w * velocities[i] + 
                           c1 * r1 * (pbest[i] - population[i]) + 
                           c2 * r2 * (gbest - population[i]))
            
            velocities[i] = np.clip(velocities[i], -max_vel, max_vel)
            
            population[i] = population[i] + velocities[i]
            
            radii = np.linalg.norm(population[i], axis=1)
            mask = radii > 1.0
            if np.any(mask):
                population[i][mask] = population[i][mask] / radii[mask][:, np.newaxis] * 0.999
        
        for i in range(pop_size):
            fitness[i] = compute_radius(population[i])
            
            if fitness[i] > pbest_fitness[i]:
                pbest[i] = population[i].copy()
                pbest_fitness[i] = fitness[i]
                
                if fitness[i] > gbest_fitness:
                    gbest = population[i].copy()
                    gbest_fitness = fitness[i]
        
        iteration += 1
        
        if time.time() - start_time > time_limit - 3:
            break
    
    return gbest

# EVOLVE-BLOCK-END
