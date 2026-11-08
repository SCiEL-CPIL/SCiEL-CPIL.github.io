"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _c3_upper_bound(x):
    """
    Compute the upper bound on C3 for a signed step function.
    Metric: h * max(|autoconvolution|)
    """
    n_fft = 2 * N
    f_x = np.fft.rfft(x, n=n_fft)
    f_conv = f_x * f_x
    conv = np.fft.irfft(f_conv, n=n_fft)[:2*N-1]
    h = 0.5 / N
    return h * np.max(np.abs(conv))

def _local_search(x, time_limit):
    """
    Fast local search using single flips.
    """
    x = x.copy()
    best_score = _c3_upper_bound(x)
    start_time = time.time()
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(N):
            old_val = x[i]
            x[i] = -old_val
            new_score = _c3_upper_bound(x)
            if new_score < best_score:
                best_score = new_score
                improved = True
            else:
                x[i] = old_val
    return x, best_score

def _genetic_algorithm(deadline):
    """
    Genetic Algorithm for optimizing the step function.
    """
    pop_size = 100
    population = np.random.choice([-1.0, 1.0], size=(pop_size, N))
    
    fitnesses = np.array([_c3_upper_bound(ind) for ind in population])
    
    best_idx = np.argmin(fitnesses)
    best_x = population[best_idx].copy()
    best_score = fitnesses[best_idx]
    
    gen = 0
    while time.time() < deadline:
        gen += 1
        new_population = []
        
        # Elitism
        elite_indices = np.argsort(fitnesses)[:5]
        for idx in elite_indices:
            new_population.append(population[idx].copy())
        
        while len(new_population) < pop_size:
            # Tournament selection
            parents = []
            for _ in range(2):
                candidates = np.random.choice(pop_size, 3, replace=False)
                best_candidate = candidates[np.argmin(fitnesses[candidates])]
                parents.append(population[best_candidate])
            
            # Uniform Crossover
            mask = np.random.choice([True, False], size=N)
            child = np.where(mask, parents[0], parents[1])
            
            # Mutation
            p_mut = 0.01
            mut_mask = np.random.random(N) < p_mut
            child[mut_mask] *= -1.0
            
            new_population.append(child)
        
        population = np.array(new_population)
        fitnesses = np.array([_c3_upper_bound(ind) for ind in population])
        
        current_best_idx = np.argmin(fitnesses)
        if fitnesses[current_best_idx] < best_score:
            best_score = fitnesses[current_best_idx]
            best_x = population[current_best_idx].copy()
            
        if time.time() > deadline - 10.0:
            break
    
    return best_x, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    x_best, score_best = _genetic_algorithm(deadline - 5.0)
    
    if time.time() < deadline - 2.0:
        time_remaining = deadline - 2.0 - time.time()
        x_best, score_best = _local_search(x_best, time_remaining)
        
    return x_best
# EVOLVE-BLOCK-END
