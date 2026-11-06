import numpy as np
import time

def construct():
    N = 24
    # Precompute the DFT matrix for the 24th roots of unity
    # M[i, j] = exp(2 * pi * i * i * j / N)
    k_idx = np.arange(N)
    m_idx = np.arange(N)
    # DFT matrix element (m, k) = exp(2 * pi * i * m * k / N)
    DFT = np.exp(2j * np.pi * np.outer(m_idx, k_idx) / N)
    # We need DFT.T such that c @ DFT.T gives values at roots
    # c is (N,), DFT.T is (N, N) where row k, col m is DFT[m, k]
    # So (c @ DFT.T)[m] = sum_k c[k] * DFT[m, k] which is P(exp(2pi i m/N))
    
    def evaluate_population(pop):
        # pop: (P, N) array of +1/-1
        # vals: (P, N) complex array
        vals = pop @ DFT.T
        return np.max(np.abs(vals), axis=1)
    
    def local_search(seq, max_steps=200):
        # seq: (N,) array of +1/-1
        # Returns improved seq and its fitness
        current_seq = seq.copy()
        current_vals = current_seq @ DFT.T
        current_fitness = np.max(np.abs(current_vals))
        
        for _ in range(max_steps):
            # Compute the effect of flipping each bit j
            # If we flip bit j, new_val[m] = current_vals[m] - 2 * current_seq[j] * DFT[m, j]
            # We want to find j that minimizes max_m |new_val[m]|
            
            # current_seq is (N,)
            # DFT is (N, N) where DFT[m, j] is the term
            # We need a matrix of shape (N, N) where row j corresponds to flipping bit j
            # And col m corresponds to the value at root m
            
            # Let's construct the matrix of new magnitudes
            # flip_effect[j, m] = |current_vals[m] - 2 * current_seq[j] * DFT[m, j]|
            
            # current_seq[:, None] is (N, 1)
            # DFT.T is (N, N) where DFT.T[j, m] = DFT[m, j]
            # So 2 * current_seq[:, None] * DFT.T is (N, N)
            # We subtract this from current_vals[None, :] (1, N)
            
            # new_vals[j, m] = current_vals[m] - 2 * current_seq[j] * DFT[m, j]
            # Note: DFT.T[j, m] is DFT[m, j].
            
            changes = 2.0 * current_seq[:, None] * DFT.T # Shape (N, N), row j, col m
            new_vals = current_vals[None, :] - changes # Shape (N, N)
            new_mags = np.abs(new_vals) # Shape (N, N)
            
            # For each j (row), find the max magnitude over m (cols)
            peak_mags = np.max(new_mags, axis=1) # Shape (N,)
            
            # Find the bit flip that results in the lowest peak magnitude
            best_j = np.argmin(peak_mags)
            best_f = peak_mags[best_j]
            
            if best_f < current_fitness - 1e-9:
                # Apply the flip
                current_seq[best_j] *= -1
                # Update current_vals efficiently? 
                # Or just recompute? Recomputing is O(N^2) which is fine.
                # But we already have new_vals[best_j] which is the new vector.
                current_vals = new_vals[best_j]
                current_fitness = best_f
            else:
                break
                
        return current_seq, current_fitness

    # Initialize population
    pop_size = 200
    # Start with random
    population = np.random.choice([-1, 1], size=(pop_size, N))
    
    # Add a few structured sequences if possible, e.g. GRS truncated
    def grs(n):
        if n == 1: return np.array([1])
        a = grs(n//2)
        b = grs(n//2)
        return np.concatenate([a, b, -b])
    
    # GRS(32) truncated to 24
    try:
        g32 = grs(32)
        population[0] = g32[:N]
    except:
        pass
        
    # Evaluate initial fitness
    fitness = evaluate_population(population)
    
    # Apply local search to initial population to get good basins
    # This is expensive if done for all, but N=24 is small. Let's do it for all to start strong.
    improved_pop = []
    improved_fit = []
    for i in range(pop_size):
        seq, fit = local_search(population[i])
        improved_pop.append(seq)
        improved_fit.append(fit)
        
    population = np.array(improved_pop)
    fitness = np.array(improved_fit)
    
    best_idx = np.argmin(fitness)
    best_fitness = fitness[best_idx]
    best_seq = population[best_idx].copy()
    
    start_time = time.time()
    time_limit = 105.0
    
    # GA Parameters
    elite_size = 20
    tournament_size = 5
    
    while time.time() - start_time < time_limit:
        # Sort population by fitness
        sort_idx = np.argsort(fitness)
        population = population[sort_idx]
        fitness = fitness[sort_idx]
        
        # Update best
        if fitness[0] < best_fitness:
            best_fitness = fitness[0]
            best_seq = population[0].copy()
            
        # Create next generation
        new_population = np.empty_like(population)
        new_fitness = np.empty_like(fitness)
        
        # Elitism: keep top elites
        new_population[:elite_size] = population[:elite_size]
        new_fitness[:elite_size] = fitness[:elite_size]
        
        idx = elite_size
        while idx < pop_size:
            # Tournament selection
            candidates = np.random.choice(pop_size, size=tournament_size, replace=False)
            cand_fitness = fitness[candidates]
            winner_idx = candidates[np.argmin(cand_fitness)]
            parent = population[winner_idx]
            
            # Mutation: Flip a few bits to explore neighbors
            # Number of flips: 1 or 2
            num_flips = 1 if np.random.random() < 0.5 else 2
            flip_indices = np.random.choice(N, size=num_flips, replace=False)
            child = parent.copy()
            child[flip_indices] *= -1
            
            # Apply local search to the child
            refined_seq, refined_fit = local_search(child)
            
            new_population[idx] = refined_seq
            new_fitness[idx] = refined_fit
            idx += 1
            
        population = new_population
        fitness = new_fitness
        
    # Final local search on the absolute best
    if time.time() - start_time < time_limit:
        final_seq, final_fit = local_search(best_seq.copy())
        if final_fit < best_fitness:
            best_seq = final_seq
            best_fitness = final_fit

    return [int(c) for c in best_seq]
