import numpy as np
import time

def construct():
    N = 24
    # Precompute the DFT matrix for the 24th roots of unity
    # M[i, j] = exp(2 * pi * i * i * j / N)
    # Note: P(z) = sum c_k z^k. At z = exp(2*pi*i*m/N), P(z) = sum c_k exp(2*pi*i*m*k/N).
    # This is the DFT of c.
    k_idx = np.arange(N)
    m_idx = np.arange(N)
    # Matrix of shape (N, N) where rows are m, cols are k
    # DFT matrix element (m, k) = exp(2 * pi * i * m * k / N)
    DFT = np.exp(2j * np.pi * np.outer(m_idx, k_idx) / N)
    
    def evaluate_population(pop):
        # pop: (P, N) array of +1/-1
        # vals: (P, N) complex array
        # vals[p, m] = sum_k pop[p, k] * DFT[m, k]
        # This is pop @ DFT.T
        vals = pop @ DFT.T
        return np.max(np.abs(vals), axis=1)
    
    # Initialize population
    pop_size = 500
    population = np.random.choice([-1, 1], size=(pop_size, N))
    
    # Start with a strong seed: Golay-Rudin-Shapiro
    # GRS sequence for length 24 (truncated from 32 or 16?)
    # GRS is defined recursively. For N=24, we can just use a good random or a known good pattern.
    # Let's use a few known good patterns if we had them, or just rely on GA.
    # Actually, let's insert the current best found by the seed program if we can guess it, 
    # or just start with random + GRS logic.
    
    # GRS construction
    def grs(n):
        if n == 1: return [1]
        a = grs(n//2)
        b = grs(n//2)
        return a + b + [-x for x in b]
    # GRS gives powers of 2. 16 is power of 2.
    # For 24, we can take GRS(16) and append 8 bits? Or just use GRS(32) and truncate?
    # GRS(32) has low peak.
    g32 = grs(32)
    population[0] = np.array(g32[:N])
    
    # Evaluate initial fitness
    fitness = evaluate_population(population)
    
    best_idx = np.argmin(fitness)
    best_fitness = fitness[best_idx]
    best_seq = population[best_idx].copy()
    
    start_time = time.time()
    time_limit = 105.0
    
    # GA Parameters
    generations = 500
    elite_size = 10
    tournament_size = 5
    
    # Local Search function
    def local_search(seq, max_steps=100):
        # Try flipping single bits
        current_fitness = np.max(np.abs(DFT @ seq))
        for _ in range(max_steps):
            improved = False
            # Try all single bit flips
            # Vectorized check: 
            # If we flip bit j, new_seq = seq.copy(); new_seq[j] *= -1
            # This is expensive to do one by one. 
            # But N=24 is small.
            # Let's just loop over j
            best_j = -1
            best_f = current_fitness
            for j in range(N):
                new_seq = seq.copy()
                new_seq[j] *= -1
                f = np.max(np.abs(DFT @ new_seq))
                if f < best_f - 1e-9:
                    best_f = f
                    best_j = j
            if best_j != -1:
                seq[best_j] *= -1
                current_fitness = best_f
                improved = True
            else:
                break
        return seq, current_fitness
    
    for gen in range(generations):
        if time.time() - start_time > time_limit:
            break
            
        # Sort population by fitness
        sort_idx = np.argsort(fitness)
        population = population[sort_idx]
        fitness = fitness[sort_idx]
        
        # Update best
        if fitness[0] < best_fitness:
            best_fitness = fitness[0]
            best_seq = population[0].copy()
            
        # Perform local search on the top few individuals to refine them
        # This is the key improvement: GA explores, LS exploits.
        for i in range(min(5, pop_size)):
            # Only do LS if we have time
            if time.time() - start_time > time_limit - 5:
                break
            refined_seq, refined_fit = local_search(population[i].copy(), max_steps=20)
            if refined_fit < fitness[i]:
                population[i] = refined_seq
                fitness[i] = refined_fit
                if refined_fit < best_fitness:
                    best_fitness = refined_fit
                    best_seq = refined_seq.copy()
        
        # Create next generation
        new_population = np.empty_like(population)
        # Elitism
        new_population[:elite_size] = population[:elite_size]
        
        idx = elite_size
        while idx < pop_size:
            # Tournament selection
            candidates = np.random.choice(pop_size, size=tournament_size, replace=False)
            cand_fitness = fitness[candidates]
            winner_idx = candidates[np.argmin(cand_fitness)]
            parent = population[winner_idx]
            
            # Mutation: Bit flip with probability 0.1
            child = parent.copy()
            flip_mask = np.random.random(N) < 0.1
            child[flip_mask] *= -1
            
            new_population[idx] = child
            idx += 1
            
        population = new_population
        fitness = evaluate_population(population)
        
    # Final local search on the absolute best
    if time.time() - start_time < time_limit:
        final_seq, final_fit = local_search(best_seq.copy(), max_steps=50)
        if final_fit < best_fitness:
            best_seq = final_seq
            best_fitness = final_fit

    return [int(c) for c in best_seq]
