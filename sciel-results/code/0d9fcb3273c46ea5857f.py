import numpy as np
import time

def construct():
    N = 24
    k_idx = np.arange(N)
    m_idx = np.arange(N)
    # DFT[m, k] = exp(2 * pi * i * m * k / N)
    DFT = np.exp(2j * np.pi * np.outer(m_idx, k_idx) / N)
    
    def evaluate_population(pop):
        # pop: (P, N)
        # vals: (P, N)
        vals = pop @ DFT.T
        return np.max(np.abs(vals), axis=1)

    def local_search(seq, max_steps=20):
        """
        Vectorized local search.
        Tries all single bit flips, picks the best, repeats.
        Then tries all double bit flips, picks the best, repeats.
        """
        current_seq = seq.copy().astype(float)
        current_dft = DFT @ current_seq
        current_max = np.max(np.abs(current_dft))
        
        # 1-flip neighborhood
        for _ in range(max_steps):
            improved = False
            # Compute DFT for all single flips
            # flip_j adds 2 * current_seq[j] * DFT[:, j] to current_dft
            # We want max over j of max_m |current_dft[m] + 2 * current_seq[j] * DFT[m, j]|
            
            # Broadcasting: 
            # current_dft shape (N, 1)
            # DFT shape (N, N) -> DFT[:, j] is (N,)
            # We need to compute for each j in 0..N-1.
            
            # Let's create an array of all flipped DFTs
            # Shape (N, N) where row j is the DFT if we flip bit j
            # term_j = 2 * current_seq[j] * DFT[:, j]
            # flipped_dfts = current_dft[:, np.newaxis] + 2 * current_seq[np.newaxis, :] * DFT
            
            flipped_dfts = current_dft[:, np.newaxis] + 2 * current_seq[np.newaxis, :] * DFT
            flipped_maxs = np.max(np.abs(flipped_dfts), axis=0)
            
            best_j = np.argmin(flipped_maxs)
            if flipped_maxs[best_j] < current_max - 1e-9:
                current_seq[best_j] *= -1
                current_dft = DFT @ current_seq
                current_max = flipped_maxs[best_j]
                improved = True
            else:
                break
        
        if not improved:
            # Try 2-flips
            for _ in range(max_steps // 2):
                improved2 = False
                # Try all pairs (i, j) with i < j
                # This is N*(N-1)/2 = 276 pairs.
                # We can vectorize this too, but it's more complex.
                # Given N=24, a loop over pairs might be okay if we are smart, 
                # or we can just do a few random 2-flips.
                # Let's do a full search over pairs for a few iterations.
                
                best_pair = None
                best_val = current_max
                # Precompute contributions
                # If we flip i and j, delta = 2*seq[i]*DFT[:,i] + 2*seq[j]*DFT[:,j]
                
                # To save time, we only do this if 1-flips stalled
                for i in range(N):
                    for j in range(i+1, N):
                        new_dft = current_dft + 2 * current_seq[i] * DFT[:, i] + 2 * current_seq[j] * DFT[:, j]
                        val = np.max(np.abs(new_dft))
                        if val < best_val - 1e-9:
                            best_val = val
                            best_pair = (i, j)
                
                if best_pair is not None:
                    i, j = best_pair
                    current_seq[i] *= -1
                    current_seq[j] *= -1
                    current_dft = DFT @ current_seq
                    current_max = best_val
                    improved2 = True
                else:
                    break
                if not improved2:
                    break
        else:
             # If 1-flips improved, we might want to restart 2-flips later, 
             # but for speed, let's just return after 1-flips converge or 
             # do one round of 2-flips at the end.
             pass

        return current_seq, current_max

    # Initialize population
    pop_size = 200
    # Use a mix of random and some structured seeds
    population = np.random.choice([-1, 1], size=(pop_size, N))
    
    # Add a few "good" seeds if we know them, or just rely on LS
    # The optimal sequence for N=24 has a peak sidelobe of ~1.258 * sqrt(24) ? No, score is max/sqrt(24).
    # Optimal score 1.2583758. Max |P| = 1.2583758 * sqrt(24) ≈ 6.15.
    # Let's just run the GA.
    
    fitness = evaluate_population(population)
    
    best_idx = np.argmin(fitness)
    best_fitness = fitness[best_idx]
    best_seq = population[best_idx].copy()
    
    # Run local search on the best initial candidate
    refined_seq, refined_fit = local_search(best_seq)
    if refined_fit < best_fitness:
        best_fitness = refined_fit
        best_seq = refined_seq.astype(int)
        population[best_idx] = best_seq
        fitness[best_idx] = best_fitness

    start_time = time.time()
    time_limit = 105.0
    
    # GA Parameters
    generations = 1000
    elite_size = 10
    
    for gen in range(generations):
        if time.time() - start_time > time_limit - 5:
            break
            
        # Sort population
        sort_idx = np.argsort(fitness)
        population = population[sort_idx]
        fitness = fitness[sort_idx]
        
        if fitness[0] < best_fitness:
            best_fitness = fitness[0]
            best_seq = population[0].copy()
            
        # Local search on top individuals
        # Limit LS calls to avoid timeout
        ls_count = 0
        max_ls = 5
        for i in range(min(5, pop_size)):
            if time.time() - start_time > time_limit - 10:
                break
            # Only LS if it's not already locally optimal (rough check: if it's one of the top few)
            # Actually, just do it on the top 3-5.
            if i < 3:
                refined_seq, refined_fit = local_search(population[i])
                if refined_fit < fitness[i]:
                    population[i] = refined_seq.astype(int)
                    fitness[i] = refined_fit
                    if refined_fit < best_fitness:
                        best_fitness = refined_fit
                        best_seq = population[i].copy()
                ls_count += 1
        
        # Create next generation
        new_population = np.empty_like(population)
        new_population[:elite_size] = population[:elite_size]
        
        idx = elite_size
        while idx < pop_size:
            # Tournament selection
            candidates = np.random.choice(pop_size, size=3, replace=False)
            winner_idx = candidates[np.argmin(fitness[candidates])]
            parent = population[winner_idx]
            
            # Mutation: flip 1 or 2 bits
            num_flips = np.random.choice([1, 2], p=[0.7, 0.3])
            child = parent.copy()
            flip_indices = np.random.choice(N, size=num_flips, replace=False)
            child[flip_indices] *= -1
            
            new_population[idx] = child
            idx += 1
            
        population = new_population
        fitness = evaluate_population(population)
        
    # Final polish
    if time.time() - start_time < time_limit:
        final_seq, final_fit = local_search(best_seq)
        if final_fit < best_fitness:
            best_seq = final_seq
            best_fitness = final_fit

    return [int(c) for c in best_seq]
