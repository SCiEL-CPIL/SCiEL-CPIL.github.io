import numpy as np
import time
import random

def construct():
    N = 256
    # Deadline set to 145s to be safe within the 150s kill limit
    deadline = time.time() + 145.0
    
    # Precompute all 4-APs as a list of index tuples
    # We need to check if cols[a] == cols[a+d] == cols[a+2d] == cols[a+3d]
    ap_indices = []
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(max_a):
            ap_indices.append((a, a+d, a+2*d, a+3*d))
    
    # Convert to numpy array for vectorized counting
    # Shape: (num_aps, 4)
    ap_array = np.array(ap_indices, dtype=np.int32)
    num_aps = len(ap_array)
    
    def count_mono(cols):
        """
        Count monochromatic 4-APs.
        cols: numpy array of 0s and 1s
        """
        # Get colors at all AP positions
        # cols[ap_array] gives shape (num_aps, 4)
        c = cols[ap_array]
        # Check if all 4 are equal
        # c[:, 0] == c[:, 1] == c[:, 2] == c[:, 3]
        mono = (c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2]) & (c[:, 2] == c[:, 3])
        return int(np.sum(mono))
        
    def repair(cols):
        """
        Ensure exactly 128 zeros and 128 ones.
        cols: numpy array of 0s and 1s
        """
        num_ones = np.sum(cols)
        target = N // 2
        
        if num_ones > target:
            # Need to flip some 1s to 0s
            excess = num_ones - target
            ones_indices = np.where(cols == 1)[0]
            # Randomly select indices to flip
            flip_idx = np.random.choice(ones_indices, size=excess, replace=False)
            cols[flip_idx] = 0
        elif num_ones < target:
            # Need to flip some 0s to 1s
            deficit = target - num_ones
            zeros_indices = np.where(cols == 0)[0]
            # Randomly select indices to flip
            flip_idx = np.random.choice(zeros_indices, size=deficit, replace=False)
            cols[flip_idx] = 1
            
        return cols

    def crossover(parent1, parent2):
        """
        Uniform crossover.
        """
        mask = np.random.randint(0, 2, size=N)
        child = np.where(mask == 1, parent1, parent2)
        return child

    def mutate(cols, mutation_rate):
        """
        Bit-flip mutation.
        """
        # Determine number of flips
        # Poisson or Binomial? Binomial with small p is standard.
        # Let's use a fixed small number or probability.
        # If we use probability, we need to repair afterwards.
        flips = np.random.binomial(N, mutation_rate)
        if flips > 0:
            flip_idx = np.random.choice(N, size=flips, replace=False)
            cols[flip_idx] = 1 - cols[flip_idx]
        return cols

    # Initialize population
    pop_size = 50
    population = []
    
    # Seed with some diverse initializations
    # 1. Random balanced
    for i in range(pop_size // 2):
        c = np.random.randint(0, 2, size=N)
        c = repair(c)
        population.append(c)
        
    # 2. Alternating
    c_alt = np.array([i % 2 for i in range(N)], dtype=np.int8)
    population.append(c_alt)
    
    # 3. Block pattern (e.g., 16 ones, 16 zeros)
    c_block = np.zeros(N, dtype=np.int8)
    for i in range(0, N, 32):
        c_block[i:i+16] = 1
    population.append(c_block)
    
    # 4. Another random
    for i in range(pop_size - len(population)):
        c = np.random.randint(0, 2, size=N)
        c = repair(c)
        population.append(c)
        
    # Evaluate initial population
    scores = np.array([count_mono(c) for c in population])
    
    best_score = np.min(scores)
    best_individual = population[np.argmin(scores)]
    
    # GA Loop
    generation = 0
    mutation_rate = 0.05
    elite_count = 5
    
    while time.time() < deadline:
        # Sort population by score
        sorted_indices = np.argsort(scores)
        sorted_pop = [population[i] for i in sorted_indices]
        sorted_scores = scores[sorted_indices]
        
        # Update global best
        if sorted_scores[0] < best_score:
            best_score = sorted_scores[0]
            best_individual = sorted_pop[0].copy()
            
        # Elitism: keep the best elite_count individuals
        new_population = sorted_pop[:elite_count].copy()
        
        # Generate offspring
        while len(new_population) < pop_size:
            # Selection: Tournament selection
            # Pick 3 random individuals and choose the best
            idx1 = np.random.randint(0, pop_size)
            idx2 = np.random.randint(0, pop_size)
            idx3 = np.random.randint(0, pop_size)
            
            s1 = scores[idx1]
            s2 = scores[idx2]
            s3 = scores[idx3]
            
            if s1 <= s2 and s1 <= s3:
                p1 = population[idx1]
            elif s2 <= s1 and s2 <= s3:
                p1 = population[idx2]
            else:
                p1 = population[idx3]
                
            # Second parent
            idx1 = np.random.randint(0, pop_size)
            idx2 = np.random.randint(0, pop_size)
            idx3 = np.random.randint(0, pop_size)
            
            s1 = scores[idx1]
            s2 = scores[idx2]
            s3 = scores[idx3]
            
            if s1 <= s2 and s1 <= s3:
                p2 = population[idx1]
            elif s2 <= s1 and s2 <= s3:
                p2 = population[idx2]
            else:
                p2 = population[idx3]
                
            # Crossover
            child = crossover(p1, p2)
            
            # Mutation
            child = mutate(child, mutation_rate)
            
            # Repair
            child = repair(child)
            
            new_population.append(child)
            
        # Evaluate new population
        # We only need to evaluate the new offspring, but for simplicity 
        # and since count_mono is fast, we can re-evaluate all or just the new ones.
        # Let's evaluate the whole new population to keep scores aligned.
        population = new_population
        scores = np.array([count_mono(c) for c in population])
        
        generation += 1
        
        # Adaptive mutation rate? 
        # If stagnation, increase mutation rate?
        # Simple heuristic: if best score hasn't changed for 10 generations, increase mutation
        # For now, keep it fixed or slightly decay.
        # Let's try a slight decay to encourage convergence later, but keep some diversity.
        # Actually, fixed 0.05 is reasonable for N=256.
        
    # Final polish: Local search on the best individual found
    # The GA might leave the best individual not fully optimized locally.
    # Perform a greedy local search on best_individual.
    
    # Precompute neighbors for local search
    # For each position i, which APs does it belong to?
    # We can reuse the ap_array structure.
    # For speed in local search, we can just do a few rounds of greedy flips.
    
    best_cols = best_individual.copy()
    current_score = count_mono(best_cols)
    
    # Greedy local search
    # Try flipping each bit if it improves the score.
    # To be fast, we can do a few passes.
    # Since we have time budget, we can do a more thorough local search here.
    
    # Re-use the delta calculation idea from the seed but optimized?
    # Or just simple hill climbing.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Randomize order to avoid bias? Or sequential?
        # Sequential is faster if we can update scores incrementally, but that's complex.
        # Let's do random order passes.
        indices = np.random.permutation(N)
        for i in indices:
            if time.time() > deadline - 5:
                break
            # Try flip
            old_val = best_cols[i]
            best_cols[i] = 1 - old_val
            new_score = count_mono(best_cols)
            if new_score < current_score:
                current_score = new_score
                improved = True
            else:
                best_cols[i] = old_val
                
    # Convert to list of ints 0/1
    return list(best_cols)

if __name__ == "__main__":
    # Quick test
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Time: {end-start:.2f}s")
    print(f"Length: {len(res)}")
    print(f"Sample: {res[:10]}")
    # Count mono
    N = 256
    ap_indices = []
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(max_a):
            ap_indices.append((a, a+d, a+2*d, a+3*d))
    ap_array = np.array(ap_indices, dtype=np.int32)
    cols = np.array(res, dtype=np.int8)
    c = cols[ap_array]
    mono = (c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2]) & (c[:, 2] == c[:, 3])
    print(f"Mono count: {np.sum(mono)}")
