import numpy as np
import time
import random

def _eval_norm_batch(coeffs_list):
    """
    Evaluate sup-norm for a batch of coefficient vectors.
    coeffs_list: List of arrays of shape (24,) with values in {+1, -1}.
    Returns: List of sup-norm values.
    """
    if not coeffs_list:
        return []
    
    n = len(coeffs_list[0])
    N = 4096
    k = np.arange(n)
    theta = 2 * np.pi * np.arange(N) / N
    # Pre-compute DFT matrix: shape (n, N)
    # DFT[j, k] = exp(i * j * theta_k)
    # We need P(z_j) = sum_k c_k z_j^k = sum_k c_k exp(i * k * theta_j)
    # So we need matrix M where M[k, j] = exp(i * k * theta_j)
    # Then P_j = sum_k c_k M[k, j]
    # This is coeffs @ M if coeffs is (batch, n) and M is (n, N).
    
    # Optimization: Since coeffs are +/- 1, we can map to 0/1 or just use them directly.
    # Let's create the matrix M.
    # To save memory/time, we can compute this once if possible, but n is fixed (24).
    # M shape: (24, 4096). This is small.
    M = np.exp(1j * np.outer(k, theta)) # shape (24, 4096)
    
    # coeffs_matrix: (batch_size, 24)
    coeffs_matrix = np.array(coeffs_list)
    
    # P_matrix: (batch_size, 4096)
    # P_matrix[i, j] = sum_k coeffs_matrix[i, k] * M[k, j]
    P_matrix = coeffs_matrix @ M
    
    mags = np.abs(P_matrix) # (batch_size, 4096)
    
    # Find max magnitude for each individual
    # To be more accurate, we should refine the peaks.
    # For speed in the GA, we approximate the sup norm by the max over the grid.
    # This is a lower bound on the true sup norm.
    # However, for ranking, it's usually sufficient.
    # We will refine the top candidates later.
    max_vals = np.max(mags, axis=1)
    
    return max_vals

def _refine_norm_fast(coeffs):
    """
    High-accuracy sup-norm evaluation for a single candidate.
    """
    n = len(coeffs)
    N = 4096
    k = np.arange(n)
    theta = 2 * np.pi * np.arange(N) / N
    M = np.exp(1j * np.outer(k, theta))
    P = coeffs @ M
    mags = np.abs(P)
    
    # Find top peaks and refine
    top_idx = np.argsort(mags)[-5:]
    max_val = 0.0
    
    for i in top_idx:
        theta0 = theta[i]
        a = theta0 - 2 * np.pi / N
        b = theta0 + 2 * np.pi / N
        
        def f(t):
            z = np.exp(1j * t)
            powers = z ** k
            val = np.dot(coeffs, powers)
            return abs(val)
        
        gr = (np.sqrt(5) - 1) / 2
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = f(c)
        fd = f(d)
        
        for _ in range(40):
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = f(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = f(d)
        
        peak = max(fc, fd)
        if peak > max_val:
            max_val = peak
    
    return max_val

def _bit_to_coeffs(bitmask):
    """Convert 24-bit integer to array of +1/-1."""
    coeffs = np.zeros(24)
    for i in range(24):
        if (bitmask >> i) & 1:
            coeffs[i] = 1.0
        else:
            coeffs[i] = -1.0
    return coeffs

def _coeffs_to_bit(coeffs):
    """Convert array of +1/-1 to 24-bit integer."""
    bitmask = 0
    for i in range(24):
        if coeffs[i] > 0:
            bitmask |= (1 << i)
    return bitmask

def _genetic_algorithm(time_budget=100.0):
    n = 24
    pop_size = 200
    start_time = time.time()
    
    # Initialize population
    # Mix of random and structured
    population = []
    # Random
    for _ in range(pop_size // 2):
        population.append(random.getrandbits(n))
    # Structured: All 0, All 1, Alternating, etc.
    population.append(0) # All -1
    population.append((1 << n) - 1) # All +1
    population.append(0b010101010101010101010101) # Alternating
    population.append(0b111111110000000000000000) # Half/Half
    
    # Pad with random if needed
    while len(population) < pop_size:
        population.append(random.getrandbits(n))
        
    # Evaluate initial population
    # Convert to coeffs for evaluation
    # To speed up, we can evaluate in batches
    # But _eval_norm_batch expects a list of arrays.
    # Let's just evaluate one by one for simplicity in the first pass, or batch them.
    # Batch is better.
    
    def eval_pop(pop):
        coeffs_list = [_bit_to_coeffs(b) for b in pop]
        scores = _eval_norm_batch(coeffs_list)
        return scores
    
    scores = eval_pop(population)
    
    # Sort population by score (ascending, lower is better)
    # We want to keep the best ones.
    # Create list of (score, bitmask)
    scored_pop = list(zip(scores, population))
    scored_pop.sort(key=lambda x: x[0])
    
    best_score = scored_pop[0][0]
    best_bit = scored_pop[0][1]
    
    # GA Loop
    while time.time() - start_time < time_budget:
        # Selection: Tournament
        # Crossover: Uniform
        # Mutation: Bit flip
        
        new_population = []
        
        # Elitism: Keep top 10%
        elite_count = max(1, pop_size // 10)
        for i in range(elite_count):
            new_population.append(scored_pop[i][1])
            
        # Fill rest with offspring
        while len(new_population) < pop_size:
            # Select parents
            # Tournament size 5
            parent1_idx = random.randint(0, pop_size - 1)
            parent2_idx = random.randint(0, pop_size - 1)
            # Simple selection: pick best of 5
            candidates = [random.randint(0, pop_size - 1) for _ in range(5)]
            p1 = min(candidates, key=lambda i: scored_pop[i][0])
            candidates = [random.randint(0, pop_size - 1) for _ in range(5)]
            p2 = min(candidates, key=lambda i: scored_pop[i][0])
            
            bit1 = scored_pop[p1][1]
            bit2 = scored_pop[p2][1]
            
            # Uniform Crossover
            mask = random.getrandbits(n)
            child = (bit1 & mask) | (bit2 & ~mask)
            
            # Mutation
            # Probability decreases over time
            progress = (time.time() - start_time) / time_budget
            mut_prob = 0.1 * (1.0 - progress) + 0.01
            for i in range(n):
                if random.random() < mut_prob:
                    child ^= (1 << i)
                    
            new_population.append(child)
            
        # Evaluate new population
        scores = eval_pop(new_population)
        scored_pop = list(zip(scores, new_population))
        scored_pop.sort(key=lambda x: x[0])
        
        if scored_pop[0][0] < best_score:
            best_score = scored_pop[0][0]
            best_bit = scored_pop[0][1]
            
    # Final Refinement on top candidates
    # Take top 5 from the last generation
    top_bits = [scored_pop[i][1] for i in range(min(5, len(scored_pop)))]
    
    # Add the global best found so far
    if best_bit not in top_bits:
        top_bits.append(best_bit)
        
    final_best_score = float('inf')
    final_best_bit = best_bit
    
    for bit in top_bits:
        coeffs = _bit_to_coeffs(bit)
        # Local Search: Hill Climbing
        improved = True
        while improved and (time.time() - start_time < time_budget + 15.0): # Allow some extra time for refinement
            improved = False
            current_score = _refine_norm_fast(coeffs)
            if current_score < final_best_score:
                final_best_score = current_score
                final_best_bit = _coeffs_to_bit(coeffs)
            
            # Try flipping each bit
            for i in range(n):
                new_bit = bit ^ (1 << i)
                new_coeffs = _bit_to_coeffs(new_bit)
                new_score = _refine_norm_fast(new_coeffs)
                if new_score < current_score:
                    bit = new_bit
                    coeffs = new_coeffs
                    improved = True
                    break # Restart loop with new bit
                
    return _bit_to_coeffs(final_best_bit)

def construct():
    n = 24
    start_time = time.time()
    time_limit = 115.0
    
    # Run GA
    best_coeffs = _genetic_algorithm(time_budget=time_limit)
    
    # Final check and return
    # Ensure output is list of ints
    return [int(c) for c in best_coeffs]
