import numpy as np
import time

N = 400

def _calc_metrics(h):
    """
    Calculate the autoconvolution, integral, and the objective score.
    Returns: max_abs_autoconv, integral, score
    """
    C = np.convolve(h, h, mode='full')
    # The task definition: "maximum absolute autoconvolution divided by squared integral"
    # The seed code excludes the middle term (lag 0) in max_abs_sid calculation:
    # max_abs_sid = np.max(np.abs(np.concatenate([C[:N-1], C[N:]])))
    # This corresponds to lags 1 to N-1.
    max_abs_sid = np.max(np.abs(np.concatenate([C[:N-1], C[N:]])))
    
    integral = np.sum(h)
    int_sq = integral * integral
    
    if int_sq == 0:
        return max_abs_sid, integral, float('inf')
        
    score = max_abs_sid / int_sq
    return max_abs_sid, integral, score

def _rudin_shapiro(n):
    """Generate Rudin-Shapiro sequence of length n (padded/truncated)."""
    seq = []
    a = {0: 1, 1: 1}
    for i in range(n):
        if i not in a:
            if i % 2 == 0:
                a[i] = a[i // 2]
            else:
                a[i] = -a[(i + 1) // 2]
        seq.append(a[i])
    return np.array(seq, dtype=np.int8)

def _local_search(h, rng, deadline, max_iterations=500):
    """
    Perform hill climbing with single-bit and block flips.
    Returns improved sequence and score.
    """
    current_h = h.copy()
    _, _, current_score = _calc_metrics(current_h)
    best_h = current_h.copy()
    best_score = current_score
    
    # Single bit flips
    for _ in range(max_iterations):
        if time.time() > deadline:
            break
        improved = False
        # Randomly select a few positions to try, or scan all? 
        # Scanning all is expensive (400 evals). 
        # Let's try random sampling of flips to be faster, or just a few.
        # Actually, for N=400, 400 convolves is ~400 * 160k ops = 64M ops. 
        # np.convolve is fast. 400 calls might take ~0.1s. 
        # So we can afford a few full scans.
        
        # Try flipping each bit once
        best_flip_idx = -1
        best_flip_score = current_score
        
        # To speed up, we can just pick a random subset of indices if time is tight, 
        # but let's stick to a simple scan if time allows.
        # We will just do a random restart of the scan or a limited scan.
        
        # Let's do a random selection of 20 indices to flip to keep it fast.
        indices = rng.choice(N, size=min(20, N), replace=False)
        for i in indices:
            current_h[i] *= -1
            _, _, s = _calc_metrics(current_h)
            if s < best_flip_score:
                best_flip_score = s
                best_flip_idx = i
                improved = True
            else:
                current_h[i] *= -1
        
        if improved:
            current_h[best_flip_idx] *= -1
            current_score = best_flip_score
            if current_score < best_score:
                best_score = current_score
                best_h = current_h.copy()
        else:
            # Try block flips if single bits fail
            block_size = rng.choice([2, 4])
            i = rng.integers(0, max(1, N - block_size + 1))
            old_block = current_h[i:i+block_size].copy()
            current_h[i:i+block_size] = -old_block
            _, _, s = _calc_metrics(current_h)
            if s < best_score:
                best_score = s
                best_h = current_h.copy()
            else:
                current_h[i:i+block_size] = old_block
                # If no improvement in block either, break to save time
                break
                
    return best_h, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    rng = np.random.default_rng(42)
    
    # Initialize Population
    pop_size = 10
    population = []
    scores = []
    
    # Seed with diverse initializations
    # 1. Rudin-Shapiro
    h_rs = _rudin_shapiro(N)
    population.append(h_rs)
    
    # 2. Random
    for _ in range(pop_size - 1):
        h_rand = rng.choice([-1, 1], size=N).astype(np.int8)
        population.append(h_rand)
        
    # Evaluate initial population
    for h in population:
        _, _, score = _calc_metrics(h)
        scores.append(score)
        
    best_h = population[0].copy()
    best_score = min(scores)
    best_idx = np.argmin(scores)
    best_h = population[best_idx].copy()
    
    # Genetic Algorithm Loop
    # We will keep the best individuals and generate offspring
    # Offspring are created by crossover and mutation, then refined by local search
    
    while time.time() < deadline:
        # Sort population by score
        sorted_indices = np.argsort(scores)
        
        # Keep top half as parents
        num_parents = pop_size // 2
        parents = [population[i].copy() for i in sorted_indices[:num_parents]]
        
        # Generate offspring
        offspring = []
        for _ in range(pop_size - num_parents):
            # Crossover
            p1 = parents[rng.integers(0, num_parents)]
            p2 = parents[rng.integers(0, num_parents)]
            
            # Uniform crossover
            mask = rng.random(N) < 0.5
            child = np.where(mask, p1, p2).astype(np.int8)
            
            # Mutation: Flip a few bits
            num_mutations = rng.integers(1, 5)
            mut_indices = rng.choice(N, size=num_mutations, replace=False)
            child[mut_indices] *= -1
            
            # Refine with local search
            child, score = _local_search(child, rng, deadline)
            offspring.append(child)
            scores.append(score)
            
            # Update global best
            if score < best_score:
                best_score = score
                best_h = child.copy()
                
        # Replace worst individuals in population with offspring
        # The worst are the last ones in sorted_indices
        worst_indices = sorted_indices[num_parents:]
        
        for i, off_idx in enumerate(worst_indices):
            # We need to map the new offspring back to the population slots
            # Since we appended to `scores` and `population` isn't updated yet, 
            # let's manage lists explicitly.
            pass
            
        # Reconstruct population
        # Keep the best parents
        new_population = [population[i].copy() for i in sorted_indices[:num_parents]]
        new_scores = [scores[i] for i in sorted_indices[:num_parents]]
        
        # Add offspring
        # Note: `scores` list now has original + offspring scores.
        # Offsets are tricky. Let's restart the list logic.
        
        # Simpler approach:
        # 1. Combine current population and offspring
        # 2. Sort by score
        # 3. Keep top pop_size
        
        # We need to align scores with population.
        # Current population: population[0..pop_size-1]
        # Current scores: scores[0..pop_size-1] (original)
        # Offspring: offspring[0..len(offspring)-1]
        # Offspring scores: stored in `scores` at indices pop_size ... pop_size+len(offspring)-1
        
        combined_pop = population + offspring
        combined_scores = scores[:pop_size] + scores[pop_size:]
        
        # Sort combined
        combined_indices = np.argsort(combined_scores)
        
        # Keep best pop_size
        population = [combined_pop[i].copy() for i in combined_indices[:pop_size]]
        scores = [combined_scores[i] for i in combined_indices[:pop_size]]
        
        # Update best
        if scores[0] < best_score:
            best_score = scores[0]
            best_h = population[0].copy()
            
    return best_h
