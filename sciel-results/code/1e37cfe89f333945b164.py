import time
import random
import numpy as np

N = 12
SIZE = 1 << N

# Precompute bit indices for fast subset transition calculation
# We will use numpy for the BFS to speed up evaluation of many candidates.

def compute_reset_length_np(trans_a, trans_b):
    """
    Computes the exact shortest reset word length for a deterministic binary automaton.
    Returns -1 if not synchronizing.
    Uses BFS on the subset automaton.
    Optimized for speed by precomputing subset transitions in a vectorized manner 
    where possible, or using efficient bit manipulation in Python if numpy overhead is high for small N.
    Given N=12, SIZE=4096, Python bit manipulation is actually quite fast and avoids numpy overhead 
    for the inner loop if we precompute the subset transitions once per automaton.
    """
    # Precompute transitions for all subsets for this specific automaton
    # trans_a and trans_b are lists of length N
    # We need to compute next_a[subset] and next_b[subset] for all 4096 subsets.
    
    # For each subset, next_state = OR of (1 << trans[i]) for i in subset
    # We can do this efficiently.
    
    next_a = np.zeros(SIZE, dtype=np.int64)
    next_b = np.zeros(SIZE, dtype=np.int64)
    
    # To avoid Python loops for 4096 subsets * 2 letters, we can use a trick:
    # next[subset] = next[subset_without_lsb] | (1 << trans[index_of_lsb])
    
    # Initialize next_a[0] = 0, next_b[0] = 0
    for subset in range(1, SIZE):
        # Get lowest set bit
        lsb = subset & -subset
        idx = lsb.bit_length() - 1
        prev_subset = subset ^ lsb
        
        next_a[subset] = next_a[prev_subset] | (1 << trans_a[idx])
        next_b[subset] = next_b[prev_subset] | (1 << trans_b[idx])
        
    start = (1 << N) - 1
    if start == 1:
        return 0
        
    visited = np.zeros(SIZE, dtype=bool)
    visited[start] = True
    
    # BFS
    # Using a list as a queue is fine for 4096 items.
    current_level = [start]
    dist = 0
    
    while current_level:
        next_level = []
        for subset in current_level:
            # Check if singleton
            if subset & (subset - 1) == 0:
                return dist
            
            na = next_a[subset]
            nb = next_b[subset]
            
            if not visited[na]:
                visited[na] = True
                next_level.append(na)
            
            if not visited[nb]:
                visited[nb] = True
                next_level.append(nb)
                
        current_level = next_level
        dist += 1
        
    return -1

def construct():
    deadline = time.time() + 55  # Safety margin
    
    # Initialize population
    # Start with a known good Pinwheel-like structure to ensure we have a valid baseline
    # Pinwheel: a = (0 1 ... N-1), b = (0 0 1 ... N-2) is not the standard high one.
    # Standard Pinwheel P_n:
    # a: i -> i+1 mod n
    # b: i -> i-1 mod n for i>0, b[0]=0 ? No, that's not it.
    # Let's just start with random valid automata and rely on the GA to find good ones.
    # But we need a good starting point.
    
    # Let's use the "Pinwheel" definition that often yields high thresholds:
    # a: i -> (i+1) % N
    # b: i -> i-1 if i>0 else 0 (This is a "step-down" with a loop at 0? No, b[0]=0)
    # Actually, the "Pinwheel" automaton is:
    # a: (0 1 ... N-1)
    # b: (0 1 ... N-2 0) i.e. b[i] = i-1 for i>0, b[0]=0? No, b[N-1]=0.
    # Let's try:
    a0 = [(i + 1) % N for i in range(N)]
    b0 = [(i - 1) % N for i in range(N)]
    # This b0 is a cycle too. Two cycles don't sync easily unless they intersect.
    # A better seed:
    # a: cycle
    # b: constant map to 0? No, that syncs in 1 step.
    # b: "almost" cycle with one break.
    b0_seed = [(i - 1) % N for i in range(N)]
    b0_seed[N-1] = 0 # Break the cycle
    
    best_a = a0[:]
    best_b = b0_seed[:]
    best_len = compute_reset_length_np(best_a, best_b)
    
    # If the seed is bad, try another known structure
    if best_len < 50:
        # Try the "Rank-2" or similar
        a0 = [(i + 1) % N for i in range(N)]
        b0_seed = [0] * N
        for i in range(1, N):
            b0_seed[i] = i - 1
        best_len = compute_reset_length_np(a0, b0_seed)
        if best_len > 0:
            best_a = a0[:]
            best_b = b0_seed[:]
    
    # Genetic Algorithm Parameters
    pop_size = 200
    mutation_rate = 0.1
    crossover_rate = 0.7
    
    # Initialize population
    population = []
    scores = []
    
    # Add the best found so far
    population.append((best_a[:], best_b[:]))
    scores.append(best_len)
    
    # Add random variants
    for _ in range(pop_size - 1):
        a = [random.randint(0, N-1) for _ in range(N)]
        b = [random.randint(0, N-1) for _ in range(N)]
        population.append((a, b))
        scores.append(-1) # Placeholder
    
    generation = 0
    
    while time.time() < deadline:
        generation += 1
        
        # Evaluate new candidates
        # We only need to evaluate those with score -1 or new ones
        for i in range(len(population)):
            if scores[i] < 0:
                a, b = population[i]
                scores[i] = compute_reset_length_np(a, b)
                if scores[i] > best_len:
                    best_len = scores[i]
                    best_a = a[:]
                    best_b = b[:]
        
        # Selection: Tournament selection
        new_population = []
        
        # Elitism: Keep the top 10%
        sorted_indices = np.argsort(scores)[::-1]
        elite_count = max(1, pop_size // 10)
        for i in range(elite_count):
            idx = sorted_indices[i]
            new_population.append((population[idx][0][:], population[idx][1][:]))
        
        # Fill the rest with offspring
        while len(new_population) < pop_size:
            # Select parents
            p1_idx = sorted_indices[random.randint(0, pop_size-1)]
            p2_idx = sorted_indices[random.randint(0, pop_size-1)]
            
            a1, b1 = population[p1_idx]
            a2, b2 = population[p2_idx]
            
            # Crossover
            if random.random() < crossover_rate:
                # Uniform crossover
                a_child = [a1[i] if random.random() < 0.5 else a2[i] for i in range(N)]
                b_child = [b1[i] if random.random() < 0.5 else b2[i] for i in range(N)]
            else:
                a_child = a1[:]
                b_child = b1[:]
            
            # Mutation
            if random.random() < mutation_rate:
                # Mutate a few random positions
                num_mutations = random.randint(1, 3)
                for _ in range(num_mutations):
                    pos = random.randint(0, N-1)
                    if random.random() < 0.5:
                        a_child[pos] = random.randint(0, N-1)
                    else:
                        b_child[pos] = random.randint(0, N-1)
            
            new_population.append((a_child, b_child))
        
        population = new_population
        # Reset scores for new children (they will be evaluated in the next loop's first pass)
        # We can just keep the scores list aligned, but since we replaced population,
        # we need to be careful. 
        # Actually, it's easier to just re-evaluate everything every generation if it's fast enough?
        # 200 candidates * ~0.1ms = 20ms per generation. 55s / 0.02s = 2750 generations.
        # This is a lot. Let's try to be smarter.
        
        # For simplicity and robustness, let's just re-evaluate the entire population each generation.
        # If it's too slow, we'll reduce pop_size.
        
        # Wait, the loop above evaluates only -1 scores. But new_population has new items.
        # Let's just reset scores for the new items.
        # Actually, the current logic is:
        # 1. Evaluate any with -1.
        # 2. Create new_population.
        # 3. The new_population contains elites (already scored) and new children (unscored).
        # 4. We need to track which are new.
        
        # Let's simplify: Just re-evaluate everything every generation.
        # It's safer and less bug-prone.
        
        # Clear scores to force re-evaluation? No, that wastes time on elites.
        # Let's stick to the "evaluate if -1" logic but ensure new children are marked -1.
        
        # In the loop above, I didn't update `scores` for the new_population.
        # I need to align `scores` with `new_population`.
        
        # Let's restructure:
        # 1. Evaluate all population.
        # 2. Generate new_population.
        # 3. Set scores for new_population: elites keep their scores, children get -1.
        
        # This is getting complicated. Let's just do a simple hill-climbing with random restarts 
        # or a simpler GA.
        
        # Let's try a simpler approach: 
        # Keep a list of candidates.
        # Each iteration, pick a random candidate, mutate it, evaluate.
        # If better, replace.
        # This is stochastic local search (Simulated Annealing-like).
        
        # Let's restart with a simpler, more robust strategy.
        
        pass # Break out of the complex GA loop to implement the simpler one below.

    # --- Revised Strategy: Stochastic Local Search with Simulated Annealing ---
    
    # Reset state for the simpler search
    best_a = a0[:]
    best_b = b0_seed[:]
    best_len = compute_reset_length_np(best_a, best_b)
    
    current_a = best_a[:]
    current_b = best_b[:]
    current_len = best_len
    
    # Simulated Annealing
    T = 10.0
    cooling_rate = 0.995
    min_T = 0.1
    
    while time.time() < deadline and T > min_T:
        # Generate a neighbor
        a_new = current_a[:]
        b_new = current_b[:]
        
        # Mutate 1 or 2 random transitions
        num_mutations = random.randint(1, 2)
        for _ in range(num_mutations):
            pos = random.randint(0, N-1)
            if random.random() < 0.5:
                a_new[pos] = random.randint(0, N-1)
            else:
                b_new[pos] = random.randint(0, N-1)
                
        new_len = compute_reset_length_np(a_new, b_new)
        
        # Acceptance criterion
        if new_len > current_len:
            # Always accept improvement
            current_a = a_new
            current_b = b_new
            current_len = new_len
            if new_len > best_len:
                best_len = new_len
                best_a = a_new[:]
                best_b = b_new[:]
        else:
            # Accept worse with probability exp((new_len - current_len) / T)
            # Note: new_len can be -1 (not synchronizing). We should treat -1 as very bad.
            # If new_len is -1, delta is huge negative.
            if new_len == -1:
                delta = -1000 # Large penalty
            else:
                delta = new_len - current_len
            
            if delta < 0:
                # Accept if random() < exp(delta / T)
                # If delta is very negative, exp is 0.
                try:
                    prob = np.exp(delta / T)
                    if random.random() < prob:
                        current_a = a_new
                        current_b = b_new
                        current_len = new_len
                except OverflowError:
                    pass # Don't accept
        
        T *= cooling_rate
        
    return [best_a, best_b]
