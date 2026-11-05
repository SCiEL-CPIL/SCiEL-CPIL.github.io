import random
import time
import numpy as np

def _get_primes(n):
    """Sieve of Eratosthenes to get primes up to n."""
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def _compute_counts(residues, primes, max_L):
    """Compute coverage counts array for given residues."""
    counts = np.zeros(max_L + 1, dtype=np.int32)
    for r, p in zip(residues, primes):
        if r == 0:
            start = p
        else:
            start = r
        if start > max_L:
            continue
        pos = np.arange(start, max_L + 1, p, dtype=np.int32)
        if len(pos) > 0:
            counts[pos] += 1
    return counts

def _find_L(counts, max_L):
    """Find largest L such that counts[1..L] >= 2."""
    below_two = np.where(counts[1:] < 2)[0]
    if len(below_two) == 0:
        return max_L
    else:
        return below_two[0]

def _get_positions(r, p, max_L):
    """Get positions covered by residue r mod p in 1..max_L."""
    if r == 0:
        start = p
    else:
        start = r
    if start > max_L:
        return np.array([], dtype=np.int32)
    return np.arange(start, max_L + 1, p, dtype=np.int32)

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s limit
    
    primes = _get_primes(1000)
    n_primes = len(primes)
    max_L = 100000
    
    # Initial solution: random residues
    random.seed(42)
    best_residues = [random.randint(0, p - 1) for p in primes]
    counts = _compute_counts(best_residues, primes, max_L)
    best_L = _find_L(counts, max_L)
    best_score = best_L
    
    # Simulated Annealing
    # We want to maximize L.
    # Temperature T starts high and cools down.
    # Move: pick 1 or 2 primes, change their residues.
    # Accept if new_L > old_L, or with probability exp((new_L - old_L) / T) if new_L < old_L.
    # Note: L is integer, so delta is integer.
    
    current_residues = best_residues[:]
    current_counts = counts.copy()
    current_L = best_L
    
    # Precompute positions for all residues for all primes?
    # Memory: sum p_i * (max_L / p_i) * 4 bytes = 168 * 100000 * 4 = 67 MB.
    # This is acceptable. It speeds up moves significantly.
    # positions_cache[i][r] = np.array of positions
    positions_cache = []
    for p in primes:
        cache_p = []
        for r in range(p):
            cache_p.append(_get_positions(r, p, max_L))
        positions_cache.append(cache_p)
        
    T = 10.0
    T_min = 0.1
    cooling_rate = 0.999
    
    # To make moves faster, we can precompute the "delta" effect?
    # No, we just apply +/- 1 to the counts array.
    # Since positions are disjoint for a given prime, we just subtract old, add new.
    
    # Optimization: Instead of copying the whole counts array for every move,
    # we can track changes? No, finding L requires scanning the whole array.
    # So we must have the full counts array to find L.
    # Copying 100k int32s is fast (~0.1ms).
    
    # Let's estimate iterations.
    # 145 seconds available.
    # Each iteration:
    #   - Pick move (1-2 primes)
    #   - Update counts (copy + update)
    #   - Find L (scan)
    #   - Accept/Reject
    #   - If accept, update current state
    
    # Let's run SA for a while, then do local search (greedy) to polish.
    
    # Phase 1: Simulated Annealing
    # Number of iterations depends on time.
    # Let's try to run as many as possible until time runs out, then polish.
    
    # Actually, a hybrid approach:
    # 1. Run SA for ~120 seconds.
    # 2. Run Greedy Local Search for the remaining ~20 seconds.
    
    # But we need to be careful with the time budget.
    # Let's use a time-based loop.
    
    # To speed up SA, we can limit the search space?
    # No, let's try random moves first.
    
    # One issue: L can be very large (up to 100000).
    # If we are already at a high L, small decreases are less critical.
    # The "energy" function is -L. We want to minimize -L.
    
    # Let's implement the loop.
    
    # We will keep the best solution found so far.
    
    # To make the SA more effective, we should probably bias moves towards
    # primes that cover "weak" spots (where count is 1 or 0).
    # But for now, let's stick to random moves to see if it beats greedy.
    
    # Actually, a "smart" random move:
    # Pick a prime i.
    # Pick a new residue r' uniformly.
    # This is what we will do.
    
    # To improve efficiency, we can precompute the list of positions for each prime/residue pair.
    # We did that in positions_cache.
    
    # Let's refine the move generation:
    # 1. Single move: change residue of one prime.
    # 2. Double move: change residues of two primes. (More expensive, but can escape local minima)
    
    # Let's start with single moves for speed, and maybe add double moves if time permits.
    
    # Also, we should periodically restart if stuck?
    # Or just let SA cool.
    
    # Let's run SA until time_limit_sa = start_time + 120.
    time_limit_sa = start_time + 120
    
    while (time.time() - start_time) < time_limit_sa:
        # Pick a prime
        i = random.randint(0, n_primes - 1)
        p = primes[i]
        old_r = current_residues[i]
        
        # Pick a new residue
        new_r = random.randint(0, p - 1)
        if new_r == old_r:
            continue
            
        # Apply move
        # old_pos = positions_cache[i][old_r]
        # new_pos = positions_cache[i][new_r]
        
        # We need to update current_counts.
        # Since we need to find L, we need the full array.
        # Let's copy it.
        new_counts = current_counts.copy()
        
        old_pos = positions_cache[i][old_r]
        new_pos = positions_cache[i][new_r]
        
        if len(old_pos) > 0:
            new_counts[old_pos] -= 1
        if len(new_pos) > 0:
            new_counts[new_pos] += 1
            
        new_L = _find_L(new_counts, max_L)
        
        # Acceptance criterion
        if new_L > current_L:
            # Always accept improvement
            current_residues[i] = new_r
            current_counts = new_counts
            current_L = new_L
            
            if current_L > best_score:
                best_score = current_L
                best_residues = current_residues[:]
                
        else:
            # Decrease or same
            delta = current_L - new_L
            if delta == 0:
                # Accept with some probability to explore?
                # If L is same, it doesn't hurt.
                # Let's accept with prob 0.5 to mix things up.
                if random.random() < 0.5:
                    current_residues[i] = new_r
                    current_counts = new_counts
            else:
                # delta > 0
                # Probability exp(-delta / T)
                # T is in units of L.
                # If T is 10, delta=1 -> exp(-0.1) ~ 0.9
                # delta=10 -> exp(-1) ~ 0.37
                # delta=50 -> exp(-5) ~ 0.0067
                prob = np.exp(-delta / T)
                if random.random() < prob:
                    current_residues[i] = new_r
                    current_counts = new_counts
                    current_L = new_L
                    
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    # Phase 2: Greedy Local Search (Polish)
    # Use the best solution found so far as the starting point for greedy.
    # This combines the global exploration of SA with the local exploitation of Greedy.
    
    time_limit_greedy = deadline - 5 # Leave 5s for final checks/return
    
    current_residues = best_residues[:]
    current_counts = _compute_counts(current_residues, primes, max_L)
    current_L = _find_L(current_counts, max_L)
    
    improved = True
    while improved and (time.time() - start_time) < time_limit_greedy:
        improved = False
        for i in range(n_primes):
            if (time.time() - start_time) > time_limit_greedy:
                break
            p = primes[i]
            old_r = current_residues[i]
            
            best_new_r = old_r
            best_new_L = current_L
            
            # Try all residues
            for new_r in range(p):
                if new_r == old_r:
                    continue
                if (time.time() - start_time) > time_limit_greedy:
                    break
                    
                new_counts = current_counts.copy()
                old_pos = positions_cache[i][old_r]
                new_pos = positions_cache[i][new_r]
                
                if len(old_pos) > 0:
                    new_counts[old_pos] -= 1
                if len(new_pos) > 0:
                    new_counts[new_pos] += 1
                    
                new_L = _find_L(new_counts, max_L)
                
                if new_L > best_new_L:
                    best_new_L = new_L
                    best_new_r = new_r
                    
            if best_new_r != old_r:
                # Apply best change
                old_pos = positions_cache[i][old_r]
                new_pos = positions_cache[i][best_new_r]
                if len(old_pos) > 0:
                    current_counts[old_pos] -= 1
                if len(new_pos) > 0:
                    current_counts[new_pos] += 1
                current_residues[i] = best_new_r
                current_L = best_new_L
                improved = True
                
                if current_L > best_score:
                    best_score = current_L
                    best_residues = current_residues[:]
                    
    return best_residues

