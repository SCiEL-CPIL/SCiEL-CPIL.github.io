import time
import random
import numpy as np

def _primes_upto(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i, is_p in enumerate(sieve) if is_p]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    primes = _primes_upto(1000)
    n_primes = len(primes)
    
    # M is the current maximum check range
    # We start small and expand as we make progress
    M = 2000
    counts = np.zeros(M + 1, dtype=np.int64)
    
    # Initialize residues
    # Random initialization is often sufficient given the strong repair loop
    residues = np.zeros(n_primes, dtype=np.int64)
    for i, p in enumerate(primes):
        residues[i] = random.randint(0, p - 1)
        # Update counts
        # residues[i] is in [0, p-1]
        # We need to add 1 to counts[r::p] for r in [1, M]
        # If r == 0, the indices are p, 2p, ...
        # If r > 0, the indices are r, r+p, ...
        # Using slicing: counts[r::p] works if r is the start index.
        # But if r=0, we want to start at p? No, 0 is not in 1..M.
        # Wait, counts is 0-indexed. counts[0] is unused (x starts at 1).
        # So we should update counts for indices >= 1.
        
        r = residues[i]
        if r == 0:
            # Indices: p, 2p, ...
            # Slicing: counts[p::p]
            counts[p::p] += 1
        else:
            # Indices: r, r+p, ...
            # Slicing: counts[r::p]
            counts[r::p] += 1
            
    def calc_L(counts, M):
        # Returns the largest L such that counts[1..L] >= 2
        if M < 1:
            return 0
        if counts[1] < 2:
            return 0
        # Find first index >= 1 where count < 2
        mask = counts[1:M+1] < 2
        if not np.any(mask):
            return M
        first_false_idx = np.argmax(mask)
        # first_false_idx is relative to counts[1:M+1]
        # So actual index is first_false_idx + 1
        # The gap is at first_false_idx + 1
        # The length L is first_false_idx
        return first_false_idx

    current_L = calc_L(counts, M)
    best_L = current_L
    best_residues = residues.copy()
    
    # Helper to check if a move is valid (doesn't break prefix [1, L])
    # move: change residue of prime i from r_old to r_new
    # It breaks coverage if any y in [1, L] with y % p == r_old has count[y] == 2
    # Because count[y] will decrease by 1.
    def is_valid_move(counts, i, r_old, r_new, L, p):
        if r_old == r_new:
            return True
        # Check if any y in [1, L] congruent to r_old mod p has count 2
        # y = r_old + k*p
        # Start from first y >= 1
        start_y = r_old
        if start_y == 0:
            start_y = p
        # Iterate y in [start_y, L] step p
        for y in range(start_y, L + 1, p):
            if counts[y] == 2:
                return False
        return True

    # Helper to apply move
    def apply_move(counts, i, r_old, r_new, p):
        if r_old != r_new:
            # Subtract old
            if r_old == 0:
                counts[p::p] -= 1
            else:
                counts[r_old::p] -= 1
            # Add new
            if r_new == 0:
                counts[p::p] += 1
            else:
                counts[r_new::p] += 1

    # Helper to revert move
    def revert_move(counts, i, r_old, r_new, p):
        if r_old != r_new:
            # Subtract new
            if r_new == 0:
                counts[p::p] -= 1
            else:
                counts[r_new::p] -= 1
            # Add old
            if r_old == 0:
                counts[p::p] += 1
            else:
                counts[r_old::p] += 1

    # Optimization Loop
    # We alternate between "Repair" (targeted moves to fix the gap) and "Perturb" (random moves to escape local minima)
    
    # Temperature for simulated annealing
    T = 10.0
    T_min = 0.1
    cooling_rate = 0.9995
    
    stuck_counter = 0
    
    while time.time() - start_time < deadline:
        # 1. Try to repair the gap
        # The gap is at current_L + 1
        gap_x = current_L + 1
        
        found_valid_move = False
        
        # Shuffle prime indices to avoid bias
        prime_indices = list(range(n_primes))
        random.shuffle(prime_indices)
        
        for i in prime_indices:
            p = primes[i]
            r_old = residues[i]
            
            # If this prime already covers the gap, it can't help by changing to cover the gap (it's already there)
            # Unless we change it to something else? No, we want to ADD coverage to gap_x.
            # If r_old == gap_x % p, it already covers it. Changing it would REMOVE coverage.
            # So we only consider primes that DO NOT cover gap_x.
            if r_old == gap_x % p:
                continue
                
            r_new = gap_x % p
            
            # Check validity: does this move break coverage for [1, current_L]?
            if is_valid_move(counts, i, r_old, r_new, current_L, p):
                # Apply move
                apply_move(counts, i, r_old, r_new, p)
                residues[i] = r_new
                
                # Recalculate L
                new_L = calc_L(counts, M)
                
                # Since we fixed the gap, new_L should be at least current_L + 1
                # But it might break later? No, is_valid_move ensured [1, current_L] is safe.
                # So new_L >= current_L + 1.
                
                if new_L > best_L:
                    best_L = new_L
                    best_residues = residues.copy()
                    
                    # If we hit the boundary M, extend
                    if new_L >= M - 10:
                        new_M = M * 2
                        old_counts = counts
                        counts = np.zeros(new_M + 1, dtype=np.int64)
                        # Rebuild counts for new range
                        # Copy old part
                        counts[:M+1] = old_counts
                        # Add contributions for new part
                        for j, pj in enumerate(primes):
                            rj = residues[j]
                            # Find first index in [M+1, new_M] congruent to rj mod pj
                            # k = rj + m*pj
                            # We need k >= M+1
                            # m >= (M+1 - rj) / pj
                            start_m = max(0, (M + 1 - rj + pj - 1) // pj)
                            first_idx = rj + start_m * pj
                            if first_idx <= new_M:
                                if rj == 0:
                                    # first_idx should be multiple of pj
                                    # counts[first_idx::pj]
                                    counts[first_idx::pj] += 1
                                else:
                                    counts[first_idx::pj] += 1
                        M = new_M
                        # Recalculate current_L
                        current_L = calc_L(counts, M)
                        if current_L > best_L:
                            best_L = current_L
                            best_residues = residues.copy()
                else:
                    current_L = new_L
                
                found_valid_move = True
                stuck_counter = 0
                break # Restart the loop to find the new gap
        
        if found_valid_move:
            continue
            
        # 2. No valid repair move found. Use Simulated Annealing to escape.
        # Pick a random prime
        i = random.randint(0, n_primes - 1)
        p = primes[i]
        r_old = residues[i]
        
        # Random new residue
        r_new = random.randint(0, p - 1)
        
        if r_new == r_old:
            continue
            
        # Apply move
        apply_move(counts, i, r_old, r_new, p)
        
        new_L = calc_L(counts, M)
        
        if new_L >= current_L:
            current_L = new_L
            residues[i] = r_new
            if new_L > best_L:
                best_L = new_L
                best_residues = residues.copy()
                # Extend M if needed
                if best_L >= M - 10:
                    new_M = M * 2
                    old_counts = counts
                    counts = np.zeros(new_M + 1, dtype=np.int64)
                    counts[:M+1] = old_counts
                    for j, pj in enumerate(primes):
                        rj = residues[j]
                        start_m = max(0, (M + 1 - rj + pj - 1) // pj)
                        first_idx = rj + start_m * pj
                        if first_idx <= new_M:
                            counts[first_idx::pj] += 1
                    M = new_M
                    current_L = calc_L(counts, M)
                    if current_L > best_L:
                        best_L = current_L
                        best_residues = residues.copy()
        else:
            # Accept with probability exp((new_L - current_L) / T)
            delta_L = new_L - current_L
            if delta_L < 0:
                prob = np.exp(delta_L / T)
                if random.random() < prob:
                    current_L = new_L
                    residues[i] = r_new
                else:
                    # Revert
                    revert_move(counts, i, r_old, r_new, p)
            else:
                # Should not happen since we checked new_L >= current_L above
                current_L = new_L
                residues[i] = r_new
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            # Restart or reset temperature?
            # Let's just keep it at T_min for now.
            # Or reheat occasionally?
            # Let's reheat if we are stuck for a long time.
            stuck_counter += 1
            if stuck_counter > 100:
                T = 5.0
                stuck_counter = 0

    # Final evaluation with a higher limit to get the true score
    # We can run _find_first_gap with a larger max_check on best_residues
    # But we have counts array up to M.
    # If best_L == M, we should check further.
    # Let's just return best_residues. The evaluator will check.
    # But to be safe, if we have time, we can extend M one last time?
    # The deadline is strict.
    
    return best_residues.tolist()

BET: {mechanism: "Vectorized coverage counting and O(1) gap detection via numpy replace slow Python loops, enabling a robust Simulated Annealing + Repair search that escapes local minima more effectively than the seed's greedy repair, likely pushing the score significantly higher.", q: 0.9, delta: [0.5, 2.0]}
