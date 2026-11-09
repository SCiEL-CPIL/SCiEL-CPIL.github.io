import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(1000)
    n_primes = len(primes)
    
    # Max L to check for score
    M = 100000
    
    def calc_L(counts):
        """Find the largest L such that every integer 1..L has count >= 2."""
        if counts[1] < 2:
            return 0
        # We want the first index i in [1, M] where counts[i] < 2
        # counts[1:] is the slice from 1 to M
        # mask is True where counts < 2
        mask = counts[1:] < 2
        if not np.any(mask):
            return M
        # np.argmax returns the index of the first True value
        # If no True values, it returns 0, but we handled that with np.any
        first_false_idx = np.argmax(mask)
        return first_false_idx

    # Initialize residues randomly
    rng = random.Random(42)
    residues = np.array([rng.randint(0, p - 1) for p in primes], dtype=np.int64)
    
    # Initialize counts
    counts = np.zeros(M + 1, dtype=np.int64)
    for i, p in enumerate(primes):
        r = residues[i]
        counts[r::p] += 1
        
    current_L = calc_L(counts)
    best_L = current_L
    best_residues = residues.copy()
    
    def directed_repair(residues, counts, time_limit):
        """
        Greedy repair: Find the first position x with count < 2.
        Pick a prime that does NOT cover x, and change its residue to cover x.
        Accept if L increases.
        """
        improved = True
        no_improve_count = 0
        max_no_improve = 5000
        while improved and time.time() < time_limit:
            improved = False
            if no_improve_count > max_no_improve:
                break
                
            L = calc_L(counts)
            if L >= M:
                break
                
            # Find the first gap
            # L is the last good index. So L+1 is the first bad index.
            # Wait, calc_L returns L such that 1..L are good.
            # So if L < M, then L+1 is the first integer with count < 2.
            # If L == 0, 1 is the first bad.
            x = L + 1
            
            if counts[x] >= 2:
                # This shouldn't happen if calc_L is correct, but just in case
                break

            # Find primes that do NOT cover x
            # A prime p covers x if residues[i] == x % p
            # We want to change a prime p such that residues[i] != x % p
            # to residues[i] = x % p.
            
            # Collect candidates
            candidates = []
            for i, p in enumerate(primes):
                if residues[i] != x % p:
                    candidates.append(i)
                    
            if not candidates:
                # All primes already cover x? That means count[x] >= 168, which is >= 2.
                # So this case implies L was calculated wrong or x is not actually bad.
                break

            # Pick a random candidate
            i = rng.choice(candidates)
            p = primes[i]
            r_old = residues[i]
            r_new = x % p
            
            # Apply change
            counts[r_old::p] -= 1
            counts[r_new::p] += 1
            residues[i] = r_new
            
            new_L = calc_L(counts)
            
            if new_L > L:
                # Improvement
                current_L_global = new_L # Need to update outer scope? 
                # We are modifying residues and counts in place.
                # We need to return the new L to the caller or handle it inside.
                # Let's just return the best residues found in this phase.
                improved = True
                no_improve_count = 0
            else:
                # Revert
                counts[r_new::p] -= 1
                counts[r_old::p] += 1
                residues[i] = r_old
                no_improve_count += 1
                
    # Phase 1: Directed Repair
    # We need to track best_L and best_residues
    # The function above modifies in place. We should save best state.
    
    # Let's restructure to a single main loop that mixes directed moves and SA moves.
    
    # Reset for the combined loop
    residues = np.array([rng.randint(0, p - 1) for p in primes], dtype=np.int64)
    counts = np.zeros(M + 1, dtype=np.int64)
    for i, p in enumerate(primes):
        r = residues[i]
        counts[r::p] += 1
        
    current_L = calc_L(counts)
    best_L = current_L
    best_residues = residues.copy()
    
    T = 5.0
    T_min = 0.1
    cooling_rate = 0.9999
    
    # Main Search Loop
    while time.time() < deadline - 5.0:
        # 1. Try Directed Repair Move
        # Find first gap
        if current_L < M:
            x = current_L + 1
            if counts[x] < 2:
                # Find primes not covering x
                # To speed up, we can just pick a random prime and check
                # But picking from candidates is better.
                # For simplicity and speed, let's just pick a random prime index
                i = rng.randint(0, n_primes - 1)
                p = primes[i]
                r_needed = x % p
                r_old = residues[i]
                
                if r_needed != r_old:
                    # Try the move
                    counts[r_old::p] -= 1
                    counts[r_needed::p] += 1
                    residues[i] = r_needed
                    
                    new_L = calc_L(counts)
                    
                    if new_L > current_L:
                        current_L = new_L
                        if new_L > best_L:
                            best_L = new_L
                            best_residues = residues.copy()
                        # Accept
                    else:
                        # Revert
                        counts[r_needed::p] -= 1
                        counts[r_old::p] += 1
                        residues[i] = r_old
                        # If this move didn't work, we might fall through to SA
                        # Or we can just continue.
                        # Let's do SA if directed failed or if we are stuck.
                        pass
        
        # 2. Simulated Annealing Move
        # This helps escape local optima of the directed repair
        i = rng.randint(0, n_primes - 1)
        p = primes[i]
        r_old = residues[i]
        r_new = rng.randint(0, p - 1)
        
        if r_new != r_old:
            counts[r_old::p] -= 1
            counts[r_new::p] += 1
            residues[i] = r_new
            
            new_L = calc_L(counts)
            
            if new_L >= current_L:
                current_L = new_L
                if new_L > best_L:
                    best_L = new_L
                    best_residues = residues.copy()
            else:
                delta = current_L - new_L
                # Use a smaller delta for acceptance if we are close to best?
                # Standard SA
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    current_L = new_L
                else:
                    # Revert
                    counts[r_new::p] -= 1
                    counts[r_old::p] += 1
                    residues[i] = r_old
        
        # Cool down
        T = max(T_min, T * cooling_rate)
        
        # Periodic restart if stuck?
        # If T is low and we haven't improved in a long time, maybe reheat?
        
    # Final Polish: Local search on small primes
    if time.time() < deadline - 2.0:
        # Use best_residues
        residues = best_residues.copy()
        counts = np.zeros(M + 1, dtype=np.int64)
        for i, p in enumerate(primes):
            r = residues[i]
            counts[r::p] += 1
        current_L = calc_L(counts)
        
        # Polish first 20 primes
        for i in range(min(20, n_primes)):
            if time.time() > deadline - 2.0:
                break
            p = primes[i]
            r_old = residues[i]
            best_r = r_old
            best_local_L = current_L
            
            for r in range(p):
                if r == r_old:
                    continue
                counts[r_old::p] -= 1
                counts[r::p] += 1
                residues[i] = r
                new_L = calc_L(counts)
                
                if new_L > best_local_L:
                    best_local_L = new_L
                    best_r = r
                else:
                    # Revert to r_old for next trial? 
                    # No, we are testing all r. 
                    # We should revert to the *current best* or just revert to r_old if we don't accept?
                    # It's safer to just calculate L for each r independently from the state with r_old.
                    # So revert to r_old after each test.
                    counts[r::p] -= 1
                    counts[r_old::p] += 1
                    residues[i] = r_old
            
            # Apply best found
            if best_r != r_old:
                counts[r_old::p] -= 1
                counts[best_r::p] += 1
                residues[i] = best_r
                current_L = best_local_L
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues.copy()
            else:
                # Ensure state is consistent
                residues[i] = r_old

    return [int(r) for r in best_residues]
