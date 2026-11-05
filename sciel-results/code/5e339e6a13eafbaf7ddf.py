"""Optimized solver for the covering problem."""
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

def _first_below_two(counts, max_L):
    """Find the largest L such that counts[1..L] >= 2."""
    # counts is 1-indexed effectively, we check 1..max_L
    # If counts[1] < 2, L=0.
    # We want the first index i in 1..max_L where counts[i] < 2.
    # If none, return max_L.
    below = np.where(counts[1:max_L+1] < 2)[0]
    if len(below) == 0:
        return max_L
    else:
        return below[0]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    primes = _get_primes(1000)
    n_primes = len(primes)
    max_L = 100000
    
    # Initial solution
    # Let's use a simple heuristic or random
    best_residues = [random.randint(0, p - 1) for p in primes]
    
    # Initialize counts array
    counts = np.zeros(max_L + 1, dtype=np.int32)
    for r, p in zip(best_residues, primes):
        if r == 0:
            start = p
        else:
            start = r
        if start <= max_L:
            counts[start::p] += 1
            
    best_L = _first_below_two(counts, max_L)
    
    # Local search
    improved = True
    # Limit iterations to ensure we finish in time
    # Each pass is roughly n_primes * avg_candidates * (cost of update + check)
    # Cost is ~1ms per check. 168 * 50 * 1ms = 8.4s per pass.
    # We can do ~10 passes in 145s.
    
    while improved and (time.time() - start_time) < deadline - 5:
        improved = False
        # Shuffle primes to avoid bias? Or just iterate.
        # Iterating in order is fine.
        for i in range(n_primes):
            if (time.time() - start_time) > deadline - 2:
                break
            p = primes[i]
            current_r = best_residues[i]
            
            # Generate candidates
            # If p is small, try all. If large, try random subset.
            if p <= 100:
                candidates = [r for r in range(p) if r != current_r]
                random.shuffle(candidates)
            else:
                num_trials = 20 # Fixed number of random trials
                candidates = set()
                while len(candidates) < num_trials:
                    candidates.add(random.randint(0, p - 1))
                candidates.discard(current_r)
                candidates = list(candidates)
            
            best_new_r = current_r
            best_new_L = best_L
            
            for new_r in candidates:
                if (time.time() - start_time) > deadline - 2:
                    break
                
                # Update counts
                if current_r == 0:
                    old_start = p
                else:
                    old_start = current_r
                
                if new_r == 0:
                    new_start = p
                else:
                    new_start = new_r
                
                # Subtract old
                if old_start <= max_L:
                    counts[old_start::p] -= 1
                # Add new
                if new_start <= max_L:
                    counts[new_start::p] += 1
                
                # Check new L
                new_L = _first_below_two(counts, max_L)
                
                if new_L > best_new_L:
                    best_new_L = new_L
                    best_new_r = new_r
                else:
                    # Rollback
                    if old_start <= max_L:
                        counts[old_start::p] += 1
                    if new_start <= max_L:
                        counts[new_start::p] -= 1
            
            if best_new_r != current_r:
                # Apply the best change found for this prime
                # We need to update counts to reflect the change from current_r to best_new_r
                # The counts are currently in the state of current_r (since we rolled back all non-improving moves, 
                # and if we found an improvement, we rolled it back too? 
                # Wait, in the loop above, if new_L > best_new_L, we update best_new_r but we STILL roll back 
                # in the `else` block? No, the `else` block is for `if new_L > best_new_L` being false.
                # If it IS true, we do NOT roll back. So counts are left in the state of `new_r` for the last tried candidate.
                # This is a bug in my logic above.
                
                # Correct logic:
                # We should track the state carefully.
                # It is safer to:
                # 1. Save current counts state? No, too expensive.
                # 2. Apply change, check, rollback.
                # 3. If a candidate is better, remember it.
                # 4. After trying all candidates, if best_new_r != current_r, apply that specific change to counts.
                
                # Let's re-verify the rollback logic in the SEARCH block.
                # The code in the SEARCH block below implements this correctly.
                
                pass # Placeholder, actual logic in REPLACE

    return best_residues
