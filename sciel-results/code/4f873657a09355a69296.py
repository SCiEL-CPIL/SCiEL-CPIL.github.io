import time
import numpy as np

def _get_primes(limit):
    """Sieve of Eratosthenes to get primes up to limit."""
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    time_limit = 118.0  # Safety margin before 150s kill
    
    primes = _get_primes(1000)
    n_primes = len(primes)
    
    # We will search for L up to a reasonable max. 
    # The task says score is L. 
    # Let's cap the check at 5000 to be safe but efficient.
    MAX_CHECK = 20000
    
    def calculate_score(residues):
        """
        Calculate the largest L such that every integer 1..L satisfies at least two congruences.
        """
        counts = np.zeros(MAX_CHECK + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            if r == 0:
                start = p
            else:
                start = r
            if start <= MAX_CHECK:
                idx = np.arange(start, MAX_CHECK + 1, p)
                counts[idx] += 1
        
        # Find the first index where count < 2
        # We care about 1..L
        if MAX_CHECK == 0:
            return 0
            
        # If all are >= 2, return MAX_CHECK
        # We want the first index where counts < 2.
        # If no such index, return MAX_CHECK.
        
        # Create a boolean array where counts < 2
        fail_mask = counts < 2
        # We only care about indices 1..MAX_CHECK
        # If fail_mask[1:] has any True, find the first one.
        if np.any(fail_mask[1:]):
            first_fail = np.argmax(fail_mask[1:]) + 1
            return first_fail - 1
        else:
            return MAX_CHECK

    def is_valid_move(residues, counts, prime_idx, new_r, x_fail):
        """
        Check if changing residues[prime_idx] to new_r keeps the prefix [1, x_fail-1] valid.
        We know x_fail is the first failure in the current state.
        We want to ensure that for all y < x_fail, the new count is >= 2.
        The count of y changes only if y is congruent to old_r or new_r mod p.
        """
        p = primes[prime_idx]
        old_r = residues[prime_idx]
        
        if old_r == new_r:
            return True
            
        # We need to check y in [1, x_fail - 1]
        # y is affected if y % p == old_r or y % p == new_r
        
        # The count decreases by 1 for y where y % p == old_r and y % p != new_r.
        # The count increases by 1 for y where y % p == new_r and y % p != old_r.
        # The count is unchanged otherwise.
        
        # Failure condition:
        # 1. y % p == old_r and y % p != new_r and counts[y] == 2.
        #    (Count drops from 2 to 1)
        
        # So we just need to check if any y in [1, x_fail-1] with y % p == old_r has counts[y] == 2.
        
        # Generate candidates
        if old_r == 0:
            start_y = p
        else:
            start_y = old_r
            
        # We only care about y < x_fail
        if start_y >= x_fail:
            return True # No affected y in the prefix
            
        # Check counts for these y's using numpy for speed
        # y = start_y, start_y+p, ... < x_fail
        y_indices = np.arange(start_y, x_fail, p)
        if np.any(counts[y_indices] == 2):
            return False
                
        return True

    best_score = 0
    best_residues = None
    
    # Initial setup: random restarts
    num_restarts = 120
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit - 5:
            break
            
        # Initialize residues
        # Random initialization
        residues = [int(np.random.randint(0, p)) for p in primes]
        
        # Calculate initial score
        current_score = calculate_score(residues)
        if current_score > best_score:
            best_score = current_score
            best_residues = residues[:]
        
        # Local search with more aggressive move acceptance
        max_no_improve = 0
        no_improve_limit = 40
        
        while time.time() - start_time < time_limit - 5:
            if max_no_improve > no_improve_limit:
                break
            
            # Find first failure
            counts = np.zeros(MAX_CHECK + 1, dtype=np.int32)
            for i, p in enumerate(primes):
                r = residues[i]
                if r == 0:
                    start = p
                else:
                    start = r
                if start <= MAX_CHECK:
                    idx = np.arange(start, MAX_CHECK + 1, p)
                    counts[idx] += 1
                    
            fail_mask = counts < 2
            if not np.any(fail_mask[1:]):
                # Maximized for this restart
                if current_score < MAX_CHECK:
                    current_score = MAX_CHECK
                break
                
            first_fail_idx = np.argmax(fail_mask[1:]) + 1
            x_fail = first_fail_idx
            
            # Try to fix x_fail - try multiple candidate moves
            move_made = False
            improved = False
            
            # Try each prime, preferring smaller primes first (they affect more numbers)
            prime_order = sorted(range(n_primes), key=lambda i: primes[i])
            
            for i in prime_order:
                p = primes[i]
                new_r = int(x_fail % p)
                if new_r == residues[i]:
                    continue
                
                # Check if this move is valid (keeps prefix [1, x_fail-1] intact)
                if is_valid_move(residues, counts, i, new_r, x_fail):
                    # Apply move
                    old_r = residues[i]
                    residues[i] = new_r
                    # Recalculate score
                    new_score = calculate_score(residues)
                    
                    if new_score > current_score:
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_residues = residues[:]
                        move_made = True
                        improved = True
                        max_no_improve = 0
                        break # Restart the while loop to find new first failure
                    elif new_score == current_score:
                        # Accept neutral moves to escape local minima
                        # But only if we've been stuck for a while
                        if max_no_improve > 3:
                            move_made = True
                            max_no_improve += 1
                            break
                        else:
                            # Revert
                            residues[i] = old_r
                    else:
                        # Revert
                        residues[i] = old_r
                        
            if not move_made:
                # Stuck. Try random perturbation with higher probability of accepting
                escaped = False
                for _ in range(30):
                    if time.time() - start_time > time_limit - 5:
                        break
                    i = int(np.random.randint(0, n_primes))
                    p = primes[i]
                    old_r = residues[i]
                    new_r = int(np.random.randint(0, p))
                    if new_r == old_r:
                        continue
                    residues[i] = new_r
                    new_score = calculate_score(residues)
                    if new_score > current_score:
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_residues = residues[:]
                        escaped = True
                        max_no_improve = 0
                        break
                    elif new_score == current_score and max_no_improve > 8:
                        # Accept neutral move
                        escaped = True
                        max_no_improve += 1
                        break
                    else:
                        # Revert
                        residues[i] = old_r
                        
                if not escaped:
                    max_no_improve += 1
                    if max_no_improve > no_improve_limit:
                        break # Restart this search instance
        
        # End of local search for this restart

    # Final polish: try to extend the best solution further
    if best_residues is not None:
        residues = best_residues[:]
        current_score = calculate_score(residues)
        if current_score < best_score:
            current_score = best_score
            
        # Try to improve further with more careful search
        for _ in range(100):
            if time.time() - start_time > time_limit - 3:
                break
                
            counts = np.zeros(MAX_CHECK + 1, dtype=np.int32)
            for i, p in enumerate(primes):
                r = residues[i]
                if r == 0:
                    start = p
                else:
                    start = r
                if start <= MAX_CHECK:
                    idx = np.arange(start, MAX_CHECK + 1, p)
                    counts[idx] += 1
                    
            fail_mask = counts < 2
            if not np.any(fail_mask[1:]):
                break
                
            first_fail_idx = np.argmax(fail_mask[1:]) + 1
            x_fail = first_fail_idx
            
            # Try all primes in order of increasing size
            prime_order = sorted(range(n_primes), key=lambda i: primes[i])
            
            moved = False
            for i in prime_order:
                p = primes[i]
                new_r = int(x_fail % p)
                if new_r == residues[i]:
                    continue
                
                if is_valid_move(residues, counts, i, new_r, x_fail):
                    old_r = residues[i]
                    residues[i] = new_r
                    new_score = calculate_score(residues)
                    if new_score > current_score:
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_residues = residues[:]
                        moved = True
                        break
                    else:
                        residues[i] = old_r
                        
            if not moved:
                break

    # Return the best residues found
    if best_residues is None:
        # Fallback to seed-like behavior if search failed
        return [int(((i + 2) // 2) % p) for i, p in enumerate(primes)]
        
    return [int(r) for r in best_residues]
