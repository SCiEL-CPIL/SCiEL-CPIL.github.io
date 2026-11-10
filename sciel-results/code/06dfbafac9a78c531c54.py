import random
import math
import time

def construct():
    # Generate primes up to 1000
    primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    n_primes = len(primes)
    
    # We will search for a solution that covers up to M
    M = 2500
    
    # Initialize residues to 0 (covers multiples of p)
    residues = [0] * n_primes
    
    def compute_counts(res):
        counts = [0] * (M + 1)
        for i in range(n_primes):
            p = primes[i]
            r = res[i]
            if r == 0:
                start = p
            else:
                start = r
            for x in range(start, M + 1, p):
                counts[x] += 1
        return counts

    def get_first_failure(counts):
        for x in range(1, M + 1):
            if counts[x] < 2:
                return x
        return M + 1

    counts = compute_counts(residues)
    current_fail = get_first_failure(counts)
    best_score = current_fail - 1
    best_residues = residues[:]
    
    start_time = time.time()
    time_limit = 110.0
    
    # Greedy Loop
    while (time.time() - start_time) < time_limit:
        if current_fail > M:
            break
            
        x_fail = current_fail
        # Find primes that do NOT cover x_fail
        candidates = []
        for i in range(n_primes):
            p = primes[i]
            if x_fail % p != residues[i]:
                candidates.append(i)
                
        if not candidates:
            # All primes cover x_fail, but count < 2?
            # This means count == 1.
            # This is impossible if n_primes > 1 and all cover it.
            # But if some don't cover it, they are in candidates.
            # If candidates is empty, all cover it. Then count >= n_primes >= 2.
            # So this case shouldn't happen if logic is correct.
            # Unless M is very small or something.
            # Force a random move to escape
            i = random.randint(0, n_primes - 1)
            p = primes[i]
            old_r = residues[i]
            new_r = random.randint(0, p - 1)
            if new_r == old_r:
                continue
            residues[i] = new_r
            start_old = old_r if old_r > 0 else p
            for x in range(start_old, M + 1, p):
                counts[x] -= 1
            start_new = new_r if new_r > 0 else p
            for x in range(start_new, M + 1, p):
                counts[x] += 1
            current_fail = get_first_failure(counts)
            if current_fail - 1 > best_score:
                best_score = current_fail - 1
                best_residues = residues[:]
            continue

        best_move = None
        best_move_gain = -1
        
        for i in candidates:
            p = primes[i]
            old_r = residues[i]
            new_r = x_fail % p
            
            # Check if this move is "safe": does it break any y < x_fail?
            is_safe = True
            # Only need to check y where p covers y (old_r) and count[y] == 1
            # If we remove p's coverage from y, count becomes 0.
            # We iterate y in range(1, x_fail)
            # Optimization: only check y that are multiples of p (offset by old_r)
            start_check = old_r if old_r > 0 else p
            for y in range(start_check, x_fail, p):
                if counts[y] == 1:
                    is_safe = False
                    break
                    
            if is_safe:
                # Calculate "gain" or just accept it?
                # Let's just accept the first safe move to keep it fast.
                # Or we can prioritize primes with larger p (fewer side effects)?
                # Let's just pick the first one.
                best_move = i
                break # Found a safe move, take it
        
        if best_move is not None:
            i = best_move
            p = primes[i]
            old_r = residues[i]
            new_r = x_fail % p
            
            # Apply move
            residues[i] = new_r
            start_old = old_r if old_r > 0 else p
            for x in range(start_old, M + 1, p):
                counts[x] -= 1
            start_new = new_r if new_r > 0 else p
            for x in range(start_new, M + 1, p):
                counts[x] += 1
            
            new_fail = get_first_failure(counts)
            current_fail = new_fail
            if new_fail - 1 > best_score:
                best_score = new_fail - 1
                best_residues = residues[:]
        else:
            # No safe move found.
            # Try a random move to escape local optimum
            i = random.choice(candidates)
            p = primes[i]
            old_r = residues[i]
            new_r = x_fail % p
            
            # Apply move even if unsafe
            residues[i] = new_r
            start_old = old_r if old_r > 0 else p
            for x in range(start_old, M + 1, p):
                counts[x] -= 1
            start_new = new_r if new_r > 0 else p
            for x in range(start_new, M + 1, p):
                counts[x] += 1
            
            new_fail = get_first_failure(counts)
            current_fail = new_fail
            if new_fail - 1 > best_score:
                best_score = new_fail - 1
                best_residues = residues[:]
                
    return best_residues
