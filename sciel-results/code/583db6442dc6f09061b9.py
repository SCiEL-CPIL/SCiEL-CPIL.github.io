import time
import random

def _is_sidon_4(subset):
    """Check if a 4-element subset is Sidon (including diagonals)."""
    a, b, c, d = sorted(subset)
    sums = [
        2*a,
        a+b, a+c, a+d,
        2*b, b+c, b+d,
        2*c, c+d,
        2*d
    ]
    return len(set(sums)) == 10

def _count_sidon_4(S):
    """Count the number of 4-element Sidon subsets in S."""
    count = 0
    n = len(S)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    if _is_sidon_4((S[i], S[j], S[k], S[l])):
                        count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    rng = random.Random(123)
    
    best_set = None
    best_count = 10**9
    
    # Try multiple random restarts
    for restart in range(200):
        if time.time() > deadline:
            break
            
        # Generate a random set
        current_set = sorted(rng.sample(range(0, 129), 12))
        
        # Initial count
        current_count = _count_sidon_4(current_set)
        
        if current_count == 0:
            return current_set
            
        # Local search: First Descent
        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_move = None
            best_new_count = current_count
            
            # Try changing each element
            for i in range(12):
                old_val = current_set[i]
                
                # Try all possible new values
                # To save time, we can skip values close to old_val? No, keep all.
                # But 128 values * 12 elements = 1536 evaluations.
                # Each evaluation requires checking 165 subsets (containing i) 
                # and 165 subsets (containing new val).
                # Actually, we can just recompute the count for the new set?
                # No, that's 495 checks. 1536 * 495 = 760k checks.
                # Incremental is 1536 * 330 = 500k checks.
                # Let's stick to full re-evaluation for simplicity first, 
                # but optimize the inner loop.
                
                # Optimization: Pre-compute which subsets are sidon?
                # The subsets change.
                
                # Let's try a simpler heuristic: 
                # Only try a few random replacements per element per iteration.
                # Or try all, but it might be slow.
                
                pass
            
            # Since full local search might be slow in Python, 
            # let's use a stochastic local search.
            
            # Pick a random element to change
            i = rng.randint(0, 11)
            old_val = current_set[i]
            
            # Try several random new values
            best_candidate = None
            best_candidate_count = current_count
            
            for _ in range(20):
                new_val = rng.randint(0, 128)
                if new_val == old_val or new_val in current_set:
                    continue
                
                new_set = current_set[:]
                new_set[i] = new_val
                new_set.sort()
                
                new_count = _count_sidon_4(new_set)
                
                if new_count < best_candidate_count:
                    best_candidate_count = new_count
                    best_candidate = new_set
                    
                if best_candidate_count == 0:
                    break
            
            if best_candidate is not None:
                current_set = best_candidate
                current_count = best_candidate_count
                improved = True
                if current_count < best_count:
                    best_count = current_count
                    best_set = current_set[:]
                if current_count == 0:
                    return current_set
            else:
                # No improvement found in this iteration
                # Maybe try a random kick?
                # Just break and restart
                break
        
        if current_count < best_count:
            best_count = current_count
            best_set = current_set[:]

    if best_count == 0:
        return best_set
    else:
        return best_set
