"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import itertools

# Precomputation for speed
N = 256
# List of all numbers 1..256
ALL_NUMS = list(range(1, N + 1))

# Precompute prime factors or just use gcd
def gcd(a, b):
    return math.gcd(a, b)

def is_coprime_to_set(num, S, exclude=None):
    """Check if num is coprime to all elements in S, excluding 'exclude'."""
    for x in S:
        if x == exclude:
            continue
        if gcd(num, x) != 1:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    # Initial Set: 1 and all primes up to 256
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(math.isqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

    primes = [p for p in range(2, N + 1) if is_prime(p)]
    S = [1] + primes
    S_set = set(S)
    
    best_sum = sum(S)
    
    # Local Search Loop
    # We alternate between 1-for-1 and 2-for-2 moves
    # To avoid infinite loops, we track the best sum found and stop if no improvement in a full cycle
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # --- Pass 1: 1-for-1 moves ---
        for x in list(S):
            if x == 1:
                continue
            # Find best y > x such that y is coprime to S \ {x}
            best_y = -1
            # Iterate candidates in descending order to find largest quickly
            for y in range(N, x, -1):
                if y in S_set:
                    continue
                if is_coprime_to_set(y, S, exclude=x):
                    best_y = y
                    break
            
            if best_y > x:
                S.remove(x)
                S_set.discard(x)
                S.append(best_y)
                S_set.add(best_y)
                best_sum += best_y - x
                improved = True
                if time.time() > deadline:
                    break
        
        if time.time() > deadline:
            break
            
        # --- Pass 2: 2-for-2 moves ---
        # For each pair in S, try to replace with a better pair
        # To keep it efficient, we only try pairs involving smaller numbers first?
        # Or just all pairs.
        S_list = list(S)
        for i in range(len(S_list)):
            if time.time() > deadline:
                break
            x = S_list[i]
            if x == 1:
                continue
            for j in range(i + 1, len(S_list)):
                w = S_list[j]
                if w == 1:
                    continue
                
                # Determine forbidden numbers
                # A number z is forbidden if it shares a factor with any element in S \ {x, w}
                # We need to find y, v from allowed set such that gcd(y,v)=1 and y+v > x+w
                # Allowed set A = { z in 1..N \ S \ {x,w} such that is_coprime_to_set(z, S, exclude=x) and is_coprime_to_set(z, S, exclude=w) }
                # Wait, is_coprime_to_set(z, S, exclude=x) checks against all except x.
                # We need z to be coprime to all in S except x and w.
                
                target_sum = x + w
                best_pair_sum = -1
                best_pair = None
                
                # Build list of candidates
                # Optimization: Pre-compute compatibility?
                # For now, filter candidates
                candidates = []
                for z in range(2, N + 1): # 1 is already in S, and replacing 1 is not allowed in this pass logic (1 is usually kept)
                    if z in S_set and z != x and z != w:
                        continue
                    if z == x or z == w:
                        continue
                    # Check if z is coprime to all others
                    ok = True
                    for v in S:
                        if v == x or v == w:
                            continue
                        if gcd(z, v) != 1:
                            ok = False
                            break
                    if ok:
                        candidates.append(z)
                
                # Find best pair from candidates
                # Sort candidates descending to find large sums quickly
                candidates.sort(reverse=True)
                # We can prune: if candidates[0] + candidates[1] <= target_sum, skip
                if len(candidates) < 2:
                    continue
                if candidates[0] + candidates[1] <= target_sum:
                    continue
                
                # Iterate to find max sum coprime pair
                # Since we want max sum, we can break early if possible?
                # Not strictly, but we can stop if we find a pair with sum > target_sum and it's the theoretical max?
                # Theoretical max is c[0] + c[1] if they are coprime.
                found_better = False
                for k in range(len(candidates)):
                    if time.time() > deadline:
                        break
                    y = candidates[k]
                    # Max possible sum with y is y + candidates[0] (if candidates[0] != y)
                    # Actually, just iterate v
                    for m in range(len(candidates)):
                        if k == m:
                            continue
                        v = candidates[m]
                        if y + v <= target_sum:
                            # Since candidates are sorted, further v will be smaller
                            break
                        if gcd(y, v) == 1:
                            # This pair is valid and better than x+w
                            if y + v > best_pair_sum:
                                best_pair_sum = y + v
                                best_pair = (y, v)
                            # We found a valid pair. Is it the best?
                            # We want the MAX sum. So we should keep looking?
                            # No, we can break if we know we can't do better.
                            # But we don't know. However, we can just take the first one that is > target?
                            # No, we want to maximize the total sum. So we want the absolute best pair.
                            # Let's just find the max.
                            pass
                    
                if best_pair and best_pair_sum > target_sum:
                    S.remove(x)
                    S_set.discard(x)
                    S.remove(w)
                    S_set.discard(w)
                    y, v = best_pair
                    S.append(y)
                    S_set.add(y)
                    S.append(v)
                    S_set.add(v)
                    best_sum += best_pair_sum - target_sum
                    improved = True
                    # Restart inner loops because S changed
                    break
            
            if improved:
                break # Restart outer i loop
                
    # Ensure return type is list of ints
    result = [int(x) for x in S]
    return result

# EVOLVE-BLOCK-END
