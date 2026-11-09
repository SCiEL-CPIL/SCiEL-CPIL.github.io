import time
import math

N = 5000

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

PRIMES = _primes(N)
# Map prime to index for faster access? Or just list.
# We need counts for each prime.
# counts is a dict: p -> list of length p

def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    points = list(range(N + 1))
    active = [True] * (N + 1)
    
    # Initialize counts
    counts = {}
    min_counts = {}
    for p in PRIMES:
        if p > N: break
        counts[p] = [0] * p
        # Fill counts
        for v in range(N + 1):
            counts[p][v % p] += 1
        min_counts[p] = min(counts[p])
        
    # We need to track which primes are "active" (p <= L)
    # L changes.
    
    while True:
        if time.time() > deadline:
            break
            
        L = len(points)
        if L == 0:
            break
            
        # Find blocked primes: p <= L and min_counts[p] > 0
        # Note: min_counts is only valid if we update it correctly.
        # We only care about p <= L.
        
        min_m = float('inf')
        p_star = None
        blocked_primes = []
        
        for p in PRIMES:
            if p > L:
                break
            if min_counts[p] > 0:
                blocked_primes.append(p)
                if min_counts[p] < min_m:
                    min_m = min_counts[p]
                    p_star = p
        
        if p_star is None:
            break
            
        # Find residues to remove for p_star
        # residues r where counts[p_star][r] == min_m
        residues_to_remove = [r for r in range(p_star) if counts[p_star][r] == min_m]
        
        # Identify points to remove
        to_remove = []
        for v in points:
            if v % p_star in residues_to_remove:
                to_remove.append(v)
                
        if not to_remove:
            # Should not happen if min_m > 0
            break
            
        # Remove them
        remove_set = set(to_remove)
        new_points = [v for v in points if v not in remove_set]
        
        # Update counts and min_counts
        for x in to_remove:
            active[x] = False
            for p in PRIMES:
                if p > L: # Only update primes that were active? 
                    # Actually, if p > L, it wasn't blocking. 
                    # But if L decreases, p might become <= L later? No, L only decreases.
                    # So we only need to update p <= L_current.
                    # However, min_counts for p > L is not used.
                    # But if we don't update them, and L increases? L never increases.
                    # So we can just update p <= L.
                    break
                r = x % p
                old_val = counts[p][r]
                counts[p][r] = old_val - 1
                # Update min if necessary
                if old_val == min_counts[p]:
                    # The min might have decreased or stayed same.
                    # If there was another element with value old_val, min stays old_val?
                    # No, if we decrement one, the value becomes old_val-1.
                    # The new min is min(old_min, new_val) is not right.
                    # The new min is the minimum of the array.
                    # If the decremented element was the unique minimum, the min increases? No, we are removing elements, counts decrease.
                    # Wait. Counts decrease.
                    # If I have [2, 2, 3]. Min is 2.
                    # I remove one from class 0. [1, 2, 3]. Min is 1.
                    # So min can decrease.
                    # Can min increase? No, we only decrement.
                    # So if old_val == min_counts[p], the new min is <= old_val.
                    # It is either old_val (if another class had old_val) or old_val-1 (if it was unique) or lower?
                    # It can't be lower than old_val-1 because we only changed one element by 1.
                    # So we just need to check if any other class has value old_val.
                    # If yes, min stays old_val.
                    # If no, min becomes old_val-1? 
                    # Wait, if I had [1, 2, 2]. Min 1.
                    # Decrement class 0. [0, 2, 2]. Min 0.
                    # If I had [2, 2, 3]. Min 2.
                    # Decrement class 0. [1, 2, 3]. Min 1.
                    # So yes, if the decremented value was the min, the new min is either the same (if dup) or decremented by 1 (if unique).
                    # We can just recalculate min(counts[p]) if old_val == min_counts[p].
                    # Cost O(p).
                    min_counts[p] = min(counts[p])
                    
        points = new_points

    return points
