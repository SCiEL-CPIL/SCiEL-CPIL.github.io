import time
import random
import math

N = 5000
START_TIME = time.time()
TIME_LIMIT = 300.0  # 300s deadline, keep margin

def _primes(limit):
    sieve = bytearray(b'\x01') * (limit + 1)
    if limit >= 0:
        sieve[0] = 0
    if limit >= 1:
        sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = 0
    return [p for p in range(2, limit + 1) if sieve[p]]

def _is_admissible(points, primes_list):
    """
    Check if the set of points is admissible.
    """
    k = len(points)
    if k == 0:
        return True
    
    # Optimization: use a set for fast lookup? No, we just need to check residues.
    # But points is a list.
    # For small k, this is fast.
    
    for p in primes_list:
        if p > k:
            break
        
        seen = bytearray(p)
        count = 0
        for v in points:
            r = v % p
            if not seen[r]:
                seen[r] = 1
                count += 1
                if count == p:
                    return False
    return True

def construct():
    global START_TIME
    start_time = START_TIME
    if start_time == 0:
        start_time = time.time()
        START_TIME = start_time
    
    primes = _primes(N)
    
    # We want to minimize the number of removed elements.
    # Strategy: Greedy selection of residue classes to ban for each prime.
    # Then local search to improve.
    
    best_removed = [False] * N
    best_size = N
    
    def calculate_removed(res_choices):
        """
        res_choices: dict {prime: residue}
        Returns a boolean list of removed elements.
        """
        removed = [False] * N
        for p, r in res_choices.items():
            for x in range(r, N, p):
                removed[x] = True
        return removed

    def get_size(removed):
        return sum(1 for x in range(N) if not removed[x])

    # 1. Greedy Initialization
    # We process primes in increasing order.
    # Why? Smaller primes have larger classes (N/p). Banning a class for p=2 removes ~2500 elements.
    # Banning a class for p=4999 removes ~1 element.
    # The constraints for small primes are "tighter" in terms of impact on the set size.
    
    res_choices = {}
    current_removed = [False] * N
    
    for p in primes:
        # Count unremoved elements in each residue class
        counts = [0] * p
        for x in range(N):
            if not current_removed[x]:
                counts[x % p] += 1
        
        # Choose residue with minimum count
        min_count = min(counts)
        # If multiple, pick one randomly or deterministically?
        # Deterministic might be fine, but randomization helps in local search later.
        # Let's pick the first one for determinism in init, randomize in LS.
        min_res = counts.index(min_count)
        
        res_choices[p] = min_res
        
        # Update current_removed
        for x in range(min_res, N, p):
            current_removed[x] = True
            
    current_size = get_size(current_removed)
    if current_size > best_size:
        best_size = current_size
        best_removed = current_removed.copy()
        best_res_choices = res_choices.copy()
    
    # 2. Local Search
    # We try to change the residue choice for a random prime and see if it reduces the union size.
    # To evaluate change efficiently, we can recalculate the total removed size.
    # Given N=5000 and ~600 primes, recalculating is O(N * pi(N)) ~ 3e6 ops.
    # This is fast enough to do many iterations.
    
    # We will run local search for a fixed number of iterations or until time limit.
    
    # We need to be careful: changing res[p] might increase the removed size.
    # We accept only if it decreases.
    
    # Let's use a simulated annealing or simple hill climbing.
    # Simple hill climbing:
    
    max_iters = 5000
    for i in range(max_iters):
        if time.time() - start_time > TIME_LIMIT - 10:
            break
            
        # Pick a random prime
        p = random.choice(primes)
        old_res = res_choices[p]
        
        # Pick a new random residue
        new_res = random.randint(0, p - 1)
        if new_res == old_res:
            continue
            
        # Calculate new removed set
        # Instead of full recalc, we can compute the delta?
        # Delta is complex because of overlaps.
        # Full recalc is safer and fast enough.
        
        temp_res_choices = res_choices.copy()
        temp_res_choices[p] = new_res
        
        new_removed = calculate_removed(temp_res_choices)
        new_size = get_size(new_removed)
        
        if new_size > current_size:
            # Improvement
            res_choices = temp_res_choices
            current_removed = new_removed
            current_size = new_size
            
            if current_size > best_size:
                best_size = current_size
                best_removed = current_removed.copy()
                best_res_choices = res_choices.copy()
        else:
            # No improvement, revert
            pass
            
    # 3. Final Polish: Try to add back elements?
    # The set defined by banned residues is already maximal with respect to *those* residue bans.
    # But maybe a different set of bans allows a larger set?
    # The local search explored the space of residue choices.
    
    # Construct the final list
    result = [x for x in range(N) if not best_removed[x]]
    
    # Sanity check
    if not _is_admissible(result, primes):
        # This should not happen if logic is correct
        # Fallback to a known valid small set
        return [0]
        
    return result

# Initial call to set start time and warm up
construct()
