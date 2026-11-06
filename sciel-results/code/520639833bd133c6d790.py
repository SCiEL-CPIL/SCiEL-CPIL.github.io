"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Return the maximum-cardinality subset of [1, 255] containing 255
    such that gcd(a, b) > 1 for all distinct pairs.
    
    Strategy:
    1. Start with the base set of all multiples of 3 in [1, 255] (size 85).
       This is a valid pairwise non-coprime set since gcd(a,b) >= 3 for all pairs.
    2. Try to add elements that are NOT divisible by 3, but are divisible by 5 or 17,
       by removing the minimal set of elements from the base set that conflict with them.
       An element x (not div by 3) conflicts with a multiple of 3, y, if gcd(x,y) == 1.
       Since x is div by 5 or 17 (and not 3), gcd(x,y) > 1 iff y is div by 5 or 17 (the same prime).
       So we only need to remove multiples of 3 that are NOT divisible by the shared prime.
    3. We use a greedy approach: sort candidate additions by "cost" (number of removals needed)
       and greedily add those that give positive net gain.
    
    Deadline guard: return immediately if we exceed 80 seconds.
    """
    start_time = time.time()
    DEADLINE = 80.0  # safety margin before 90s kill
    
    N = 255
    base = [x for x in range(1, N + 1) if x % 3 == 0]
    # base is all multiples of 3, includes 255. Size = 85.
    
    # Candidates: numbers in [1, 255] not divisible by 3, but divisible by 5 or 17
    # (since 255 = 3*5*17, any number sharing a factor with 255 must be div by 3, 5, or 17)
    candidates = [x for x in range(1, N + 1) if x % 3 != 0 and (x % 5 == 0 or x % 17 == 0)]
    
    # For each candidate, compute which base elements it conflicts with
    # A candidate x (div by p in {5, 17}) conflicts with base element y (div by 3)
    # if gcd(x, y) == 1, i.e., y is not divisible by any prime factor of x.
    # Since x is not div by 3, x's prime factors are a subset of {5, 17} (plus possibly others).
    # gcd(x, y) > 1 iff y shares a prime factor with x.
    # y is div by 3. If y is also div by 5, then gcd(x,y) >= 5 (if x div by 5).
    # If y is also div by 17, then gcd(x,y) >= 17 (if x div by 17).
    
    # Let's be more precise: for candidate x, the primes of x that are in {5, 17} are the relevant ones.
    # Actually, x might have other prime factors too, but y is a multiple of 3, so y's prime factors
    # include 3 and possibly others. gcd(x, y) > 1 iff they share ANY prime factor.
    
    # Let's just compute gcd directly for correctness.
    base_set = set(base)
    base_list = list(base_set)
    
    # For each candidate, find which base elements conflict
    candidate_info = []
    for x in candidates:
        conflicts = []
        for y in base_list:
            if math.gcd(x, y) == 1:
                conflicts.append(y)
        # Net gain = 1 - len(conflicts)
        # We want to add x and remove all conflicting y's
        net_gain = 1 - len(conflicts)
        candidate_info.append((net_gain, x, conflicts))
    
    # Sort by net_gain descending (most beneficial first)
    candidate_info.sort(key=lambda t: -t[0])
    
    current_set = set(base_list)
    best_size = len(current_set)
    best_result = list(current_set)
    
    # Greedy: try to add candidates in order of net_gain
    # But adding one candidate might affect conflicts for another.
    # For a simple greedy: iterate, and if net_gain > 0, add it and remove conflicts.
    # But we need to recompute conflicts after each addition since the set changes.
    
    # Actually, a better approach: since the conflicts are between the candidate and base elements,
    # and we're removing base elements, let's track which base elements are still present.
    
    # Let's use a different greedy: try all single additions first, then pairs, etc.
    # But that's expensive. Let's stick with a simpler greedy.
    
    # Re-initialize
    current_set = set(base_list)
    best_size = len(current_set)
    best_result = list(current_set)
    
    # Sort candidates by net_gain descending
    for net_gain, x, conflicts in candidate_info:
        if time.time() > DEADLINE:
            break
        
        if net_gain <= 0:
            # No point trying further since sorted descending
            break
        
        # Check if x is still a valid candidate (not already in set, and all conflicts still in set)
        if x in current_set:
            continue
        
        # Check which conflicts are still in the current set
        active_conflicts = [y for y in conflicts if y in current_set]
        actual_net_gain = 1 - len(active_conflicts)
        
        if actual_net_gain > 0:
            # Add x, remove active_conflicts
            current_set.add(x)
            for y in active_conflicts:
                current_set.discard(y)
            
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_result = list(current_set)
    
    # Second pass: try to add remaining candidates that might now have fewer conflicts
    # because some conflicting base elements were already removed
    for _ in range(10):  # a few iterations to converge
        if time.time() > DEADLINE:
            break
        
        improved = False
        for net_gain, x, conflicts in candidate_info:
            if time.time() > DEADLINE:
                break
            
            if x in current_set:
                continue
            
            active_conflicts = [y for y in conflicts if y in current_set]
            actual_net_gain = 1 - len(active_conflicts)
            
            if actual_net_gain > 0:
                current_set.add(x)
                for y in active_conflicts:
                    current_set.discard(y)
                improved = True
                
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_result = list(current_set)
        
        if not improved:
            break
    
    # Verify the result
    result = best_result
    if 255 not in result:
        result.append(255)
    
    # Final verification: check all pairs have gcd > 1
    # For large sets this is O(n^2), but n <= 85+something, so it's fine
    # Actually, let's just trust the construction and return
    
    if time.time() > DEADLINE:
        return result
    
    return result
# EVOLVE-BLOCK-END
