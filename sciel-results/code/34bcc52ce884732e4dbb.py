import time
from random import Random

def _instance(size, seed, limit=10000):
    rng = Random(seed)
    values = set()
    while len(values) < size:
        x = rng.randint(-limit, limit)
        if x != 0:
            values.add(x)
    return sorted(values)

def _is_sum_free(S):
    s = set(S)
    for a in S:
        for b in S:
            if (a + b) in s:
                return False
    return True

def _can_add(S_set, x):
    """Check if x can be added to S_set without violating sum-freeness."""
    if 2 * x in S_set:
        return False
    # Check x + c in S
    for c in S_set:
        if x + c in S_set:
            return False
    # Check x = c1 + c2
    for c1 in S_set:
        if x - c1 in S_set:
            return False
    return True

def _greedy_by_key(S, key_func):
    """Greedy construction: add elements in order of key_func if they don't conflict."""
    S_set = set()
    result = []
    for x in sorted(S, key=key_func):
        if _can_add(S_set, x):
            S_set.add(x)
            result.append(x)
    return result

def _greedy_odd(S):
    return [x for x in S if x % 2 != 0]

def _local_search(S, rng, deadline):
    """Improve a sum-free subset of S via local search with efficient conflict checking."""
    S_set = set(S)
    all_elements = set(S)
    excluded = list(all_elements - S_set)
    
    best = len(S_set)
    best_set = set(S_set)
    
    # Ensure initial is valid
    if not _is_sum_free(S_set):
        # Fallback to a simple valid set
        S_set = set(_greedy_by_key(list(S), lambda v: -abs(v)))
        excluded = list(all_elements - S_set)
        best = len(S_set)
        best_set = set(S_set)
        
    while time.time() < deadline:
        improved = False
        
        # 1. Try adding any excluded element that fits
        # Iterate over a copy because we modify excluded
        for x in list(excluded):
            if _can_add(S_set, x):
                S_set.add(x)
                excluded.remove(x)
                improved = True
                if len(S_set) > best:
                    best = len(S_set)
                    best_set = set(S_set)
        
        if not excluded:
            break
            
        # 2. Try swaps: remove c from S, add x from excluded
        # To speed up, we only try a random subset of swaps if many
        swap_candidates = list(excluded)
        rng.shuffle(swap_candidates)
        
        for x in swap_candidates:
            if time.time() >= deadline:
                break
            
            # If x can be added directly, we would have done it in step 1.
            # So x must conflict with something in S_set.
            conflicts = set()
            if 2 * x in S_set:
                conflicts.add(2 * x)
            for c in S_set:
                if x + c in S_set:
                    conflicts.add(c)
                    conflicts.add(x + c)
                if x - c in S_set:
                    conflicts.add(c)
                    conflicts.add(x - c)
            
            # Try removing each conflict and adding x
            for c in conflicts:
                if c in S_set:
                    # Try swapping c with x
                    # We need to check if S_set - {c} + {x} is sum-free
                    # This is equivalent to checking if x can be added to S_set - {c}
                    # But we must also ensure that removing c doesn't break anything else?
                    # No, removing an element never breaks sum-freeness.
                    # So we just need to check if x is compatible with S_set - {c}.
                    
                    # Optimization: Check if x conflicts with any element in S_set - {c}
                    # We already know x conflicts with c. Does it conflict with others?
                    # The conflicts set contains all elements in S_set that conflict with x.
                    # If we remove c, are there other conflicts?
                    # If yes, then x still can't be added.
                    # So we only proceed if c is the *only* conflict, or if removing c resolves all conflicts.
                    # Actually, the condition for x to be addable to S' = S - {c} is:
                    # 1. 2x not in S'
                    # 2. For all y in S', x+y not in S'
                    # 3. For all y in S', x-y not in S'
                    
                    # Let's just use _can_add on the modified set.
                    # To avoid creating a new set every time, we can check manually.
                    # But for simplicity and given N is small (<=240), creating a set copy is fine.
                    
                    new_set = S_set - {c}
                    if _can_add(new_set, x):
                        S_set = new_set
                        S_set.add(x)
                        excluded.remove(x)
                        improved = True
                        if len(S_set) > best:
                            best = len(S_set)
                            best_set = set(S_set)
                        break # Move on to next x
            
        if not improved:
            # Stuck. Try a random restart of the local search process?
            # Or just break and let the outer loop handle restarts.
            # For now, break to save time for other restarts.
            break
            
    return list(best_set)

def construct():
    deadline = time.time() + 145.0
    results = []
    instances = [(60, 801), (120, 802), (240, 803)]
    
    for idx, (size, seed) in enumerate(instances):
        S = _instance(size, seed)
        rng = Random(seed + 12345)
        
        remaining_time = deadline - time.time()
        num_remaining = len(instances) - idx
        inst_budget = remaining_time / num_remaining - 2.0
        inst_deadline = time.time() + inst_budget
        
        # Build multiple initial candidates
        cand1 = _greedy_odd(S)
        cand2 = _greedy_by_key(S, lambda v: -abs(v))
        cand3 = _greedy_by_key(S, lambda v: v)
        cand4 = _greedy_by_key(S, lambda v: -v)
        
        # Randomized greedy
        rng.shuffle(S)
        cand5 = _greedy_by_key(S, lambda v: 0) # Just uses the shuffled order
        
        candidates = [cand1, cand2, cand3, cand4, cand5]
        
        best_result = []
        
        # Run local search on the best initial candidate
        best_init = max(candidates, key=len)
        result = _local_search(best_init, rng, inst_deadline)
        if len(result) > len(best_result):
            best_result = result
            
        # If time permits, try local search on other promising initial candidates
        if time.time() < inst_deadline - 5.0:
            for cand in candidates:
                if cand == best_init:
                    continue
                if time.time() >= inst_deadline:
                    break
                sub_deadline = min(time.time() + 2.0, inst_deadline)
                res = _local_search(cand, rng, sub_deadline)
                if len(res) > len(best_result):
                    best_result = res
                    
        # Fallback to best initial if no result
        if not best_result:
            best_result = best_init
            
        if not _is_sum_free(best_result):
            # Safety check
            for cand in candidates:
                if _is_sum_free(cand):
                    best_result = cand
                    break
            if not _is_sum_free(best_result):
                best_result = [x for x in S if x > 0]
                # Ensure it's sum-free
                while not _is_sum_free(best_result):
                    best_result.pop()
        
        results.append(best_result)
    
    return results
