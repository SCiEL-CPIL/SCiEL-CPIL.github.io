import itertools
import time
import random

# Global cache for cap sets
_cap_set_cache = {}
_initialized = False
_deadline = time.time() + 290  # Safety margin

def _is_cap_set(vecs):
    """Check if a list of vectors is a cap set."""
    # For small sets, O(N^2) check is fine.
    # A set is a cap set if no 3 distinct points form an AP.
    # x, y, z form AP iff x + z = 2y (mod 3)
    # This is equivalent to x, y, z being distinct and collinear.
    vec_set = set(vecs)
    for x in vecs:
        for y in vecs:
            if x == y: continue
            # z = 2y - x
            z = tuple((2*y[i] - x[i]) % 3 for i in range(len(x)))
            if z in vec_set and z != x and z != y:
                return False
    return True

def _greedy_cap_set(n, max_iters=1000, timeout=10):
    """
    Construct a large cap set using greedy randomization.
    """
    start_time = time.time()
    all_vecs = list(itertools.product([0, 1, 2], repeat=n))
    random.shuffle(all_vecs)
    
    best_set = []
    best_size = 0
    
    # Try multiple random orderings
    while len(best_set) < best_size or time.time() - start_time < timeout:
        if time.time() > _deadline:
            break
        current_set = []
        current_set_list = []
        # Greedy insertion
        for v in all_vecs:
            # Check if adding v creates an AP
            is_valid = True
            # Optimization: Only check against existing set
            for u in current_set_list:
                # Check if v, u, w form AP for some w in set?
                # No, just check if v completes an AP with any pair in set.
                # Actually, checking if v is in AP with any u in set is O(N^2) per v.
                # Faster: For each u in set, check if the third point w is in set.
                w = tuple((2*v[i] - u[i]) % 3 for i in range(n))
                if w in current_set and w != u and w != v:
                    is_valid = False
                    break
                # Also check if u is the middle?
                # If v, u, w is AP, then 2u = v+w.
                # If v, w, u is AP, then 2w = v+u.
                # The check above covers "v and u are endpoints, w is middle" and "v is middle, u is endpoint"?
                # Let's be precise.
                # An AP is {a, b, c} with 2b = a+c.
                # When adding v to S, we check if there exist a, c in S such that 2b = a+c where b is one of {v, a, c}.
                # Case 1: v is middle. 2v = a+c. a,c in S.
                # Case 2: a is middle. 2a = v+c. c in S.
                # Case 3: c is middle. 2c = v+a. a in S.
                
                # My loop above:
                # w = 2v - u. This checks if v is the middle? No.
                # If w = 2v - u, then 2v = u + w. So v is the middle.
                # We need to check if w is in S.
                # This covers Case 1.
                
                # We also need to check if u is the middle.
                # 2u = v + c => c = 2u - v.
                c = tuple((2*u[i] - v[i]) % 3 for i in range(n))
                if c in current_set and c != u and c != v:
                    is_valid = False
                    break
                
                # We also need to check if c (some other point) is the middle?
                # If we check all u in S, and check if v is middle (w in S) and if u is middle (c in S), do we cover all?
                # If the AP is {a, b, c} and v is one of them.
                # If v=a: b is middle. 2b = a+c => c = 2b - a = 2b - v.
                #   Here b is in S. We iterate u=b.
                #   We check c = 2u - v. If c in S, invalid.
                # If v=b: b is middle. 2b = a+c => a = 2b - c.
                #   Here a,c in S.
                #   We iterate u=a.
                #   We check w = 2v - u = 2b - a = c. If c in S, invalid.
                # If v=c: b is middle. 2b = a+c => a = 2b - c = 2b - v.
                #   Here b is in S. We iterate u=b.
                #   We check c' = 2u - v = a. If a in S, invalid.
                
                # So checking "v is middle" (w in S) and "u is middle" (c in S) for all u in S covers all cases.
            
            if is_valid:
                current_set.add(v)
                current_set_list.append(v)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)
            
        # Shuffle for next iteration
        random.shuffle(all_vecs)
        
        # Early stop if we hit a known bound or time
        if best_size >= (3**n) // (n + 1) * 1.1: # Rough upper bound
            break

    return best_set

def priority(vector, n):
    global _cap_set_cache, _initialized
    
    if n not in _cap_set_cache:
        if time.time() > _deadline:
            # Fallback to static heuristic if out of time
            # Prefer vectors with sum 0?
            return sum(vector) % 3
        # Compute cap set
        # For small n, use exact or known values
        if n <= 6:
             # Use a simpler exhaustive or pre-computed logic if possible, 
             # but greedy with many restarts works well for n<=6
             pass
        
        cap_set = _greedy_cap_set(n, timeout=60)
        _cap_set_cache[n] = cap_set
        _initialized = True

    if vector in _cap_set_cache[n]:
        return 1.0
    else:
        return 0.0
