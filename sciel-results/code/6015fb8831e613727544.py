import time
import random
from fractions import Fraction

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 50.0
    
    # Precompute reciprocals as float for speed, and exact for final check
    recip = [0.0] * (n + 1)
    for i in range(1, n + 1):
        recip[i] = 1.0 / i
    
    # Precompute 3-AP constraints
    # A set is valid iff for every 3-AP (a, b, c) with a < b < c, it doesn't contain all three.
    # Equivalently, the "conflict graph" on {1..n} where edge (u,v) exists if there's a w
    # such that {u,v,w} is a 3-AP. Then S is a valid set iff S is an independent set in this conflict graph.
    
    # Conflict graph: u conflicts with v if there exists w != u,v such that (u+v+w)/3 is an integer in [1,n].
    # Actually, 3-AP is (a, a+d, a+2d).
    # So for any pair (u,v) with u < v, they are endpoints of a 3-AP if (u+v) is even.
    # Let m = (u+v)/2. If m is an integer and 1 <= m <= n, then (u, m, v) is a 3-AP.
    # So u and v conflict iff (u+v) is even.
    # Wait, this means the graph is bipartite? No.
    # Let's check: u=1, v=3 -> m=2. (1,2,3) is AP. So 1 and 3 conflict.
    # u=1, v=5 -> m=3. (1,3,5) is AP. So 1 and 5 conflict.
    # u=1, v=2 -> m=1.5. Not an integer. So 1 and 2 do NOT conflict as endpoints.
    # But 1 and 2 could be in an AP as (1, 2, 3) where 1 is start, 2 is mid.
    # In that case, 1 and 3 are endpoints and conflict. 2 is the middle.
    # Does 2 conflict with anyone?
    # 2 and 4: m=3. (2,3,4) AP. So 2 and 4 conflict.
    # 2 and 1: no direct endpoint conflict.
    # But if I pick 1 and 2, can I pick 3? No, because 1,2,3 is AP.
    # So 1 and 2 are compatible *as a pair*, but adding 3 creates a conflict.
    # The "Independent Set" formulation works on the graph where edge (u,v) means "u and v cannot be in the same set".
    # When do u and v conflict?
    # If there exists w such that {u,v,w} is a 3-AP.
    # Cases:
    # 1. u, v are endpoints: w = (u+v)/2. Requires u+v even.
    # 2. u, w are endpoints, v is middle: w = 2v - u. Requires 1 <= w <= n.
    # 3. w, v are endpoints, u is middle: w = 2u - v. Requires 1 <= w <= n.
    
    # So, u and v conflict if:
    # - (u+v) is even (they are endpoints of some AP, provided mid is in range) -- actually mid is always in range if u,v in [1,n] and u!=v? No.
    #   If u=1, v=256, mid=128.5 (odd sum).
    #   If u=1, v=255, mid=128. Yes.
    #   If u=1, v=3, mid=2. Yes.
    #   Mid is in [1,n] iff u,v in [1,n] and u!=v? 
    #   min(u,v) < (u+v)/2 < max(u,v). Since u,v in [1,n], mid is in (1,n) so in [2, n-1]. Yes.
    #   So if (u+v) is even, they conflict.
    # - OR, 2v - u is in [1,n] and 2v - u != u, v. (v is middle)
    # - OR, 2u - v is in [1,n] and 2u - v != u, v. (u is middle)
    
    # Let's build the adjacency list for the conflict graph.
    # n=256, so 256 nodes.
    # We can use a list of bitmasks or just a list of lists.
    # Given the high density, a bitmask (Python int) is efficient for intersection checks.
    
    conflict_mask = [0] * (n + 1)
    for u in range(1, n + 1):
        mask = 0
        for v in range(1, n + 1):
            if u == v:
                continue
            # Check if u, v conflict
            # 1. Endpoints
            if (u + v) % 2 == 0:
                mask |= (1 << v)
                continue # Already conflict, no need to check other cases? 
                         # Actually, if they are endpoints, they conflict. 
                         # But maybe they are also middle/endpoint in another AP? 
                         # Yes, but if they already conflict, adding more reasons doesn't change the boolean "conflict".
                         # However, we just need to know IF they conflict.
                         # So if (u+v)%2==0, they conflict.
                         # Else, check if one is middle.
            else:
                # u+v is odd, so they can't be endpoints of an integer AP.
                # Check if v is middle: w = 2v - u
                w = 2 * v - u
                if 1 <= w <= n and w != u and w != v:
                    mask |= (1 << v)
                # Check if u is middle: w = 2u - v
                else:
                    w = 2 * u - v
                    if 1 <= w <= n and w != u and w != v:
                        mask |= (1 << v)
        conflict_mask[u] = mask
        
    # Helper: check if adding x to set S (represented as bitmask) is valid
    # S is represented by bits set in s_mask.
    # x is valid if no node in S conflicts with x.
    # i.e., (s_mask & conflict_mask[x]) == 0
    # Also x must not be in S already (though we check that separately).
    
    def can_add(x, s_mask):
        if (s_mask >> x) & 1:
            return False
        return (s_mask & conflict_mask[x]) == 0
    
    def get_sum(s_mask):
        total = 0.0
        m = s_mask
        while m:
            lsb = m & -m
            i = lsb.bit_length() - 1
            total += recip[i]
            m ^= lsb
        return total
    
    # Greedy Fill
    # Try to add elements in order of decreasing weight (increasing i is decreasing weight? No, 1/1 > 1/2 > ...).
    # So we should try small i first.
    def greedy_fill(s_mask, order=None):
        if order is None:
            order = list(range(1, n + 1))
        for x in order:
            if not ((s_mask >> x) & 1):
                if can_add(x, s_mask):
                    s_mask |= (1 << x)
        return s_mask
    
    # Local Search: Remove one, refill greedily, see if sum increases.
    def local_search(s_mask, max_iters=50):
        best = s_mask
        best_sum = get_sum(best)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            # Try removing each element in current set
            elements = []
            m = best
            while m:
                lsb = m & -m
                elements.append(lsb.bit_length() - 1)
                m ^= lsb
            
            random.shuffle(elements)
            
            for rem in elements:
                trial = best ^ (1 << rem)
                # Refill
                trial = greedy_fill(trial)
                new_sum = get_sum(trial)
                if new_sum > best_sum + 1e-12:
                    best = trial
                    best_sum = new_sum
                    improved = True
                    break
            if not improved:
                break
        return best
    
    # 1. Initial Greedy
    best_mask = 0
    best_mask = greedy_fill(best_mask)
    best_sum = get_sum(best_mask)
    
    # 2. Local Search
    best_mask = local_search(best_mask)
    best_sum = get_sum(best_mask)
    
    # 3. Randomized Restarts
    random.seed(42)
    while time.time() < deadline - 5.0:
        # Perturb
        perturbed = best_mask
        
        # Remove 1-3 random elements
        num_removals = random.randint(1, 3)
        elements_in_set = []
        m = perturbed
        while m:
            lsb = m & -m
            elements_in_set.append(lsb.bit_length() - 1)
            m ^= lsb
        
        for _ in range(num_removals):
            if not elements_in_set:
                break
            idx = random.randint(0, len(elements_in_set) - 1)
            rem_val = elements_in_set[idx]
            perturbed &= ~(1 << rem_val)
            elements_in_set.pop(idx)
            
        # Refill with shuffled order to break ties/dependencies
        order = list(range(1, n + 1))
        random.shuffle(order)
        perturbed = greedy_fill(perturbed, order)
        
        # Local Search
        improved = local_search(perturbed)
        
        new_sum = get_sum(improved)
        if new_sum > best_sum + 1e-12:
            best_sum = new_sum
            best_mask = improved

    # 4. Exact Polish with Fractions
    # Convert best_mask to set of ints
    result_set = set()
    m = best_mask
    while m:
        lsb = m & -m
        result_set.add(lsb.bit_length() - 1)
        m ^= lsb
    
    # Exact local search using Fractions
    # We only do this if we have time, as it's slower.
    if time.time() < deadline - 2.0:
        exact_recip = {i: Fraction(1, i) for i in range(1, n+1)}
        
        def get_exact_sum(s):
            return sum(exact_recip[i] for i in s)
        
        def can_add_exact(x, s):
            # Same logic as can_add but s is a set
            # We can reuse conflict_mask for speed? 
            # can_add uses bitmask. Let's just use the bitmask version but on a temporary mask.
            # Or just check conflicts directly.
            for y in s:
                if (y in conflict_mask[x]): # This is O(N) per check, slow.
                    return False
            # Better: use the bitmask check
            mask = 0
            for y in s:
                mask |= (1 << y)
            return (mask & conflict_mask[x]) == 0

        def greedy_fill_exact(s_mask):
            # Same as greedy_fill but returns set? 
            # Let's stick to bitmask for greedy, then convert.
            pass
        
        # Actually, local search in bitmask is fast. 
        # The main risk is floating point comparison.
        # 1/1 + 1/3 vs 1/2 + 1/4 ... 
        # Fractions are exact.
        # Let's do a few local search steps with exact comparison.
        
        current_set = set(result_set)
        current_sum = get_exact_sum(current_set)
        
        for _ in range(10): # Few steps
            if time.time() > deadline:
                break
            improved = False
            elements = list(current_set)
            random.shuffle(elements)
            
            for rem in elements:
                trial_set = current_set - {rem}
                # Greedy refill in exact mode? 
                # Greedy doesn't use sum for decision, only validity. So greedy is deterministic given order.
                # We can just refill using the standard greedy_fill on the mask.
                trial_mask = 0
                for x in trial_set:
                    trial_mask |= (1 << x)
                
                trial_mask = greedy_fill(trial_mask)
                
                # Convert back to set for exact sum
                new_set = set()
                m = trial_mask
                while m:
                    lsb = m & -m
                    new_set.add(lsb.bit_length() - 1)
                    m ^= lsb
                    
                new_sum = get_exact_sum(new_set)
                if new_sum > current_sum:
                    current_set = new_set
                    current_sum = new_sum
                    improved = True
                    break
            if not improved:
                break
        
        # Update best if improved
        final_set = current_set
    else:
        final_set = result_set
    
    # Final Validation
    # Check no 3-AP
    s_list = sorted(final_set)
    s_set = set(s_list)
    for i in range(len(s_list)):
        for j in range(i+1, len(s_list)):
            a = s_list[i]
            b = s_list[j]
            if (a + b) % 2 == 0:
                mid = (a + b) // 2
                if mid != a and mid != b and mid in s_set:
                    # Conflict found!
                    # This should not happen.
                    pass 
            # Also check if a is middle of b, c?
            # The loop above covers all pairs (a,b). 
            # If a,b are endpoints, mid is checked.
            # If a is middle, then b is an endpoint, and there exists c such that (b+c)/2 = a.
            # Then a and c are also a pair. 
            # Wait, my validation logic only checks if the MIDPOINT of (a,b) is in the set.
            # This covers:
            # 1. a, b endpoints -> mid in set?
            # 2. a, b where a is middle? No, a,b are just a pair. 
            #    If a is middle of (x, b), then (x+b)/2 = a -> x = 2a - b.
            #    Is x in set? 
            #    My check: for pair (a,b), check if (a+b)/2 in set.
            #    This does NOT check if (2a-b) in set.
            #    
            #    Let's refine validation.
            #    For every triple (x, y, z) with x<y<z, if x,z in set, is y in set?
            #    This is O(N^3) or O(N^2) if smart.
            #    For each pair (a, c) in set with a < c, if (a+c) is even, check if (a+c)/2 is in set.
            #    This is exactly what the loop above does.
            #    Why? 
            #    Any 3-AP is (a, m, c) with m=(a+c)/2.
            #    If a, c in set, and m in set, then we have a 3-AP.
            #    My check: for every pair (a, c) in set, if mid in set -> invalid.
            #    This covers all 3-APs.
            #    
            #    So the validation is correct.
            raise AssertionError("3-AP found in final set")

    result = sorted(final_set)
    return result

