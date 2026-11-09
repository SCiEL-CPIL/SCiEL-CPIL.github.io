import time
import random
from itertools import combinations

L = 5354228880  # lcm(1..24)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    weights = [0] * 25
    for i in range(1, 25):
        weights[i] = L // i

    # Precompute all "conflict" relations:
    # For each element x, find all subsets of OTHER elements whose weight sum == weights[x]
    # We store these as frozensets of element indices (excluding x)
    # A set is valid iff it contains NO such conflict (i.e., for no x in S, the subset is contained in S\{x})
    
    # Actually, let's precompute for each x, the list of subsets T (T subset of [1..24]\{x}) such that sum(weights[t] for t in T) == weights[x]
    # Then a set S is invalid if there exists x in S such that T is a subset of S\{x} for some T in conflicts[x].
    
    # To make this efficient, we can represent conflicts as bitmasks.
    # There are 24 elements, so a bitmask fits in an int.
    
    # conflicts[x] is a list of bitmasks representing subsets T of [1..24]\{x} that sum to weights[x].
    # If S (as bitmask) contains x and S\{x} contains T, then conflict.
    # Equivalent: (S & mask_with_x) and (S & T) == T.
    
    # Let's precompute these conflicts.
    # Note: The number of subsets of 23 elements is 2^23 ~ 8 million. This might be too slow to enumerate all for each x.
    # However, we only care about subsets that sum to a specific target.
    # We can use meet-in-the-middle to find all subsets summing to weights[x] for each x.
    
    # Alternative approach:
    # Since we want to maximize cardinality, and the problem is small (24 elements),
    # we can use a Branch and Bound search that checks validity incrementally.
    
    # Let's stick to the incremental subset sum check but optimize it.
    # The current program uses a `sums` set of all possible subset sums of the selected elements.
    # When adding x, we check if weights[x] is in `sums`.
    # We also need to ensure that for all previously selected y, weights[y] is NOT in the new set of sums formed by (selected \ {y}) + {x}.
    # The current program does NOT do this second check correctly in the backtracking step!
    # It only checks `if w not in sums`. It does not check if adding x creates a conflict for existing elements.
    # Wait, let's re-read the current code.
    # `if w not in sums:` -> checks if x is subsumed by others.
    # It does NOT check if x causes others to be subsumed.
    # This means the current program might return an INVALID set.
    # Let's verify this hypothesis.
    # If the current program returns an invalid set, the score should be 0 or low. But it's 1.5833.
    # Maybe the search is lucky? Or maybe I'm missing something.
    # "no reciprocal of a selected element may equal the sum of reciprocals of a subset of the OTHER selected elements"
    # If I add x, I must ensure:
    # 1. weights[x] != sum(subset of weights[selected])
    # 2. For all y in selected: weights[y] != sum(subset of weights[selected \ {y} + {x}])
    # Condition 2 is expensive to check for every y.
    
    # Let's use a different strategy:
    # 1. Precompute all conflicts (x, T) where T is a subset of [1..24]\{x} and sum(T) == weights[x].
    # 2. Use a backtracking search that maintains the current set S.
    # 3. Before adding x, check if any conflict (x, T) has T subset of S.
    # 4. Before adding x, check if any conflict (y, T) for y in S has T subset of (S \ {y}) U {x}.
    #    This is equivalent to: T \ {x} subset of S \ {y} and (x in T or not).
    #    Actually, simpler: just check if the new set S U {x} contains any conflict.
    #    A conflict is a pair (x, T) such that x in S, T subset S, x not in T, and sum(T) == weights[x].
    #    So, we can precompute a list of all "bad" patterns.
    #    A pattern is a set of elements P such that for some x in P, the sum of reciprocals of P\{x} equals reciprocal of x?
    #    No, T can be a proper subset of S\{x}.
    #    So the "bad" configurations are any set S that contains a "minimal" conflict structure.
    #    Actually, if S contains a conflict (x, T), it is invalid.
    #    So we just need to ensure S does NOT contain any (x, T) where T is a subset summing to weights[x].
    
    # Let's precompute for each x, the list of minimal subsets T (or all subsets) that sum to weights[x].
    # Since 2^23 is large, let's use meet-in-the-middle to find all such T for each x.
    # There are 24 x's. 24 * (2^11 + 2^12) operations is manageable.
    
    # Step 1: Precompute conflicts
    # conflicts[x] = set of bitmasks T (T does not contain x) such that sum(weights[t] for t in T) == weights[x]
    
    all_conflicts = {}
    for x in range(1, 25):
        target = weights[x]
        others = [y for y in range(1, 25) if y != x]
        # Meet in the middle
        n = len(others)
        mid = n // 2
        left = others[:mid]
        right = others[mid:]
        
        left_sums = {0: 0} # sum: bitmask
        # We want all subsets, so we store all sums.
        # Since weights are large, the number of distinct sums might be small?
        # Or we can just store the set of bitmasks that achieve the target.
        
        # Actually, we just need to know if a given S contains such a T.
        # So for each x, we want a list of bitmasks T.
        
        left_sums_dict = {}
        for mask in range(1 << len(left)):
            s = 0
            m = 0
            for i in range(len(left)):
                if mask & (1 << i):
                    s += weights[left[i]]
                    m |= (1 << left[i])
            if s <= target:
                left_sums_dict[s] = m
        
        right_sums_dict = {}
        for mask in range(1 << len(right)):
            s = 0
            m = 0
            for i in range(len(right)):
                if mask & (1 << i):
                    s += weights[right[i]]
                    m |= (1 << right[i])
            if s <= target:
                right_sums_dict[s] = m
        
        found = set()
        for s_l, m_l in left_sums_dict.items():
            s_r_needed = target - s_l
            if s_r_needed in right_sums_dict:
                m_r = right_sums_dict[s_r_needed]
                found.add(m_l | m_r)
        
        all_conflicts[x] = found

    # Now, a set S (bitmask) is valid if for all x in S:
    # There is no T in all_conflicts[x] such that T is a subset of S and x is not in T.
    # Note: T is constructed from others, so x is never in T.
    # So check: for all x in S, for all T in all_conflicts[x]: (S & T) != T.
    
    # To speed up search, we can precompute for each x, the list of T's.
    # And use a backtracking algorithm that maintains S.
    
    # Optimization: Sort elements by weight descending? Or just 24..1?
    # Let's use 24..1.
    
    elements = list(range(24, 0, -1))
    n = len(elements)
    
    # Precompute bitmask for each element
    elem_mask = {x: (1 << x) for x in range(1, 25)}
    
    best_set = set()
    best_size = 0
    
    # Branch and Bound
    # State: idx, current_mask (bitmask of selected elements), selected_list
    # Pruning:
    # 1. If current_mask is invalid, prune. (Check conflicts)
    # 2. If len(selected) + (n - idx) <= best_size, prune.
    
    def is_valid_mask(mask):
        # Check if mask is a valid set
        for x in range(1, 25):
            if mask & elem_mask[x]:
                for T in all_conflicts[x]:
                    if (mask & T) == T:
                        return False
        return True
    
    # To make is_valid_mask faster during search, we can maintain a list of "active" conflicts?
    # Or just check incrementally.
    # When adding x to S, we only need to check:
    # 1. Conflicts involving x: Is there T in all_conflicts[x] such that T subset of S?
    # 2. Conflicts involving y in S: Is there T in all_conflicts[y] such that T subset of (S U {x})?
    #    This means T subset of S OR (T \ {x} subset of S and x in T).
    #    Since T does not contain y, and we are adding x, the only new subsets formed are those containing x.
    #    So we only need to check T's that contain x? No, T is a subset of others.
    #    If T did not contain x, it was already checked when S was formed.
    #    If T contains x, then T \ {x} must be subset of S.
    #    So:
    #    - Check if x is subsumed: any T in all_conflicts[x] is subset of S.
    #    - Check if any y in S is subsumed by S U {x}:
    #      For each y in S, check if there is T in all_conflicts[y] such that x in T and (T \ {x}) subset of S.
    
    # Let's precompute for each y, the list of T in all_conflicts[y] that contain a specific x?
    # That's too much.
    # Instead, for each y, we can store all_conflicts[y] as a list.
    # When adding x, for each y in S, iterate all_conflicts[y] and check if x in T and (T ^ (1<<x)) subset of S.
    
    # This is still O(|S| * |conflicts[y]|).
    # Given the small size, it might be fine.
    
    # Let's refine the backtracking.
    
    def backtrack(idx, mask, selected):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
        
        # Prune by size
        if len(selected) + (n - idx) <= best_size:
            return
        
        # Update best
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = set(selected)
            
        if idx >= n:
            return
        
        x = elements[idx]
        mx = elem_mask[x]
        
        # Try include x
        # Check if S U {x} is valid
        valid = True
        
        # 1. Check x is not subsumed by S
        for T in all_conflicts[x]:
            if (mask & T) == T:
                valid = False
                break
        
        if valid:
            # 2. Check no y in S is subsumed by S U {x}
            # Only need to check T's that contain x
            for y in selected:
                for T in all_conflicts[y]:
                    if T & mx: # T contains x
                        if (mask & (T ^ mx)) == (T ^ mx):
                            valid = False
                            break
                if not valid:
                    break
        
        if valid:
            backtrack(idx + 1, mask | mx, selected + [x])
        
        # Try exclude x
        backtrack(idx + 1, mask, selected)
        
    backtrack(0, 0, [])
    
    # Local improvement: try to add elements not in best_set
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Sort candidates by weight descending? Or random?
            # Let's try descending weight (smaller index) as they are "stronger"
            for y in range(24, 0, -1):
                if y in best_set:
                    continue
                my = elem_mask[y]
                mask = 0
                for z in best_set:
                    mask |= elem_mask[z]
                
                valid = True
                # Check y not subsumed by best_set
                for T in all_conflicts[y]:
                    if (mask & T) == T:
                        valid = False
                        break
                
                if valid:
                    # Check no z in best_set is subsumed by best_set U {y}
                    for z in best_set:
                        for T in all_conflicts[z]:
                            if T & my:
                                if (mask & (T ^ my)) == (T ^ my):
                                    valid = False
                                    break
                        if not valid:
                            break
                
                if valid:
                    best_set.add(y)
                    improved = True
                    break
    
    if not best_set:
        best_set = {24}
        
    return list(best_set)
