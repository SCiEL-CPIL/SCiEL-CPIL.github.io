import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Proven valid tail: for a >= 171, any b,c > a gives b+c > 2a >= 2*171=342
    # but b+c mod a = 0 requires b+c >= a (true) and b+c = k*a for some k
    # Since b,c <= 256, b+c <= 512. For a >= 171, k can be 1 or 2.
    # k=1: b+c = a, but b > a and c >= b > a, so b+c > 2a > a. Impossible.
    # k=2: b+c = 2a. With b,c in [a+1, 256], b+c in [2a+2, 512]. 
    #   2a in [342, 512]. Need b+c = 2a with b > a, c >= b.
    #   b >= a+1, c >= b >= a+1, so b+c >= 2a+2 > 2a. Impossible!
    # So [171, 256] is valid.
    
    base_tail = list(range(171, 257))  # 86 elements
    
    # Build conflict graph: edge between x,y if they cannot both be in S
    # Two elements conflict if there exists a third element z (or role assignment) 
    # such that the triple constraint is violated.
    # 
    # For a set S, the constraint is: for every a<b<=c in S, a does NOT divide b+c.
    # 
    # When adding x to S, conflicts arise if:
    # 1. x is the smallest (a=x): need b,c in S with x<b<=c and x|(b+c)
    # 2. x is middle (b=x): need a in S with a<x, c in S with c>=x, a|(x+c)  
    # 3. x is largest (c=x): need a,b in S with a<b<=x, a|(b+x)
    #
    # For pairwise conflicts (for MIS formulation), we say x and y conflict if
    # adding x would be blocked by y being in S (in any role assignment).
    
    # Precompute conflict pairs
    # conflict[x][y] = True if x and y cannot both be in a valid set
    # This is approximate: actual validity depends on the full set, but this gives
    # a good heuristic for greedy selection.
    
    # For a more accurate approach: two elements x < y conflict if:
    # - y is a, and there exist b,c > y such that y|(b+c) -- but this depends on other elements
    # - x is a, y is b or c, and x|(y + z) for some z in S
    # 
    # Simplified: x and y (x<y) are "potentially conflicting" if:
    # - x divides y + y = 2y (when x is a, y is b=c)
    # - x divides y + z for some z > y that might be in S (hard to precompute)
    #
    # Let's use a different approach: greedy with dynamic conflict counting
    
    def can_add(S, x):
        """Check if adding x to set S (where S is a set of ints) keeps validity."""
        if x in S:
            return True
        
        S_list = sorted(S)
        
        # Case 1: x is smallest (a=x), need b,c in S with x < b <= c and x | (b+c)
        for i, b in enumerate(S_list):
            if b <= x:
                continue
            for c in S_list[i:]:  # c >= b
                if (b + c) % x == 0:
                    return False
        
        # Case 2: x is middle (b=x), need a in S with a < x, c in S with c >= x, a | (x+c)
        for a in S_list:
            if a >= x:
                break
            for c in S_list:
                if c < x:
                    continue
                if (x + c) % a == 0:
                    return False
        
        # Case 3: x is largest (c=x), need a,b in S with a < b <= x, a | (b+x)
        for i, a in enumerate(S_list):
            for j in range(i+1, len(S_list)):
                b = S_list[j]
                if b > x:
                    break
                if (b + x) % a == 0:
                    return False
        
        return True
    
    def try_greedy(S_initial, order):
        """Try greedy addition in given order. S_initial is a list."""
        S = set(S_initial)
        for x in order:
            if x in S:
                continue
            if can_add(S, x):
                S.add(x)
        return S
    
    # Candidate orders for adding elements 1..170
    candidates_to_add = list(range(1, 171))
    
    best_set = set(base_tail)
    best_size = len(base_tail)
    
    # Strategy: Model as weighted greedy on conflict graph
    # Weight = 1/(1 + number of potential conflicts)
    # Potential conflicts for x: count of y in [1,256]\{x} such that x and y might conflict
    
    def count_potential_conflicts(x, current_set):
        """Count how many elements in current_set would conflict with x if added."""
        S_list = sorted(current_set)
        count = 0
        
        # Case 1: x is smallest (a=x), need b,c in S with x < b <= c and x | (b+c)
        for i, b in enumerate(S_list):
            if b <= x:
                continue
            for c in S_list[i:]:  # c >= b
                if (b + c) % x == 0:
                    count += 1
                    break  # One conflict is enough to block
        
        # Case 2: x is middle (b=x), need a in S with a < x, c in S with c >= x, a | (x+c)
        for a in S_list:
            if a >= x:
                break
            for c in S_list:
                if c < x:
                    continue
                if (x + c) % a == 0:
                    count += 1
                    break
        
        # Case 3: x is largest (c=x), need a,b in S with a < b <= x, a | (b+x)
        for i, a in enumerate(S_list):
            for j in range(i+1, len(S_list)):
                b = S_list[j]
                if b > x:
                    break
                if (b + x) % a == 0:
                    count += 1
                    break
        
        return count
    
    def greedy_with_conflict_priority(S_initial, candidates, deadline):
        """Greedy that picks the candidate with fewest conflicts at each step."""
        S = set(S_initial)
        remaining = [x for x in candidates if x not in S]
        
        while remaining:
            if time.time() > deadline:
                break
            
            # For each remaining candidate, count conflicts with current S
            # Pick the one with fewest conflicts (tie-break: smaller value first)
            best_candidate = None
            best_conflicts = float('inf')
            
            for x in remaining:
                if x in S:
                    continue
                conflicts = count_potential_conflicts(x, S)
                if conflicts < best_conflicts or (conflicts == best_conflicts and (best_candidate is None or x < best_candidate)):
                    best_conflicts = conflicts
                    best_candidate = x
            
            if best_candidate is None:
                break
            
            if can_add(S, best_candidate):
                S.add(best_candidate)
                remaining.remove(best_candidate)
            else:
                # Shouldn't happen if count_potential_conflicts is accurate, but be safe
                remaining.remove(best_candidate)
        
        return S
    
    # Order 1: descending (170, 169, ..., 1)
    order_desc = list(range(170, 0, -1))
    S1 = try_greedy(base_tail, order_desc)
    if len(S1) > best_size:
        best_set = S1
        best_size = len(S1)
    
    if time.time() > deadline:
        return sorted(best_set)
    
    # Order 2: ascending (1, 2, ..., 170)
    order_asc = list(range(1, 171))
    S2 = try_greedy(base_tail, order_asc)
    if len(S2) > best_size:
        best_set = S2
        best_size = len(S2)
    
    if time.time() > deadline:
        return sorted(best_set)
    
    # Order 3: Conflict-priority greedy (MIS-inspired)
    S3 = greedy_with_conflict_priority(base_tail, candidates_to_add, deadline)
    if len(S3) > best_size:
        best_set = S3
        best_size = len(S3)
    
    if time.time() > deadline:
        return sorted(best_set)
    
    # Order 4: mix - try from the middle outward
    mid = 85
    order_mid = []
    for d in range(86):
        if mid - d >= 1:
            order_mid.append(mid - d)
        if mid + d <= 170 and mid + d != mid - d:
            order_mid.append(mid + d)
    S4 = try_greedy(base_tail, order_mid)
    if len(S4) > best_size:
        best_set = S4
        best_size = len(S4)
    
    if time.time() > deadline:
        return sorted(best_set)
    
    # Try to extend the tail further: maybe [165, 256] or lower is valid?
    def is_valid_set(S):
        S_list = sorted(S)
        for i, a in enumerate(S_list):
            for j in range(i+1, len(S_list)):
                b = S_list[j]
                for k in range(j, len(S_list)):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        return False
        return True
    
    # Try to lower the tail start
    for tail_start in range(170, 140, -1):
        if time.time() > deadline:
            break
        tail = list(range(tail_start, 257))
        if is_valid_set(tail):
            # Greedily add elements below tail_start using conflict priority
            order_below = list(range(tail_start - 1, 0, -1))
            S_new = greedy_with_conflict_priority(tail, order_below, deadline)
            if len(S_new) > best_size:
                best_set = S_new
                best_size = len(S_new)
            break
    
    if time.time() > deadline:
        return sorted(best_set)
    
    # Final: try a few more random-ish orders using deterministic pseudo-random
    rng = random.Random(42)
    
    for trial in range(10):
        if time.time() > deadline:
            break
        order = candidates_to_add[:]
        rng.shuffle(order)
        S_t = try_greedy(base_tail, order)
        if len(S_t) > best_size:
            best_set = S_t
            best_size = len(S_t)
    
    # Also try conflict-priority with different initial seeds
    for trial in range(5):
        if time.time() > deadline:
            break
        rng.shuffle(candidates_to_add)
        S_t = greedy_with_conflict_priority(base_tail, candidates_to_add, deadline)
        if len(S_t) > best_size:
            best_set = S_t
            best_size = len(S_t)
    
    return sorted(best_set)
