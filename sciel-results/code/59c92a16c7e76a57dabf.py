import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill

    N = 256
    recip = [0.0] + [1.0 / i for i in range(1, N + 1)]

    # Precompute conflicts: for each x, list of (y, z) such that {x, y, z} forms a 3-AP
    conflict_pairs = [[] for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            c = 2 * b - a
            if c > N or c <= b:
                continue
            # a, b, c is a 3-AP
            conflict_pairs[a].append((b, c))
            conflict_pairs[b].append((a, c))
            conflict_pairs[c].append((a, b))

    def can_add(x, S_mask):
        """Check if x can be added to set S_mask."""
        for y, z in conflict_pairs[x]:
            if (S_mask & (1 << y)) and (S_mask & (1 << z)):
                return False
        return True

    def get_float_sum(S_mask):
        """Compute floating-point reciprocal sum from bitmask."""
        total = 0.0
        for x in range(1, N + 1):
            if S_mask & (1 << x):
                total += recip[x]
        return total

    def greedy_fill(forbidden_mask, S_mask, order=None):
        """Fill S by adding elements in a given order."""
        S = S_mask
        if order is None:
            order = range(1, N + 1)
        for x in order:
            if forbidden_mask & (1 << x):
                continue
            if S & (1 << x):
                continue
            if can_add(x, S):
                S |= (1 << x)
        return S

    # Step 1: Get initial solution via greedy (ascending order)
    S_asc = greedy_fill(0, 0, order=range(1, N + 1))
    initial_sum = get_float_sum(S_asc)
    
    best_S = S_asc
    best_sum = initial_sum

    # Also try greedy with different orderings
    # Try descending order (prioritize small numbers with high reciprocals)
    S_desc = greedy_fill(0, 0, order=range(N, 0, -1))
    desc_sum = get_float_sum(S_desc)
    if desc_sum > best_sum:
        best_S = S_desc
        best_sum = desc_sum

    # Try a few random orderings
    for trial in range(20):
        if time.time() > deadline - 30:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        S_rand = greedy_fill(0, 0, order=order)
        rand_sum = get_float_sum(S_rand)
        if rand_sum > best_sum:
            best_S = S_rand
            best_sum = rand_sum

    # Step 2: Local search with multiple move types
    if time.time() < deadline - 10:
        S = best_S
        sum_float = get_float_sum(S)
        S_elements = [x for x in range(1, N + 1) if S & (1 << x)]
        
        improved = True
        while improved and (time.time() - start_time) < deadline - 5:
            improved = False
            
            # Try 1-Swap: remove x, add y (where y has higher reciprocal)
            best_gain = 0
            best_x = -1
            best_y = -1
            
            for y in range(1, N + 1):
                if (time.time() - start_time) > deadline - 5:
                    break
                if S & (1 << y):
                    continue
                
                for x in range(1, N + 1):
                    if not (S & (1 << x)):
                        continue
                    
                    gain = recip[y] - recip[x]
                    if gain <= 1e-15:
                        continue
                    
                    S_new = S & ~(1 << x)
                    if can_add(y, S_new):
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                            best_y = y
            
            if best_y != -1:
                S = S & ~(1 << best_x)
                S |= (1 << best_y)
                sum_float += best_gain
                improved = True
                continue
            
            # Try adding a single element if possible
            added = False
            for y in range(1, N + 1):
                if (time.time() - start_time) > deadline - 5:
                    break
                if S & (1 << y):
                    continue
                if can_add(y, S):
                    S |= (1 << y)
                    sum_float += recip[y]
                    S_elements.append(y)
                    improved = True
                    added = True
                    break
            
            if not added:
                # Try 2-Swap: remove x1, x2, add y1, y2
                best_2swap_gain = 0
                best_x1 = best_x2 = best_y1 = best_y2 = -1
                
                # Only try this if we have time
                if (time.time() - start_time) < deadline - 10:
                    for y1 in range(1, N + 1):
                        if (time.time() - start_time) > deadline - 10:
                            break
                        if S & (1 << y1):
                            continue
                        
                        for y2 in range(y1 + 1, N + 1):
                            if S & (1 << y2):
                                continue
                            
                            # Check if y1 and y2 can both be added together
                            # First check they don't conflict with each other
                            y1y2_conflict = False
                            for y, z in conflict_pairs[y1]:
                                if y == y2 or z == y2:
                                    y1y2_conflict = True
                                    break
                            if y1y2_conflict:
                                continue
                            
                            for x1 in range(1, N + 1):
                                if not (S & (1 << x1)):
                                    continue
                                
                                for x2 in range(x1 + 1, N + 1):
                                    if not (S & (1 << x2)):
                                        continue
                                    
                                    gain = recip[y1] + recip[y2] - recip[x1] - recip[x2]
                                    if gain <= 1e-15:
                                        continue
                                    
                                    S_no = S & ~(1 << x1) & ~(1 << x2)
                                    if can_add(y1, S_no) and can_add(y2, S_no):
                                        # Check y1 and y2 together
                                        if can_add(y2, S_no | (1 << y1)):
                                            if gain > best_2swap_gain:
                                                best_2swap_gain = gain
                                                best_x1, best_x2, best_y1, best_y2 = x1, x2, y1, y2
                                if (time.time() - start_time) > deadline - 10:
                                    break
                        if (time.time() - start_time) > deadline - 10:
                            break
                
                if best_y1 != -1:
                    S = S & ~(1 << best_x1) & ~(1 << best_x2)
                    S |= (1 << best_y1) | (1 << best_y2)
                    sum_float += best_2swap_gain
                    improved = True
                    continue
                
                # Try remove + greedy refill
                for x in S_elements:
                    if (time.time() - start_time) > deadline - 5:
                        break
                    if not (S & (1 << x)):
                        continue
                    
                    S_no_x = S & ~(1 << x)
                    S_new = greedy_fill(1 << x, S_no_x)
                    sum_new = get_float_sum(S_new)
                    gain = sum_new - sum_float
                    
                    if gain > 1e-15:
                        S = S_new
                        sum_float = sum_new
                        S_elements = [x for x in range(1, N + 1) if S & (1 << x)]
                        improved = True
                        break
        
        if sum_float > best_sum:
            best_S = S
            best_sum = sum_float

    # Step 3: Try to improve by removing elements that block many additions
    if time.time() < deadline - 5:
        S = best_S
        sum_float = get_float_sum(S)
        
        # For each element, compute how many potential additions it blocks
        for x in range(1, N + 1):
            if (time.time() - start_time) > deadline - 5:
                break
            if not (S & (1 << x)):
                continue
            
            # Count how many y not in S cannot be added because of x
            blocked_count = 0
            for y in range(1, N + 1):
                if S & (1 << y):
                    continue
                if not can_add(y, S):
                    # Check if removing x would allow y to be added
                    S_no_x = S & ~(1 << x)
                    if can_add(y, S_no_x):
                        blocked_count += 1
            
            # If x blocks many additions and has low reciprocal, consider removing it
            if blocked_count > 3 and recip[x] < 0.02:
                S_no_x = S & ~(1 << x)
                S_new = greedy_fill(1 << x, S_no_x)
                sum_new = get_float_sum(S_new)
                if sum_new > sum_float + 1e-15:
                    S = S_new
                    sum_float = sum_new
        
        if sum_float > best_sum:
            best_S = S
            best_sum = sum_float

    # Convert bitmask to list
    result = [x for x in range(1, N + 1) if best_S & (1 << x)]
    
    # Validate
    S_set = set(result)
    valid = True
    for a in S_set:
        for b in S_set:
            if a >= b:
                continue
            s = a + b
            if s % 2 == 0:
                mid = s // 2
                if mid in S_set and mid != a and mid != b:
                    valid = False
                    break
        if not valid:
            break
            
    if not valid:
        # Fallback to greedy
        result = [x for x in range(1, N + 1) if S_asc & (1 << x)]
        
    return result
