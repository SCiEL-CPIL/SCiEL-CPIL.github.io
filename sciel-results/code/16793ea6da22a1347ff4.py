"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s limit
    N = 256
    
    # Precompute conflict graph: conflicts[i] is a bitmask of numbers that conflict with i
    # Two numbers a, b conflict if there exists c such that {a, b, c} forms a 3-term AP
    # i.e., a + c = 2b or b + c = 2a or a + b = 2c
    # For a given x, it conflicts with y if:
    #   - x is middle: y + z = 2x for some z, i.e., z = 2x - y must be in [1,N] and z != y
    #   - x is end: y + x = 2z for some z, i.e., z = (y+x)/2 must be integer, in [1,N], z != x, z != y
    #   - x is other end: x + z = 2y for some z, i.e., z = 2y - x must be in [1,N], z != x, z != y
    # Actually, we need: adding x to S creates a 3-AP iff there exist a, b in S such that
    # {a, b, x} contains a 3-term AP. This happens iff:
    #   - 2x - a = b for some a != b in S (x is middle)
    #   - (a + x) / 2 = b for some a != b != x in S, (a+x) even (x is end, a is other end)
    #   - (b + x) / 2 = a for some a != b != x in S, (b+x) even (x is end, b is other end)
    # But (a+x)/2 = b and (b+x)/2 = a are symmetric. So:
    #   x conflicts with pair {a, b} if a + b = 2x or (a+x) even and (a+x)/2 in S or (b+x) even and (b+x)/2 in S
    # For precomputation: x and y are "conflict partners" if there exists z such that {x, y, z} is a 3-AP.
    # This means: x + y = 2z (z is middle) or x + z = 2y (y is middle) or y + z = 2x (x is middle)
    # Case 1: x + y = 2z => z = (x+y)/2, need x+y even, z in [1,N], z != x, z != y
    # Case 2: x + z = 2y => z = 2y - x, need z in [1,N], z != x, z != y
    # Case 3: y + z = 2x => z = 2x - y, need z in [1,N], z != x, z != y
    # So x and y conflict if any of these hold.
    
    # For efficient can_add: given S (as a bitmask), x can be added iff for all a in S,
    #   - 2x - a is not in S (and 2x-a != a)  [x as middle]
    #   - (a+x) is even and (a+x)//2 is not in S (and (a+x)//2 != a, (a+x)//2 != x) [x as end]
    # Equivalently, using precomputed conflict masks:
    #   x can be added iff (conflict_mask[x] & S_mask) == 0
    # where conflict_mask[x] has bit y set iff x and y cannot both be in a 3-AP-free set.
    
    # Build conflict masks
    conflict_mask = [0] * (N + 1)  # 1-indexed
    for x in range(1, N + 1):
        mask = 0
        for y in range(1, N + 1):
            if x == y:
                continue
            # Check if {x, y, z} forms a 3-AP for some z
            found = False
            # Case 1: x + y = 2z, z = (x+y)/2
            if (x + y) % 2 == 0:
                z = (x + y) // 2
                if 1 <= z <= N and z != x and z != y:
                    found = True
            # Case 2: x + z = 2y, z = 2y - x
            z = 2 * y - x
            if 1 <= z <= N and z != x and z != y:
                found = True
            # Case 3: y + z = 2x, z = 2x - y
            z = 2 * x - y
            if 1 <= z <= N and z != x and z != y:
                found = True
            if found:
                mask |= (1 << y)
        conflict_mask[x] = mask
    
    def can_add_mask(S_mask, x):
        """Check if x can be added to set represented by S_mask."""
        return (conflict_mask[x] & S_mask) == 0
    
    def reciprocal_sum(S):
        return sum(1.0 / x for x in S)
    
    def greedy_construction(order=None):
        """Greedy construction given an order of elements to consider."""
        if order is None:
            order = list(range(1, N + 1))
        S_mask = 0
        for x in order:
            if 1 <= x <= N and can_add_mask(S_mask, x):
                S_mask |= (1 << x)
        return S_mask
    
    def mask_to_set(S_mask):
        return [x for x in range(1, N + 1) if (S_mask >> x) & 1]
    
    def local_search(S_mask, current_score, max_iters=5000):
        """Local search with add and swap moves."""
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            
            # Try adding elements
            for x in range(1, N + 1):
                if not ((S_mask >> x) & 1) and can_add_mask(S_mask, x):
                    S_mask |= (1 << x)
                    current_score += 1.0 / x
                    improved = True
                    break
            
            # Try swap moves: remove x, add y
            if not improved and time.time() < deadline:
                S_list = mask_to_set(S_mask)
                for x in S_list:
                    S_mask_no_x = S_mask & ~(1 << x)
                    for y in range(1, N + 1):
                        if y == x:
                            continue
                        if not ((S_mask_no_x >> y) & 1) and can_add_mask(S_mask_no_x, y):
                            new_score = current_score - 1.0 / x + 1.0 / y
                            if new_score > current_score + 1e-15:
                                S_mask = S_mask_no_x | (1 << y)
                                current_score = new_score
                                improved = True
                                break
                    if improved:
                        break
        
        return S_mask, current_score
    
    # Initial greedy construction (smallest first for max reciprocal)
    S_mask = greedy_construction()
    S_set = mask_to_set(S_mask)
    best_score = reciprocal_sum(S_set)
    best_mask = S_mask
    
    # Local search refinement
    S_mask, best_score = local_search(S_mask, best_score)
    best_mask = S_mask
    
    # Multiple randomized restarts
    random.seed(42)
    for restart in range(50):
        if time.time() > deadline - 5:  # leave 5s buffer
            break
        
        # Create randomized order: bias toward small numbers
        order = list(range(1, N + 1))
        # Assign random priorities but bias small numbers
        weighted_order = []
        for x in order:
            weight = 1.0 / x + random.random() * 0.5
            weighted_order.append((weight, x))
        weighted_order.sort(key=lambda t: t[0])
        order = [x for _, x in weighted_order]
        
        S_mask = greedy_construction(order)
        S_set = mask_to_set(S_mask)
        score = reciprocal_sum(S_set)
        S_mask, score = local_search(S_mask, score)
        
        if score > best_score:
            best_score = score
            best_mask = S_mask
    
    # Final polish: try all possible single additions and swaps one more time
    S_mask, best_score = local_search(best_mask, best_score, max_iters=10000)
    best_mask = S_mask
    
    return mask_to_set(best_mask)
# EVOLVE-BLOCK-END
