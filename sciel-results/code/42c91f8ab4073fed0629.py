"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _is_valid(S):
    """Check if set S is valid using float with tight tolerance."""
    S_list = list(S)
    for a in S_list:
        target = 1.0 / a
        others = [x for x in S_list if x != a]
        n = len(others)
        if n == 0:
            continue
        
        mid = n // 2
        left = others[:mid]
        right = others[mid:]
        
        left_sums = set()
        for mask in range(1, 1 << len(left)):
            s = 0.0
            for i in range(len(left)):
                if mask & (1 << i):
                    s += 1.0 / left[i]
            left_sums.add(round(s, 15))
        
        right_sums = set()
        for mask in range(1, 1 << len(right)):
            s = 0.0
            for i in range(len(right)):
                if mask & (1 << i):
                    s += 1.0 / right[i]
            right_sums.add(round(s, 15))
        
        t = round(target, 15)
        for ls in left_sums:
            if round(t - ls, 15) in right_sums:
                return False
        if t in left_sums or t in right_sums:
            return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    best_set = None
    best_size = 0
    
    # Try different starting sets and orderings
    start_sets = [
        list(range(13, 25)),
        list(range(10, 25)),
        list(range(8, 25)),
    ]
    
    add_orders = [
        list(range(1, 13)),
        list(range(12, 0, -1)),
        list(range(1, 13)),
    ]
    
    for si, start in enumerate(start_sets):
        if time.time() > deadline:
            break
        S = start[:]
        order = add_orders[si % len(add_orders)]
        
        for a in order:
            if time.time() > deadline:
                break
            if a not in S:
                S_new = S + [a]
                if _is_valid(S_new):
                    S = S_new
        
        # Try adding remaining
        for a in range(1, 25):
            if time.time() > deadline:
                break
            if a not in S:
                S_new = S + [a]
                if _is_valid(S_new):
                    S = S_new
        
        S.sort()
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
    
    if best_set is None or len(best_set) == 0:
        return list(range(13, 25))
    
    return best_set
# EVOLVE-BLOCK-END
