"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # Safety margin before 90s kill
    
    # Known valid set: {63,...,128}, size 66
    best_set = set(range(63, 129))
    best_size = 66
    
    def can_add(x, S):
        """Check if adding x to set S creates a forbidden triple."""
        S_set = S  # already a set
        S_list = sorted(S)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                # Triple (x, a, b): need x+a, x+b, a+b all in S ∪ {x}
                sa = x + a
                sb = x + b
                ab = a + b
                if (sa in S_set or sa == x) and (sb in S_set or sb == x) and (ab in S_set or ab == x):
                    return False
        return True
    
    # Try greedy addition from below 63 with different orders
    orders = [
        list(range(1, 63)),
        list(range(62, 0, -1)),
        list(range(1, 63, 2)),
        list(range(62, 0, -2)),
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        current_set = set(range(63, 129))
        for x in order:
            if time.time() > deadline:
                break
            if can_add(x, current_set):
                current_set.add(x)
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
    
    # Try different base intervals
    for m in range(58, 64):
        if time.time() > deadline:
            break
        base = set(range(m, 129))
        # Quick check: is base valid?
        # For base [m,128], sums in base are [2m+1, 128]
        # Need to verify no forbidden triple in base itself
        # This is expensive; skip if too slow
        if time.time() > deadline:
            break
        
        # Check if base is valid by checking all triples (expensive but m is close to 63)
        base_list = sorted(base)
        base_valid = True
        for i in range(len(base_list)):
            if time.time() > deadline:
                base_valid = False
                break
            a = base_list[i]
            for j in range(i + 1, len(base_list)):
                b = base_list[j]
                for k in range(j + 1, len(base_list)):
                    c = base_list[k]
                    if (a + b) in base and (a + c) in base and (b + c) in base:
                        base_valid = False
                        break
                if not base_valid:
                    break
            if not base_valid:
                break
        
        if not base_valid:
            continue
        
        for order in [list(range(1, m)), list(range(m - 1, 0, -1))]:
            if time.time() > deadline:
                break
            current_set = set(base)
            for x in order:
                if time.time() > deadline:
                    break
                if can_add(x, current_set):
                    current_set.add(x)
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
    
    result = [int(x) for x in sorted(best_set)]
    return result

# EVOLVE-BLOCK-END
