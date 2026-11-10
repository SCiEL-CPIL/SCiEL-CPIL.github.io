"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Find the longest nonempty interval [start, start+length-1] contained in [1, 1008]
    such that for every x, y in the interval (including x=y), x*y % 1009 != 1.
    """
    deadline = time.time() + 85  # 85s deadline, 5s safety margin
    
    P = 1009
    MAX_VAL = 1008  # residues 1..1008
    
    # Precompute modular inverses
    inv = [0] * (P + 1)
    for x in range(1, P):
        inv[x] = pow(x, P - 2, P)
    
    # Identify self-inverse residues (x^2 ≡ 1 mod P)
    self_inverse = set()
    for x in range(1, P):
        if x * x % P == 1:
            self_inverse.add(x)
    
    # Assign group IDs: self-inverse elements get -1 (forbidden),
    # each inverse pair gets a unique positive ID
    group_id = [0] * (P + 1)
    group_counter = 0
    assigned = [False] * (P + 1)
    for x in range(1, P):
        if x in self_inverse:
            group_id[x] = -1
        elif not assigned[x]:
            group_counter += 1
            y = inv[x]
            group_id[x] = group_counter
            group_id[y] = group_counter
            assigned[x] = True
            assigned[y] = True
    
    # Two-pointer / sliding window approach
    # freq[g] = number of elements with group ID g in current window
    freq = {}
    left = 1
    best_len = 0
    best_s = 1
    
    for right in range(1, MAX_VAL + 1):
        g = group_id[right]
        if g == -1:
            # Self-inverse element: window must exclude it
            # Reset window to start after this element
            freq.clear()
            left = right + 1
            if left > MAX_VAL:
                break
            continue
        else:
            freq[g] = freq.get(g, 0) + 1
            # If we now have two elements from the same inverse pair, shrink from left
            while freq.get(g, 0) > 1:
                lg = group_id[left]
                if lg != -1:
                    freq[lg] -= 1
                    if freq[lg] == 0:
                        del freq[lg]
                left += 1
        
        current_len = right - left + 1
        if current_len > best_len:
            best_len = current_len
            best_s = left
    
    # Verify the result
    if best_len > 0:
        s = best_s
        e = s + best_len - 1
        valid = True
        for x in range(s, e + 1):
            if x in self_inverse:
                valid = False
                break
            y = inv[x]
            if s <= y <= e:
                valid = False
                break
        if valid:
            return [s, best_len]
    
    # Fallback: try a simple valid interval
    return [2, 30]
# EVOLVE-BLOCK-END
