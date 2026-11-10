import numpy as np

def construct():
    p = 1009
    # Precompute inverses
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    best_len = 0
    best_start = 1
    
    for s in range(1, p):
        # We want to find the first e >= s such that [s, e] is invalid.
        # [s, e] is invalid if there exists x in [s, e] with inv[x] in [s, e].
        # As we increment e, we only need to check if the new element e creates a conflict.
        # A conflict is created if inv[e] is in [s, e].
        # Note: if inv[e] is in [s, e-1], it's a conflict. If inv[e] == e, it's a conflict.
        # If inv[e] > e, it's not in [s, e] yet, so no conflict with e itself.
        # However, could adding e invalidate a previous pair? No, the set of pairs only grows.
        # If [s, e-1] was valid, no pair in [s, e-1] conflicts.
        # The only new pairs involving e are (e, x) for x in [s, e-1].
        # (e, x) conflicts if e * x == 1 => x == inv[e].
        # So we just need to check if inv[e] is in [s, e].
        
        max_valid_e = s - 1 # Initially no valid e >= s found? No, we start checking from s.
        
        for e in range(s, p):
            # Check if e creates a conflict in [s, e]
            if s <= inv[e] <= e:
                # Interval [s, e] is invalid.
                # So max valid end is e - 1.
                length = (e - 1) - s + 1
                if length > best_len:
                    best_len = length
                    best_start = s
                break
            else:
                # If we reach the end of the range without breaking
                if e == p - 1:
                    length = (p - 1) - s + 1
                    if length > best_len:
                        best_len = length
                        best_start = s
    
    return [best_start, best_len]
