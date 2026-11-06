"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def _is_dissociated(subset):
    n = len(subset)
    sums = set()
    for mask in range(1 << n):
        s = 0.0
        for i in range(n):
            if mask & (1 << i):
                s += 1.0 / subset[i]
        s_rounded = round(s, 12)
        if s_rounded in sums:
            return False
        sums.add(s_rounded)
    return True

def construct():
    # Try a large set of larger numbers
    candidate = list(range(10, 23))  # 13 elements
    if _is_dissociated(candidate):
        return candidate
    
    # Try smaller sizes
    for size in range(12, 0, -1):
        candidate = list(range(23 - size, 23))
        if _is_dissociated(candidate):
            return candidate
    
    return [1]
# EVOLVE-BLOCK-END
