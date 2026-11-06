"""Optimized Witt design S(5,6,12) construction via PSL(2,11) group action."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

# Precompute PSL(2,11) permutations at module load time
# This avoids recomputation on every construct() call

def _build_psl2_11():
    """Generate all 660 distinct permutations of PSL(2,11) on 12 points (0..10, inf=11)."""
    inv = [0] * 11
    for i in range(1, 11):
        inv[i] = pow(i, 9, 11)
    
    perms = []
    seen = set()
    
    for a in range(11):
        for b in range(11):
            for c in range(11):
                for d in range(11):
                    if (a * d - b * c) % 11 != 1:
                        continue
                    perm = [0] * 12
                    for x in range(11):
                        num = (a * x + b) % 11
                        den = (c * x + d) % 11
                        if den == 0:
                            perm[x] = 11
                        else:
                            perm[x] = (num * inv[den]) % 11
                    if c == 0:
                        perm[11] = 11
                    else:
                        perm[11] = (a * inv[c]) % 11
                    t = tuple(perm)
                    if t not in seen:
                        seen.add(t)
                        perms.append(t)
    return perms

# Build once at import time
_PSL_PERMS = _build_psl2_11()

def construct():
    start = time.time()
    deadline = 145.0  # Safety margin before 150s wall-clock limit
    
    # Known base block with stabilizer of order 5 in PSL(2,11)
    # Orbit size = 660 / 5 = 132, which is exactly the Witt design
    base = (0, 1, 2, 3, 4, 5)
    
    # Generate orbit of base block under PSL(2,11)
    orbit = set()
    for p in _PSL_PERMS:
        # Apply permutation to each element of base block
        b0 = p[base[0]]
        b1 = p[base[1]]
        b2 = p[base[2]]
        b3 = p[base[3]]
        b4 = p[base[4]]
        b5 = p[base[5]]
        # Sort for canonical representation
        arr = [b0, b1, b2, b3, b4, b5]
        arr.sort()
        orbit.add((arr[0], arr[1], arr[2], arr[3], arr[4], arr[5]))
    
    # Verify orbit size
    if len(orbit) != 132:
        # Fallback: try alternative base blocks
        for alt_base in [(0,1,3,4,5,9), (0,1,2,4,5,8)]:
            orbit = set()
            for p in _PSL_PERMS:
                b0 = p[alt_base[0]]
                b1 = p[alt_base[1]]
                b2 = p[alt_base[2]]
                b3 = p[alt_base[3]]
                b4 = p[alt_base[4]]
                b5 = p[alt_base[5]]
                arr = [b0, b1, b2, b3, b4, b5]
                arr.sort()
                orbit.add((arr[0], arr[1], arr[2], arr[3], arr[4], arr[5]))
            if len(orbit) == 132:
                break
    
    # Verify no 5-subset is repeated (validity check)
    seen_fives = set()
    valid = True
    for block in orbit:
        for i in range(6):
            sub = block[:i] + block[i+1:]
            if sub in seen_fives:
                valid = False
                break
            seen_fives.add(sub)
        if not valid:
            break
    
    if valid and len(seen_fives) == 792:
        return [list(b) for b in orbit]
    
    # Fallback: greedy packing (should not be reached)
    FIVE_SUBSETS = list(combinations(range(12), 5))
    FIVE_TO_IDX = {s: i for i, s in enumerate(FIVE_SUBSETS)}
    SIX_SUBSETS = list(combinations(range(12), 6))
    covered = bytearray(792)
    current_blocks = []
    
    for b in SIX_SUBSETS:
        idxs = [FIVE_TO_IDX[sub] for sub in combinations(b, 5)]
        if not any(covered[i] for i in idxs):
            current_blocks.append(list(b))
            for i in idxs:
                covered[i] = 1
    
    return current_blocks
# EVOLVE-BLOCK-END
