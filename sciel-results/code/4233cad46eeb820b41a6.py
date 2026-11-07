"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def _lcm(a, b):
    return a * b // math.gcd(a, b)

# Compute LCM of 1..12
_LCM = 1
for i in range(1, 13):
    _LCM = _lcm(_LCM, i)
# _LCM = 27720

# Max possible L is bounded. The problem states L <= 144.
# Let's use a slightly larger bound for safety in mask precomputation.
MAX_L = 200
N = 12

# Precompute masks for each i in 1..12 and each residue r in 0..i-1
# mask[i][r] is a bitmask of length MAX_L where bit k is 1 if k % i == r
precomputed_masks = [[0] * i for i in range(N + 1)]
for i in range(1, N + 1):
    for r in range(i):
        mask = 0
        for k in range(MAX_L):
            if k % i == r:
                mask |= (1 << k)
        precomputed_masks[i][r] = mask

def _can_match(m, L):
    """
    Check if (m, m+L] contains distinct b_1...b_N with i | b_i.
    Uses precomputed masks and DFS.
    """
    # Get masks for each i
    masks = []
    for i in range(1, N + 1):
        r = (-(m + 1)) % i
        # We only care about bits 0 to L-1
        # precomputed_masks[i][r] has bits up to MAX_L-1.
        # We need to mask out bits >= L.
        # Since L <= MAX_L, we can just use the precomputed mask.
        # The DFS will only consider bits < L if we check availability correctly.
        # Actually, the mask contains bits for all k < MAX_L.
        # We must ensure that the chosen k is < L.
        # We can create a limit mask: (1 << L) - 1.
        limit_mask = (1 << L) - 1
        masks.append(precomputed_masks[i][r] & limit_mask)
        
    # Sort by number of options (most constrained first)
    # We store (popcount, mask, i)
    sorted_masks = []
    for idx, mask in enumerate(masks):
        if mask == 0:
            return False
        sorted_masks.append((mask.bit_count(), mask, idx))
    sorted_masks.sort(key=lambda x: x[0])
    
    # DFS
    def dfs(k, used_mask):
        if k == N:
            return True
        _, mask, _ = sorted_masks[k]
        avail = mask & ~used_mask
        while avail:
            bit = avail & (-avail)
            avail -= bit
            if dfs(k + 1, used_mask | bit):
                return True
        return False
    
    return dfs(0, 0)

def _f_val(m):
    """
    Find the least positive L such that (m, m+L] contains distinct b_1...b_N with i | b_i.
    """
    # Lower bound: N
    # Upper bound: MAX_L
    lo = 1
    hi = MAX_L
    
    # Find a valid hi (should be found within MAX_L)
    while not _can_match(m, hi):
        hi *= 2
        if hi > 1000: # Safety
            break
    
    # Binary search for the minimum L
    while lo < hi:
        mid = (lo + hi) // 2
        if _can_match(m, mid):
            hi = mid
        else:
            lo = mid + 1
    
    return lo

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80 seconds, leaving 10s margin
    
    max_m = 1000000
    lcm_period = _LCM  # 27720
    
    best_f = 0
    best_m = 0
    
    # Iterate over residues
    # We can optimize by checking if we have enough time
    for r in range(lcm_period):
        if time.time() > deadline:
            break
        
        f = _f_val(r)
        if f > best_f:
            best_f = f
            best_m = r
    
    return int(best_m)

# EVOLVE-BLOCK-END
