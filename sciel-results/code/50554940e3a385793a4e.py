import time

_P = 1009
_START = time.time()
_DEADLINE = 90.0
_MARGIN = 5.0

def construct():
    # 1. Precompute quadratic residues and their smallest roots
    # is_res[i] is True if i is a non-zero quadratic residue mod _P
    is_res = [False] * (_P)
    # root_list[r] stores the smallest root x in [1, _P-1] such that x^2 % _P == r
    # Using a list instead of dict for O(1) access with lower overhead
    root_list = [0] * _P
    
    for x in range(1, _P):
        r = (x * x) % _P
        if r == 0:
            continue
        if not is_res[r]:
            is_res[r] = True
            root_list[r] = x

    # 2. Build a bitset of residues
    # Bit i is 1 if i is a non-zero quadratic residue
    M = 0
    for i in range(1, _P):
        if is_res[i]:
            M |= (1 << i)
            
    # 3. Find all valid starts s such that s, s+1, s+2, s+3, s+4 are residues
    # We want bit s in the result to be 1 if bits s..s+4 in M are 1.
    # M >> k has bit s set if M has bit s+k set.
    # So ANDing M, M>>1, M>>2, M>>3, M>>4 gives 1 at s if all 5 are residues.
    V = M & (M >> 1) & (M >> 2) & (M >> 3) & (M >> 4)
    
    # We need s >= 1. Bit 0 in V corresponds to s=0.
    # Since 0 is not a residue, bit 0 in M is 0, so bit 0 in V is 0.
    # We also need s+4 <= _P - 1 => s <= _P - 5.
    # The bitset M only has bits up to _P - 1 set.
    # If s > _P - 5, then s+4 > _P - 1, so one of the checks would look at a bit > _P - 1, which is 0 in M.
    # So valid s are naturally bounded.
    
    # Find the least significant set bit in V
    # If V is 0, no solution found (should not happen)
    if V == 0:
        raise RuntimeError("No run of 5 consecutive non-zero quadratic residues found")
    
    # Least set bit index
    # (V & -V) isolates the least significant set bit.
    # To get the index, we can use bit_length - 1.
    s = (V & -V).bit_length() - 1
    
    # Verify s is within valid range [1, _P - 5]
    if s < 1 or s > _P - 5:
        # Fallback to linear scan if bitset logic has edge case issues
        for s_check in range(1, _P - 4):
            if all(is_res[s_check + i] for i in range(5)):
                s = s_check
                break
        else:
            raise RuntimeError("No valid start found even after fallback")
            
    roots = [root_list[s + i] for i in range(5)]
    return {'start': s, 'roots': roots}
