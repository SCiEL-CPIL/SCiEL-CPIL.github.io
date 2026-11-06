"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Find the minimum start s (1 <= s <= 1004) such that s, s+1, ..., s+4 are
    all quadratic residues modulo 1009, with explicit non-zero roots for each.
    
    Strategy:
    1. Precompute an array `roots_arr` where roots_arr[r] is the smallest root x
       such that x^2 mod p == r, or 0 if not a residue.
    2. Scan s from 1 upwards. For each s, check if all of s..s+4 have non-zero roots.
    3. Return the first valid s with its roots.
    4. Include a deadline guard for safety.
    """
    p = 1009
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin before 90s kill
    
    # Precompute roots using an array for O(1) access
    # roots_arr[r] = smallest x in [1, p-1] such that x^2 % p == r, else 0
    roots_arr = [0] * p
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:  # Exclude zero residues
            if roots_arr[r] == 0:  # Keep the first (smallest) root found
                roots_arr[r] = x
    
    # Scan for the minimum start s
    # s ranges from 1 to 1004 (so that s+4 <= 1008, no wraparound into 1009+)
    for s in range(1, 1005):
        if time.time() > deadline:
            # If we hit the deadline (unlikely for this small p), return best found so far?
            # Since we scan from 1 up, if we haven't found one, there is no valid s.
            # But to be safe against infinite loops or hangs, we break.
            break
        
        # Check if s, s+1, s+2, s+3, s+4 are all quadratic residues
        valid = True
        result_roots = []
        for i in range(5):
            r = s + i
            root = roots_arr[r]
            if root == 0:
                valid = False
                break
            result_roots.append(root)
        
        if valid:
            return {'start': s, 'roots': result_roots}
    
    raise RuntimeError('No valid run found')
# EVOLVE-BLOCK-END
