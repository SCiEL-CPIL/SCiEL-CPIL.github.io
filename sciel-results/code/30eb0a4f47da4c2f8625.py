"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Find the minimal start s (1 <= s <= 1004) such that s, s+1, s+2, s+3, s+4
    are all non-zero quadratic residues modulo 1009, and return the start and
    one root for each residue.
    
    Approach: Precompute a bitmask of quadratic residues mod 1009 (bits 1..1008),
    then use bitwise operations to check runs of 5 consecutive residues in O(1)
    per candidate s. Scan s from 1 upward to find the minimal valid start.
    """
    start_time = time.time()
    deadline = start_time + 60.0  # Safety margin before 90s kill
    
    p = 1009
    
    # Build a bitmask where bit i is set if i is a non-zero quadratic residue mod p
    # We use a list of integers for the bitmask (Python int supports arbitrary length)
    # Bit i (0-indexed) represents residue value i
    # We care about residues 1..1008 (excluding 0)
    
    # Create a set of quadratic residues
    qr_set = set()
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            qr_set.add(r)
            if r not in roots:
                roots[r] = x
    
    # Build a bitmask: bit i set if i is in qr_set
    # Python integers can be arbitrarily large, so we can use a single int
    mask = 0
    for r in qr_set:
        mask |= (1 << r)
    
    # For a given s, we need bits s, s+1, s+2, s+3, s+4 all set
    # That is: (mask >> s) & 0b11111 == 0b11111
    # But we must also ensure s+4 <= 1008 (since residues are 1..1008)
    # So s can range from 1 to 1004
    
    # Scan s from 1 upward to find the minimal valid start
    for s in range(1, 1005):
        # Check if bits s through s+4 are all set
        if (mask >> s) & 0b11111 == 0b11111:
            # Verify s+4 <= 1008 (i.e., s <= 1004, which is guaranteed by range)
            # Extract roots
            result_roots = []
            valid = True
            for i in range(5):
                val = s + i
                if val > 1008 or val < 1:
                    valid = False
                    break
                if val not in roots:
                    valid = False
                    break
                result_roots.append(roots[val])
            if valid:
                return {'start': s, 'roots': result_roots}
        
        # Check time periodically
        if time.time() > deadline:
            # This should not happen for this small problem, but safety guard
            raise RuntimeError("Time limit exceeded")
    
    raise RuntimeError('No valid run found')
# EVOLVE-BLOCK-END
