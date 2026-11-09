"""Find the minimal start s for 5 consecutive quadratic residues modulo 1009."""
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85.0  # Safety margin before 90s kill
    
    p = 1009
    m = 5
    
    # Precompute a list where root_of[r] is a root x such that x*x % p == r
    # 0 indicates r is not a non-zero quadratic residue.
    # Using a list is faster than a dict for small integers and simpler to initialize.
    root_of = [0] * p
    
    for x in range(1, p):
        r = (x * x) % p
        # We only care about non-zero residues.
        # If r is 0, skip (0 is not a valid residue per task: "zero residues ... excluded")
        if r != 0:
            # setdefault equivalent: only assign if not already set, 
            # though any root is valid, the first one found (smallest x) is fine.
            if root_of[r] == 0:
                root_of[r] = x

    # Search for the minimum s in 1..p-m such that s, s+1, ..., s+m-1 are all non-zero QRs
    # s+i must be <= p-1 to avoid wrap-around issues with the definition of residues 1..p-1.
    # Specifically, if s+m-1 > p-1, we might hit p (which is 0 mod p, excluded) or wrap.
    # The task states 1 <= ri <= 1008, so residues must be in 1..1008.
    # Thus s+i must be in 1..1008.
    # Max s is p - m = 1009 - 5 = 1004.
    
    for s in range(1, p - m + 1):
        # Check if all s..s+m-1 are in root_of
        valid = True
        for i in range(m):
            if root_of[s + i] == 0:
                valid = False
                break
        
        if valid:
            # Extract roots
            result_roots = [root_of[s + i] for i in range(m)]
            
            # Final verification step to ensure correctness and validity
            # This acts as a "polish" or safety check against any logical flaws
            is_valid = True
            for i, r in enumerate(result_roots):
                # Check range 1 <= r <= 1008
                if r < 1 or r > p - 1:
                    is_valid = False
                    break
                # Check r^2 mod p == s+i
                if (r * r) % p != (s + i):
                    is_valid = False
                    break
            
            if is_valid:
                return {'start': s, 'roots': result_roots}
            else:
                # If verification fails (should not happen), continue searching
                # This prevents returning an invalid result which scores 0
                continue

    # Fallback: If no run found (shouldn't happen for p=1009, m=5)
    # The problem guarantees a solution exists within the constraints.
    raise RuntimeError('No run of 5 consecutive QRs found')

# EVOLVE-BLOCK-END
