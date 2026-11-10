"""Find the minimum s such that s, s+1, s+2, s+3, s+4 are all non-zero quadratic residues mod 1009."""
import time

# EVOLVE-BLOCK-START
def construct():
    p = 1009
    m = 5
    deadline = time.time() + 85.0  # safety margin before 90s kill

    # Precompute: for each value v in 1..p-1, store a root r such that r^2 % p == v.
    # We want the smallest s (1 <= s <= 1004) such that s, s+1, ..., s+4 are all
    # non-zero quadratic residues mod p.
    #
    # Strategy:
    # 1. Build a boolean array is_residue[v] = True if v is a non-zero quadratic residue mod p.
    # 2. Build a roots dict mapping each residue v to some root r with r^2 % p == v.
    # 3. Scan s from 1 upward; the first s where all of s..s+4 are residues is optimal.
    #
    # Since p=1009 is small, this is extremely fast.

    # Step 1: Determine which values are quadratic residues
    is_residue = [False] * p
    roots = {}
    for x in range(1, p):
        r2 = (x * x) % p
        if r2 != 0:
            is_residue[r2] = True
            if r2 not in roots:
                roots[r2] = x

    # Step 2: Find the smallest s such that s, s+1, ..., s+4 are all residues
    # Note: s+i must be in range 1..1008 (since 1<=ri<=1008 and ri^2 mod 1009 = s+i)
    # Actually s+i mod 1009 should be in 1..1008, but since s <= 1004 and i <= 4,
    # s+4 <= 1008, so s+i is always in 1..1008. No wraparound needed.
    
    for s in range(1, p - m + 1):  # s from 1 to 1004
        # Check if s, s+1, ..., s+4 are all quadratic residues
        valid = True
        for i in range(m):
            v = s + i
            if not is_residue[v]:
                valid = False
                break
        if valid:
            result_roots = [roots[s + i] for i in range(m)]
            # Verify: each root squared mod p should equal s+i
            for i in range(m):
                assert (result_roots[i] * result_roots[i]) % p == s + i
            return {'start': s, 'roots': result_roots}

    # Fallback: should not reach here for p=1009, but just in case
    raise RuntimeError('No run of 5 consecutive quadratic residues found')
# EVOLVE-BLOCK-END
