"""Rewritten program using bitwise AND of shifted residue bitsets for O(1) minimal-start search."""
import time

def construct():
    deadline = time.time() + 80  # safety margin before 90s wall-clock limit
    p = 1009
    m = 5
    
    # Build a bitset where bit r is set iff r is a quadratic residue mod p (r != 0)
    # and r is in range 1..p-1
    # We need to check if s, s+1, ..., s+4 are all quadratic residues (non-zero)
    # s ranges from 1 to p-1-m = 1004
    # s+4 ranges from 5 to 1008, all within 1..1008, so no wraparound issues
    
    # Create bitset as a Python int for arbitrary precision
    # bit i set means residue i is a quadratic residue (and non-zero)
    residue_bits = 0
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            residue_bits |= (1 << r)
    
    # Now find the smallest s in [1, 1004] such that bits s, s+1, s+2, s+3, s+4 are all set
    # This is equivalent to: (residue_bits >> s) & 0x1F == 0x1F
    # because bits s through s+4 being set means the lower 5 bits of (residue_bits >> s) are all 1
    
    for s in range(1, 1005):
        if time.time() > deadline:
            # Should not happen for this small problem, but guard just in case
            raise RuntimeError("Deadline exceeded during search")
        if (residue_bits >> s) & 0x1F == 0x1F:
            # Found minimal s; now find roots for s, s+1, ..., s+4
            roots = []
            for i in range(m):
                target = s + i
                # Find a root x in [1, 1008] such that x^2 mod p == target
                # We can search directly
                found = False
                for x in range(1, p):
                    if (x * x) % p == target:
                        roots.append(x)
                        found = True
                        break
                if not found:
                    raise RuntimeError(f"No root found for residue {target}")
            return {'start': s, 'roots': roots}
    
    raise RuntimeError('No baseline run found')
