"""Rewritten program using bitset-based search for minimal run-start."""
import time

def construct():
    p = 1009
    start_time = time.time()
    deadline = start_time + 60.0  # 60s safety margin within 90s limit
    
    # Build bitset: bit i is set if residue i (mod p) is a quadratic residue with a non-zero root
    # We need residues 1..1008 (excluding 0)
    # For each x in 1..p-1, compute x*x % p, set that bit
    # Also store one root for each residue
    
    # Use a Python integer as a bitset (p=1009 bits, easily fits)
    residue_bits = 0
    roots = {}  # residue -> root
    
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:  # exclude zero residue
            residue_bits |= (1 << r)
            if r not in roots:  # keep the first (smallest x) root for determinism
                roots[r] = x
    
    # Now search for the smallest s in 1..1004 such that residues s, s+1, s+2, s+3, s+4
    # are all quadratic residues (and all in range 1..1008)
    # Since s <= 1004 and s+4 <= 1008, all residues are in valid range
    
    # For a given s, we need bits s, s+1, s+2, s+3, s+4 all set in residue_bits
    # We can check: ((residue_bits >> s) & 0x1F) == 0x1F
    
    for s in range(1, 1005):
        if ((residue_bits >> s) & 0x1F) == 0x1F:
            # Found the minimal s
            r_list = [s, s+1, s+2, s+3, s+4]
            root_list = [roots[r] for r in r_list]
            return {'start': s, 'roots': root_list}
    
    # Should not reach here, but just in case
    raise RuntimeError('No valid run found')
