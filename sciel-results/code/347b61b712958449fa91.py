"""Rewritten program using bitset approach for finding consecutive quadratic residues."""
import time

def construct():
    p = 1009
    deadline = time.time() + 80  # safety margin before 90s limit
    
    # Precompute roots: for each residue r (1..p-1), store one root x such that x^2 ≡ r (mod p)
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r != 0 and r not in roots:
            roots[r] = x
    
    # Build a bitset where bit i is set if i is a quadratic residue (1 <= i <= 1008)
    # We need to find the smallest s (1 <= s <= 1004) such that s, s+1, s+2, s+3, s+4 are all residues
    # Use an integer as a bitset: bit i represents whether residue i exists
    bitset = 0
    for r in roots:
        bitset |= (1 << r)
    
    # We need to find the smallest s where bits s, s+1, s+2, s+3, s+4 are all set
    # Create a mask that has bits 0-4 set: 0b11111 = 31
    mask = 0b11111
    
    # For each possible s from 1 to 1004, check if (bitset >> s) & mask == mask
    # This is a single bitwise operation per s
    best_s = None
    for s in range(1, 1005):
        if (bitset >> s) & mask == mask:
            best_s = s
            break
    
    if best_s is None:
        # Fallback: this should not happen, but handle gracefully
        raise RuntimeError('No run of 5 consecutive quadratic residues found')
    
    result_roots = [roots[best_s + i] for i in range(5)]
    return {'start': best_s, 'roots': result_roots}
