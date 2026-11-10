"""Find the earliest run of 5 consecutive quadratic residues modulo 1009."""

def construct():
    p = 1009
    
    # Precompute: for each residue value, store a root
    # r^2 mod p = residue
    # We iterate r from 1 to (p-1)//2 to get distinct residues
    root_of = {}
    for r in range(1, (p - 1) // 2 + 1):
        residue = (r * r) % p
        if residue not in root_of:
            root_of[residue] = r
            
    # Build a bitset: bit i is set if i is a QR mod p (i in 1..1008)
    # Using an integer where bit position i represents value i
    bitset = 0
    for res in root_of:
        bitset |= (1 << res)
        
    # To find s where s, s+1, s+2, s+3, s+4 are all residues:
    # We compute the AND of the bitset shifted by 0, 1, 2, 3, 4.
    # If bit s is set in the result, it means bits s, s+1, s+2, s+3, s+4 
    # were all set in the original bitset.
    candidates = bitset
    for shift in range(1, 5):
        candidates &= (bitset >> shift)
        
    # Search for the minimum s such that 1 <= s <= 1004 and bit s is set in candidates
    # s+4 must be <= 1008, so s <= 1004.
    # Also s >= 1.
    for s in range(1, 1005):
        if (candidates >> s) & 1:
            return {
                'start': s,
                'roots': [root_of[s + i] for i in range(5)]
            }
            
    raise RuntimeError('No run of 5 consecutive quadratic residues found')
