import time

def construct():
    p = 1009
    
    # Precompute roots and a bitmask for non-zero quadratic residues
    # root_arr[i] will store a root r such that r*r % p == i, or 0 if i is not a QR
    root_arr = [0] * p
    mask = 0
    
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            # We only need one root per residue. 
            # If root_arr[r] is 0, it means we haven't seen this residue yet.
            if root_arr[r] == 0:
                root_arr[r] = x
            # Set the bit for this residue in the mask
            mask |= (1 << r)
    
    # Search for the minimum s such that s, s+1, s+2, s+3, s+4 are all non-zero QRs
    # s ranges from 1 to 1004 (since s+4 <= 1008)
    # We use the bitmask to check all 5 positions at once.
    # (mask >> s) shifts the bits so that bit 0 corresponds to value s.
    # We check if the lowest 5 bits (0x1F) are all set.
    
    for s in range(1, 1005):
        # Check if bits s, s+1, s+2, s+3, s+4 are all set in mask
        if ((mask >> s) & 0x1F) == 0x1F:
            # All 5 values are non-zero quadratic residues.
            # Retrieve the roots
            roots = [root_arr[s + i] for i in range(5)]
            return {'start': s, 'roots': roots}
            
    # This should not happen for p=1009, but just in case
    raise RuntimeError("No valid run found")
