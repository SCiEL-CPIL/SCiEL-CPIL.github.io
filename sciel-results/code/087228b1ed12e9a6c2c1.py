import time

def construct():
    """
    Finds the minimum start s for a run of 5 consecutive non-zero quadratic residues mod 1009.
    Uses a bitset representation for O(1) per-candidate consecutive-residue checks.
    Returns a dictionary with 'start' and 'roots'.
    """
    p = 1009
    m = 5
    
    # Deadline guard: safety margin before 90s kill
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Precompute:
    # 1. root_of[r] = one witness x such that x*x % p == r (for non-zero QRs)
    # 2. qr_bitset: bit i set iff i is a non-zero quadratic residue mod p
    root_of = [0] * p
    qr_bitset = 0
    
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            # Set the bit for residue r
            qr_bitset |= (1 << r)
            # Store one root (smallest x found first, since we iterate ascending)
            if root_of[r] == 0:
                root_of[r] = x
    
    # Mask for 5 consecutive bits: 0b11111
    mask5 = (1 << m) - 1  # 31
    
    # Search for the minimum s in ascending order
    # s ranges from 1 to p - m (1004) to ensure s+m-1 < p
    for s in range(1, p - m + 1):
        # Check if bits s through s+4 are all set in qr_bitset
        # (qr_bitset >> s) & mask5 == mask5  iff  bits s..s+4 are all 1
        if (qr_bitset >> s) & mask5 == mask5:
            roots = [root_of[s + i] for i in range(m)]
            return {'start': s, 'roots': roots}
        
        # Safety check (should never trigger for p=1009, m=5)
        if time.time() > deadline:
            break
    
    # Fallback, though mathematically a solution is guaranteed to exist for p=1009, m=5
    raise RuntimeError("No valid run found")
