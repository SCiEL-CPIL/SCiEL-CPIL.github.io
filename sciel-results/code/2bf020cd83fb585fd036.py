import time

P = 1009
M = 5
DEADLINE = time.time() + 85.0  # Safety margin before 90s limit

def construct():
    # 1. Precompute the roots for non-zero quadratic residues
    # qr_root[v] = r such that r*r % P == v, or 0 if v is not a non-zero QR
    qr_root = [0] * P
    for x in range(1, P):
        r = (x * x) % P
        if r != 0 and qr_root[r] == 0:
            qr_root[r] = x

    # 2. Build the bitmask for non-zero quadratic residues
    # Bit v is 1 if v is a non-zero quadratic residue
    mask = 0
    for v in range(1, P):
        if qr_root[v] != 0:
            mask |= (1 << v)

    # 3. Find the minimal s in [1, P-M] such that s..s+4 are all QRs
    # We check if the 5 bits starting at position s are all 1.
    # (mask >> s) & 0x1F == 0x1F
    target_pattern = (1 << M) - 1  # 0b11111 = 31
    
    for s in range(1, P - M + 1):
        # Check if bits s, s+1, s+2, s+3, s+4 are all set
        if ((mask >> s) & target_pattern) == target_pattern:
            roots = [qr_root[s + i] for i in range(M)]
            return {'start': s, 'roots': roots}
        
        # Deadline check (should not trigger for this small problem)
        if time.time() > DEADLINE:
             break

    # Fallback: if loop didn't return (should not happen for p=1009)
    # Re-scan with explicit checks just in case
    for s in range(1, P - M + 1):
        if all(qr_root[s + i] != 0 for i in range(M)):
            roots = [qr_root[s + i] for i in range(M)]
            return {'start': s, 'roots': roots}
            
    raise RuntimeError('No run of 5 consecutive nonzero QRs found')
