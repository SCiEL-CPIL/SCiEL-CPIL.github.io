import time

# EVOLVE-BLOCK-START
def construct():
    p = 1009
    deadline = time.time() + 59.0  # Safety margin
    
    # Precompute Legendre symbol and smallest roots
    # legendre[x] = 1 if QR, -1 if QNR, 0 if x=0
    # root[r] = x where x^2 mod p == r
    legendre = [0] * p
    root = [0] * p
    
    half_p = (p - 1) // 2
    
    for x in range(1, p):
        # Calculate Legendre symbol
        # pow(x, half_p, p) returns 1 or p-1
        ls = pow(x, half_p, p)
        if ls == 1:
            legendre[x] = 1
            # Store smallest root for the residue x
            # Note: x is the root, x*x%p is the residue.
            # Wait, the loop variable x is the base.
            # The residue is r = x*x % p.
            # We need root[r] = x (smallest).
            r = (x * x) % p
            if root[r] == 0:
                root[r] = x
        else:
            legendre[x] = -1
            
    # Note: The loop above sets legendre[x] based on x being a residue?
    # NO. Legendre symbol is defined for the value x.
    # chi(x) = 1 if x is a quadratic residue.
    # My loop calculates chi(x) for each x.
    # So legendre[x] is 1 if x is a QR.
    # This is correct.
    
    # Find smallest s such that s..s+4 are all QRs
    # This means legendre[s] == 1, ..., legendre[s+4] == 1
    # Sum of legendre[s..s+4] must be 5.
    
    # Initial window sum for s=1
    window_sum = sum(legendre[1:6])
    
    for s in range(1, 1005):
        if time.time() > deadline:
            raise TimeoutError("Computation timed out")
            
        if window_sum == 5:
            roots = [root[s + i] for i in range(5)]
            # Verify roots are non-zero and valid
            # root[r] is 0 if r is not a QR (or r=0).
            # Since window_sum==5, s..s+4 are QRs, so roots should be non-zero.
            return {'start': s, 'roots': roots}
            
        # Slide window
        if s < 1004:
            window_sum -= legendre[s]
            window_sum += legendre[s + 5]
            
    raise RuntimeError('No valid run found')
# EVOLVE-BLOCK-END
