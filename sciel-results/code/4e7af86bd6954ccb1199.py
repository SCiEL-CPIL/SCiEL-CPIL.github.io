import time

def construct():
    p = 1009
    deadline = time.time() + 55  # Safety margin
    
    # Precompute QRs and roots
    # roots[i] = smallest x such that x*x % p == i, or 0 if not QR
    roots = [0] * p
    qr_bits = 0
    
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            qr_bits |= (1 << r)
            if roots[r] == 0:
                roots[r] = x
                
    mask = 0b11111
    
    for s in range(1, 1005):
        if time.time() > deadline:
            # Fallback: return best found so far? 
            # Since we scan from 1, if we haven't found one, 
            # and we hit deadline, we can't just return nothing.
            # But for p=1009, this loop is instant.
            break
            
        if (qr_bits >> s) & mask == mask:
            # Verify roots exist (should be true if bit is set)
            result_roots = [roots[s + i] for i in range(5)]
            # Check if any root is 0 (should not happen)
            if all(r != 0 for r in result_roots):
                return {'start': s, 'roots': result_roots}
                
    # If not found (should not happen), raise or return last valid?
    # The task implies a solution exists.
    raise RuntimeError('No valid run found')
