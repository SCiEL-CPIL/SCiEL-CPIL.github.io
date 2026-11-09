import time
import numpy as np

def construct():
    P = 1009
    M = 5
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Compute residues
    x = np.arange(1, P)
    r = (x * x) % P
    
    # Create boolean mask of QRs
    qr_mask = np.zeros(P, dtype=bool)
    qr_mask[r] = True
    
    # Create roots array (smallest root)
    roots = np.zeros(P, dtype=int)
    # We want the smallest x for each r.
    # Since x is increasing, we can just use np.minimum where valid?
    # Or just iterate?
    # Numpy advanced indexing: roots[r] = x doesn't handle duplicates well (last one wins).
    # We want first one (smallest x).
    # So we iterate or use a trick.
    # Since P is small, loop is fine.
    for i in range(len(x)):
        if roots[r[i]] == 0:
            roots[r[i]] = x[i]
            
    # Find runs of M True values in qr_mask[1:P]
    # qr_mask[0] is False (0 is not QR in our definition, or we don't care)
    # We need s, s+1, ..., s+4 to be True.
    # This is equivalent to a convolution or sliding sum.
    
    # Sliding window sum of M elements
    # We want sum == M
    # qr_mask[1:] is the relevant part
    # Actually, qr_mask[0] is index 0. We care about 1..P-1.
    
    # Use cumsum trick for sliding window
    # Or just simple loop, it's fast enough.
    
    for s in range(1, P - M + 1):
        if time.time() > deadline:
            break
        # Check if s..s+M-1 are all True
        # qr_mask[s:s+M].all()
        if np.all(qr_mask[s:s+M]):
            current_roots = [int(roots[s + i]) for i in range(M)]
            return {'start': s, 'roots': current_roots}
            
    return {'start': P - M, 'roots': [1, 1, 1, 1, 1]}
