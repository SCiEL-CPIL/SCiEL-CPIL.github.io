import numpy as np

def construct():
    p = 1009
    
    # Build the QR indicator array using vectorized NumPy operations
    # r is 1..p-1, compute r^2 mod p
    r = np.arange(1, p, dtype=np.int64)
    residues = (r * r) % p
    
    # Create boolean array: is[i] = True if i is a non-zero QR
    is_qr = np.zeros(p, dtype=bool)
    is_qr[residues] = True
    is_qr[0] = False  # exclude zero
    
    # Store roots: for each residue value, keep the smallest root
    # Initialize with large value, then fill
    roots = np.full(p, p, dtype=np.int64)
    # For each r in 1..p-1, if roots[res] is still unfilled, set it
    # Vectorized: we can do this by iterating or using a trick
    # Since p is small (1009), a simple loop is fine but let's use argsort trick
    # Actually, let's just do it simply since p is small
    for ri in range(1, p):
        res = (ri * ri) % p
        if roots[res] == p:
            roots[res] = ri
    
    # Find all positions s where is_qr[s], is_qr[s+1], ..., is_qr[s+4] are all True
    # This is a sliding window of 5 all-True values
    # We can compute: valid = is_qr[:-4] & is_qr[1:-3] & is_qr[2:-2] & is_qr[3:-1] & is_qr[4:]
    valid = is_qr[:-4] & is_qr[1:-3] & is_qr[2:-2] & is_qr[3:-1] & is_qr[4:]
    
    # valid[i] corresponds to start position i+1 (since is_qr is 0-indexed)
    # We need s in range 1..1004, which means index 0..1003 in valid
    # But valid has length p-4 = 1005, indices 0..1004
    # s = i + 1, so i ranges from 0 to 1003 for s in 1..1004
    # valid[1004] would correspond to s=1005 which is out of range
    
    # Find the first True in valid[0:1004] (indices 0 to 1003)
    if not np.any(valid[:1004]):
        raise RuntimeError("No solution found")
    
    idx = np.argmax(valid[:1004])  # first occurrence of True
    s = idx + 1
    
    # Extract roots for s, s+1, s+2, s+3, s+4
    roots_list = [int(roots[s + i]) for i in range(5)]
    
    return {
        'start': int(s),
        'roots': roots_list
    }
