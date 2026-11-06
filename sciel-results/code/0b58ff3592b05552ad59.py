import time

# EVOLVE-BLOCK-START
def construct():
    p = 1009
    deadline = time.time() + 59.0  # Safety margin
    
    # Precompute quadratic residues and their smallest roots
    # root[r] = x where x^2 mod p == r, 1 <= x <= p-1
    root = [0] * p
    bitset = 0
    
    for x in range(1, p):
        r = (x * x) % p
        if root[r] == 0:
            root[r] = x
        bitset |= (1 << r)
        
    mask = 0b11111  # 5 consecutive bits set
    
    # Find the smallest s such that s, s+1, ..., s+4 are all non-zero QRs
    # Use a sliding window count for efficiency
    # Count consecutive QRs starting at each position
    # is_qr[i] = 1 if i is a non-zero QR, else 0
    is_qr = [0] * (p + 5)  # extra space to avoid bounds issues
    for r in range(1, p):
        if (bitset >> r) & 1:
            is_qr[r] = 1
    
    # Sliding window: count of 1s in window of size 5
    # window_sum for positions s..s+4
    window_sum = sum(is_qr[1:6])  # s=1: positions 1,2,3,4,5
    for s in range(1, 1005):
        if time.time() > deadline:
            raise TimeoutError("Computation timed out")
        
        if window_sum == 5:
            roots = [root[s + i] for i in range(5)]
            return {'start': s, 'roots': roots}
        
        # Slide window: remove is_qr[s], add is_qr[s+5]
        if s < 1004:
            window_sum -= is_qr[s]
            window_sum += is_qr[s + 5]
            
    raise RuntimeError('No valid run found')
# EVOLVE-BLOCK-END
