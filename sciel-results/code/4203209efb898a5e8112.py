import time

# EVOLVE-BLOCK-START

def construct():
        p = 1009
        m = 5
        
        # is_qr[r] is True if r is a non-zero quadratic residue mod p
        is_qr = [False] * (p + 1)
        root_map = {}
        
        for x in range(1, p):
            r = (x * x) % p
            if r == 0:
                continue
            is_qr[r] = True
            if r not in root_map:
                root_map[r] = x
        
        # Sliding window to find first s such that s..s+m-1 are all QRs
        # Count of QRs in current window
        count = 0
        for i in range(m):
            if is_qr[i + 1]: # indices 1..m
                count += 1
        
        if count == m:
            s = 1
            roots = [root_map[s + i] for i in range(m)]
            return {'start': s, 'roots': roots}
            
        for s in range(2, p - m + 1):
            # Remove s-1, add s+m-1
            if is_qr[s - 1]:
                count -= 1
            if is_qr[s + m - 1]:
                count += 1
                
            if count == m:
                roots = [root_map[s + i] for i in range(m)]
                return {'start': s, 'roots': roots}
                
        raise RuntimeError("No run found")

    # EVOLVE-BLOCK-END
