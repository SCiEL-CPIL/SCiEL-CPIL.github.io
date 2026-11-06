import numpy as np
import time

def construct():
    N = 17
    MIN_DIST = 6
    start_time = time.time()
    DEADLINE = 85
    
    # Use a known good seed: the [17,7,6] QR subcode (128 words)
    # Construct [17,8,5] QR code
    g = [1, 1, 0, 1, 1, 1, 0, 0, 1, 1]
    
    def gf2_mod(p, m):
        r = list(p)
        while len(r) >= len(m):
            if r[-1]:
                s = len(r) - len(m)
                for i in range(len(m)):
                    r[s+i] ^= m[i]
                while len(r) > 1 and not r[-1]:
                    r.pop()
        return r
    
    qr = []
    for msg in range(256):
        m = [(msg>>i)&1 for i in range(8)]
        sh = m + [0]*9
        rem = gf2_mod(sh, g)
        c = [0]*17
        for i in range(9): c[i] ^= rem[i]
        for i in range(8): c[9+i] = m[i]
        qr.append(sum(c[i]<<i for i in range(17)))
    
    C = [w for w in qr if bin(w).count('1')%2==0]
    
    all_words = np.arange(2**N, dtype=np.uint32)
    bits = np.zeros((2**N, N), dtype=np.uint8)
    for i in range(N): bits[:,i] = (all_words>>i)&1
    
    valid = np.ones(2**N, dtype=bool)
    for w in C:
        wb = np.array([(w>>i)&1 for i in range(N)], dtype=np.uint8)
        valid &= (np.sum(bits^wb, axis=1) >= MIN_DIST)
    
    for idx in range(2**N):
        if time.time()-start_time > DEADLINE: break
        if not valid[idx]: continue
        C.append(int(all_words[idx]))
        wb = np.array([(all_words[idx]>>i)&1 for i in range(N)], dtype=np.uint8)
        valid &= (np.sum(bits^wb, axis=1) >= MIN_DIST)
    
    return C
