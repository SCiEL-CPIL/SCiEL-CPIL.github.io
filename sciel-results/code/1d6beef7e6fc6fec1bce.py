import numpy as np
import time
import random

def construct():
    N = 18
    MIN_DIST = 6
    start_time = time.time()
    DEADLINE = 85  # Safety margin before 90s kill

    def time_remaining():
        return DEADLINE - (time.time() - start_time)

    # 1. Try Extended QR Code Construction
    try:
        # GF(256) arithmetic
        # Primitive polynomial: x^8 + x^4 + x^3 + x^2 + 1
        def gf_mult(a, b):
            res = 0
            while b:
                if b & 1:
                    res ^= a
                b >>= 1
                a <<= 1
                if a & 0x100:
                    a ^= 0x11D # x^8 + x^4 + x^3 + x^2 + 1
            return res

        # Find primitive element gamma of GF(256)
        # 0x11D is primitive
        gamma = 2 # alpha = 2 is usually primitive for this poly?
        # Let's verify order of 2
        # 2^8 = 256 = 1 + 16 + 8 + 4 = 29? No.
        # In GF(256), 2 is just the element '2'.
        # Let's assume gamma = 2 is primitive.
        
        # alpha = gamma^15
        alpha = 1
        for _ in range(15):
            alpha = gf_mult(alpha, gamma)
        
        # Verify alpha^17 == 1
        check = 1
        for _ in range(17):
            check = gf_mult(check, alpha)
        if check != 1:
            raise Exception("alpha is not 17th root")
            
        # Compute minimal polynomial of alpha
        # Roots: alpha^(2^k) for k=0..7
        roots = []
        r = alpha
        for _ in range(8):
            roots.append(r)
            r = gf_mult(r, r) # r = r^2
            
        # g(x) = prod (x - root_i)
        # Start with 1
        g = [1] # coefficient of x^0
        for root in roots:
            # Multiply g(x) by (x - root) = x + root (in char 2)
            new_g = [0] * (len(g) + 1)
            for i in range(len(g)):
                new_g[i+1] ^= g[i]       # x * g[i]
                new_g[i] ^= gf_mult(g[i], root) # -root * g[i]
            g = new_g
            
        # g is now [c_0, c_1, ..., c_8]
        # g(x) = c_0 + c_1 x + ... + c_8 x^8
        # We want generator matrix rows to be shifts of this polynomial.
        # Standard cyclic code generator matrix:
        # Row i has g(x) shifted by i positions.
        # Let's represent rows as integers of length 17.
        
        row_ints = []
        for i in range(9):
            val = 0
            # Coefficients of x^i * g(x)
            # g(x) = sum_{k=0}^8 c_k x^k
            # x^i g(x) = sum_{k=0}^8 c_k x^{k+i}
            for k in range(9):
                if g[k] != 0: # g[k] is 0 or 1
                    pos = k + i
                    if pos < 17:
                        val |= (1 << pos)
            
            # Parity bit
            parity = bin(val).count('1') % 2
            if parity:
                val |= (1 << 17)
            row_ints.append(val)
        
        # Generate all 512 codewords
        codes = []
        for m in range(512):
            c = 0
            for j in range(9):
                if (m >> j) & 1:
                    c ^= row_ints[j]
            codes.append(c)
        
        # Verify minimum distance
        min_d = N + 1
        for i in range(len(codes)):
            for j in range(i + 1, len(codes)):
                d = (codes[i] ^ codes[j]).bit_count()
                if d < min_d:
                    min_d = d
                    if min_d < MIN_DIST:
                        break
            if min_d < MIN_DIST:
                break
        
        if min_d >= MIN_DIST:
            return codes
    except Exception:
        pass

    # Fallback: Greedy with local search improvement
    def greedy_construct():
        codes = [0]
        candidates = list(range(2**N))
        random.shuffle(candidates)
        
        for cand in candidates:
            if time_remaining() < 5:
                break
            ok = True
            for c in codes:
                if (cand ^ c).bit_count() < MIN_DIST:
                    ok = False
                    break
            if ok:
                codes.append(cand)
        
        return codes

    def local_search_improve(codes, max_time=30):
        """Try to improve code size by local search: remove a word, try to add two."""
        codes = list(codes)
        deadline = time.time() + max_time
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            n = len(codes)
            for i in range(n):
                if time.time() > deadline:
                    break
                codes_without_i = codes[:i] + codes[i+1:]
                removed = codes[i]
                
                added = []
                candidates = []
                for delta in range(1, 100):
                    c1 = removed ^ (1 << delta)
                    if c1 < 2**N:
                        candidates.append(c1)
                    c2 = removed ^ ((1 << delta) - 1)
                    if c2 < 2**N:
                        candidates.append(c2)
                
                random.shuffle(candidates)
                
                for cand in candidates[:200]:
                    if len(added) >= 2:
                        break
                    ok = True
                    for c in codes_without_i + added:
                        if (cand ^ c).bit_count() < MIN_DIST:
                            ok = False
                            break
                    if ok:
                        added.append(cand)
                
                if len(added) >= 2:
                    codes = codes_without_i + added
                    improved = True
                    break
        
        return codes

    # Run greedy
    codes = greedy_construct()
    
    if time_remaining() > 10:
        codes = local_search_improve(codes, max_time=max(5, time_remaining() - 10))
    
    return codes
