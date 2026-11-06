import time
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s limit, keep 5s margin

    # 1. GF(2^8) Arithmetic
    IRRED = 0b100011101
    
    def gf_mul(a, b):
        res = 0
        while b:
            if b & 1:
                res ^= a
            b >>= 1
            a <<= 1
            if a & 0x100:
                a ^= IRRED
        return res

    def gf_pow(a, n):
        res = 1
        while n:
            if n & 1:
                res = gf_mul(res, a)
            a = gf_mul(a, a)
            n >>= 1
        return res

    # 2. Find a primitive 17th root of unity
    alpha = None
    for i in range(2, 256):
        if gf_pow(i, 17) == 1:
            if i != 1:
                alpha = i
                break
    
    if alpha is None:
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words

    # 3. Compute Minimal Polynomial g(x)
    poly = [1]
    
    for j in range(8):
        root = gf_pow(alpha, 2**j)
        new_poly = [0] * (len(poly) + 1)
        for k in range(len(poly)):
            new_poly[k] ^= gf_mul(root, poly[k])
            new_poly[k+1] ^= poly[k]
        poly = new_poly
        
    # 4. Generate Codewords
    basis = []
    for i in range(9):
        vec = [0] * 17
        for j in range(9):
            vec[i+j] = poly[j]
        basis.append(vec)
        
    codewords = []
    for mask in range(512):
        word = [0] * 17
        for i in range(9):
            if mask & (1 << i):
                for j in range(17):
                    word[j] ^= basis[i][j]
        parity = sum(word) % 2
        val = 0
        for i in range(17):
            if word[i]:
                val |= (1 << i)
        if parity:
            val |= (1 << 17)
        codewords.append(val)
    
    # 5. Greedy refinement: try to add nonlinear codewords
    N = 18
    MIN_DIST = 6
    existing_set = set(codewords)
    
    def to_bits(val):
        return np.array([(val >> i) & 1 for i in range(N)], dtype=np.uint8)
    
    existing_bits = np.array([to_bits(w) for w in codewords])
    
    # Generate candidates in weight order
    by_weight = defaultdict(list)
    for w in range(1 << N):
        by_weight[w.bit_count()].append(w)
    
    max_check = 30000
    checked = 0
    
    for weight in range(N + 1):
        if time.time() > deadline or checked >= max_check:
            break
        for cand in by_weight[weight]:
            if time.time() > deadline or checked >= max_check:
                break
            checked += 1
            if cand in existing_set:
                continue
            
            cand_bits = to_bits(cand)
            xor = existing_bits ^ cand_bits[np.newaxis, :]
            min_d = xor.sum(axis=1).min()
            
            if min_d >= MIN_DIST:
                codewords.append(cand)
                existing_set.add(cand)
                existing_bits = np.vstack([existing_bits, cand_bits[np.newaxis, :]])
    
    return codewords
