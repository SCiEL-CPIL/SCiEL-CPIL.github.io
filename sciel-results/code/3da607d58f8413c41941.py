import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 17
    MIN_DIST = 6
    
    def time_left():
        return deadline - time.time()
    
    # GF(2^8) arithmetic
    PRIM_POLY = 0x11D
    
    def gf_mul(a, b):
        res = 0
        for i in range(8):
            if (b >> i) & 1:
                res ^= a
            a <<= 1
            if a & 0x100:
                a ^= PRIM_POLY
        return res

    def gf_pow(a, p):
        res = 1
        for _ in range(p):
            res = gf_mul(res, a)
        return res

    # Find alpha (primitive element)
    alpha = None
    for a in range(2, 256):
        if gf_pow(a, 17) == 1 and gf_pow(a, 1) != 1:
            alpha = a
            break
            
    if alpha is None:
        return greedy_construct(deadline)

    # Quadratic residues mod 17
    Q = {1, 2, 4, 8, 9, 13, 15, 16}
    
    # Calculate generator polynomial g(x) = prod_{q in Q} (x + alpha^q)
    g_coeffs = [1]
    
    for q in Q:
        root = gf_pow(alpha, q)
        new_g = [0] * (len(g_coeffs) + 1)
        for i in range(len(g_coeffs)):
            new_g[i] ^= gf_mul(g_coeffs[i], root)
            new_g[i+1] ^= g_coeffs[i]
        g_coeffs = new_g
        
    # Convert to binary polynomial coefficients
    bin_g = [1 if c == 1 else 0 for c in g_coeffs]
    while len(bin_g) < 9:
        bin_g.append(0)
        
    # Convert bin_g to integer for fast shifting
    g_int = 0
    for i in range(9):
        if bin_g[i]:
            g_int |= (1 << i)
            
    # Generate all 512 codewords of the [17,9,6] code
    base_code = []
    
    for m in range(512):
        c = 0
        for i in range(9):
            if (m >> i) & 1:
                c ^= (g_int << i)
        
        # Reduce mod x^17 - 1
        # deg(m) <= 8, deg(g) = 8. deg(prod) <= 16.
        # So c is already in [0, 2^17).
        
        # Check even weight
        if c.bit_count() % 2 == 0:
            base_code.append(c)
            
    code = base_code[:]
    code_set = set(code)
    
    rng = random.Random(42)
    
    # Precompute all weight-7 to weight-10 candidates
    candidate_pool = []
    for w in range(7, 11):
        for combo in combinations(range(N), w):
            c = 0
            for p in combo:
                c |= (1 << p)
            if c not in code_set:
                candidate_pool.append(c)
    
    rng.shuffle(candidate_pool)
    
    # Vectorized greedy: add candidates that are compatible
    # Process in batches for speed
    batch_size = 2000
    added = 0
    for batch_start in range(0, len(candidate_pool), batch_size):
        if time_left() < 3.0:
            break
        batch = candidate_pool[batch_start:batch_start + batch_size]
        
        # Convert code to bit matrix for vectorized operations
        code_bits = np.array([[ (c >> i) & 1 for i in range(N)] for c in code], dtype=np.uint8)
        
        for cand in batch:
            if time_left() < 3.0:
                break
            if cand in code_set:
                continue
            cand_bits = np.array([(cand >> i) & 1 for i in range(N)], dtype=np.uint8)
            diffs = code_bits ^ cand_bits
            dists = diffs.sum(axis=1)
            if dists.min() >= MIN_DIST:
                code.append(cand)
                code_set.add(cand)
                added += 1
    
    # Second phase: random search with vectorized checks
    for _ in range(100):
        if time_left() < 3.0:
            break
        
        for _ in range(1000):
            if time_left() < 3.0:
                break
            w = rng.randint(6, 12)
            positions = rng.sample(range(N), w)
            candidate = 0
            for p in positions:
                candidate |= (1 << p)
            
            if candidate in code_set:
                continue
            
            code_bits = np.array([[ (c >> i) & 1 for i in range(N)] for c in code], dtype=np.uint8)
            cand_bits = np.array([(candidate >> i) & 1 for i in range(N)], dtype=np.uint8)
            diffs = code_bits ^ cand_bits
            dists = diffs.sum(axis=1)
            if dists.min() >= MIN_DIST:
                code.append(candidate)
                code_set.add(candidate)
    
    return code

def greedy_construct(deadline):
    code = [0]
    code_set = {0}
    rng = random.Random(42)
    while time.time() < deadline:
        w = rng.randint(3, 14)
        positions = rng.sample(range(17), w)
        c = 0
        for p in positions:
            c |= (1 << p)
        if c not in code_set:
            ok = True
            for w2 in code:
                if (c ^ w2).bit_count() < 6:
                    ok = False
                    break
            if ok:
                code.append(c)
                code_set.add(c)
    return code
