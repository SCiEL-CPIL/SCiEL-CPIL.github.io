import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 17
    MIN_DIST = 6
    
    def time_left():
        return deadline - time.time()

    # 1. Construct the [17,8,6] QR code correctly
    # Find the generator polynomial g(x) of degree 9
    # g(x) = (x+1) * m(x), where m(x) is the degree 8 factor of x^17-1 with m(1)=0
    
    def find_m_poly():
        # x^17 + 1
        target = (1 << 17) | 1
        # Iterate monic polynomials of degree 8
        for mask in range(256):
            m = (1 << 8) | mask
            # Check if m divides target
            rem = target
            for i in range(17 - 8, -1, -1):
                if (rem >> i) & 1:
                    rem ^= (m << i)
            if rem == 0:
                # Check m(1) == 0 (even number of 1s)
                if bin(m).count('1') % 2 == 0:
                    return m
        return None

    m_poly = find_m_poly()
    
    if m_poly is None:
        # Fallback to a simple greedy if construction fails
        return greedy_construct(deadline)
        
    # g(x) = (x+1) * m(x)
    # (x+1) is 0b11 (3)
    # Polynomial multiplication in GF(2)
    def poly_mul(a, b):
        res = 0
        while b:
            if b & 1:
                res ^= a
            a <<= 1
            b >>= 1
        return res

    g_poly = poly_mul(3, m_poly)
    
    # Generate all 256 codewords
    # c(x) = m(x) * g(x) mod (x^17 - 1)
    # Actually, for cyclic codes, c(x) = u(x) g(x) where deg(u) < 17-9=8
    # Since g(x) has degree 9, u(x) has degree < 8.
    # The code is linear, so we can just generate all multiples.
    
    code = []
    code_set = set()
    
    # Precompute g_poly for fast shifting?
    # We need to compute u(x) * g(x) mod (x^17+1)
    # Since deg(u) < 8 and deg(g) = 9, deg(u*g) < 17.
    # So no reduction mod x^17+1 is needed for the product itself, 
    # UNLESS we view the code as cyclic. 
    # Wait, the codewords are just the multiples.
    # c = u * g. Since deg(c) < 17, it fits in 17 bits.
    
    for u in range(256):
        c = poly_mul(u, g_poly)
        # Ensure c is within 17 bits (it should be, max deg 16)
        if c >= (1 << 17):
            c ^= (1 << 17) # Reduction mod x^17+1, though shouldn't happen
        code.append(c)
        code_set.add(c)
        
    # Verify base code validity (debug check, removed in final for speed if confident)
    # for i in range(len(code)):
    #    for j in range(i+1, len(code)):
    #       if (code[i] ^ code[j]).bit_count() < 6:
    #           pass # Should not happen

    # 2. Greedy Expansion
    rng = random.Random(42)
    
    # Candidate pool: Weights 6 to 12
    # To speed up, we can just random sample
    # Or systematic for low weights
    
    def can_add(cand):
        if cand in code_set:
            return False
        # Check distance
        # Optimization: Use numpy for vectorized distance check if code is large?
        # Code size is ~256+. 256 * 17 ops is small.
        for w in code:
            if (cand ^ w).bit_count() < MIN_DIST:
                return False
        return True

    # Systematic pass for weight 6, 7, 8
    # This helps fill small holes
    for w in range(6, 9):
        if time_left() < 10.0: break
        for combo in combinations(range(N), w):
            if time_left() < 5.0: break
            c = 0
            for p in combo:
                c |= (1 << p)
            if c not in code_set:
                if can_add(c):
                    code.append(c)
                    code_set.add(c)
                    
    # Randomized greedy for the rest
    # Focus on weights 8, 9, 10 which are optimal for packing
    for _ in range(200000):
        if time_left() < 5.0:
            break
        # Weight distribution: more 8/9
        r = rng.random()
        if r < 0.4:
            w = 8
        elif r < 0.8:
            w = 9
        else:
            w = rng.randint(6, 12)
            
        positions = rng.sample(range(N), w)
        c = 0
        for p in positions:
            c |= (1 << p)
            
        if c not in code_set:
            if can_add(c):
                code.append(c)
                code_set.add(c)
                
    # Local Search / Swap (Simple version)
    # Try to swap a word with two others? Too complex for remaining time.
    # Just do a final random polish
    for _ in range(50000):
        if time_left() < 2.0:
            break
        w = rng.randint(8, 10)
        positions = rng.sample(range(N), w)
        c = 0
        for p in positions:
            c |= (1 << p)
        if c not in code_set:
            if can_add(c):
                code.append(c)
                code_set.add(c)
                
    return code

def greedy_construct(deadline):
    # Fallback: Pure random greedy
    code = [0]
    code_set = {0}
    rng = random.Random(42)
    N = 17
    MIN_DIST = 6
    
    while time.time() < deadline - 1:
        w = rng.randint(6, 12)
        positions = rng.sample(range(N), w)
        c = 0
        for p in positions:
            c |= (1 << p)
        if c not in code_set:
            ok = True
            for w2 in code:
                if (c ^ w2).bit_count() < MIN_DIST:
                    ok = False
                    break
            if ok:
                code.append(c)
                code_set.add(c)
    return code
