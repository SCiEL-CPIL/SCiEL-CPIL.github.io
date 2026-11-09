import time
import random
import numpy as np

def construct():
    start_time = time.time()
    
    # 1. Attempt deterministic construction for n=49
    sol_49 = build_f49_coloring()
    if sol_49:
        return sol_49
        
    # 2. Fallback to seed 32 if something goes wrong
    return get_seed_solution()

def build_f49_coloring():
    # Field F_49 = F_7[x] / (x^2 - 3)
    # Element is (a, b) => a + b*alpha
    # alpha^2 = 3
    
    def add(e1, e2):
        return ((e1[0] + e2[0]) % 7, (e1[1] + e2[1]) % 7)
        
    def sub(e1, e2):
        return ((e1[0] - e2[0]) % 7, (e1[1] - e2[1]) % 7)
        
    def mul(e1, e2):
        a1, b1 = e1
        a2, b2 = e2
        # (a1 + b1*a)(a2 + b2*a) = a1a2 + a1b2a + b1a2a + b1b2a^2
        # a^2 = 3
        real = (a1*a2 + 3*b1*b2) % 7
        imag = (a1*b2 + b1*a2) % 7
        return (real, imag)
        
    # Find primitive element
    # Order is 48.
    # Try 2 (1,0) ?
    # Let's just iterate through all non-zero elements to find one of order 48.
    # There are 48 non-zero elements.
    
    elements = []
    for a in range(7):
        for b in range(7):
            if a == 0 and b == 0:
                continue
            elements.append((a,b))
            
    def pow(e, n):
        res = (1, 0)
        base = e
        while n > 0:
            if n % 2 == 1:
                res = mul(res, base)
            base = mul(base, base)
            n //= 2
        return res
        
    primitive = None
    for e in elements:
        # Check if order is 48
        # e^48 should be 1
        # e^d should not be 1 for d | 48, d < 48
        # Divisors of 48: 1, 2, 3, 4, 6, 8, 12, 16, 24
        if pow(e, 48) == (1,0):
            is_prim = True
            for d in [24, 16, 12, 8, 6, 4, 3, 2]:
                if pow(e, d) == (1,0):
                    is_prim = False
                    break
            if is_prim:
                primitive = e
                break
                
    if primitive is None:
        return None
        
    g = primitive
    # Generate subgroup H of index 4
    # H = <g^4>
    # Cosets: H, gH, g^2H, g^3H
    
    # Map element to color
    # Color 0: H
    # Color 1: gH
    # Color 2: g^2H
    # Color 3: g^3H
    
    # Precompute powers of g
    g_powers = [ (1,0) ]
    for i in range(1, 48):
        g_powers.append(mul(g_powers[-1], g))
        
    # Map element to discrete log
    # log_g[e] = i if e == g^i
    log_map = {}
    for i, e in enumerate(g_powers):
        log_map[e] = i
        
    def get_color(e):
        # e is non-zero
        i = log_map[e]
        return i % 4
        
    # Build coloring for n=49
    n = 49
    # Vertices are the elements in elements list
    # We need a consistent ordering. The list 'elements' is fine.
    
    colors = []
    # We need to map vertex index to field element
    # vertices = elements
    # for i < j:
    #   diff = sub(elements[i], elements[j])
    #   color = get_color(diff)
    
    for i in range(n):
        for j in range(i+1, n):
            diff = sub(elements[i], elements[j])
            # diff is non-zero because i != j
            c = get_color(diff)
            colors.append(c)
            
    # Verification (optional but safe)
    # Build matrix
    mat = np.zeros((n, n), dtype=np.int8)
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            mat[i,j] = colors[idx]
            mat[j,i] = colors[idx]
            idx += 1
            
    # Check for monochromatic triangles
    # This is O(n^3) = 49^3 ~ 117k operations. Very fast.
    for i in range(n):
        for j in range(i+1, n):
            c = mat[i,j]
            for k in range(j+1, n):
                if mat[i,k] == c and mat[j,k] == c:
                    return None # Invalid, fallback
                    
    return {'n': n, 'colors': colors}

def get_seed_solution():
    # Same as before
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    return {'n': 32, 'colors': colors}
