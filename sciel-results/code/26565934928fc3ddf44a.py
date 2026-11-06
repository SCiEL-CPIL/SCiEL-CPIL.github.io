"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def _evaluate(coeffs):
    num = 0
    den = 1
    for i, c in enumerate(coeffs):
        if c == 0:
            continue
        d = i + 1
        g = gcd(den, d)
        den = den // g * d
        num = num * (d // g) + c * (den // d)
    g = gcd(abs(num), den)
    if g:
        num //= g
        den //= g
    return num, den

def _abs_val(num, den):
    return abs(num) / den

def construct():
    deadline = time.time() + 50
    
    best_val = float('inf')
    best_coeffs = [0] * 28 + [1, -1]  # default: 1/29 - 1/30 = 1/870
    
    def update(coeffs):
        nonlocal best_val, best_coeffs
        num, den = _evaluate(coeffs)
        if num == 0:
            return
        val = _abs_val(num, den)
        if val < best_val:
            best_val = val
            best_coeffs = list(coeffs)
    
    # Pairs
    for i in range(30):
        for j in range(i + 1, 30):
            c = [0] * 30
            c[i] = 1
            c[j] = -1
            update(c)
    
    # Triples
    for i in range(30):
        for j in range(i + 1, 30):
            for k in range(j + 1, 30):
                if time.time() > deadline - 15:
                    break
                for s in [(1,-1,-1),(1,1,-1),(1,-1,1)]:
                    c = [0] * 30
                    c[i] = s[0]
                    c[j] = s[1]
                    c[k] = s[2]
                    update(c)
    
    # Random local search
    random.seed(42)
    for restart in range(500):
        if time.time() > deadline - 5:
            break
        coeffs = [0] * 30
        n = random.randint(3, 15)
        pos = random.sample(range(30), n)
        for p in pos:
            coeffs[p] = random.choice([-1, 1])
        
        for step in range(80):
            if time.time() > deadline - 5:
                break
            num, den = _evaluate(coeffs)
            if num == 0:
                idx = random.randint(0, 29)
                coeffs[idx] = random.choice([-1, 0, 1])
                continue
            cur = _abs_val(num, den)
            if cur < best_val:
                best_val = cur
                best_coeffs = list(coeffs)
            
            idx = random.randint(0, 29)
            old = coeffs[idx]
            new = random.choice([-1, 0, 1])
            if new == old:
                continue
            coeffs[idx] = new
            n2, d2 = _evaluate(coeffs)
            if n2 == 0:
                coeffs[idx] = old
            else:
                v2 = _abs_val(n2, d2)
                if v2 >= cur:
                    coeffs[idx] = old
    
    return best_coeffs
# EVOLVE-BLOCK-END
