import numpy as np
from itertools import combinations
import time

def construct():
    start_time = time.time()
    
    # GF(2^8) with primitive polynomial x^8 + x^4 + x^3 + x^2 + 1
    GF_POLY = 0x11D
    
    def gf_mul(a, b):
        result = 0
        for _ in range(8):
            if b & 1:
                result ^= a
            b >>= 1
            a <<= 1
            if a & 0x100:
                a ^= GF_POLY
        return result
    
    def gf_pow(base, exp):
        result = 1
        for _ in range(exp):
            result = gf_mul(result, base)
        return result
    
    # Find element of order 17
    alpha = None
    for a in range(2, 256):
        if gf_pow(a, 17) == 1:
            # Check it's not of smaller order (17 is prime, so order is 1 or 17)
            if a != 1:
                alpha = a
                break
    assert alpha is not None
    
    # Non-residues mod 17
    residues = set(pow(i, 2, 17) for i in range(1, 17))
    non_residues = sorted(set(range(1, 17)) - residues)
    # non_residues = [3, 5, 6, 7, 10, 11, 12, 14]
    
    # Compute generator polynomial g(x) = prod_{i in non_residues} (x - alpha^i)
    # In GF(2), subtraction = addition
    g = [1]  # polynomial coefficients, g[j] = coeff of x^j
    for i in non_residues:
        ai = gf_pow(alpha, i)
        new_g = [0] * (len(g) + 1)
        for j, coeff in enumerate(g):
            new_g[j+1] ^= coeff  # x * coeff * x^j
            new_g[j] ^= gf_mul(coeff, ai)  # ai * coeff * x^j
        g = new_g
    
    # Verify g is binary (all coeffs are 0 or 1)
    for coeff in g:
        if coeff not in (0, 1):
            # This shouldn't happen if non_residues form a complete cyclotomic coset
            pass
    
    # g should have degree 8, so len(g) = 9
    assert len(g) == 9, f"Generator polynomial degree {len(g)-1}"
    
    # Generate all 512 codewords
    words_17 = set()
    for u in range(512):
        word = 0
        for i in range(9):
            if (u >> i) & 1:
                for j in range(9):
                    if g[j]:
                        word ^= 1 << ((i + j) % 17)
        words_17.add(word)
    
    assert len(words_17) == 512
    
    # Extend to [18,9,6]: add parity bit at position 17
    words_18 = []
    for w in words_17:
        parity = w.bit_count() & 1
        words_18.append(w | (parity << 17))
    
    # Shorten: keep codewords with bit 17 = 0, remove bit 17
    words = []
    for w in words_18:
        if (w >> 17) & 1 == 0:
            words.append(w & 0x1FFFF)
    
    words = list(set(words))
    assert len(words) == 256, f"Got {len(words)} words after shortening"
    
    # Verify minimum distance >= 6
    # (Optional, for debugging)
    
    # Hill climbing: add more words
    def build_forbidden(word_list):
        forbidden = np.zeros(131072, dtype=bool)
        for w in word_list:
            forbidden[w] = True
            for dist in range(1, 6):
                for delta in deltas_by_dist[dist]:
                    forbidden[w ^ delta] = True
        return forbidden
    
    # Precompute deltas
    deltas_by_dist = {}
    for dist in range(1, 6):
        deltas = []
        for positions in combinations(range(17), dist):
            delta = 0
            for pos in positions:
                delta |= (1 << pos)
            deltas.append(delta)
        deltas_by_dist[dist] = deltas
    
    # Greedy addition
    while time.time() - start_time < 55:  # leave 5s buffer
        forbidden = build_forbidden(words)
        valid = np.where(~forbidden)[0]
        if len(valid) == 0:
            break
        # Pick the first valid candidate (or could be smarter)
        new_w = int(valid[0])
        words.append(new_w)
    
    return words
