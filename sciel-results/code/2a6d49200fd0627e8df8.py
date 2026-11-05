import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 57.0  # Safety margin for 60s limit
    
    # 1. Construct the [17,8,6] linear code (shortened extended QR code) - 256 words
    # Work in GF(2^8)
    def gf_mul(a, b, poly=0x1D):
        res = 0
        for i in range(8):
            if (b & 1):
                res ^= a
            b >>= 1
            if (a & 0x80):
                a = (a << 1) ^ poly
            else:
                a <<= 1
            a &= 0xFF
        return res

    def gf_pow(a, n, poly=0x1D):
        res = 1
        while n:
            if n & 1:
                res = gf_mul(res, a, poly)
            a = gf_mul(a, a, poly)
            n >>= 1
        return res

    # Find a primitive 17th root of unity in GF(2^8)
    alpha = None
    for a in range(1, 256):
        if gf_pow(a, 17) == 1 and a != 1:
            alpha = a
            break

    if alpha is None:
        # Fallback: seed code
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words

    # Quadratic residues mod 17
    Q = [1, 2, 4, 8, 9, 13, 15, 16]
    
    # Build generator polynomial g(x) = prod_{i in Q} (x + alpha^i)
    g_coeffs = [1]
    for i in Q:
        root = gf_pow(alpha, i)
        new_g = [0] * (len(g_coeffs) + 1)
        for j in range(len(g_coeffs)):
            new_g[j+1] ^= g_coeffs[j]
            new_g[j] ^= gf_mul(g_coeffs[j], root)
        g_coeffs = new_g
        
    # Generator matrix G (9x17) for [17,9,5] QR code
    G = np.zeros((9, 17), dtype=np.uint8)
    for k in range(9):
        for j in range(9):
            bit = g_coeffs[j]
            if bit:
                pos = (j + k) % 17
                G[k, pos] = 1

    # Generate all 512 codewords of [17,9,5]
    words_17 = []
    for u in range(512):
        word = np.zeros(17, dtype=np.uint8)
        for i in range(9):
            if (u >> i) & 1:
                word ^= G[i]
        words_17.append(word)
        
    # Extend to [18,9,6] with parity bit
    words_18 = []
    for w in words_17:
        weight = np.sum(w)
        parity = weight % 2
        w_ext = np.append(w, parity)
        words_18.append(w_ext)
        
    # Shorten to [17,8,6]: keep words with 0 at position 17
    words_17_short = []
    for w in words_18:
        if w[17] == 0:
            w_short = w[:17]
            val = 0
            for i in range(17):
                if w_short[i]:
                    val |= (1 << i)
            words_17_short.append(val)
    
    if len(words_17_short) != 256:
        # Fallback
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words

    # 2. Greedy extension: try to add more words beyond the 256 linear code
    words = list(words_17_short)
    
    # Precompute for fast distance checking
    # Use a set for O(1) membership, and track words as list
    word_set = set(words)
    
    def is_valid(candidate):
        """Check if candidate has distance >= 6 from all existing words."""
        for w in words:
            if (candidate ^ w).bit_count() < 6:
                return False
        return True
    
    # Try to add words via random sampling with multiple strategies
    random.seed(12345)
    
    # Strategy 1: Pure random sampling
    attempts = 0
    max_attempts = 500000
    while time.time() < deadline - 5.0 and attempts < max_attempts:
        candidate = random.getrandbits(17)
        if candidate not in word_set:
            if is_valid(candidate):
                words.append(candidate)
                word_set.add(candidate)
        attempts += 1
    
    # Strategy 2: Try structured candidates - words at distance 6 from existing words
    # For each existing word, generate candidates at exactly distance 6
    # This is more targeted but expensive. Try a limited subset.
    if time.time() < deadline - 3.0:
        # Try adding complements of existing words (distance 17 from each, so always valid if not already present)
        # Actually complement has distance 17 from original, but may conflict with others
        for w in list(words):
            comp = w ^ 0x1FFFF  # 17-bit complement
            if comp not in word_set:
                if is_valid(comp):
                    words.append(comp)
                    word_set.add(comp)
    
    # Strategy 3: Try words formed by flipping 6 bits from existing words
    if time.time() < deadline - 2.0:
        # Sample a subset of existing words and try small perturbations
        import itertools
        sample_words = random.sample(words, min(50, len(words)))
        for w in sample_words:
            if time.time() > deadline - 2.0:
                break
            # Try flipping exactly 6 random bits
            for _ in range(20):
                mask = 0
                bits = random.sample(range(17), 6)
                for b in bits:
                    mask |= (1 << b)
                candidate = w ^ mask
                if candidate not in word_set:
                    if is_valid(candidate):
                        words.append(candidate)
                        word_set.add(candidate)
    
    # Final cleanup: ensure all words are distinct and valid
    # The greedy process ensures validity, but let's verify count
    return words
