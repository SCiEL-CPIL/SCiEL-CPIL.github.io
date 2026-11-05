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

    # 2. Vectorized Nonlinear Extension
    words = np.array(words_17_short, dtype=np.uint32)
    word_set = set(words_17_short)
    
    # Precompute a popcount lookup table for 8-bit values
    popcount_table = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)
    
    def fast_popcount(x):
        """Fast popcount for 17-bit integers using lookup table"""
        return popcount_table[x & 0xFF] + popcount_table[(x >> 8) & 0xFF]
    
    def check_candidate(c):
        """Check if candidate c has distance >= 6 from all words. Returns True if valid."""
        if c in word_set:
            return False
        for w in words:
            if fast_popcount(c ^ w) < 6:
                return False
        return True
    
    def check_batch_fast(candidates):
        """
        candidates: numpy array of uint32
        returns: list of candidates that are valid (distance >= 6 from all words)
        """
        if len(candidates) == 0:
            return []
        
        valid_candidates = []
        for c in candidates:
            if c in word_set:
                continue
            is_valid = True
            for w in words:
                if fast_popcount(c ^ w) < 6:
                    is_valid = False
                    break
            if is_valid:
                valid_candidates.append(c)
        return valid_candidates

    random.seed(12345)
    
    # Strategy 1: Weight-based search with structured candidates
    # Generate all words of specific weights that might be valid
    import itertools
    
    def get_words_of_weight(w):
        masks = []
        for comb in itertools.combinations(range(17), w):
            mask = 0
            for i in comb:
                mask |= (1 << i)
            masks.append(mask)
        return np.array(masks, dtype=np.uint32)
    
    # Try weights 6 through 10
    for w in [6, 7, 8, 9, 10]:
        if time.time() > deadline - 5.0:
            break
            
        candidates = get_words_of_weight(w)
        # Shuffle to avoid bias
        np.random.shuffle(candidates)
        
        # Check in chunks
        chunk_size = 1000
        for i in range(0, len(candidates), chunk_size):
            if time.time() > deadline - 5.0:
                break
            chunk = candidates[i:i+chunk_size]
            valids = check_batch_fast(chunk)
            for v in valids:
                words = np.append(words, v)
                word_set.add(v)
                
    # Strategy 2: Local search - try flipping bits of existing codewords
    # This is more structured than random search
    if time.time() < deadline - 5.0:
        # For each existing word, try all single-bit and double-bit flips
        base_words = list(words)
        n_base = len(base_words)
        
        for idx in range(n_base):
            if time.time() > deadline - 5.0:
                break
            base = base_words[idx]
            
            # Try single bit flips
            for bit_pos in range(17):
                candidate = base ^ (1 << bit_pos)
                if check_candidate(candidate):
                    words = np.append(words, candidate)
                    word_set.add(candidate)
            
            # Try double bit flips
            for bit1 in range(17):
                if time.time() > deadline - 5.0:
                    break
                for bit2 in range(bit1+1, 17):
                    candidate = base ^ (1 << bit1) ^ (1 << bit2)
                    if check_candidate(candidate):
                        words = np.append(words, candidate)
                        word_set.add(candidate)
    
    # Strategy 3: Random search with vectorized check (if time remains)
    if time.time() < deadline - 5.0:
        batch_size = 5000
        while time.time() < deadline - 2.0:
            candidates = np.array([random.getrandbits(17) for _ in range(batch_size)], dtype=np.uint32)
            valids = check_batch_fast(candidates)
            for v in valids:
                words = np.append(words, v)
                word_set.add(v)
                
            if len(words) >= 258:
                break

    # Final return as list of integers
    return list(words)
