import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 57.0  # Safety margin for 90s limit (generous)

    # 1. Construct the [17,8,6] linear code (shortened extended QR code)
    # This code has 2^8 = 256 codewords.
    
    # Work in GF(2^8)
    # Primitive polynomial: x^8 + x^4 + x^3 + x^2 + 1 (0x1D)
    
    def gf_mul(a, b, poly=0x1D):
        # Multiplication in GF(2^8)
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
        # Fallback: Return seed code if construction fails (should not happen)
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words

    # 2. Construct the generator polynomial for the [17,9,5] QR code
    # The roots are alpha^i for i in the set of quadratic residues mod 17.
    # Quadratic residues mod 17: 1, 2, 4, 8, 9, 13, 15, 16
    Q = [1, 2, 4, 8, 9, 13, 15, 16]
    
    # g(x) = prod_{i in Q} (x - alpha^i)
    # In GF(2), subtraction is addition, so (x - a) = (x + a)
    
    g_coeffs = [1]  # Start with 1
    
    for i in Q:
        root = gf_pow(alpha, i)
        # Multiply g(x) by (x + root)
        new_g = [0] * (len(g_coeffs) + 1)
        for j in range(len(g_coeffs)):
            new_g[j+1] ^= g_coeffs[j]
            new_g[j] ^= gf_mul(g_coeffs[j], root)
        g_coeffs = new_g
        
    # g_coeffs has 9 elements (degree 8)
    # The generator matrix G for the [17,9,5] code is 9x17.
    # Row k is x^k * g(x) mod (x^17 - 1)
    
    G = np.zeros((9, 17), dtype=np.uint8)
    
    for k in range(9):
        for j in range(9):
            bit = g_coeffs[j]
            if bit:
                pos = (j + k) % 17
                G[k, pos] = 1

    # 3. Generate all 512 codewords of the [17,9,5] code
    words_17 = []
    for u in range(512):
        word = np.zeros(17, dtype=np.uint8)
        for i in range(9):
            if (u >> i) & 1:
                word ^= G[i]
        words_17.append(word)
        
    # 4. Extend to [18,9,6]
    # Add a parity bit to make the weight even
    words_18 = []
    for w in words_17:
        weight = np.sum(w)
        parity = weight % 2
        w_ext = np.append(w, parity)
        words_18.append(w_ext)
        
    # 5. Shorten to [17,8,6]
    # Shorten at the last coordinate (index 17).
    # Keep only codewords with 0 at position 17.
    # Remove position 17.
    
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
        # Fallback: Return seed code if construction fails
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words

    # 6. Attempt to extend the code beyond 256 words using a greedy search with precomputed popcount
    # The current best known is 258. We try to add more words.
    
    # Precompute popcount for all 17-bit numbers? No, that's 2^17 = 131072 entries.
    # Or we can just use bit_count() which is fast in Python 3.10+.
    # Let's use a lookup table for 8-bit and 9-bit halves to be safe and fast.
    
    # Actually, let's just use a simple greedy approach with a bit_count check.
    # To speed it up, we can precompute the distances from each existing codeword to all possible new codewords?
    # That's 256 * 131072 operations. Too slow.
    
    # Better approach: Iterate through all 131072 possible words.
    # For each word, check if it's valid against all current codewords.
    # If valid, add it.
    # This is O(N * K) where N is number of candidates (131072) and K is current code size.
    # Initially K=256. 131072 * 256 ~ 33M operations. Might be too slow in Python.
    
    # Optimization: Use a set of current codewords.
    # For a new candidate c, we need dist(c, c_i) >= 6 for all c_i in code.
    
    # Let's try a smarter greedy:
    # 1. Start with the 256 linear codewords.
    # 2. Sort all other 131072 - 256 candidates by some heuristic (e.g., weight, or distance to centroid).
    # 3. Try to add them one by one.
    
    # Even faster:
    # Precompute a "conflict" map?
    
    # Let's try the simple approach but with a time check.
    # If it's too slow, we just return the 256 words.
    
    code = set(words_17_short)
    codewords_list = list(code)
    
    # Precompute popcount for all 17-bit values?
    # popcount_table = [i.bit_count() for i in range(1 << 17)]
    # This takes ~0.1s and 1MB memory. Worth it.
    
    popcount_table = [0] * (1 << 17)
    for i in range(1, 1 << 17):
        popcount_table[i] = popcount_table[i >> 1] + (i & 1)
    
    # Helper function to check if a word is valid
    def is_valid(word):
        # Check against all existing codewords
        for c in codewords_list:
            if popcount_table[word ^ c] < 6:
                return False
        return True

    # Candidates: all 17-bit words not in the code
    # We can iterate through all 131072 words.
    # To make it faster, we can skip words that are too close to the first few codewords.
    
    # Let's iterate through all possible words.
    # We can optimize by checking against a subset of codewords first?
    # No, we must check all.
    
    # However, 131072 * 256 checks is ~33M XORs and table lookups.
    # In Python, this might take 1-2 seconds.
    # As we add more words, it gets slower.
    # If we add 10 words, it's 131072 * 266 ~ 35M.
    
    # Let's try it.
    
    # To speed up, we can use numpy?
    # No, Python loop with list of ints is okay for 33M ops if optimized.
    # Actually, 33M Python loop iterations might take ~1-3 seconds.
    # With time limit 57s, we have plenty of room.
    
    # Let's try to add words.
    # We iterate through all 131072 words.
    # We can randomize the order or use a specific order.
    # Random order might find valid words faster?
    # Or just sequential.
    
    # Let's use a random permutation of all words to avoid bias.
    # But generating a permutation of 131072 items takes time/memory.
    # Let's just iterate sequentially.
    
    # Actually, a better strategy:
    # The linear code is a subspace.
    # The cosets of the linear code partition the space.
    # There are 131072 / 256 = 512 cosets.
    # Each coset has 256 words.
    # Within a coset, all words have the same distance to the linear code?
    # No, but the distances between words in the same coset are the same as distances in the linear code.
    # Wait, if c1, c2 are in the same coset, c1 = u + v1, c2 = u + v2 where v1, v2 in C.
    # dist(c1, c2) = dist(v1, v2).
    # Since min dist of C is 6, any two distinct words in the same coset have dist >= 6.
    # So we can pick at most 1 word from each coset?
    # No, we can pick multiple words from a coset as long as they are mutually distant.
    # But since they are all mutually distant (min dist 6), we can pick ANY subset of a coset.
    # However, we must also ensure distance to words in OTHER cosets is >= 6.
    
    # So the problem reduces to:
    # Pick a subset of cosets, and for each selected coset, pick a subset of its 256 words.
    # This is complex.
    
    # Let's stick to the greedy search.
    
    # Optimization:
    # Instead of checking all 256 codewords for each candidate,
    # we can maintain a list of "forbidden" distances?
    
    # Let's just run the loop.
    
    # To make it faster, we can break early if we find a conflict.
    
    # We will iterate through all 131072 words.
    # We will skip the ones already in the code.
    
    # We can speed this up by using a list of codewords and checking.
    
    # Let's try to add as many as possible.
    
    # We'll use a list for codewords to allow fast iteration.
    # We'll use a set for O(1) membership check.
    
    # We'll iterate through all 131072 words.
    # For each word, if it's not in the set, check validity.
    
    # To speed up validity check, we can precompute the "ball" of radius 5 around each codeword?
    # No, that's 256 * sum_{i=0}^5 C(17, i) ~ 256 * 10000 ~ 2.5M entries.
    # We can create a boolean array of size 131072 marking "forbidden" words.
    # Initially, mark all words within distance 5 of any codeword as forbidden.
    # Then, for any new candidate, just check if it's forbidden.
    # When we add a new codeword, we update the forbidden array.
    
    # This is a great optimization!
    
    # forbidden = bytearray(1 << 17)
    # For each codeword c, for each word w with dist(c, w) <= 5, set forbidden[w] = 1.
    
    # How many words are within distance 5 of a codeword?
    # Sum_{i=0}^5 C(17, i) = 1 + 17 + 136 + 680 + 2380 + 6188 = 9402.
    # So for each codeword, we mark ~9400 words as forbidden.
    # 256 codewords * 9400 = 2.4M operations to initialize.
    # This is very fast.
    
    # Then, to find a new word, we just scan for any word where forbidden[w] == 0.
    # When we add it, we mark its neighbors as forbidden.
    
    # This reduces the complexity significantly.
    
    forbidden = bytearray(1 << 17)
    
    # Precompute neighbors within distance 5 for all 17-bit words?
    # Or just generate them on the fly?
    # Generating neighbors of a word c:
    # All words w such that popcount(w ^ c) <= 5.
    # We can iterate through all masks m with popcount(m) <= 5 and set w = c ^ m.
    
    # There are 9402 such masks.
    # Precomputing the list of masks is good.
    
    masks = []
    for m in range(1 << 17):
        if popcount_table[m] <= 5:
            masks.append(m)
            
    # Initialize forbidden array
    for c in codewords_list:
        for m in masks:
            forbidden[c ^ m] = 1
            
    # Now, find new words.
    # We can scan through all 131072 words and pick the first one not forbidden.
    # Or we can maintain a list of available words?
    # Scanning is O(2^17) per new word.
    # If we add 10 words, that's 10 * 131072 = 1.3M operations. Very fast.
    
    # Let's do this.
    
    added = 0
    while time.time() < deadline - 1.0:
        # Find next available word
        new_word = -1
        for w in range(1 << 17):
            if not forbidden[w]:
                new_word = w
                break
        
        if new_word == -1:
            break
            
        # Add new_word
        code.add(new_word)
        codewords_list.append(new_word)
        added += 1
        
        # Mark its neighbors as forbidden
        for m in masks:
            forbidden[new_word ^ m] = 1
            
    # Return the list of codewords
    return list(code)
