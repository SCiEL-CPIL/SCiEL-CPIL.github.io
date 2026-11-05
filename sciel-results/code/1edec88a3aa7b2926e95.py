import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # 1. Construct the [17,8,6] linear code (256 words)
    # Re-using the logic from the current program but encapsulated
    
    def gf_mul(a, b, poly=0x1d):
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

    def gf_pow(a, n, poly=0x1d):
        res = 1
        while n:
            if n & 1:
                res = gf_mul(res, a, poly)
            a = gf_mul(a, a, poly)
            n >>= 1
        return res

    alpha = None
    for a in range(1, 256):
        if gf_pow(a, 17) == 1 and a != 1:
            alpha = a
            break
            
    if alpha is None:
        return [0]

    Q = [1, 2, 4, 8, 9, 13, 15, 16]
    g_coeffs = [1]
    for i in Q:
        root = gf_pow(alpha, i)
        new_g = [0] * (len(g_coeffs) + 1)
        for j in range(len(g_coeffs)):
            new_g[j+1] ^= g_coeffs[j]
            new_g[j] ^= gf_mul(g_coeffs[j], root)
        g_coeffs = new_g
        
    G = np.zeros((9, 17), dtype=np.uint8)
    for k in range(9):
        for j in range(9):
            if g_coeffs[j]:
                G[k, (j + k) % 17] = 1
                
    words_17 = []
    for u in range(512):
        word = np.zeros(17, dtype=np.uint8)
        for i in range(9):
            if (u >> i) & 1:
                word ^= G[i]
        words_17.append(word)
        
    words_18 = []
    for w in words_17:
        parity = np.sum(w) % 2
        words_18.append(np.append(w, parity))
        
    initial_code = []
    for w in words_18:
        if w[17] == 0:
            val = 0
            for i in range(17):
                if w[i]:
                    val |= (1 << i)
            initial_code.append(val)
            
    if len(initial_code) != 256:
        return initial_code

    current_code = list(initial_code)
    
    # Helper to check if a word can be added
    def can_add(word, code_list):
        for w in code_list:
            if bin(word ^ w).count('1') < 6:
                return False
        return True

    # Helper to find a random word that fits
    def find_random_word(code_list, attempts=100):
        for _ in range(attempts):
            # Generate random word
            w = random.getrandbits(17)
            if can_add(w, code_list):
                return w
        return None

    # Helper to find a pair of words that fit
    def find_random_pair(code_list, attempts=100):
        for _ in range(attempts):
            w1 = random.getrandbits(17)
            if not can_add(w1, code_list):
                continue
            # Temporarily add w1 to check w2? Or just check w2 against code_list + w1
            w2 = random.getrandbits(17)
            if bin(w1 ^ w2).count('1') < 6:
                continue
            if not can_add(w2, code_list):
                continue
            return (w1, w2)
        return None

    # Local Search: Try to replace 1 word with 2
    # We iterate through the code, pick a word, remove it, and try to find 2 new ones.
    
    # To speed up, we can precompute distances or use a better random strategy.
    # Random might be slow. Let's try to bias the random generation towards "nearby" words or just use a fixed seed for reproducibility if needed, but random is fine.
    
    # Optimization: can_add is O(N). N=256. 100 attempts * 256 checks * 2 words = ~50k ops per try.
    # 55 seconds is plenty for many tries.
    
    # Let's run a loop that tries to improve the size.
    # We can also try to "kick" the solution by removing a random word and not replacing it immediately,
    # then trying to fill it back up with 2 words.
    
    best_code = current_code[:]
    best_size = len(current_code)
    
    # We will use a simple hill climbing / random restarts approach.
    # Since we start at a local optimum (maximal linear code), we need to break it.
    # Strategy:
    # 1. Pick a random word w_idx from current_code.
    # 2. Remove it.
    # 3. Try to find 2 words that fit in the reduced set.
    # 4. If found, add them.
    # 5. If not, try to find 1 word that fits (restore size) or revert.
    
    # To make it faster, we can limit the number of attempts per "kick".
    
    random.seed(42) # For reproducibility if needed, though not strictly required
    
    iterations = 0
    while time.time() < deadline - 5.0:
        iterations += 1
        # Pick a random word to remove
        idx = random.randint(0, len(current_code) - 1)
        removed_word = current_code.pop(idx)
        
        reduced_code = current_code
        
        # Try to find 2 new words
        pair = find_random_pair(reduced_code, attempts=50)
        
        if pair:
            current_code.append(pair[0])
            current_code.append(pair[1])
            # If we improved, update best
            if len(current_code) > best_size:
                best_size = len(current_code)
                best_code = current_code[:]
            # Continue searching from this new state
        else:
            # Try to find 1 new word to restore size
            single = find_random_word(reduced_code, attempts=20)
            if single:
                current_code.append(single)
            else:
                # Revert
                current_code.append(removed_word)
                
        # If we are stuck at the same size for a while, we might want to restart or change strategy.
        # But for now, just keep trying.
        
        # Periodic check
        if iterations % 100 == 0 and time.time() > deadline - 10.0:
             break

    return best_code
