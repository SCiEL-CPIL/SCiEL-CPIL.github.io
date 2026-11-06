import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    N = 17
    MIN_DIST = 6
    MASK = (1 << N) - 1
    
    # 1. Construct the 256-word linear code (Shortened [18,9,6] QR code)
    # GF(256) construction
    def gf_mul(a, b, poly=0x11D):
        res = 0
        while b:
            if b & 1:
                res ^= a
            b >>= 1
            a <<= 1
            if a & 0x100:
                a ^= poly
        return res & 0xFF

    def gf_pow(a, n, poly=0x11D):
        res = 1
        base = a
        while n:
            if n & 1:
                res = gf_mul(res, base, poly)
            base = gf_mul(base, base, poly)
            n >>= 1
        return res

    def get_primitive_17th_root():
        for candidate in range(2, 256):
            if gf_pow(candidate, 17) == 1 and candidate != 1:
                return candidate
        return None

    beta = get_primitive_17th_root()
    linear_code = set()
    
    if beta:
        # Construct generator polynomial
        poly_coeffs = [1]
        current_beta = beta
        for _ in range(8):
            new_coeffs = [0] * (len(poly_coeffs) + 1)
            for i in range(len(poly_coeffs)):
                new_coeffs[i+1] = poly_coeffs[i]
            for i in range(len(poly_coeffs)):
                new_coeffs[i] ^= gf_mul(poly_coeffs[i], current_beta)
            poly_coeffs = new_coeffs
            current_beta = gf_mul(current_beta, current_beta)
        
        g_bits = []
        for c in poly_coeffs:
            if c == 1:
                g_bits.append(1)
            elif c == 0:
                g_bits.append(0)
            else:
                g_bits.append(1)
        
        if len(g_bits) == 9 and g_bits[0] == 1:
            # Generate [17, 9] code
            codewords_17 = []
            for u in range(512):
                c = 0
                for k in range(9):
                    if (u >> k) & 1:
                        for j in range(9):
                            if g_bits[j]:
                                c ^= 1 << (j + k)
                codewords_17.append(c)
            
            # Extend to 18
            extended_codewords = []
            for c in codewords_17:
                parity = bin(c).count('1') % 2
                ec = c | (parity << 17)
                extended_codewords.append(ec)
            
            # Shorten (fix bit 17 to 0)
            shortened_codewords = []
            for ec in extended_codewords:
                if (ec >> 17) & 1 == 0:
                    shortened_codewords.append(ec & MASK)
            
            linear_code = set(shortened_codewords)
    
    # Fallback if linear code construction fails or is empty
    if not linear_code:
        linear_code = {0}
        # Greedy fill
        for i in range(1, 1 << N):
            if time.time() > deadline - 5.0:
                break
            valid = True
            for w in linear_code:
                if (i ^ w).bit_count() < MIN_DIST:
                    valid = False
                    break
            if valid:
                linear_code.add(i)

    current_code = set(linear_code)
    best_code = set(current_code)
    best_size = len(current_code)
    
    random.seed(42)
    
    # Helper to check if a candidate can be added
    def can_add(cand, code_set):
        for w in code_set:
            if (cand ^ w).bit_count() < MIN_DIST:
                return False
        return True

    # Local Search Loop
    # Strategy: 
    # 1. Try to add random words.
    # 2. If stuck, remove a random word to "loosen" the constraint, then try to add.
    # 3. If we find a larger set, keep it.
    # 4. If we shrink too much, revert to best.
    
    attempts_since_add = 0
    kick_threshold = 500  # After 500 failed adds, try a kick
    
    while time.time() < deadline - 2.0:
        # Try to add a random word
        cand = random.randint(0, MASK)
        
        if cand in current_code:
            continue
            
        if can_add(cand, current_code):
            current_code.add(cand)
            attempts_since_add = 0
            if len(current_code) > best_size:
                best_code = set(current_code)
                best_size = len(current_code)
                # Optional: print or log progress, but keep quiet for speed
        else:
            attempts_since_add += 1
            if attempts_since_add >= kick_threshold:
                # Kick: remove a random word to create space
                # We don't want to remove too many, so we remove 1 and try to add 2-3
                # If we fail to regain size, we might revert, but for now we just proceed
                # Actually, a better kick is to remove a word that is "redundant" or just random
                # Let's remove a random word
                to_remove = random.choice(list(current_code))
                current_code.remove(to_remove)
                attempts_since_add = 0
                kick_threshold = random.randint(100, 300) # Randomize next kick threshold
                
                # Try to add several words immediately after removal
                for _ in range(10):
                    if time.time() > deadline - 2.0:
                        break
                    new_cand = random.randint(0, MASK)
                    if new_cand not in current_code and can_add(new_cand, current_code):
                        current_code.add(new_cand)
                        if len(current_code) > best_size:
                            best_code = set(current_code)
                            best_size = len(current_code)
    
    return list(best_code)
