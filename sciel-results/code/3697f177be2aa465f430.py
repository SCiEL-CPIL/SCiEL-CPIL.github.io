import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    N = 17
    D = 6
    R = D - 1  # Radius 5
    
    # 1. Generate the [17, 9, 6] shortened QR code (256 words)
    # Using the same GF(2^8) logic as the seed, but optimized
    PRIMITIVE_POLY = 0x11D
    
    def gf_mul(a, b):
        res = 0
        while b:
            if b & 1:
                res ^= a
            b >>= 1
            a <<= 1
            if a & 0x100:
                a ^= PRIMITIVE_POLY
        return res

    def gf_pow(a, e):
        res = 1
        while e:
            if e & 1:
                res = gf_mul(res, a)
            a = gf_mul(a, a)
            e >>= 1
        return res

    # Find primitive 17th root of unity
    alpha = 2
    if gf_pow(alpha, 17) != 1:
        alpha = gf_pow(alpha, 15)
        if gf_pow(alpha, 17) != 1:
            for x in range(1, 256):
                if gf_pow(x, 17) == 1 and x != 1:
                    alpha = x
                    break

    # Compute minimal polynomial of alpha^3
    roots = []
    for k in range(8):
        exp = (3 * pow(2, k, 17)) % 17
        roots.append(gf_pow(alpha, exp))
        
    p = [1]
    for r in roots:
        new_p = [0] * (len(p) + 1)
        for i in range(len(p)):
            new_p[i + 1] ^= p[i]
            new_p[i] ^= gf_mul(p[i], r)
        p = new_p
        
    g_coeffs = [0 if c == 0 else 1 for c in p]
    
    linear_code = []
    for a_val in range(512):
        c_bits = [0] * 17
        for i in range(9):
            if (a_val >> i) & 1:
                for j in range(9):
                    c_bits[i + j] ^= g_coeffs[j]
        word_int = 0
        for i in range(17):
            if c_bits[i]:
                word_int |= (1 << i)
        if word_int.bit_count() % 2 == 0:
            linear_code.append(word_int)
            
    # 2. Prepare for greedy extension
    # Precompute Hamming ball offsets of radius 5
    # This is a list of bit-masks (integers) to XOR with the center word
    offsets = [0]
    # Single bit flips
    for i in range(N):
        offsets.append(1 << i)
    # Double bit flips
    for i in range(N):
        for j in range(i+1, N):
            offsets.append((1 << i) | (1 << j))
    # Triple bit flips
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                offsets.append((1 << i) | (1 << j) | (1 << k))
    # Quad bit flips
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                for l in range(k+1, N):
                    offsets.append((1 << i) | (1 << j) | (1 << k) | (1 << l))
    # Quint bit flips
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                for l in range(k+1, N):
                    for m in range(l+1, N):
                        offsets.append((1 << i) | (1 << j) | (1 << k) | (1 << l) | (1 << m))
                        
    # Convert offsets to a numpy array for faster indexing? 
    # Actually, we just need to mark `forbidden[word ^ offset] = True`
    # We can use a boolean numpy array of size 2^17
    forbidden = np.zeros(2**N, dtype=bool)
    
    # Mark initial linear code
    for w in linear_code:
        for off in offsets:
            forbidden[w ^ off] = True
            
    current_code = list(linear_code)
    best_code = list(current_code)
    best_len = len(best_code)
    
    # Greedy Extension Loop
    # We try multiple random restarts/orders to find a better packing
    # Since the space is small, we can be aggressive
    
    for restart in range(1000): # Limit restarts, but check time
        if time.time() > deadline:
            break
            
        # Reset forbidden based on current best code? 
        # Or just continue from current? 
        # If we just continue, we might get stuck. 
        # Let's try to add more words to the current set.
        
        # Strategy: 
        # 1. Find all valid candidates.
        # 2. If none, break inner loop.
        # 3. Pick one (randomly or heuristically).
        # 4. Add it.
        
        # To allow "swaps" or better global search, we'd need more complex logic.
        # Simple greedy is often enough for small gains.
        # Let's try a simple greedy pass on the current `current_code`.
        
        # Re-initialize forbidden for this pass? No, it's cumulative.
        # But if we want to try a different order, we should probably start fresh from `best_code`
        # or just keep adding.
        
        # Let's try to add as many as possible in one go.
        added_in_pass = 0
        while True:
            if time.time() > deadline:
                break
                
            # Find candidates not in forbidden
            # This is O(2^17) which is ~131k checks. 
            # We can do this with numpy:
            valid = np.where(~forbidden)[0]
            if len(valid) == 0:
                break
                
            # Pick a candidate
            # Random pick to break symmetry
            idx = random.randint(0, len(valid) - 1)
            new_word = int(valid[idx])
            
            # Add to code
            current_code.append(new_word)
            added_in_pass += 1
            
            # Update forbidden
            for off in offsets:
                forbidden[new_word ^ off] = True
                
        if len(current_code) > best_len:
            best_code = list(current_code)
            best_len = len(best_code)
            
        # If we couldn't add anything, we are stuck. 
        # We could try to remove a word and try again, but that's complex.
        # For now, if no progress, we can try a fresh start from best_code with a different random seed
        # or just stop if we've done many restarts.
        
        if added_in_pass == 0:
            # Try to perturb: remove a random word from best_code and restart greedy?
            # This is a simple local search step.
            if len(best_code) > 10 and random.random() < 0.1:
                # Remove a random word
                rem_idx = random.randint(0, len(best_code) - 1)
                current_code = [w for i, w in enumerate(best_code) if i != rem_idx]
                
                # Rebuild forbidden
                forbidden = np.zeros(2**N, dtype=bool)
                for w in current_code:
                    for off in offsets:
                        forbidden[w ^ off] = True
            else:
                # No progress and no perturbation, maybe try a different random seed order
                # Or just continue to next restart which does nothing if added=0
                # To avoid infinite loop, we can break if we do many empty restarts
                pass
        else:
            # We made progress, keep going
            pass

    return best_code
