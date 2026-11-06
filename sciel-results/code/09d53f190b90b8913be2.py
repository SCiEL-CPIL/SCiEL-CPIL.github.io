"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 17
    MIN_DIST = 6
    
    # 1. Construct the 256-word [17,8,6] shortened QR code
    # Generator polynomial for [18,9,6] QR code: x^9 + x^7 + x^3 + x^2 + x + 1
    # Coefficients: 1, 0, 0, 1, 0, 0, 1, 1, 1, 1 (x^9 to x^0)
    # Let's represent the generator matrix rows.
    # g(x) = x^9 + x^7 + x^3 + x^2 + x + 1
    # Powers: 9, 7, 3, 2, 1, 0
    
    g_powers = [9, 7, 3, 2, 1, 0]
    
    # Build generator matrix G (9x18)
    G = np.zeros((9, 18), dtype=np.uint8)
    for i in range(9):
        for p in g_powers:
            pos = (i + p) % 18
            G[i, pos] = 1
            
    # Generate all 512 codewords of the [18,9,6] code
    # Codeword is sum of selected rows of G
    # We only need those with bit 0 = 0 (shortening condition)
    # Then remove bit 0
    
    qr_words = []
    # Iterate over all 9-bit messages
    # To be faster, we can just iterate and check bit 0
    # But 512 is small.
    for msg in range(512):
        # Calculate codeword
        # Efficiently: use numpy
        # bits = (G.T @ (msg bits)) % 2
        # msg_bits = np.array([(msg >> i) & 1 for i in range(9)])
        # cw = (G.T @ msg_bits) % 2
        
        # Manual loop for speed in Python if numpy overhead is high? 
        # 512 * 9 * 18 is small.
        msg_bits = np.array([(msg >> i) & 1 for i in range(9)], dtype=np.uint8)
        cw = (G.T @ msg_bits) % 2
        
        # Check shortening condition: bit 0 must be 0
        if cw[0] == 0:
            # Remove bit 0, keep bits 1..17
            # Shift right by 1
            word_int = 0
            for j in range(1, 18):
                if cw[j]:
                    word_int |= (1 << (j-1))
            qr_words.append(word_int)
            
    # qr_words should have 256 elements
    # Verify uniqueness
    qr_words = list(set(qr_words))
    
    if len(qr_words) < 256:
        # Fallback if generation failed (should not happen)
        # Use random greedy
        qr_words = []
        for _ in range(256):
             # This fallback is bad, but prevents crash
             pass
        # Actually, let's just use the seed if this fails?
        # No, the QR code construction is standard.
        
    current_words = qr_words[:]
    
    def get_bits(words):
        n_words = len(words)
        bits = np.zeros((n_words, N), dtype=np.uint8)
        for i, w in enumerate(words):
            for j in range(N):
                if w & (1 << j):
                    bits[i, j] = 1
        return bits

    current_bits = get_bits(current_words)
    best_words = current_words[:]
    best_size = len(current_words)
    
    def int_to_bits(w):
        b = np.zeros(N, dtype=np.uint8)
        for j in range(N):
            if w & (1 << j):
                b[j] = 1
        return b

    def min_dist_to_code(candidate_int, bits):
        cand_bits = int_to_bits(candidate_int)
        xor = bits ^ cand_bits
        dists = xor.sum(axis=1)
        return int(np.min(dists))
        
    def is_valid(candidate_int, bits):
        return min_dist_to_code(candidate_int, bits) >= MIN_DIST

    def generate_candidate():
        # Weight 8 or 9 is good for distance 6
        w = random.randint(7, 10)
        positions = random.sample(range(N), w)
        word = 0
        for p in positions:
            word |= (1 << p)
        return word

    # Local Search Loop
    # Strategy: 
    # 1. Try to add a random word.
    # 2. If it fails, identify conflicting words.
    # 3. If conflict size is small (1 or 2), try to swap them out and add the new word + potentially more.
    # 4. Accept only if size increases.
    
    attempts = 0
    max_attempts = 50000 # Safety
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        
        candidate = generate_candidate()
        
        # Check validity
        cand_bits = int_to_bits(candidate)
        xor = current_bits ^ cand_bits
        dists = xor.sum(axis=1)
        
        if np.min(dists) >= MIN_DIST:
            # Valid, add it
            current_words.append(candidate)
            current_bits = np.vstack([current_bits, cand_bits])
            
            if len(current_words) > best_size:
                best_size = len(current_words)
                best_words = current_words[:]
        else:
            # Find conflicting indices
            bad_indices = np.where(dists < MIN_DIST)[0]
            
            # If conflict is small, try a swap
            if len(bad_indices) == 1:
                idx = bad_indices[0]
                old_word = current_words[idx]
                
                # Remove old_word, add candidate
                # New size is same. 
                # Try to add ANOTHER word to the new set to increase size.
                
                # Temporarily update
                temp_words = current_words[:]
                temp_bits = current_bits.copy()
                temp_words[idx] = candidate
                temp_bits[idx] = cand_bits
                
                # Try to add a new random word
                new_candidate = generate_candidate()
                new_cand_bits = int_to_bits(new_candidate)
                
                xor2 = temp_bits ^ new_cand_bits
                dists2 = xor2.sum(axis=1)
                
                if np.min(dists2) >= MIN_DIST:
                    # Success! We swapped 1 for 1, and added 1 more. Net +1.
                    current_words = temp_words
                    current_bits = temp_bits
                    current_words.append(new_candidate)
                    current_bits = np.vstack([current_bits, new_cand_bits])
                    
                    if len(current_words) > best_size:
                        best_size = len(current_words)
                        best_words = current_words[:]
                        
            elif len(bad_indices) == 2:
                # Try to swap 2 for 1 + 2 new?
                # Or just 2 for 1 (size -1, bad)
                # Or 2 for 2 (size 0, neutral)
                # Let's try 2 for 1 (candidate) + 1 new?
                # Remove bad_indices[0], bad_indices[1]
                # Add candidate
                # Try to add new_candidate
                
                idx1, idx2 = bad_indices[0], bad_indices[1]
                
                temp_words = current_words[:]
                temp_bits = current_bits.copy()
                
                # Remove idx2 first to keep indices stable? 
                # Easier: create a new list excluding these two
                new_words_list = [w for i, w in enumerate(current_words) if i not in bad_indices]
                new_bits = np.vstack([current_bits[i] for i in range(len(current_words)) if i not in bad_indices])
                
                # Add candidate
                new_words_list.append(candidate)
                new_bits = np.vstack([new_bits, cand_bits])
                
                # Try to add new_candidate
                new_candidate = generate_candidate()
                new_cand_bits = int_to_bits(new_candidate)
                
                xor2 = new_bits ^ new_cand_bits
                dists2 = xor2.sum(axis=1)
                
                if np.min(dists2) >= MIN_DIST:
                    # Size: -2 + 1 + 1 = 0. No gain.
                    # Try to add ANOTHER new_candidate2?
                    new_candidate2 = generate_candidate()
                    new_cand_bits2 = int_to_bits(new_candidate2)
                    
                    xor3 = new_bits ^ new_cand_bits2
                    dists3 = xor3.sum(axis=1)
                    
                    # Check dist between new_candidate and new_candidate2
                    dist_12 = int_to_bits(new_candidate) ^ new_cand_bits2
                    d_12 = dist_12.sum()
                    
                    if np.min(dists3) >= MIN_DIST and d_12 >= MIN_DIST:
                         # Size: -2 + 1 + 1 + 1 = +1
                         current_words = new_words_list
                         current_bits = new_bits
                         current_words.append(new_candidate)
                         current_bits = np.vstack([current_bits, new_cand_bits])
                         current_words.append(new_candidate2)
                         current_bits = np.vstack([current_bits, new_cand_bits2])
                         
                         if len(current_words) > best_size:
                             best_size = len(current_words)
                             best_words = current_words[:]
                # If no gain, discard (revert implicitly)
            # If conflict > 2, skip (too costly to repair)

    return best_words

# EVOLVE-BLOCK-END
