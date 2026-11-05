import numpy as np
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80 seconds deadline
    
    # Generator matrix for the [17,8,6] code
    G = np.array([
        [1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1],
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1]
    ], dtype=np.uint8)
    
    # Generate all 256 codewords
    code_words = []
    for i in range(256):
        coeffs = [(i >> j) & 1 for j in range(8)]
        word = np.sum(np.array(coeffs)[:, None] * G, axis=0) % 2
        val = 0
        for bit in word:
            val = (val << 1) | bit
        code_words.append(val)
        
    code_set = set(code_words)
    code_list = list(code_set)
    
    # Helper to convert int to bits array
    def int_to_bits(val, n=17):
        return np.array([(val >> i) & 1 for i in range(n)], dtype=np.uint8)
        
    # Verify minimum distance of the linear code
    min_dist = 17
    for i in range(len(code_list)):
        bits_i = int_to_bits(code_list[i])
        for j in range(i+1, len(code_list)):
            bits_j = int_to_bits(code_list[j])
            dist = np.sum(bits_i ^ bits_j)
            if dist < min_dist:
                min_dist = dist
                if min_dist < 6:
                    # Fallback to seed code
                    words = []
                    for slope in range(16):
                        for intercept in range(2):
                            word = 0
                            for point in range(16):
                                bit = ((slope & point).bit_count() + intercept) % 2
                                word |= bit << point
                            words.append(word)
                    return words
                    
    # If we are here, the linear code is valid with min distance >= 6
    code_matrix = np.array([int_to_bits(w) for w in code_list])
    
    # Generate candidates at distance exactly 6 from the zero word
    cand_bits = []
    for comb in combinations(range(17), 6):
        bits = np.zeros(17, dtype=np.uint8)
        for pos in comb:
            bits[pos] = 1
        cand_bits.append(bits)
    cand_matrix = np.array(cand_bits)
    
    # Try to add candidates
    for _ in range(2): # Try to add at most 2 more words
        if time.time() > deadline:
            break
            
        found = False
        # Compute distances between all candidates and all codewords
        diff = cand_matrix[:, None, :] ^ code_matrix[None, :, :]
        dists = np.sum(diff, axis=2)
        min_dists = np.min(dists, axis=1)
        valid_indices = np.where(min_dists >= 6)[0]
        
        if len(valid_indices) > 0:
            idx = valid_indices[0]
            new_word_bits = cand_matrix[idx]
            new_val = 0
            for bit in new_word_bits:
                new_val = (new_val << 1) | bit
            if new_val not in code_set:
                code_set.add(new_val)
                code_list.append(new_val)
                code_matrix = np.vstack([code_matrix, new_word_bits[None, :]])
                found = True
        
        if not found:
            break
            
    return code_list
