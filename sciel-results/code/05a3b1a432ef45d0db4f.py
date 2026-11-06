import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 90s limit
    
    N = 17
    MIN_DIST = 6
    
    # Step 1: Construct the [17, 9, 6] QR code (256 words)
    # Quadratic residues modulo 17: {1, 2, 4, 8, 9, 13, 15, 16}
    # Include 0 to get dimension 9
    residues = [0, 1, 2, 4, 8, 9, 13, 15, 16]
    
    gen_row = [0] * N
    for r in residues:
        gen_row[r] = 1
    
    # Generate circulant matrix
    circ = np.zeros((N, N), dtype=np.int8)
    for i in range(N):
        for j in range(N):
            circ[i, j] = gen_row[(j - i) % N]
    
    G = circ[:9, :]
    
    # Pre-convert rows to integers
    g_ints = []
    for i in range(9):
        val = 0
        for j in range(N):
            if G[i, j] == 1:
                val |= (1 << j)
        g_ints.append(val)
    
    # Generate all 2^9 = 256 codewords
    words = []
    for mask in range(256):
        w = 0
        for i in range(9):
            if (mask >> i) & 1:
                w ^= g_ints[i]
        words.append(w)
    
    # Verify minimum distance
    min_w = N
    for w in words:
        if w == 0:
            continue
        weight = bin(w).count('1')
        if weight < min_w:
            min_w = weight
    
    if min_w < MIN_DIST:
        # Fallback: use greedy from scratch
        words = [0]
    
    # Step 2: Try to add more words using local search
    # Convert to set for fast lookup, and keep list for iteration
    code_set = set(words)
    code_list = list(words)
    
    # Precompute all weights for quick access
    def popcount(x):
        return bin(x).count('1')
    
    def min_dist_to_code(x, code_list):
        md = N
        for w in code_list:
            d = (x ^ w).bit_count()
            if d < md:
                md = d
                if md < MIN_DIST:
                    break
        return md
    
    # Greedy addition phase
    random.seed(12345)
    
    def try_add_greedy(code_list, code_set, remaining_time_func):
        """Try to add words greedily"""
        added = 0
        attempts = 0
        max_attempts = 500000
        
        while time.time() < remaining_time_func() and attempts < max_attempts:
            attempts += 1
            cand = random.getrandbits(N)
            
            # Quick check: is cand already in code?
            if cand in code_set:
                continue
            
            # Check distance to all codewords
            valid = True
            for w in code_list:
                d = (cand ^ w).bit_count()
                if d < MIN_DIST:
                    valid = False
                    break
            
            if valid:
                code_list.append(cand)
                code_set.add(cand)
                added += 1
        
        return added
    
    # Run greedy addition
    remaining = deadline - time.time()
    if remaining > 5.0:
        try_add_greedy(code_list, code_set, lambda: deadline - 10.0)
    
    # Step 3: Local search - try to replace one word with two words
    # This is the key to escaping the linear code limit
    def local_search_replace_one_with_two(code_list, code_set, time_budget):
        """Try to replace one codeword with two codewords that are farther apart"""
        improved = True
        while improved and time.time() < time_budget:
            improved = False
            n = len(code_list)
            
            for i in range(n):
                if time.time() > time_budget:
                    break
                
                w_i = code_list[i]
                
                # Try removing w_i and seeing if we can add two new words
                # that cover more space
                # First, find the "closest neighbor" of w_i
                closest_dist = N
                closest_idx = -1
                for j in range(n):
                    if j == i:
                        continue
                    d = (w_i ^ code_list[j]).bit_count()
                    if d < closest_dist:
                        closest_dist = d
                        closest_idx = j
                
                if closest_dist >= MIN_DIST + 2:
                    # w_i is well-separated, try to split
                    # Generate candidates near w_i
                    for _ in range(20):
                        # Perturb w_i
                        perturb_mask = random.getrandbits(N)
                        # Keep weight change small
                        if popcount(perturb_mask) > 4:
                            continue
                        
                        cand1 = w_i ^ perturb_mask
                        if cand1 in code_set or cand1 == 0:
                            continue
                        
                        # Check if cand1 is valid (distance >= MIN_DIST to all others except w_i)
                        valid1 = True
                        for j in range(n):
                            if j == i:
                                continue
                            d = (cand1 ^ code_list[j]).bit_count()
                            if d < MIN_DIST:
                                valid1 = False
                                break
                        
                        if not valid1:
                            continue
                        
                        # Now try to find a second word cand2
                        # such that both cand1 and cand2 are valid and far from each other
                        for _ in range(10):
                            perturb_mask2 = random.getrandbits(N)
                            if popcount(perturb_mask2) > 4:
                                continue
                            cand2 = w_i ^ perturb_mask2
                            
                            if cand2 == cand1 or cand2 in code_set:
                                continue
                            
                            # Check cand2 distance to all (except w_i)
                            valid2 = True
                            for j in range(n):
                                if j == i:
                                    continue
                                d = (cand2 ^ code_list[j]).bit_count()
                                if d < MIN_DIST:
                                    valid2 = False
                                    break
                            
                            if not valid2:
                                continue
                            
                            # Check distance between cand1 and cand2
                            d12 = (cand1 ^ cand2).bit_count()
                            if d12 < MIN_DIST:
                                continue
                            
                            # Also check that removing w_i and adding cand1, cand2
                            # doesn't violate anything (already checked above)
                            # Replace w_i with cand1 and cand2
                            code_list[i] = cand1
                            code_list.append(cand2)
                            code_set.discard(w_i)
                            code_set.add(cand1)
                            code_set.add(cand2)
                            improved = True
                            break
                        
                        if improved:
                            break
                
                if improved:
                    break
        
        return len(code_list)
    
    # Run local search
    remaining = deadline - time.time()
    if remaining > 10.0:
        local_search_replace_one_with_two(code_list, code_set, deadline - 5.0)
    
    # Step 4: Another greedy pass after local search
    remaining = deadline - time.time()
    if remaining > 5.0:
        try_add_greedy(code_list, code_set, lambda: deadline - 2.0)
    
    # Step 5: More local search if time permits
    remaining = deadline - time.time()
    if remaining > 5.0:
        local_search_replace_one_with_two(code_list, code_set, deadline - 2.0)
    
    # Ensure all words are valid and distinct
    final_set = set()
    final_list = []
    for w in code_list:
        if w not in final_set:
            # Verify this word has distance >= MIN_DIST to all previously added
            valid = True
            for v in final_list:
                if (w ^ v).bit_count() < MIN_DIST:
                    valid = False
                    break
            if valid:
                final_set.add(w)
                final_list.append(w)
    
    return final_list
