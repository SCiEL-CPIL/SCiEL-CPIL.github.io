import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s margin for 90s limit
    
    N = 17
    MIN_DIST = 6
    
    # 1. Construct the [17, 9, 6] Quadratic Residue Code (256 words)
    residues = [0, 1, 2, 4, 8, 9, 13, 15, 16]
    
    gen_row = [0] * N
    for r in residues:
        gen_row[r] = 1
    
    # Pre-convert generator rows to integers
    g_ints = []
    for i in range(9):
        val = 0
        for j in range(N):
            if gen_row[(j - i) % N]:
                val |= (1 << j)
        g_ints.append(val)
    
    # Generate all 2^9 codewords
    words = []
    for mask in range(2**9):
        w = 0
        for i in range(9):
            if (mask >> i) & 1:
                w ^= g_ints[i]
        words.append(w)
    
    # Verify minimum distance for linear code
    min_w = N
    for w in words:
        if w == 0:
            continue
        weight = bin(w).count('1')
        if weight < min_w:
            min_w = weight
        if min_w < MIN_DIST:
            break
    
    if min_w < MIN_DIST:
        # Fallback to Reed-Muller based greedy
        rm_words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                rm_words.append(word)
        code = list(rm_words)
        random.seed(42)
        while time.time() < deadline:
            added = False
            for _ in range(500):
                cand = random.randint(0, 2**N - 1)
                valid = all((cand ^ w).bit_count() >= MIN_DIST for w in code)
                if valid:
                    code.append(cand)
                    added = True
                    break
            if not added:
                break
        return code
    
    # We have a valid 256-word linear code. Try to extend it nonlinearly.
    code = list(words)
    
    # Convert to numpy array for vectorized distance checks
    code_arr = np.array(code, dtype=np.int32)
    
    random.seed(12345)
    
    def check_candidate(cand, code_list, code_np):
        """Check if cand is valid (min distance >= MIN_DIST to all in code_list)."""
        xor_vals = code_np ^ cand
        # Compute popcount using bit operations
        total = np.zeros(len(code_np), dtype=np.int32)
        for b in range(N):
            total += ((xor_vals >> b) & 1).astype(np.int32)
        return total.min() >= MIN_DIST
    
    def check_candidate_excluding(cand, code_list, code_np, exclude_idx):
        """Check if cand is valid, excluding code_list[exclude_idx]."""
        xor_vals = code_np ^ cand
        total = np.zeros(len(code_np), dtype=np.int32)
        for b in range(N):
            total += ((xor_vals >> b) & 1).astype(np.int32)
        # Exclude the one at exclude_idx
        if len(total) > 1:
            # Temporarily set the excluded one to a large value
            saved = total[exclude_idx]
            total[exclude_idx] = N  # Max possible distance is N
            result = total.min() >= MIN_DIST
            total[exclude_idx] = saved
            return result
        else:
            return True
    
    # Phase 1: Greedy addition with random and local search candidates
    fail_count = 0
    max_fails = 5000
    
    while time.time() < deadline - 10.0 and fail_count < max_fails:
        added = False
        
        # Try a batch of random candidates
        batch_size = 5000
        candidates = np.random.randint(0, 2**N, size=batch_size)
        
        for cand_int in candidates:
            if time.time() > deadline - 10.0:
                break
            if check_candidate(int(cand_int), code, code_arr):
                code.append(int(cand_int))
                code_arr = np.append(code_arr, int(cand_int))
                added = True
                fail_count = 0
                break
        
        # If random didn't work, try local search (flip bits of existing words)
        if not added:
            local_added = False
            for _ in range(200):
                if time.time() > deadline - 10.0:
                    break
                base_idx = random.randint(0, len(code) - 1)
                base = code[base_idx]
                num_flips = random.randint(1, 4)
                flips = random.sample(range(N), num_flips)
                cand = base
                for pos in flips:
                    cand ^= (1 << pos)
                if check_candidate(cand, code, code_arr):
                    code.append(cand)
                    code_arr = np.append(code_arr, cand)
                    local_added = True
                    added = True
                    fail_count = 0
                    break
            if not local_added:
                fail_count += 1
    
    # Phase 2: Local search via swaps to escape local optima
    # Try to replace one word with two new words (or just move a word to a better position)
    # This allows escaping the linear code structure
    
    swap_attempts = 0
    max_swap_attempts = 10000
    
    while time.time() < deadline - 5.0 and swap_attempts < max_swap_attempts:
        swap_attempts += 1
        
        # Pick a random word to potentially replace
        idx = random.randint(0, len(code) - 1)
        old_word = code[idx]
        
        # Try to find a new word that can replace old_word and allow adding another word
        # Strategy: Try to find a word that is valid when old_word is removed
        # Then see if we can add TWO words instead of one
        
        # First, try to find a replacement word
        found_replacement = False
        for _ in range(100):
            if time.time() > deadline - 5.0:
                break
            
            # Generate candidate by flipping bits of old_word or random
            if random.random() < 0.5:
                # Flip some bits of old_word
                num_flips = random.randint(1, 5)
                flips = random.sample(range(N), num_flips)
                cand = old_word
                for pos in flips:
                    cand ^= (1 << pos)
            else:
                # Random candidate
                cand = random.randint(0, 2**N - 1)
            
            if cand == old_word:
                continue
            
            # Check if cand is valid when old_word is removed
            if check_candidate_excluding(cand, code, code_arr, idx):
                # Replace old_word with cand
                code[idx] = cand
                code_arr[idx] = cand
                found_replacement = True
                break
        
        if not found_replacement:
            continue
        
        # Now try to add a new word to the code (which now has same size but different structure)
        added_new = False
        for _ in range(200):
            if time.time() > deadline - 5.0:
                break
            
            if random.random() < 0.5:
                base_idx = random.randint(0, len(code) - 1)
                base = code[base_idx]
                num_flips = random.randint(1, 4)
                flips = random.sample(range(N), num_flips)
                cand = base
                for pos in flips:
                    cand ^= (1 << pos)
            else:
                cand = random.randint(0, 2**N - 1)
            
            if check_candidate(cand, code, code_arr):
                code.append(cand)
                code_arr = np.append(code_arr, cand)
                added_new = True
                break
    
    return code
