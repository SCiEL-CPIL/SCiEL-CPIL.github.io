import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin before 90s kill
    
    N = 17
    MIN_DIST = 6
    
    # 1. Construct the 256-word linear code: [17, 9, 6] shortened QR code
    # GF(256) with primitive polynomial x^8 + x^4 + x^3 + x^2 + 1 (0x11D)
    poly = 0x11D
    
    def gf_mul(a, b):
        res = 0
        while b:
            if b & 1:
                res ^= a
            b >>= 1
            a <<= 1
            if a & 0x100:
                a ^= poly
        return res & 0xFF

    def gf_pow(a, n):
        res = 1
        base = a
        while n:
            if n & 1:
                res = gf_mul(res, base)
            base = gf_mul(base, base)
            n >>= 1
        return res

    def get_primitive_17th_root():
        for candidate in range(2, 256):
            val = gf_pow(candidate, 17)
            if val == 1 and candidate != 1:
                return candidate
        return None

    beta = get_primitive_17th_root()
    if beta is None:
        # Fallback seed
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words

    # Build generator polynomial G(x) of degree 8
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
        
    g_bits = [0] * 9
    for i, c in enumerate(poly_coeffs):
        if c == 1:
            g_bits[i] = 1
        elif c == 0:
            g_bits[i] = 0
        else:
            g_bits[i] = 1  # Fallback

    # Generate 256 codewords
    codewords_17 = []
    for u in range(512):
        c = 0
        for k in range(9):
            if (u >> k) & 1:
                for j in range(9):
                    if g_bits[j]:
                        c ^= 1 << (j + k)
        codewords_17.append(c)
        
    # Extend to 18 bits
    extended_codewords = []
    for c in codewords_17:
        parity = bin(c).count('1') % 2
        ec = c | (parity << 17)
        extended_codewords.append(ec)
        
    # Shorten by fixing bit 17 to 0
    shortened_codewords = []
    for ec in extended_codewords:
        if (ec >> 17) & 1 == 0:
            shortened_codewords.append(ec & ((1 << 17) - 1))
            
    shortened_codewords = list(set(shortened_codewords))
    
    if len(shortened_codewords) < 32:
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words

    # Sort for determinism
    shortened_codewords.sort()
    current_code = shortened_codewords
    best_code = list(current_code)
    
    # 2. Randomized Local Search / Simulated Annealing
    # Goal: Add words to break linearity and reach > 256.
    
    # Precompute distances from each current codeword to all possible candidates? 
    # 2^17 = 131072 candidates. 256 codewords.
    # Distance matrix is 256 x 131072. Too large for direct storage if we recompute often, 
    # but we can compute on the fly or use numpy vectorization.
    
    # Let's use a simpler approach:
    # 1. Generate a pool of candidate words not in the code.
    # 2. For each candidate, check min distance to existing code.
    # 3. If min_dist >= 6, add it.
    # 4. If min_dist < 6, we are stuck. 
    #    To escape, we can try "mutating" an existing codeword.
    #    If we replace codeword c_i with c_i' (c_i' = c_i XOR mask), 
    #    we need dist(c_i', c_j) >= 6 for all j != i, and dist(c_i', new_candidate) >= 6.
    #    This is complex.
    
    # Alternative Strategy: "Greedy with Backtracking" or "Local Search on the Code Set"
    # Since we want to maximize size, and we are at 256 (linear limit), we need to find nonlinear additions.
    # The Milshtein code is nonlinear.
    
    # Let's try a direct search for additions first.
    # If we can't add any, we try to modify the set.
    
    # Efficient distance check using NumPy
    # Represent code as a matrix of bits? 256x17.
    # Candidates as 131072x17.
    # This is 131072 * 256 * 17 operations ~ 500M. Might be slow in pure Python loop, but NumPy can handle it in chunks.
    
    code_arr = np.array(current_code, dtype=np.uint8)
    # code_arr is 256x17? No, it's 256x1 integers.
    # Let's unpack bits.
    def unpack_bits(words, n):
        # words: list of ints
        # returns: numpy array (len(words), n) of uint8
        arr = np.zeros((len(words), n), dtype=np.uint8)
        for i in range(n):
            mask = 1 << i
            arr[:, i] = (np.array(words, dtype=np.uint64) & mask).astype(np.uint8)
        return arr

    # This unpacking is slow if done repeatedly. 
    # Let's stick to Python integer bit_count for the inner loop if we do it smartly, 
    # or precompute a distance lookup table for small distances?
    
    # Actually, 256 codewords is small. 
    # Let's try to add random candidates.
    
    # Generate all 131072 words? No, that's too many to iterate through naively if we do it many times.
    # Let's generate random candidates.
    
    # However, the "hole" in the linear code is specific.
    # A better heuristic: 
    # 1. Identify "tight" pairs (distance 6).
    # 2. Try to shift one of them to free up space for a new word?
    
    # Let's try a simpler local search:
    # 1. Start with the 256 words.
    # 2. Generate a random word W.
    # 3. Check if W can be added (min dist >= 6).
    # 4. If yes, add it.
    # 5. If no, find the word C in the code closest to W (dist < 6).
    # 6. Try to replace C with W? No, W is too close to C.
    # 7. Try to find a neighbor W' of W such that W' is far from all code words?
    
    # Let's try "Simulated Annealing" on the set of codewords.
    # State: Set of codewords S.
    # Move: Remove one word, add a different word.
    # Cost: Number of conflicts (pairs with dist < 6). We want 0 conflicts.
    # Objective: Maximize |S| subject to 0 conflicts.
    
    # Since we start with 0 conflicts, we must maintain 0 conflicts.
    # So the move is: Remove C_i, Add W_new.
    # Constraints:
    # 1. W_new not in S \ {C_i}
    # 2. dist(W_new, C_j) >= 6 for all j != i
    
    # This is equivalent to: Can we find a W_new that is at distance >= 6 from all words in S \ {C_i}?
    # If yes, we can swap C_i -> W_new. Size remains 256.
    # But we want size > 256.
    # So we should also try "Add" moves if possible.
    
    # If we can't add, we swap to change the configuration, hoping to create a "gap" where an add is possible later.
    
    # Let's implement this.
    
    # Precompute a list of all words? No.
    # We need to sample W_new.
    # To check if W_new is valid against S \ {C_i}, we need to check distances to 255 words.
    # This is fast in Python if we use bit_count.
    
    # Optimization: 
    # If we maintain the code as a list, checking 255 distances is ~255 * 10ns = 2.5us.
    # We can do ~100,000 such checks per second in Python? Maybe 10,000.
    # 45 seconds * 10,000 = 450,000 swaps.
    # This might be enough to explore the space.
    
    # To find W_new, we can sample randomly.
    # But random sampling might miss the specific nonlinear structure.
    # Better: Sample W_new from the "neighborhood" of existing words?
    # Or, use the fact that the dual of the QR code has certain properties?
    
    # Let's try random sampling first.
    
    random.seed(42)
    
    # Helper: check if word w is valid against a set of words (list of ints)
    def is_valid(w, words_list):
        for c in words_list:
            if w == c:
                return False
            if (w ^ c).bit_count() < MIN_DIST:
                return False
        return True
    
    # To speed up is_valid, we can early exit.
    # Also, we can precompute nothing, but we can sort the list? No.
    
    # Let's run the search.
    
    current_size = len(current_code)
    max_size_found = current_size
    
    # We will attempt to add words first.
    # If we add a word, we increase size.
    # If we can't add, we try swaps.
    
    # To make swaps effective, we should target "bottleneck" words.
    # A word is a bottleneck if it prevents many potential additions.
    # Approximation: Words involved in many distance-6 pairs.
    
    # Let's count distance-6 pairs for each word.
    pair_counts = [0] * len(current_code)
    for i in range(len(current_code)):
        for j in range(i+1, len(current_code)):
            d = (current_code[i] ^ current_code[j]).bit_count()
            if d == MIN_DIST:
                pair_counts[i] += 1
                pair_counts[j] += 1
    
    # Sort indices by pair count (descending) to pick "busy" words for removal.
    busy_indices = sorted(range(len(current_code)), key=lambda i: pair_counts[i], reverse=True)
    
    # We will iterate through a schedule of moves.
    # 1. Try to add a random word.
    # 2. If fail, try to swap a busy word with a random word.
    
    # To speed up "Try to add", we can check against the whole list.
    # To speed up "Try to swap", we check against list minus the removed word.
    
    # Let's use a set for O(1) membership, but we need distance checks which are O(N).
    
    # We'll keep a list for iteration and a set for membership.
    code_set = set(current_code)
    
    steps = 0
    max_steps = 200000  # Adjust based on speed
    
    # We need to be careful with time.
    # Let's measure the time of one is_valid check.
    
    while time.time() < deadline - 2.0:
        steps += 1
        
        # 1. Try to add a word
        # Sample a random word
        # To bias towards "interesting" words, maybe sample from a pool?
        # For now, pure random.
        
        # Optimization: Instead of pure random, we can try words that are "almost" valid.
        # But let's start with random.
        
        success_add = False
        for _ in range(5):  # Try 5 random candidates per step
            w = random.randint(0, (1 << N) - 1)
            if w in code_set:
                continue
            
            # Check validity
            valid = True
            for c in current_code:
                if (w ^ c).bit_count() < MIN_DIST:
                    valid = False
                    break
            
            if valid:
                current_code.append(w)
                code_set.add(w)
                max_size_found = max(max_size_found, len(current_code))
                success_add = True
                # Update pair counts? Expensive. Let's skip updating pair counts for now,
                # or recompute periodically.
                break
        
        if success_add:
            continue
            
        # 2. Try to swap
        # Pick a busy word to remove
        # We need to recompute pair counts occasionally as the code changes.
        # For now, use the initial pair counts or a random index if we don't want to recompute.
        # Let's recompute pair counts every 1000 steps or if size changes.
        
        if steps % 1000 == 0:
            pair_counts = [0] * len(current_code)
            for i in range(len(current_code)):
                for j in range(i+1, len(current_code)):
                    d = (current_code[i] ^ current_code[j]).bit_count()
                    if d == MIN_DIST:
                        pair_counts[i] += 1
                        pair_counts[j] += 1
            busy_indices = sorted(range(len(current_code)), key=lambda i: pair_counts[i], reverse=True)
        
        # Pick a random busy word (top 10)
        if busy_indices:
            # Pick from top 10
            idx = random.choice(busy_indices[:10])
        else:
            idx = random.randint(0, len(current_code) - 1)
        
        removed_word = current_code[idx]
        
        # Try to find a replacement w
        # w must be valid against current_code \ {removed_word}
        
        # To make this efficient, we can check against the list, skipping idx.
        
        found_replacement = False
        for _ in range(5):
            w = random.randint(0, (1 << N) - 1)
            if w in code_set:
                continue
            
            valid = True
            for j, c in enumerate(current_code):
                if j == idx:
                    continue
                if (w ^ c).bit_count() < MIN_DIST:
                    valid = False
                    break
            
            if valid:
                # Perform swap
                current_code[idx] = w
                code_set.remove(removed_word)
                code_set.add(w)
                found_replacement = True
                break
        
        # If no replacement found, do nothing (keep the code as is)
        # This is a "stall". We could try more aggressive moves, but for now, let's just continue.
        
        # Occasionally, we might want to try to "repair" a conflict if we allowed one?
        # No, we are maintaining 0 conflicts.
        
        # To avoid getting stuck in a local minimum where no random swap works,
        # we can try "guided" swaps.
        # E.g., take a word C_i, flip one bit to get C_i'.
        # Check if C_i' is valid against the rest.
        # If yes, swap.
        
        if not found_replacement and steps % 10 == 0:
            # Try bit-flip mutations on a busy word
            idx = random.choice(busy_indices[:10]) if busy_indices else random.randint(0, len(current_code)-1)
            removed_word = current_code[idx]
            
            for bit in range(N):
                w = removed_word ^ (1 << bit)
                if w in code_set:
                    continue
                
                valid = True
                for j, c in enumerate(current_code):
                    if j == idx:
                        continue
                    if (w ^ c).bit_count() < MIN_DIST:
                        valid = False
                        break
                
                if valid:
                    current_code[idx] = w
                    code_set.remove(removed_word)
                    code_set.add(w)
                    break

    # Final check: ensure all distances are >= 6
    # If we made a mistake, we should return the best valid code.
    # Since we only add/swap if valid, the current code should be valid.
    # But let's verify to be safe, and if invalid, return the initial 256.
    
    # Verification is O(N^2). 256^2 = 65k checks. Fast.
    final_code = list(current_code)
    valid_final = True
    for i in range(len(final_code)):
        for j in range(i+1, len(final_code)):
            if (final_code[i] ^ final_code[j]).bit_count() < MIN_DIST:
                valid_final = False
                break
        if not valid_final:
            break
    
    if not valid_final:
        return best_code # Or initial code
    
    return final_code

