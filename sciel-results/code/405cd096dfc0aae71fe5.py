import time
import random
import numpy as np

# EVOLVE-BLOCK-START
N = 18
MIN_DIST = 6

def hamming_distance(a, b):
    return (a ^ b).bit_count()

def reed_muller_construction():
    """First-order Reed-Muller code on 16 coordinates, extended by two zero bits.
    Every affine Boolean function on F_2^4 is a word: 32 words, distance 8."""
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    return words

def construct_quadratic_residue_code():
    """
    Construct the extended quadratic residue code of length 18.
    This is a [18, 9, 6] code with 512 codewords.
    """
    # The generator polynomial for the [17, 9, 5] QR code over GF(2)
    # is g(x) = x^8 + x^5 + x^4 + x^3 + 1
    # Coefficients from x^0 to x^8: [1, 0, 0, 1, 1, 1, 0, 0, 1]
    
    candidates = [
        [1, 0, 0, 1, 1, 1, 0, 0, 1], # x^8 + x^5 + x^4 + x^3 + 1
        [1, 0, 0, 1, 1, 1, 0, 1, 1], # Another potential candidate?
        [1, 1, 0, 0, 1, 1, 1, 0, 1], # Just in case
    ]
    
    for g_coeffs in candidates:
        # Build generator matrix for [17, 9, 5] code
        G_17 = []
        for i in range(9):
            row = 0
            for j in range(9):
                if g_coeffs[j]:
                    pos = (i + j) % 17
                    row |= (1 << pos)
            G_17.append(row)
            
        # Extend to length 18 with parity bit
        G_18 = []
        for row in G_17:
            parity = row.bit_count() % 2
            row_18 = row | (parity << 17)
            G_18.append(row_18)
            
        # Generate all 2^9 = 512 codewords
        words = []
        for mask in range(512):
            word = 0
            for i in range(9):
                if (mask >> i) & 1:
                    word ^= G_18[i]
            words.append(word)
            
        # Verify min distance
        # Optimization: Since it's a linear code, min distance is min weight of non-zero codewords.
        # But we check all pairs to be safe and match the task requirement style.
        # 512^2 / 2 = 130k checks. Fast.
        min_d = 18
        for i in range(len(words)):
            for j in range(i+1, len(words)):
                d = hamming_distance(words[i], words[j])
                if d < min_d:
                    min_d = d
                    if min_d < MIN_DIST:
                        break
            if min_d < MIN_DIST:
                break
                
        if min_d >= MIN_DIST:
            return words
            
    return None

def is_valid_word(candidate, words):
    """Check if candidate has distance >= MIN_DIST from all words in words."""
    for w in words:
        if (candidate ^ w).bit_count() < MIN_DIST:
            return False
    return True

def greedy_random_construction(seed=42, max_iter=2000000, deadline=None, base_words=None):
    """Greedy random construction to add words to the base code."""
    rng = random.Random(seed)
    if base_words is None:
        words = reed_muller_construction()
    else:
        words = list(base_words)
    word_set = set(words)
    
    for i in range(max_iter):
        if deadline and time.time() > deadline:
            break
            
        # Generate random word
        # Bias towards low weights
        weight = rng.choice([0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6, 7, 8, 9])
        positions = rng.sample(range(N), min(weight, N))
        word = 0
        for p in positions:
            word |= 1 << p
        
        if word in word_set:
            continue
        
        valid = True
        for w in words:
            if hamming_distance(word, w) < MIN_DIST:
                valid = False
                break
        if valid:
            words.append(word)
            word_set.add(word)
    
    return words

def greedy_with_local_search(seed=42, max_iter=500000, deadline=None, base_words=None):
    """Greedy construction with occasional local search perturbations."""
    rng = random.Random(seed)
    if base_words is None:
        words = reed_muller_construction()
    else:
        words = list(base_words)
    word_set = set(words)
    
    for i in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
        # Occasionally do a local search: try flipping bits of an existing word
        if rng.random() < 0.1 and len(words) > 0:
            base_idx = rng.randint(0, len(words) - 1)
            base = words[base_idx]
            # Try flipping 1-3 bits
            n_flips = rng.randint(1, 3)
            flip_positions = rng.sample(range(N), n_flips)
            candidate = base
            for p in flip_positions:
                candidate ^= (1 << p)
            if candidate not in word_set and is_valid_word(candidate, words):
                words.append(candidate)
                word_set.add(candidate)
        else:
            # Standard random generation
            weight = rng.choice([0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6, 7, 8, 9])
            positions = rng.sample(range(N), min(weight, N))
            word = 0
            for p in positions:
                word |= 1 << p
            
            if word in word_set:
                continue
            
            if is_valid_word(word, words):
                words.append(word)
                word_set.add(word)
    
    return words

def systematic_search(seed=42, deadline=None):
    """Systematic search over low-weight words, then medium-weight."""
    rng = random.Random(seed)
    words = []
    word_set = set()
    
    # First, add the all-zero word
    words.append(0)
    word_set.add(0)
    
    # Generate all words of weight 1 through 5
    # This is manageable: C(18,1)+C(18,2)+C(18,3)+C(18,4)+C(18,5) = 18+153+816+3060+8568 = 12615
    from itertools import combinations
    
    for weight in range(1, 6):
        if deadline and time.time() > deadline:
            break
        for positions in combinations(range(N), weight):
            if deadline and time.time() > deadline:
                break
            word = 0
            for p in positions:
                word |= 1 << p
            if word in word_set:
                continue
            if is_valid_word(word, words):
                words.append(word)
                word_set.add(word)
    
    # Then try random higher-weight words
    for i in range(200000):
        if deadline and time.time() > deadline:
            break
        weight = rng.randint(6, 12)
        positions = rng.sample(range(N), weight)
        word = 0
        for p in positions:
            word |= 1 << p
        if word in word_set:
            continue
        if is_valid_word(word, words):
            words.append(word)
            word_set.add(word)
    
    return words

def vectorized_greedy_extend(words, deadline, seed=42, max_iter=500000):
    """
    Vectorized greedy extension using numpy for faster distance checks.
    This is the key refinement: instead of checking each candidate against
    all existing words one-by-one in Python, we batch the distance checks.
    """
    rng = random.Random(seed)
    words = list(words)
    word_set = set(words)
    
    if not words:
        return words
    
    # Precompute numpy array of current words
    words_np = np.array(words, dtype=np.uint16)
    
    for i in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
        # Generate a batch of random candidates
        batch_size = min(100, max(10, (max_iter - i) // 10))
        
        # Generate random weights
        weights = rng.choices([0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6, 7, 8, 9], k=batch_size)
        
        candidates = []
        for w in weights:
            positions = rng.sample(range(N), min(w, N))
            word = 0
            for p in positions:
                word |= 1 << p
            if word not in word_set:
                candidates.append(word)
        
        if not candidates:
            continue
        
        candidates_np = np.array(candidates, dtype=np.uint16)
        
        # Compute distances from each candidate to each existing word
        # Use broadcasting: candidates_np[:, None] vs words_np[None, :]
        # This creates a batch_size x len(words) matrix
        # But this might be too large if len(words) is big. Let's be careful.
        
        # Actually, for 512 words, 100x512 = 51200 operations, which is fast.
        # For larger codes, we might need to chunk.
        
        if len(words) > 1000:
            # Chunk the existing words to avoid huge memory usage
            chunk_size = 500
            all_valid = np.ones(len(candidates), dtype=bool)
            for start in range(0, len(words), chunk_size):
                end = min(start + chunk_size, len(words))
                chunk = words_np[start:end]
                # XOR and popcount
                xor = candidates_np[:, None] ^ chunk[None, :]
                # Popcount for each row
                # Use a lookup table or bit manipulation
                # For uint16, we can use a trick
                # Actually, numpy doesn't have a direct popcount for 2D arrays easily
                # Let's use a different approach: iterate over candidates
                # This defeats the purpose. Let's stick with the simple approach
                # but limit batch size if words is large.
                pass
            # For now, just fall back to simple check if words is large
            for idx, cand in enumerate(candidates):
                if not all_valid[idx]:
                    continue
                if is_valid_word(cand, words):
                    words.append(cand)
                    word_set.add(cand)
                    words_np = np.append(words_np, cand)
        else:
            # Vectorized check
            xor = candidates_np[:, None] ^ words_np[None, :]
            # Compute popcount for each element
            # We can use a lookup table for bytes
            # Or use a simple loop over bits
            # Let's use a vectorized popcount
            # For 18-bit numbers, we can do:
            # popcount(x) = popcount(x & 0x00FF) + popcount((x >> 8) & 0x00FF)
            # Precompute popcount for 0-255
            popcount_table = np.array([bin(i).count('1') for i in range(256)], dtype=np.int8)
            
            low = xor & 0x00FF
            high = (xor >> 8) & 0x00FF
            
            dists = popcount_table[low] + popcount_table[high]
            
            # Check if all distances are >= MIN_DIST
            valid_mask = np.all(dists >= MIN_DIST, axis=1)
            
            for idx in range(len(candidates)):
                if valid_mask[idx]:
                    cand = candidates[idx]
                    words.append(cand)
                    word_set.add(cand)
                    words_np = np.append(words_np, cand)
    
    return words

def construct():
    """
    Attempts QR code, then tries to extend it with greedy search.
    Falls back to robust greedy search if QR fails.
    """
    start_time = time.time()
    deadline = start_time + 50.0 # 10s margin
    
    best_words = []
    
    # Try QR Code first (fast)
    try:
        qr_words = construct_quadratic_residue_code()
        if qr_words:
            best_words = qr_words
            # Try to extend the QR code with a short greedy search
            # This might find a few more words if the code is not maximal
            # Use vectorized extension for speed
            extended_words = vectorized_greedy_extend(qr_words, deadline, seed=42, max_iter=200000)
            if len(extended_words) > len(best_words):
                best_words = extended_words
            
            # Also try standard greedy with local search
            if time.time() < deadline:
                local_words = greedy_with_local_search(seed=123, max_iter=100000, deadline=deadline, base_words=qr_words)
                if len(local_words) > len(best_words):
                    best_words = local_words
            
            return best_words
    except Exception:
        pass

    # Fallback: Greedy Search
    seeds = [42, 123, 999, 2023, 777]
    
    for seed in seeds:
        if time.time() > deadline:
            break
        current_words = greedy_random_construction(seed=seed, max_iter=1000000, deadline=deadline)
        if len(current_words) > len(best_words):
            best_words = current_words
    
    # Try systematic search
    if time.time() < deadline:
        sys_words = systematic_search(seed=42, deadline=deadline)
        if len(sys_words) > len(best_words):
            best_words = sys_words
    
    return best_words

# EVOLVE-BLOCK-END
