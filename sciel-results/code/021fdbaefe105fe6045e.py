import time
import random

def construct():
    N = 18
    MIN_DIST = 6
    TOTAL_BUDGET = 55.0
    t_start = time.time()
    deadline = t_start + TOTAL_BUDGET

    # Build the extended quadratic-residue code of length 18
    # QR(17) is a [17, 6, 5] code. Its extension is [18, 6, 6] with 64 words.
    # The quadratic residues mod 17 are: 1, 2, 4, 8, 9, 13, 15, 16
    # In 0-indexed positions (subtract 1): {0, 1, 3, 7, 8, 12, 14, 15}
    
    row0_positions = [0, 1, 3, 7, 8, 12, 14, 15]
    
    # Build generator matrix (6 rows, circulant shifts)
    G = []
    for i in range(6):
        row = 0
        for pos in row0_positions:
            shifted = (pos + i) % 17
            row |= (1 << shifted)
        G.append(row)
    
    # Compute all 2^6 = 64 codewords
    qr_words_18 = []
    for mask in range(64):
        word = 0
        for i in range(6):
            if (mask >> i) & 1:
                word ^= G[i]
        # Add parity bit at position 17
        parity = word.bit_count() % 2
        extended = word | (parity << 17)
        qr_words_18.append(extended)
    
    # Verify minimum distance
    def min_distance(words):
        min_d = N + 1
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                d = (words[i] ^ words[j]).bit_count()
                if d < min_d:
                    min_d = d
        return min_d
    
    md = min_distance(qr_words_18)
    if md < MIN_DIST:
        # Fallback to Reed-Muller if construction is wrong
        rm_words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                rm_words.append(word)
        best_words = list(rm_words)
    else:
        best_words = list(qr_words_18)
    
    best_size = len(best_words)
    
    # Efficient greedy extension with early termination
    def can_add_fast(candidate, current_words):
        for w in current_words:
            if (candidate ^ w).bit_count() < MIN_DIST:
                return False
        return True
    
    current_words = list(best_words)
    current_set = set(current_words)
    
    # Greedy extension: iterate through all 2^18 words
    # Use a blocked approach to be more efficient
    added_any = True
    while added_any and time.time() < deadline - 8.0:
        added_any = False
        for candidate in range(1 << N):
            if candidate in current_set:
                continue
            if can_add_fast(candidate, current_words):
                current_words.append(candidate)
                current_set.add(candidate)
                added_any = True
                if len(current_words) > best_size:
                    best_words = list(current_words)
                    best_size = len(current_words)
            if time.time() > deadline - 5.0:
                break
    
    # Local search: try bit flips on existing codewords to find new valid words
    if time.time() < deadline - 5.0:
        rng = random.Random(42)
        flipped_candidates = set()
        for w in current_words:
            for bit in range(N):
                flipped = w ^ (1 << bit)
                flipped_candidates.add(flipped)
        
        # Shuffle for randomness
        candidate_list = list(flipped_candidates)
        rng.shuffle(candidate_list)
        
        for candidate in candidate_list:
            if candidate in current_set:
                continue
            if can_add_fast(candidate, current_words):
                current_words.append(candidate)
                current_set.add(candidate)
                if len(current_words) > best_size:
                    best_words = list(current_words)
                    best_size = len(current_words)
            if time.time() > deadline - 5.0:
                break
    
    # Try a few random restarts with greedy extension
    if time.time() < deadline - 10.0:
        rng2 = random.Random(12345)
        for restart in range(5):
            if time.time() > deadline - 5.0:
                break
            
            # Create a random valid seed
            seed_words = []
            seed_set = set()
            attempts = 0
            while len(seed_words) < 12 and attempts < 50000:
                w = rng2.randint(0, (1 << N) - 1)
                if w not in seed_set:
                    valid = True
                    for sw in seed_words:
                        if (w ^ sw).bit_count() < MIN_DIST:
                            valid = False
                            break
                    if valid:
                        seed_words.append(w)
                        seed_set.add(w)
                attempts += 1
            
            if len(seed_words) < 5:
                continue
            
            # Greedy extend
            extended = list(seed_words)
            ext_set = set(extended)
            for candidate in range(1 << N):
                if candidate in ext_set:
                    continue
                if can_add_fast(candidate, extended):
                    extended.append(candidate)
                    ext_set.add(candidate)
                if time.time() > deadline - 5.0:
                    break
            
            if len(extended) > best_size:
                best_words = extended
                best_size = len(extended)
    
    return best_words
