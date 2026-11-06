"""Construct largest binary code of length 18 with min distance 6."""
import time
import random
import numpy as np

def construct():
    N = 18
    MIN_DIST = 6
    DEADLINE = time.time() + 82  # Safety margin before 90s limit
    
    # Start with extended Reed-Muller RM(1,4) punctured to length 18
    # RM(1,4) has 32 codewords of length 16, all weight 0, 8, or 16
    # We need length 18, so we pad with 2 zeros (or use a different construction)
    
    # Better: use the [18,10,6] extended QR code shortening, or greedy from RM
    # Let's use a vectorized greedy approach starting from a good seed
    
    # Seed: generate Reed-Muller-like codewords of length 18
    # Use quadratic forms over GF(2)^4 mapped to 18 positions
    seed_words = []
    # Positions 0-15 correspond to 4-bit vectors, positions 16-17 are extra
    # For each linear function f(x) = a·x + b, create a word
    for a in range(16):
        for b in range(2):
            word = 0
            for x in range(16):
                bit = (bin(a & x).count('1') + b) % 2
                if bit:
                    word |= (1 << x)
            # Positions 16, 17: set to make even weight (for distance preservation)
            w = bin(word).count('1')
            if w % 2 == 0:
                pass  # positions 16,17 stay 0
            else:
                word |= (1 << 16)  # flip one extra bit
            seed_words.append(word)
    
    # Verify seed words have pairwise distance >= 6
    # RM(1,4) has min distance 8, so after padding it should be fine
    # But let's verify and remove any violations
    valid_seed = []
    for w in seed_words:
        ok = True
        for v in valid_seed:
            if (w ^ v).bit_count() < MIN_DIST:
                ok = False
                break
        if ok:
            valid_seed.append(w)
    
    words = list(valid_seed)
    words_set = set(words)
    
    # Precompute distances for fast checking
    # Use numpy for vectorized distance computation
    words_arr = np.array(words, dtype=np.int64)
    
    def try_add_word(candidate, words_arr, words_set):
        """Try to add candidate word to the code. Returns True if added."""
        if candidate in words_set:
            return False
        # Check distance to all existing words
        diffs = words_arr ^ candidate
        # Count bits using numpy
        min_dist = np.min([bin(d).count('1') for d in diffs])
        if min_dist >= MIN_DIST:
            words_set.add(candidate)
            return True
        return False
    
    # Faster: use lookup table for bit counts of 18-bit values
    # Precompute popcount for all 2^18 values
    popcount_table = np.zeros(1 << N, dtype=np.int8)
    for i in range(1, 1 << N):
        popcount_table[i] = popcount_table[i >> 1] + (i & 1)
    
    def vectorized_greedy_pass(words_arr, words_set, rng, max_candidates=100000):
        """Greedy pass: try random candidates and add valid ones."""
        added = 0
        for _ in range(max_candidates):
            if time.time() > DEADLINE:
                break
            candidate = rng.getrandbits(N)
            if candidate in words_set:
                continue
            # Check distance to all existing words using popcount table
            diffs = words_arr ^ candidate
            distances = popcount_table[diffs]
            if np.min(distances) >= MIN_DIST:
                words_set.add(candidate)
                words_arr = np.append(words_arr, candidate)
                added += 1
        return added
    
    rng = random.Random(42)
    
    # Phase 1: Structured search - try all weight-6 and weight-7 words
    # This is expensive (C(18,6)=18564, C(18,7)=31824) but gives good coverage
    if time.time() < DEADLINE - 20:
        # Try weight 6 words first
        from itertools import combinations
        for pos in combinations(range(N), 6):
            if time.time() > DEADLINE - 15:
                break
            candidate = 0
            for p in pos:
                candidate |= (1 << p)
            if candidate in words_set:
                continue
            diffs = words_arr ^ candidate
            distances = popcount_table[diffs]
            if np.min(distances) >= MIN_DIST:
                words_set.add(candidate)
                words_arr = np.append(words_arr, candidate)
        
        # Try weight 7 words
        for pos in combinations(range(N), 7):
            if time.time() > DEADLINE - 10:
                break
            candidate = 0
            for p in pos:
                candidate |= (1 << p)
            if candidate in words_set:
                continue
            diffs = words_arr ^ candidate
            distances = popcount_table[diffs]
            if np.min(distances) >= MIN_DIST:
                words_set.add(candidate)
                words_arr = np.append(words_arr, candidate)
    
    # Phase 2: Random greedy with multiple restarts
    best_words = list(words_set)
    best_count = len(best_words)
    
    for restart in range(50):
        if time.time() > DEADLINE - 5:
            break
        
        # Each restart: try to add random words
        current_words_set = set(words_set)
        current_arr = words_arr.copy()
        
        # Try adding random words
        for _ in range(200000):
            if time.time() > DEADLINE - 3:
                break
            candidate = rng.getrandbits(N)
            if candidate in current_words_set:
                continue
            diffs = current_arr ^ candidate
            distances = popcount_table[diffs]
            if np.min(distances) >= MIN_DIST:
                current_words_set.add(candidate)
                current_arr = np.append(current_arr, candidate)
        
        if len(current_words_set) > best_count:
            best_count = len(current_words_set)
            best_words = list(current_words_set)
            words_set = current_words_set
            words_arr = current_arr
    
    # Phase 3: Local improvement - try to replace words to allow more additions
    if time.time() < DEADLINE - 5:
        # Try swapping: remove one word, see if we can add two new ones
        for _ in range(10000):
            if time.time() > DEADLINE - 3:
                break
            if len(words_set) < 2:
                break
            # Pick a random word to remove
            idx = rng.randrange(len(words_set))
            words_list = list(words_set)
            removed = words_list[idx]
            words_set.remove(removed)
            current_arr = np.array([w for w in words_set], dtype=np.int64)
            
            # Try to add new words
            added_any = False
            for _ in range(5000):
                if time.time() > DEADLINE - 2:
                    break
                candidate = rng.getrandbits(N)
                if candidate in words_set:
                    continue
                diffs = current_arr ^ candidate
                distances = popcount_table[diffs]
                if np.min(distances) >= MIN_DIST:
                    words_set.add(candidate)
                    current_arr = np.append(current_arr, candidate)
                    added_any = True
            
            if not added_any:
                # Put back the removed word
                words_set.add(removed)
    
    # Final verification
    final_words = list(words_set)
    if len(final_words) < 2:
        # Fallback: return at least the seed
        return seed_words[:1] if seed_words else [0]
    
    # Verify all pairs have distance >= MIN_DIST
    min_d = N + 1
    for i in range(len(final_words)):
        for j in range(i + 1, len(final_words)):
            d = (final_words[i] ^ final_words[j]).bit_count()
            if d < min_d:
                min_d = d
            if min_d < MIN_DIST:
                break
        if min_d < MIN_DIST:
            break
    
    if min_d >= MIN_DIST:
        return final_words
    
    # If verification failed, do a conservative cleanup
    cleaned = []
    for w in final_words:
        ok = True
        for v in cleaned:
            if (w ^ v).bit_count() < MIN_DIST:
                ok = False
                break
        if ok:
            cleaned.append(w)
    
    return cleaned if cleaned else [0]
