"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

# We will use a randomized local search approach:
# 1. Start with a greedy covering (or the known direct-sum construction as a baseline).
# 2. Iteratively try to remove codewords that are "redundant" (i.e., their covered words
#    are all covered by other codewords) or swap them for better ones.
# 3. Use a time budget to return the best result found.

def _hamming_ball_indices(word, n, q, radius):
    """Yield indices of all words within Hamming distance <= radius from word."""
    # For n=8, q=3, radius=1: 1 + 8*2 = 17 words
    # We represent words as integers in base q.
    word_int = 0
    for i in range(n):
        word_int = word_int * q + word[i]
    
    yield word_int
    
    for pos in range(n):
        for val in range(q):
            if val == word[pos]:
                continue
            new_word = list(word)
            new_word[pos] = val
            new_int = 0
            for i in range(n):
                new_int = new_int * q + new_word[i]
            yield new_int

def _word_to_int(word, q):
    result = 0
    for i in range(len(word)):
        result = result * q + word[i]
    return result

def _int_to_word(x, n, q):
    word = []
    for i in range(n):
        word.append(x % q)
        x //= q
    return word[::-1]

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second safety margin before 120s limit
    
    total_words = q ** n  # 6561 for q=3, n=8
    
    # Baseline: direct sum construction (486 words)
    if n >= 4 and q == 3:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        baseline = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n-4)]
    else:
        # Greedy baseline for other parameters
        baseline = []
        covered = np.zeros(total_words, dtype=bool)
        for _ in range(10000):  # cap
            best_pos = -1
            best_gain = 0
            # Random sampling for speed
            for _ in range(200):
                word = [random.randint(0, q-1) for _ in range(n)]
                word_int = _word_to_int(word, q)
                gain = 0
                for idx in _hamming_ball_indices(word, n, q, radius):
                    if not covered[idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_pos = word_int
                    best_word = word
            if best_gain == 0:
                break
            baseline.append(best_word)
            for idx in _hamming_ball_indices(best_word, n, q, radius):
                covered[idx] = True
    
    # Convert baseline to integer representation for speed
    codewords = [_word_to_int(w, q) for w in baseline]
    best_codewords = codewords[:]
    
    # Build coverage structure: for each word in space, which codewords cover it
    # cover_count[i] = number of codewords within distance <= radius of word i
    cover_count = np.zeros(total_words, dtype=np.int32)
    for cw in codewords:
        for idx in _hamming_ball_indices(_int_to_word(cw, n, q), n, q, radius):
            cover_count[idx] += 1
    
    # Verify coverage
    if np.any(cover_count == 0):
        # Not fully covered, need to fix
        pass
    
    # Local search: try to remove redundant codewords
    # A codeword is redundant if removing it doesn't leave any word uncovered
    # i.e., for all words it covers, cover_count[word] >= 2
    
    def is_redundant(cw_idx):
        cw = codewords[cw_idx]
        word = _int_to_word(cw, n, q)
        for idx in _hamming_ball_indices(word, n, q, radius):
            if cover_count[idx] <= 1:
                return False
        return True
    
    # Iteratively remove redundant codewords
    removed = True
    while removed:
        removed = False
        for i in range(len(codewords) - 1, -1, -1):
            if time.time() > deadline - 10:
                break
            if is_redundant(i):
                cw = codewords[i]
                word = _int_to_word(cw, n, q)
                for idx in _hamming_ball_indices(word, n, q, radius):
                    cover_count[idx] -= 1
                codewords.pop(i)
                removed = True
        else:
            if time.time() > deadline - 10:
                break
    
    if len(codewords) < len(best_codewords):
        best_codewords = codewords[:]
    
    # Try to add codewords to reduce... no, we want to minimize.
    # Try swaps: remove one, add one, see if we can get smaller set
    # Actually, let's try a different approach: greedy removal with randomization
    
    # Restart from baseline and do randomized greedy removal
    random.seed(42)
    for trial in range(20):
        if time.time() > deadline - 5:
            break
        
        # Start from baseline
        cw_list = [_word_to_int(w, q) for w in baseline]
        cc = np.zeros(total_words, dtype=np.int32)
        for cw in cw_list:
            word = _int_to_word(cw, n, q)
            for idx in _hamming_ball_indices(word, n, q, radius):
                cc[idx] += 1
        
        # Randomized removal
        removed_any = True
        while removed_any:
            if time.time() > deadline - 5:
                break
            removed_any = False
            # Shuffle indices
            indices = list(range(len(cw_list)))
            random.shuffle(indices)
            for i in indices:
                if time.time() > deadline - 5:
                    break
                cw = cw_list[i]
                word = _int_to_word(cw, n, q)
                redundant = True
                for idx in _hamming_ball_indices(word, n, q, radius):
                    if cc[idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    for idx in _hamming_ball_indices(word, n, q, radius):
                        cc[idx] -= 1
                    cw_list.pop(i)
                    removed_any = True
                    if len(cw_list) < len(best_codewords):
                        best_codewords = cw_list[:]
        
        if len(cw_list) < len(best_codewords):
            best_codewords = cw_list[:]
    
    # Final polish: try to remove any remaining redundant codewords
    if time.time() < deadline - 2:
        cc = np.zeros(total_words, dtype=np.int32)
        for cw in best_codewords:
            word = _int_to_word(cw, n, q)
            for idx in _hamming_ball_indices(word, n, q, radius):
                cc[idx] += 1
        
        changed = True
        while changed and time.time() < deadline - 1:
            changed = False
            for i in range(len(best_codewords) - 1, -1, -1):
                cw = best_codewords[i]
                word = _int_to_word(cw, n, q)
                redundant = True
                for idx in _hamming_ball_indices(word, n, q, radius):
                    if cc[idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    for idx in _hamming_ball_indices(word, n, q, radius):
                        cc[idx] -= 1
                    best_codewords.pop(i)
                    changed = True
    
    # Verify final coverage
    if time.time() < deadline:
        cc = np.zeros(total_words, dtype=np.int32)
        for cw in best_codewords:
            word = _int_to_word(cw, n, q)
            for idx in _hamming_ball_indices(word, n, q, radius):
                cc[idx] += 1
        if np.any(cc == 0):
            # Fall back to baseline
            best_codewords = [_word_to_int(w, q) for w in baseline]
    
    # Convert back to list of lists
    return [_int_to_word(cw, n, q) for cw in best_codewords]
# EVOLVE-BLOCK-END
