"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def _build_linear_code():
    """Construct the 256-word [17,8,6] code by shortening the [18,9,6] QR code.
    
    The extended quadratic residue code of length 18 has parameters [18,9,6].
    Shortening by one coordinate gives a [17,8,6] code with 2^8 = 256 words.
    """
    N = 17
    
    # Quadratic residues mod 18 (including 0): {0, 1, 4, 7, 9, 10, 16}
    h = np.zeros(18, dtype=np.uint8)
    for pos in [0, 1, 4, 7, 9, 10, 16]:
        h[pos] = 1
    
    # Generator matrix for [18,10,5] code: 10 cyclic shifts of h
    G18 = np.zeros((10, 18), dtype=np.uint8)
    for i in range(10):
        G18[i] = np.roll(h, i)
    
    # The [18,9,6] code is the even-like subcode
    G_even = np.zeros((9, 18), dtype=np.uint8)
    for i in range(9):
        G_even[i] = (G18[i] + G18[i+1]) % 2
    
    # Generate all 512 codewords of the [18,9,6] code
    codewords_18 = set()
    for mask in range(512):
        cw = np.zeros(18, dtype=np.uint8)
        for i in range(9):
            if mask & (1 << i):
                cw = (cw + G_even[i]) % 2
        val = 0
        for j in range(18):
            if cw[j]:
                val |= (1 << j)
        codewords_18.add(val)
    
    # Shorten by coordinate 0: keep codewords with bit 0 = 0, remove bit 0
    codewords_17 = set()
    for cw in codewords_18:
        if (cw & 1) == 0:
            codewords_17.add(cw >> 1)
    
    return list(codewords_17)


def _popcount_array(arr):
    """Vectorized popcount for numpy array of uint32."""
    # Use a lookup table approach for speed
    # For 17-bit numbers, split into bytes
    result = np.zeros(arr.shape, dtype=np.uint8)
    for shift in range(0, 32, 8):
        byte = (arr >> shift) & 0xFF
        # Lookup table for popcount of 0-255
        if shift == 0:
            lut = np.array([0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,
                           1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
                           1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
                           2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
                           1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
                           2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
                           2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
                           3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8], dtype=np.uint8)
        elif shift == 8:
            lut = np.array([0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,
                           1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
                           1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
                           2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
                           1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
                           2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
                           2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
                           3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8], dtype=np.uint8)
        elif shift == 16:
            lut = np.array([0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,
                           1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
                           1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
                           2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
                           1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
                           2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
                           2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
                           3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8], dtype=np.uint8)
        elif shift == 24:
            lut = np.array([0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,
                           1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
                           1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
                           2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
                           1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
                           2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
                           2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
                           3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8], dtype=np.uint8)
        else:
            lut = np.array([0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,
                           1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
                           1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
                           2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
                           1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
                           2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
                           2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
                           3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8], dtype=np.uint8)
        result += lut[byte]
    return result


def construct():
    """Construct the largest set of 17-bit words with pairwise Hamming distance at least 6.
    
    Uses a nonlinear local search starting from the 256-word linear code.
    """
    start_time = time.time()
    deadline = start_time + 85  # 85 second deadline with safety margin
    
    N = 17
    MIN_DIST = 6
    
    # Start with the linear code
    try:
        words = _build_linear_code()
    except Exception:
        # Fallback: use the RM code
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
    
    if not words:
        return [0]
    
    best_words = list(words)
    best_count = len(best_words)
    word_set = set(best_words)
    
    rng = np.random.RandomState(42)
    
    # Precompute all words for random sampling
    all_words = np.array(list(range(2**N)), dtype=np.uint32)
    rng.shuffle(all_words)
    
    # Phase 1: Try to add words greedily
    words_arr = np.array(list(word_set), dtype=np.uint32)
    
    for trial in range(len(all_words)):
        if time.time() > deadline - 20:
            break
        
        candidate = int(all_words[trial])
        if candidate in word_set:
            continue
        
        # Check if candidate can be added
        xored = words_arr ^ candidate
        dists = _popcount_array(xored)
        
        if np.min(dists) >= MIN_DIST:
            word_set.add(candidate)
            best_words.append(candidate)
            best_count += 1
            words_arr = np.append(words_arr, candidate)
    
    # Phase 2: Try swap operations - remove one word, add two
    # This allows us to escape local optima
    for iteration in range(200):
        if time.time() > deadline - 15:
            break
        
        # Try random swaps
        for swap_attempt in range(50):
            if time.time() > deadline - 15:
                break
            
            # Pick a random existing word to potentially remove
            idx = rng.randint(len(best_words))
            remove_word = best_words[idx]
            
            # Try to find two words that can replace it
            # First, temporarily remove the word
            temp_set = set(best_words)
            temp_set.discard(remove_word)
            temp_arr = np.array(list(temp_set), dtype=np.uint32)
            
            # Try to add two new words
            added = []
            for add_attempt in range(100):
                if time.time() > deadline - 15:
                    break
                
                # Pick a random candidate
                cand_idx = rng.randint(2**N)
                candidate = cand_idx
                
                if candidate in temp_set or candidate in added:
                    continue
                
                # Check distance to all temp words
                xored = temp_arr ^ candidate
                dists = _popcount_array(xored)
                
                if np.min(dists) < MIN_DIST:
                    continue
                
                # Check distance to already added words in this swap
                valid = True
                for prev_added in added:
                    d = (candidate ^ prev_added).bit_count()
                    if d < MIN_DIST:
                        valid = False
                        break
                
                if valid:
                    added.append(candidate)
                    temp_set.add(candidate)
                    temp_arr = np.append(temp_arr, candidate)
                    if len(added) == 2:
                        break
            
            # If we added 2 words, accept the swap
            if len(added) == 2:
                best_words = list(temp_set)
                word_set = temp_set
                best_count = len(best_words)
    
    return best_words
# EVOLVE-BLOCK-END
