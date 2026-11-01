import time
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    # For the specific case: q=3, n=8, radius=1
    total_words = q ** n  # 6561
    num_coords = n
    alphabet = q
    
    # Convert word index to list of digits
    def idx_to_digits(idx):
        digits = [0] * num_coords
        for i in range(num_coords - 1, -1, -1):
            digits[i] = idx % alphabet
            idx //= alphabet
        return digits
    
    # Convert digits to index
    def digits_to_idx(digits):
        idx = 0
        for d in digits:
            idx = idx * alphabet + d
        return idx
    
    # Precompute: for each codeword index, the set of words it covers (as a list of indices)
    # Each codeword covers itself + all words at Hamming distance 1
    # Total: 1 + n*(q-1) = 1 + 8*2 = 17 words per codeword
    
    # We'll use a greedy set cover approach
    # uncovered: boolean array of size total_words
    uncovered = np.ones(total_words, dtype=np.uint8)
    codewords = []
    
    # Precompute coverage for each possible codeword
    # coverage[cw_idx] = list of word indices covered by codeword cw_idx
    # This is memory intensive: 6561 * 17 integers. Let's compute on the fly or use a smarter approach.
    
    # Actually, let's precompute coverage as a 2D array: coverage[cw_idx, j] = word_idx
    # Shape: (total_words, 17)
    cover_size = 1 + num_coords * (alphabet - 1)  # 17
    coverage = np.zeros((total_words, cover_size), dtype=np.int32)
    
    for cw_idx in range(total_words):
        digits = idx_to_digits(cw_idx)
        # The codeword itself
        coverage[cw_idx, 0] = cw_idx
        j = 1
        for pos in range(num_coords):
            for val in range(alphabet):
                if val == digits[pos]:
                    continue
                new_digits = digits[:]
                new_digits[pos] = val
                coverage[cw_idx, j] = digits_to_idx(new_digits)
                j += 1
    
    # Greedy set cover: at each step, pick the codeword that covers the most uncovered words
    # We need to efficiently track which codewords cover which uncovered words.
    # 
    # Approach: maintain a count for each codeword of how many uncovered words it covers.
    # When we add a codeword, we mark its covered words as covered, and for each newly covered
    # word, we decrement the count of all codewords that also cover that word.
    #
    # For that, we need: for each word, which codewords cover it.
    # A word w is covered by codeword c iff d(c,w) <= 1, i.e., c = w or c differs from w in exactly 1 position.
    # So word w is covered by: w itself, and for each position, for each other value, the word with that value at that position.
    # That's also 17 codewords per word.
    
    # covers_word[word_idx] = list of codeword indices that cover word_idx
    covers_word = np.zeros((total_words, cover_size), dtype=np.int32)
    for w_idx in range(total_words):
        digits = idx_to_digits(w_idx)
        covers_word[w_idx, 0] = w_idx
        j = 1
        for pos in range(num_coords):
            for val in range(alphabet):
                if val == digits[pos]:
                    continue
                new_digits = digits[:]
                new_digits[pos] = val
                covers_word[w_idx, j] = digits_to_idx(new_digits)
                j += 1
    
    # count[cw_idx] = number of uncovered words covered by codeword cw_idx
    count = np.zeros(total_words, dtype=np.int32)
    # Initialize: for each uncovered word (all are uncovered initially), increment count for each cw that covers it
    for w_idx in range(total_words):
        for j in range(cover_size):
            cw_idx = covers_word[w_idx, j]
            count[cw_idx] += 1
    
    num_covered = 0
    max_iter = 100000
    iter_count = 0
    
    while num_covered < total_words and iter_count < max_iter:
        if time.time() > deadline:
            break
        iter_count += 1
        
        # Find codeword with max count
        best_cw = np.argmax(count)
        if count[best_cw] == 0:
            break
        
        # Add this codeword
        codewords.append(best_cw)
        
        # For each word covered by best_cw, if it was uncovered, mark as covered
        # and decrement count for all codewords that cover that word
        for j in range(cover_size):
            w_idx = coverage[best_cw, j]
            if uncovered[w_idx]:
                uncovered[w_idx] = 0
                num_covered += 1
                # Decrement count for all codewords that cover w_idx
                for k in range(cover_size):
                    cw_idx = covers_word[w_idx, k]
                    count[cw_idx] -= 1
    
    # Convert codeword indices to lists of digits
    result = [idx_to_digits(cw) for cw in codewords]
    
    # Ensure distinctness (should be by construction)
    seen = set()
    unique_result = []
    for cw in result:
        t = tuple(cw)
        if t not in seen:
            seen.add(t)
            unique_result.append(cw)
    
    return unique_result
