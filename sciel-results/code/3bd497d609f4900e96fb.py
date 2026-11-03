import numpy as np
import time
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for safety
    
    # Total words in the space
    total_words = q ** n
    
    # Generate all words as indices 0 to total_words - 1
    # We can map an index to its tuple representation if needed, 
    # but we can work with indices directly for coverage.
    
    # Precompute neighbors for each word
    # neighbors[i] is a list of indices covered by word i (Hamming distance <= radius)
    # Since radius=1, neighbors are i itself and words differing in 1 position.
    
    # Helper to convert index to base-q digits
    def index_to_digits(idx, n, q):
        digits = [0] * n
        for i in range(n):
            digits[i] = idx % q
            idx //= q
        return digits

    # Helper to convert digits to index
    def digits_to_index(digits, q):
        idx = 0
        for d in reversed(digits):
            idx = idx * q + d
        return idx

    # Precompute coverage sets
    # cover_mask[i] will be a numpy boolean array or a set of indices?
    # Using a numpy array of shape (total_words, total_words) is 2187*2187 ~ 4.7M booleans.
    # That's about 4.7 MB. Very manageable.
    # cover_mask[i, j] is True if j is within distance radius of i.
    
    # Actually, for the greedy step, we need to count how many *uncovered* words are covered by candidate i.
    # If we maintain `uncovered` as a boolean array of size total_words,
    # gain[i] = np.dot(cover_mask[i], uncovered)
    
    cover_mask = np.zeros((total_words, total_words), dtype=bool)
    
    for i in range(total_words):
        digits_i = index_to_digits(i, n, q)
        # i covers itself
        cover_mask[i, i] = True
        # i covers neighbors at distance 1
        for pos in range(n):
            for val in range(q):
                if val == digits_i[pos]:
                    continue
                new_digits = digits_i.copy()
                new_digits[pos] = val
                j = digits_to_index(new_digits, q)
                cover_mask[i, j] = True

    # Greedy construction
    uncovered = np.ones(total_words, dtype=bool)
    code = []
    
    while np.any(uncovered):
        if time.time() > deadline:
            break
            
        # Find the codeword that covers the most uncovered words
        # gains = cover_mask @ uncovered
        # best_idx = np.argmax(gains)
        
        # To save memory/time, we can compute gains in chunks or just use the dot product.
        # total_words is 2187, so cover_mask is 2187x2187.
        # Dot product is fast.
        gains = np.dot(cover_mask, uncovered)
        
        best_idx = np.argmax(gains)
        
        # If no gain, we have a problem (should not happen if radius >= 0 and space is finite)
        if gains[best_idx] == 0:
            break
            
        code.append(best_idx)
        
        # Update uncovered
        uncovered[cover_mask[best_idx]] = False
        
        # Optional: Check for early termination or improvements?
        # For now, just greedy.

    # Post-processing: Try to remove redundant codewords
    # Sort code by some heuristic? Or just try to remove each one.
    # If removing a codeword leaves the code still covering everything, remove it.
    
    def is_valid(code_indices):
        if not code_indices:
            return False
        # Create a boolean array of covered words
        # We can sum the cover masks of the selected codewords
        # If any sum > 0, it's covered.
        # But we need to check if ALL words are covered.
        # covered_count = np.sum(cover_mask[code_indices], axis=0)
        # return np.all(covered_count > 0)
        
        # More efficient:
        # Initialize a boolean array
        covered = np.zeros(total_words, dtype=bool)
        for idx in code_indices:
            covered |= cover_mask[idx]
        return np.all(covered)

    # Try to remove codewords one by one
    # We can repeat this process until no more removals are possible.
    changed = True
    while changed and time.time() < deadline:
        changed = False
        new_code = []
        for idx in code:
            # Try removing idx
            if is_valid([x for x in code if x != idx]):
                # It is safe to remove
                changed = True
                # Do not add idx to new_code
            else:
                new_code.append(idx)
        code = new_code

    # Convert indices back to lists of integers
    result = [index_to_digits(idx, n, q) for idx in code]
    
    return result
