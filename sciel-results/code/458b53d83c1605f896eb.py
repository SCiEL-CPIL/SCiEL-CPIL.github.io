import time
import numpy as np

def construct(q, n, radius):
    deadline = time.time() + 115
    
    # Precompute powers of q
    pow_q = np.array([q**i for i in range(n)], dtype=np.int64)
    total = q**n
    
    # Generate all words as integers
    all_words = np.arange(total, dtype=np.int64)
    
    # Convert word integer to vector
    digits = np.zeros((total, n), dtype=np.int64)
    tmp = all_words.copy()
    for i in range(n):
        digits[:, i] = tmp % q
        tmp //= q
        
    # Precompute the list of words covered by each codeword
    # For radius 1, q=3, n=8: 1 + 8*2 = 17 words
    num_covered = 1 + n * (q - 1)
    cover_indices = np.zeros((total, num_covered), dtype=np.int64)
    
    for c in range(total):
        d = digits[c]
        idx = 0
        cover_indices[c, idx] = c
        idx += 1
        for i in range(n):
            for v in range(q):
                if v != d[i]:
                    cover_indices[c, idx] = c + (v - d[i]) * pow_q[i]
                    idx += 1
                    
    # Build reverse mapping: word -> list of codewords covering it
    # Using a list of lists for flexibility, or a sparse matrix if needed.
    # Given small size, list of lists is fine.
    word_to_codewords = [[] for _ in range(total)]
    for c in range(total):
        for w in cover_indices[c]:
            word_to_codewords[int(w)].append(c)

    # Greedy Set Cover using NumPy for argmax
    selected = []
    selected_bool = np.zeros(total, dtype=bool)
    # uncovered_counts[c] = number of words in cover_indices[c] that are not yet covered
    uncovered_counts = np.full(total, num_covered, dtype=np.int32)
    word_covered = np.zeros(total, dtype=bool)
    
    for step in range(total):
        if time.time() > deadline - 5:
            break
            
        # Find best codeword: max uncovered count among unselected
        # Mask selected codewords to -1 so they are not chosen
        # We can do this by setting their count to -1 temporarily or using np.where
        # Efficient way: keep an array of counts, set selected to -1
        # But we need to restore? No, we don't need counts for selected anymore.
        
        # To avoid modifying the array and restoring, we can just use np.argmax on a view
        # or simply:
        # counts_to_check = uncovered_counts.copy()
        # counts_to_check[selected_bool] = -1
        # best_c = np.argmax(counts_to_check)
        
        # Even faster: just use np.argmax on uncovered_counts, but ensure selected are low.
        # Let's just update uncovered_counts[selected] = -1 when we select.
        
        best_c = int(np.argmax(uncovered_counts))
        
        if uncovered_counts[best_c] <= 0:
            break # All covered or no valid candidate
        
        # Select best_c
        selected_bool[best_c] = True
        selected.append(best_c)
        uncovered_counts[best_c] = -1 # Mark as selected/invalid for future argmax
        
        # Update coverage
        for w in cover_indices[best_c]:
            w_int = int(w)
            if not word_covered[w_int]:
                word_covered[w_int] = True
                # Decrement counts for all codewords covering w_int
                for c_prime in word_to_codewords[w_int]:
                    if not selected_bool[c_prime]:
                        uncovered_counts[c_prime] -= 1
                        
    # Local Search: Redundancy Removal
    # Check if any selected codeword is redundant
    # A codeword c is redundant if for all w in cover_indices[c], cover_count[w] > 1
    
    # Re-calculate cover counts from scratch for safety and simplicity, 
    # or maintain them. Maintaining is harder with the above update logic 
    # because we didn't track counts per word, only per codeword.
    # Let's rebuild cover_count for the selected set.
    cover_count = np.zeros(total, dtype=np.int32)
    for c in selected:
        for w in cover_indices[c]:
            cover_count[int(w)] += 1
            
    changed = True
    while changed and time.time() < deadline - 1:
        changed = False
        # Iterate in reverse to handle indices safely
        for i in range(len(selected) - 1, -1, -1):
            if time.time() > deadline - 1:
                break
            c = selected[i]
            is_redundant = True
            for w in cover_indices[c]:
                if cover_count[int(w)] <= 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                # Remove c
                for w in cover_indices[c]:
                    cover_count[int(w)] -= 1
                selected_bool[c] = False
                selected.pop(i)
                changed = True
                
    # Convert selected indices to codeword vectors
    if not selected:
        # Fallback: return all words (should not happen for valid covering)
        return [digits[i].tolist() for i in range(total)]
    
    selected_codes = [digits[c].tolist() for c in selected]
    
    return selected_codes
