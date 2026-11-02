import itertools
import time

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Total number of words
    total_words = q ** n
    
    # Precompute all words as tuples
    all_words = list(itertools.product(range(q), repeat=n))
    
    # Map word to index
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute ball masks for each word
    # ball_mask[i] is a bitmask of all words within 'radius' of all_words[i]
    ball_masks = [0] * total_words
    
    # Helper to get index from word
    def get_idx(word):
        return word_to_idx[word]
        
    # Calculate Hamming distance
    def hamming_dist(w1, w2):
        return sum(1 for a, b in zip(w1, w2) if a != b)
        
    # Precomputation step
    for i, w in enumerate(all_words):
        mask = 0
        # Iterate through all words to check distance? 
        # 729 * 729 = 531k checks. Fast enough.
        for j, v in enumerate(all_words):
            if hamming_dist(w, v) <= radius:
                mask |= (1 << j)
        ball_masks[i] = mask
        
    # Initialize uncovered set as all 1s
    # Using an integer as a bitset
    uncovered = (1 << total_words) - 1
    
    code = []
    
    # Symmetry breaking: Fix the first codeword to all zeros
    # This is valid because the automorphism group is transitive on the space.
    first_word = (0,) * n
    first_idx = get_idx(first_word)
    code.append(list(first_word))
    uncovered &= ~ball_masks[first_idx]
    
    # Greedy Set Cover
    while uncovered != 0:
        if time.time() > deadline:
            break
            
        best_idx = -1
        best_count = 0
        
        # Find the word that covers the most uncovered words
        # Optimization: We can skip words that are already in the code? 
        # No, adding a word already in code is useless, but its mask is already subtracted.
        # Actually, if a word is in code, its ball is already covered. 
        # But we just need to pick a new word.
        # We can iterate through all indices.
        
        for i in range(total_words):
            # Optional: skip if this word is already in code? 
            # It won't add new coverage, so count will be 0.
            # But checking membership is O(1) with a set.
            # Let's just compute count.
            covered = uncovered & ball_masks[i]
            count = covered.bit_count()
            if count > best_count:
                best_count = count
                best_idx = i
                # Early exit if we cover all? Unlikely in one step unless r is large.
                # if count == uncovered.bit_count(): break # Not possible to cover all remaining with one word usually
                
        if best_idx == -1:
            break
            
        # Add the best word
        code.append(list(all_words[best_idx]))
        uncovered &= ~ball_masks[best_idx]
        
    return code
