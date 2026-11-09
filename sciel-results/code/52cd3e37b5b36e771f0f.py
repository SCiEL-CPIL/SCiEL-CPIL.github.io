def construct():
    import time
    import numpy as np
    
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s kill
    
    N = 17
    MIN_D = 6
    
    # Precompute weights for all 2^17 words using numpy
    all_words = np.arange(1 << N, dtype=np.int64)
    # Efficient weight calculation: count bits
    # We can use a lookup table or bit manipulation
    # For simplicity and speed with numpy, let's use a helper
    # But 2^17 is small, so even a python loop might be ok if vectorized
    
    # Let's use a vectorized approach for weight calculation
    # Create a 17-bit mask
    # Actually, let's just use the standard python method but optimized
    # Or use numpy's popcount if available? No, standard numpy doesn't have popcount.
    # Let's use a bit trick:
    # x = x - ((x >> 1) & 0x55555555)
    # ... this is complex.
    
    # Simpler: Use a precomputed table for 8-bit chunks
    # 2^17 = 131072.
    # Let's create a table for 0..255
    pop_table = np.array([bin(i).count('1') for i in range(256)], dtype=np.int8)
    
    # Vectorized weight calculation
    w = all_words
    weights = (pop_table[w & 0xFF] + 
               pop_table[(w >> 8) & 0xFF] + 
               pop_table[(w >> 16) & 0xFF])
    
    # Greedy linear code construction
    # We want to find a basis of size 8 (for 256 words)
    # But we can also try to add non-linear words later.
    
    # Strategy:
    # 1. Find a good linear code (up to 256 words)
    # 2. Try to add non-linear words to expand the code
    
    # For the linear code, we scan candidates.
    # To speed up, we can prioritize candidates with higher weight?
    # Or just scan in order.
    
    # Let's use a greedy approach with numpy for fast distance checks.
    # current_code is a numpy array of codewords.
    # When adding a new candidate g, we check min distance to all existing codewords.
    
    current_code = np.array([0], dtype=np.int64)
    basis = []
    
    # We want to add up to 8 basis vectors.
    # But scanning 131k candidates for each of 8 steps is 1M checks.
    # Each check is a numpy operation of size 2^k.
    # Total operations: sum_{k=0}^7 131072 * 2^k = 131072 * (2^8 - 1) = 131072 * 255 = 33M.
    # This is fast in numpy.
    
    # However, we need to find the *first* valid candidate or a *good* one.
    # If we scan in order 1..131071, we might find a bad basis early.
    # Let's try to find a basis that allows for 8 vectors.
    
    # To improve the chance of finding a large code, we can use a heuristic:
    # Pick candidates that are "far" from existing ones.
    # But for linear codes, any basis works if it's valid.
    
    # Let's just scan.
    for g in range(1, 1 << N):
        if time.time() > deadline:
            break
            
        # Check if g can be added
        # We need wt(g ^ c) >= MIN_D for all c in current_code
        # This is equivalent to min(weights[g ^ current_code]) >= MIN_D
        
        # Compute distances
        xor_vals = g ^ current_code
        min_dist = np.min(weights[xor_vals])
        
        if min_dist >= MIN_D:
            # Add g to basis
            basis.append(g)
            # Update current_code
            # New code = current_code union (current_code ^ g)
            new_code = np.concatenate([current_code, current_code ^ g])
            current_code = new_code
            
            if len(basis) >= 8:
                break
    
    # Now we have a linear code of size 2^k.
    # If k=8, we have 256 words.
    # If k<8, we have fewer.
    
    # Try to add non-linear words.
    # We want to add words that are at distance >= MIN_D from all existing words.
    # This is a greedy packing problem.
    
    # We can iterate through all candidates and add them if they fit.
    # But we must be careful not to add words that are in the linear code.
    # Also, we should prioritize words that "fit" well.
    
    # To speed up, we can use a vectorized check.
    # But adding one word at a time is O(M * N) where M is current size.
    # If we add many words, this becomes slow.
    
    # Let's try to add as many as possible.
    # We'll iterate through candidates in a random order or sorted by weight?
    # High weight words are more likely to be far from low weight words.
    
    # Let's create a list of all words not in current_code.
    # But checking membership is slow.
    
    # Instead, let's just iterate through all words and check if they can be added.
    # We'll keep a set of added words for fast lookup.
    
    added_set = set(current_code.tolist())
    
    # To make the greedy step faster, we can use a "forbidden" mask.
    # A word w is forbidden if wt(w ^ c) < MIN_D for some c in code.
    # We can precompute the forbidden set?
    # For each c in code, the forbidden words are those within distance < MIN_D.
    # This is a ball of radius MIN_D-1.
    # The size of a ball of radius 5 in 17 dimensions is sum_{i=0}^5 C(17,i) = 1+17+136+680+2380+6188 = 9402.
    # So each codeword forbids ~9400 words.
    # With 256 codewords, that's 256 * 9400 = 2.4M forbidden entries.
    # This is manageable.
    
    # Let's create a boolean array `forbidden` of size 2^N.
    # forbidden[w] = True if w is within distance < MIN_D of any codeword.
    
    forbidden = np.zeros(1 << N, dtype=bool)
    
    # For each codeword, mark the ball of radius MIN_D-1 as forbidden.
    # We can do this by iterating over all words and checking distance? No, too slow.
    # We need to generate the ball efficiently.
    
    # Alternative: For each candidate w, check if it's forbidden.
    # This is what we did before.
    
    # Let's try a different approach:
    # We'll iterate through all words in a specific order (e.g., decreasing weight) and add them if not forbidden.
    
    # Order by weight (descending)
    # But sorting 131k items is fast.
    order = np.argsort(-weights)
    
    # Now iterate through order and try to add words.
    # But we must check against the *current* code, which grows.
    
    # This is a greedy algorithm.
    # We'll use the `forbidden` array to track which words are already "covered" by the code.
    # Initially, we mark the balls around the linear code words as forbidden.
    
    # Let's precompute the balls.
    # For each c in current_code:
    #   for each w in all_words:
    #     if weights[w ^ c] < MIN_D:
    #       forbidden[w] = True
    
    # This is O(|code| * 2^N) = 256 * 131072 = 33M operations.
    # This is fast in numpy if vectorized.
    
    # But we can do it even faster.
    # We know the distance is < MIN_D.
    # We can iterate over all masks with weight < MIN_D and mark c ^ mask as forbidden.
    
    # Precompute all masks with weight < MIN_D.
    masks = []
    for m in range(1 << N):
        if weights[m] < MIN_D:
            masks.append(m)
    
    masks = np.array(masks, dtype=np.int64)
    
    # Now for each c in current_code, mark c ^ masks as forbidden.
    # We can do this in a loop over current_code.
    
    for c in current_code:
        # Mark c ^ masks as forbidden
        # But we must ensure we don't go out of bounds? No, XOR stays in range.
        forbidden[c ^ masks] = True
    
    # Now iterate through candidates in order of decreasing weight (or random) and add if not forbidden.
    # We should also update the forbidden set when we add a new word.
    
    # To speed up the update, we can just mark the new ball as forbidden.
    
    # Let's iterate through all words in `order` (decreasing weight).
    # If a word is not forbidden, add it and mark its ball as forbidden.
    
    for w in order:
        if time.time() > deadline:
            break
        
        if not forbidden[w]:
            # Add w to code
            added_set.add(w)
            # Mark w ^ masks as forbidden
            forbidden[w ^ masks] = True
    
    # Convert back to list of integers
    result = list(added_set)
    
    return result
