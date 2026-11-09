def construct():
    # Base Reed-Muller code (32 words)
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    
    # Convert to set for fast lookup
    word_set = set(words)
    
    # Greedy addition
    # We iterate through all possible 18-bit words
    # To be efficient, we only check candidates that are not in word_set
    # and have distance >= 6 from all current words.
    
    # Optimization: 
    # Instead of checking against all words, we can just check against the set.
    # But checking against all 512 words for 262k candidates is 131 million ops.
    # 131M ops in Python might take ~10-20 seconds.
    # This is feasible within 60s.
    
    # We will iterate from 0 to 2**18 - 1
    # But we can skip words already in the set.
    
    # To improve ordering, we can prioritize words with higher weight?
    # Or just sequential.
    
    # We will add a word if it is valid.
    
    # We must ensure we don't exceed time.
    # We'll use a simple loop.
    
    # Note: The base 32 words are already in word_set.
    
    # We iterate through all possible words
    for w in range(2**18):
        if w in word_set:
            continue
            
        # Check distance against all existing words
        valid = True
        for existing in word_set:
            if (w ^ existing).bit_count() < 6:
                valid = False
                break
        
        if valid:
            word_set.add(w)
            # If we reach 512, we can stop? 
            # Or keep going to maximize?
            # The task asks for the largest possible.
            # But 512 is likely the max.
            # If we find 512, we can break to save time?
            # Or continue to see if we can go higher?
            # Given the Hamming bound is 1524, we might go higher.
            # But the hint says 512 is a reference.
            # I will continue to try to find more, but with a time check?
            # Python doesn't have easy time checks in a tight loop without overhead.
            # I will just run the full loop.
            
    return list(word_set)
