import time
import random

def construct():
    """Construct a large binary code of length 17 with minimum distance 6."""
    deadline = time.time() + 85  # Safety margin before 90s limit
    
    N = 17
    MIN_DIST = 6
    TOTAL_CANDIDATES = 2 ** N
    
    # Generate the first-order Reed-Muller code RM(1,4) on 16 coordinates
    # This has 32 words, length 16, min distance 8.
    # We pad with a 0 bit to make length 17.
    seed_words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            # Bit 16 is 0
            seed_words.append(word)
    
    seed_set = set(seed_words)
    
    def greedy_expand(initial_words, order):
        """
        Expand a set of words by greedily adding candidates from 'order'.
        Returns the final list of words.
        """
        words = list(initial_words)
        word_set = set(words)
        
        for cand in order:
            if cand in word_set:
                continue
            
            # Check Hamming distance to all existing words
            valid = True
            for w in words:
                if (cand ^ w).bit_count() < MIN_DIST:
                    valid = False
                    break
            
            if valid:
                words.append(cand)
                word_set.add(cand)
                
        return words

    def local_search(current_words, time_limit):
        """
        Try to improve the code size by swapping words and adding new ones.
        """
        words = list(current_words)
        word_set = set(words)
        best_words = list(words)
        best_size = len(words)
        
        # Pre-compute list of all non-codewords for random selection
        # This is expensive, so we just pick random integers and check
        
        start_time = time.time()
        
        while time.time() < time_limit:
            # Pick a random word to remove
            idx = random.randint(0, len(words) - 1)
            w_remove = words[idx]
            
            # Remove it
            words.pop(idx)
            word_set.remove(w_remove)
            
            # Try to add a random candidate
            # We pick a random integer from the full space
            # To increase chance of finding a good one, we might bias towards 
            # words that are "close" to the removed word? No, random is fine.
            added_any = False
            for _ in range(10): # Try a few candidates
                cand = random.randint(0, TOTAL_CANDIDATES - 1)
                if cand in word_set:
                    continue
                
                # Check validity
                valid = True
                for w in words:
                    if (cand ^ w).bit_count() < MIN_DIST:
                        valid = False
                        break
                
                if valid:
                    words.append(cand)
                    word_set.add(cand)
                    added_any = True
                    break
            
            # If we added a word, size increased or stayed same (if we added 1, removed 1)
            # Actually, we removed 1. If we added 1, size is same.
            # We need to try to add MORE words now that space is rearranged.
            
            if len(words) > best_size:
                best_words = list(words)
                best_size = len(words)
            
            # Try to greedily add any valid words to the current state
            # This is a bit expensive, so we do it sparingly
            # Just a few random additions
            for _ in range(5):
                cand = random.randint(0, TOTAL_CANDIDATES - 1)
                if cand in word_set:
                    continue
                valid = True
                for w in words:
                    if (cand ^ w).bit_count() < MIN_DIST:
                        valid = False
                        break
                if valid:
                    words.append(cand)
                    word_set.add(cand)
                    if len(words) > best_size:
                        best_words = list(words)
                        best_size = len(words)

        return best_words

    best_code = []
    best_size = 0
    
    # Run multiple greedy passes with different random orderings
    # Each pass takes ~1-2 seconds. We can run ~30-40 passes.
    
    while time.time() < deadline - 10: # Keep 10s for local search
        # Create a random order of all candidates
        # To save time creating a full list and shuffling, we can just 
        # iterate through range and use a random skip? No, shuffle is fine.
        # 131072 elements shuffle is fast.
        order = list(range(TOTAL_CANDIDATES))
        random.shuffle(order)
        
        current_code = greedy_expand(seed_words, order)
        current_size = len(current_code)
        
        if current_size > best_size:
            best_size = current_size
            best_code = current_code
            
        # If we found a very large code, we can stop early
        if best_size >= 258:
            break
            
    # If we have time, run local search on the best code found
    if time.time() < deadline - 2 and best_code:
        improved_code = local_search(best_code, deadline - 2)
        if len(improved_code) > best_size:
            best_code = improved_code
            best_size = len(improved_code)
            
    return best_code
