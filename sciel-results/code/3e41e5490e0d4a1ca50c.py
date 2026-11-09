import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    N = 17
    MIN_DIST = 6
    MAX_SPACE = 1 << N
    
    # Precompute all masks with bit_count <= 5
    # These are the vectors within distance 5
    masks = []
    for m in range(MAX_SPACE):
        if m.bit_count() <= 5:
            masks.append(m)
            
    # words list and forbidden array
    words = []
    forbidden = [False] * MAX_SPACE
    
    def add_word(w):
        if forbidden[w]:
            return False
        # Mark w and its neighbors as forbidden
        for m in masks:
            idx = w ^ m
            if idx < MAX_SPACE: # Always true for XOR
                forbidden[idx] = True
        words.append(w)
        return True

    # 1. Initialize with RM(1,4) code (32 words)
    # These are affine functions on F_2^4.
    # We embed them in 17 bits (last bit 0, or just use 16 bits and 0 for 17th).
    # The RM code on 16 coords has distance 8.
    # In 17 coords (with 0 padding), distance is still 8.
    # This is a valid code.
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            # word is 16 bits. 17th bit is 0.
            add_word(word)
            
    # 2. Greedy fill
    # We will iterate through the space.
    # To avoid bias, we can shuffle the order or just use range.
    # Using range is deterministic.
    # Let's do multiple passes.
    
    while time.time() < deadline:
        added_any = False
        # Iterate through all possible words
        # We can optimize by only checking those not yet forbidden, 
        # but checking `if not forbidden[w]` is O(1).
        
        # To make it more robust, we can try to add words in a random order 
        # if the deterministic order gets stuck, but usually deterministic 
        # greedy on a good seed is strong.
        
        # Let's iterate 0 to MAX_SPACE-1
        for w in range(MAX_SPACE):
            if not forbidden[w]:
                if add_word(w):
                    added_any = True
            # Check time periodically to avoid exceeding deadline mid-loop
            if (w & 0xFF) == 0 and time.time() > deadline:
                break
        
        if not added_any:
            # No more words can be added with this greedy strategy
            # We could try a local search (swap) here, but 
            # for now, let's see if we get a good number.
            # If we have time, we can try random restarts of the greedy process?
            # No, the current set is already built.
            # A simple improvement: try to remove a word and see if 2 can be added?
            # This is complex.
            
            # Let's just break if we can't add more.
            break
            
    return words
