import time
import random

def construct():
    deadline = time.time() + 55  # Safety margin for 60s limit
    
    N = 18
    MIN_DIST = 6
    
    # 1. Initial Reed-Muller [16,5,8] code (extended with 2 zeros to length 18? 
    #    The seed code used 16 bits. The task says 18-bit words.
    #    The seed code put the 16-bit pattern in the lower 16 bits.
    #    This is a valid [18,5,8] code? No, distance is still 8.
    #    We can use the 2 extra bits to increase size or distance.
    #    Since we want max size, we start with these 32 and try to add more.
    
    base_words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            base_words.append(word)
            
    code = list(base_words)
    code_set = set(code)
    
    # Generate all possible words
    all_words = list(range(2**N))
    
    # Shuffle to avoid bias
    random.shuffle(all_words)
    
    # Greedy addition
    for w in all_words:
        if time.time() > deadline:
            break
            
        if w in code_set:
            continue
            
        # Check distance to all existing codewords
        valid = True
        for cw in code:
            if (w ^ cw).bit_count() < MIN_DIST:
                valid = False
                break
                
        if valid:
            code.append(w)
            code_set.add(w)
            
    # If we have time, try a second pass with a different shuffle?
    # Or just return what we have.
    # Given the Hamming bound allows ~1500, and we start with 32, 
    # a single greedy pass might get to 50-100.
    # Let's add a second pass if time permits.
    
    if time.time() < deadline - 10:
        random.shuffle(all_words)
        for w in all_words:
            if time.time() > deadline:
                break
            if w in code_set:
                continue
            valid = True
            for cw in code:
                if (w ^ cw).bit_count() < MIN_DIST:
                    valid = False
                    break
            if valid:
                code.append(w)
                code_set.add(w)
                
    return code
