import time
import random

def construct():
    N = 17
    MIN_DIST = 6
    MAX_WORDS = 2**N
    # Use a safety margin of 80s
    deadline = time.time() + 80.0
    
    best_words = []
    best_size = 0
    
    # List of all possible non-zero words
    # We always include 0 in the code
    candidates = list(range(1, MAX_WORDS))
    
    while time.time() < deadline:
        random.shuffle(candidates)
        current_words = [0]
        
        for c in candidates:
            # Check if c can be added
            can_add = True
            for cw in current_words:
                if (c ^ cw).bit_count() < MIN_DIST:
                    can_add = False
                    break
            
            if can_add:
                current_words.append(c)
                
        if len(current_words) > best_size:
            best_size = len(current_words)
            best_words = current_words
            
            # Optional: Early stop if we hit theoretical max or a very high value
            if best_size >= 340:
                break

    return best_words
