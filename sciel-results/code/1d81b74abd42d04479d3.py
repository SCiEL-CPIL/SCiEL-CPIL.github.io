import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 90s kill
    
    N = 17
    MIN_DIST = 6
    
    words = []
    
    # Phase 1: Add RM(1,4) words (32 words, distance 8 on 16 bits)
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    
    # Phase 2: Greedy addition - try all candidates in forward order
    def can_add(candidate, words_list):
        for w in words_list:
            if (candidate ^ w).bit_count() < MIN_DIST:
                return False
        return True
    
    for candidate in range(1, 1 << N):
        if time.time() > deadline - 15.0:
            break
        if can_add(candidate, words):
            words.append(candidate)
    
    # Phase 3: Try reverse order to catch missed candidates
    if time.time() < deadline - 15.0:
        for candidate in range((1 << N) - 1, 0, -1):
            if time.time() > deadline - 5.0:
                break
            if can_add(candidate, words):
                words.append(candidate)
    
    # Phase 4: Try a few more passes with shuffled-like orderings
    if time.time() < deadline - 10.0:
        # Try candidates with weight in a specific range
        for target_weight in range(1, N):
            if time.time() > deadline - 5.0:
                break
            for candidate in range(1, 1 << N):
                if candidate.bit_count() != target_weight:
                    continue
                if time.time() > deadline - 5.0:
                    break
                if can_add(candidate, words):
                    words.append(candidate)
    
    return words
