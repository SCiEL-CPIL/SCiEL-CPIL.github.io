"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s out of 90s budget
    
    N = 18
    MIN_DIST = 6
    
    # First-order Reed-Muller code on 16 coordinates, extended by two zero bits.
    # Every affine Boolean function on F_2^4 is a word: 32 words, distance 8.
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    
    # Try to extend the code by adding words greedily
    # We'll try random words and check if they can be added
    rng = random.Random(42)
    
    def can_add(w, words):
        for existing in words:
            if (w ^ existing).bit_count() < MIN_DIST:
                return False
        return True
    
    # First, try to add all weight-6 words systematically
    # But that's too many (18C6 = 18564), so let's be smart about it
    
    # Try adding random words
    attempts = 0
    while time.time() < deadline - 5:
        # Try random words with various weights
        for weight in range(3, 16):
            w = 0
            bits = list(range(N))
            rng.shuffle(bits)
            for b in bits[:weight]:
                w |= (1 << b)
            if can_add(w, words):
                words.append(w)
        
        attempts += 1
        if attempts % 100 == 0:
            break
    
    # Also try systematic search for low-weight words
    # Weight 3: C(18,3) = 816
    from itertools import combinations
    for combo in combinations(range(N), 3):
        w = 0
        for b in combo:
            w |= (1 << b)
        if can_add(w, words):
            words.append(w)
    
    # Weight 4: C(18,4) = 3060
    for combo in combinations(range(N), 4):
        if time.time() > deadline - 5:
            break
        w = 0
        for b in combo:
            w |= (1 << b)
        if can_add(w, words):
            words.append(w)
    
    # Weight 5: C(18,5) = 8568
    for combo in combinations(range(N), 5):
        if time.time() > deadline - 5:
            break
        w = 0
        for b in combo:
            w |= (1 << b)
        if can_add(w, words):
            words.append(w)
    
    # Weight 6: C(18,6) = 18564
    for combo in combinations(range(N), 6):
        if time.time() > deadline - 5:
            break
        w = 0
        for b in combo:
            w |= (1 << b)
        if can_add(w, words):
            words.append(w)
    
    return words
# EVOLVE-BLOCK-END
