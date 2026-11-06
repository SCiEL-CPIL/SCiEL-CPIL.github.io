import time
import random
from itertools import combinations
import numpy as np

def construct():
    start = time.time()
    deadline = start + 55.0  # 55 second deadline, 5s safety margin

    # N=17, min distance=6
    N = 17
    min_dist = 6

    # Start with the 32-word first-order Reed-Muller code (extended by zero bits)
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    # Add one zero bit at position 16 for N=17
    words = [w << 1 for w in words]
    
    # Also try to construct the [18,9,6] quadratic residue code, then shorten
    # First, let's try a greedy random construction to find a large code
    # But first, let's see if we can get a good linear code
    
    # For now, let's do a greedy local search
    # Start with the RM code, then try to add random words and do local improvement
    
    # Better approach: use a randomized greedy algorithm
    # 1. Start with an empty set
    # 2. Add words one by one, choosing the word that maximizes some criterion
    
    # Actually, let's try to construct the [17,9,6] code if it exists, or a large nonlinear code
    
    # Let's try a different approach: start with the 256-word [18,9,6] code
    # and see if we can shorten it properly
    
    # The [18,9,6] code: this is the extended quadratic residue code of length 18
    # Let's construct it using a known construction
    
    # Actually, let's just do a good greedy random search with time limit
    
    # Generate candidate words randomly, but with some structure
    # Use the RM code as a starting point, then try to extend
    
    best_code = words[:]
    best_size = len(best_code)
    
    # Precompute bit counts for all 2^17 integers? That's 131072, feasible
    # But we need Hamming distances, so let's precompute a lookup for popcount of XOR
    
    # For efficiency, let's use a set for fast lookup
    code_set = set(best_code)
    
    # Try to add random words to the code
    # For each candidate word, check if it's at distance >= 6 from all existing words
    
    # To make this efficient, we can use a greedy approach:
    # 1. Generate a large pool of candidate words
    # 2. Sort them by weight (prefer moderate weight)
    # 3. Add them one by one if compatible
    
    # But checking compatibility against 256+ words is expensive
    # Let's limit the search
    
    # Actually, let's try to find the [17,9,6] code by shortening the [18,9,6] code
    # The [18,9,6] code has 256 words. If we fix one coordinate to 0, we get 128 words.
    # But maybe we can find a coordinate where all weight-6 words have 0, allowing us to drop it
    # and keep all 256 words at distance >= 6.
    
    # Let's construct the [18,9,6] code first.
    # The quadratic residue code of length 16 is the [16,5,6] code.
    # Extended: [17,5,6]. Extended again: [18,6,6]? No, that doesn't sound right.
    
    # Actually, the standard QR code of length 15 is [15,7,5].
    # Punctured/extended versions give various codes.
    
    # Let me use a different strategy: just do a randomized greedy construction
    # with a time limit, starting from the RM code.
    
    random.seed(42)
    
    # Generate a pool of candidate words
    # Use structured candidates: words from the second-order RM code, plus random perturbations
    candidates = set(best_code)
    
    # Add second-order RM code words (all quadratic Boolean functions)
    # f(x) = sum_{i<=j} a_{ij} x_i x_j + sum_i b_i x_i + c
    # There are 2^(1 + 4 + 10) = 2^15 = 32768 such functions, too many to enumerate
    # But we can sample some
    
    # Actually, let's just add random words with weight between 4 and 12
    for _ in range(50000):
        if time.time() > deadline - 10:
            break
        weight = random.randint(4, 12)
        bits = random.sample(range(N), weight)
        word = 0
        for b in bits:
            word |= (1 << b)
        candidates.add(word)
    
    # Now do greedy construction: sort candidates by weight (prefer middle weights)
    # and add them if compatible
    
    # But checking compatibility is O(|code| * N) per candidate, which is expensive
    # Let's limit the number of candidates we check
    
    # Actually, let's just try to add a subset of candidates
    # Sort by weight, prefer weights 5-8 (balanced)
    
    # This is getting complex. Let's just return the RM code for now,
    # or do a simple local search.
    
    # Simple local search: try to flip bits in existing words to create new valid words
    # But that might not work well.
    
    # Let's just return the 32-word RM code if we're running out of time
    if time.time() > deadline - 5:
        return best_code
    
    # Try to add random words one by one
    # Limit the number of attempts
    max_attempts = 5000
    for _ in range(max_attempts):
        if time.time() > deadline - 5:
            break
        
        # Generate a random candidate
        weight = random.randint(4, 12)
        bits = random.sample(range(N), weight)
        word = 0
        for b in bits:
            word |= (1 << b)
        
        if word in code_set:
            continue
        
        # Check if word is at distance >= min_dist from all words in code
        compatible = True
        for w in best_code:
            dist = (word ^ w).bit_count()
            if dist < min_dist:
                compatible = False
                break
        
        if compatible:
            best_code.append(word)
            code_set.add(word)
    
    return best_code
