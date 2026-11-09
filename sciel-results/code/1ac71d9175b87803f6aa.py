"""Construct binary error-correcting code of length 18, distance 6 via greedy + local search."""
import random
import numpy as np
from itertools import combinations

N = 18
MIN_DIST = 6

def popcount(x):
    return bin(x).count('1')

def hamming(x, y):
    return popcount(x ^ y)

def min_dist_to_set(word, code):
    """Return minimum Hamming distance from word to any codeword in code."""
    md = N + 1
    for c in code:
        d = popcount(word ^ c)
        if d < md:
            md = d
            if md < MIN_DIST:
                break
    return md

def greedy_add(code, word):
    """Try to add word to code, return True if successful."""
    for c in code:
        if popcount(word ^ c) < MIN_DIST:
            return False
    code.append(word)
    return True

def local_search_improve(code, max_iters=200):
    """Try to add more codewords via local search: mutate random words and try to add."""
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        # Try to add a new codeword by mutating a random base
        base = random.randint(0, (1 << N) - 1)
        # Try adding base directly
        if min_dist_to_set(base, code) >= MIN_DIST:
            code.append(base)
            improved = True
            continue
        # Try single-bit flips
        for i in range(N):
            w = base ^ (1 << i)
            if min_dist_to_set(w, code) >= MIN_DIST:
                code.append(w)
                improved = True
                break
        if improved:
            continue
        # Try double-bit flips
        for i in range(N):
            for j in range(i+1, N):
                w = base ^ (1 << i) ^ (1 << j)
                if min_dist_to_set(w, code) >= MIN_DIST:
                    code.append(w)
                    improved = True
                    break
            if improved:
                break
    return code

def greedy_construction(seed):
    """Greedy construction with random ordering."""
    rng = random.Random(seed)
    # Start with 0
    code = [0]
    # Generate candidate words in random order
    candidates = list(range(1, 1 << N))
    rng.shuffle(candidates)
    for w in candidates:
        if min_dist_to_set(w, code) >= MIN_DIST:
            code.append(w)
            if len(code) >= 512:  # Early stop if we hit reference size
                break
    return code

def construct():
    """Construct the largest possible binary code of length 18, distance 6."""
    best_code = []
    
    # Strategy 1: Greedy with different random seeds
    for seed in range(10):
        code = greedy_construction(seed)
        # Local search to improve
        code = local_search_improve(code, max_iters=50)
        if len(code) > len(best_code):
            best_code = code
    
    # Strategy 2: Start from Reed-Muller and extend
    # First-order Reed-Muller RM(1,4) has 32 words, distance 8
    rm_code = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            rm_code.append(word)
    
    # Try to extend RM code with additional words
    extended = list(rm_code)
    rng = random.Random(42)
    candidates = list(range(1 << N))
    rng.shuffle(candidates)
    for w in candidates:
        if w in extended:
            continue
        if min_dist_to_set(w, extended) >= MIN_DIST:
            extended.append(w)
            if len(extended) >= 512:
                break
    
    extended = local_search_improve(extended, max_iters=100)
    if len(extended) > len(best_code):
        best_code = extended
    
    # Strategy 3: More aggressive local search from best
    best_code = local_search_improve(best_code, max_iters=200)
    
    # Ensure we have at least the RM code
    if len(best_code) < 32:
        best_code = rm_code
    
    return best_code
