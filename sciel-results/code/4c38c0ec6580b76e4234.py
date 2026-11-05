"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np


def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius `radius`.
    For the canonical task q=3, n=6, radius=1, we aim for ~73 words.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    if q != 3 or n != 6 or radius != 1:
        # Fallback: simple greedy for other parameters
        return _greedy_fallback(q, n, radius, start_time, deadline)

    # Specialized for q=3, n=6, radius=1
    total_words = 3 ** n  # 729
    ball_size = 1 + n * (q - 1)  # 1 + 6*2 = 13

    # Precompute all words as tuples and their integer representations
    all_words = list(itertools.product(range(3), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute balls: for each word index, list of indices within Hamming distance <= radius
    balls = []
    for w in all_words:
        ball = [word_to_idx[w]]
        for pos in range(n):
            orig = w[pos]
            for new_val in range(3):
                if new_val != orig:
                    new_w = list(w)
                    new_w[pos] = new_val
                    ball.append(word_to_idx[tuple(new_w)])
        balls.append(ball)

    # Convert balls to numpy arrays for faster operations
    balls_np = [np.array(b, dtype=np.int32) for b in balls]

    # Greedy construction: repeatedly pick the codeword that covers the most uncovered words
    uncovered = np.ones(total_words, dtype=np.int32)
    code = []
    
    # Precompute for each potential codeword which words it covers
    # We'll use a priority approach
    
    # Initial: pick codeword covering most uncovered
    while uncovered.sum() > 0 and time.time() < deadline - 30.0:
        # Find codeword that covers most uncovered words
        best_idx = -1
        best_count = -1
        for idx in range(total_words):
            if uncovered[idx] == 0:
                continue
            # Count how many uncovered words are in this ball
            b = balls_np[idx]
            count = uncovered[b].sum()
            if count > best_count:
                best_count = count
                best_idx = idx
        
        if best_idx == -1:
            break
        
        # Add this codeword
        code.append(all_words[best_idx])
        # Update uncovered
        b = balls_np[best_idx]
        uncovered[b] = 0

    # Now try to reduce the code using local search
    # We'll try:
    # 1. Remove redundant codewords
    # 2. Swap codewords to potentially allow further removals
    
    # Build coverage count: for each word, how many codewords cover it
    counts = np.zeros(total_words, dtype=np.int32)
    for c in code:
        idx = word_to_idx[c]
        b = balls_np[idx]
        counts[b] += 1
    
    # Try to remove redundant codewords
    improved = True
    while improved and time.time() < deadline - 20.0:
        improved = False
        for i in range(len(code)):
            idx = word_to_idx[code[i]]
            b = balls_np[idx]
            # Can we remove this codeword? All words it covers must have count >= 2
            if np.all(counts[b] >= 2):
                # Remove it
                counts[b] -= 1
                code.pop(i)
                improved = True
                break  # Restart scan

    # Now try local search with swaps
    # For each pair of codewords, try replacing one with another to see if we can then remove more
    # This is expensive, so we'll do a limited search
    
    best_code = code[:]
    best_size = len(code)
    
    # Simulated annealing-like approach: try random swaps and see if we can reduce size
    for iteration in range(5000):
        if time.time() > deadline - 10.0:
            break
        
        if len(code) >= best_size:
            continue
        
        # Try a random swap: replace code[i] with a random other word
        i = random.randint(0, len(code) - 1)
        new_word_idx = random.randint(0, total_words - 1)
        
        old_word = code[i]
        old_idx = word_to_idx[old_word]
        new_word = all_words[new_word_idx]
        new_idx = word_to_idx[new_word]
        
        # Check if new_word is already in code
        if new_word in code:
            continue
        
        # Tentatively swap
        code[i] = new_word
        
        # Update counts
        counts[balls_np[old_idx]] -= 1
        counts[balls_np[new_idx]] += 1
        
        # Check validity: all counts must be >= 1
        if np.any(counts < 1):
            # Revert
            counts[balls_np[new_idx]] -= 1
            counts[balls_np[old_idx]] += 1
            code[i] = old_word
            continue
        
        # Try to remove redundant codewords
        removed_any = False
        for j in range(len(code)):
            if j == i:
                continue
            idx_j = word_to_idx[code[j]]
            b_j = balls_np[idx_j]
            if np.all(counts[b_j] >= 2):
                # Remove code[j]
                counts[b_j] -= 1
                code.pop(j)
                removed_any = True
                if j < i:
                    i -= 1
                break
        
        if len(code) < best_size:
            best_code = code[:]
            best_size = len(code)
    
    # Final cleanup: remove any redundant codewords
    for i in range(len(code)):
        idx = word_to_idx[code[i]]
        b = balls_np[idx]
        if np.all(counts[b] >= 2):
            counts[b] -= 1
            code.pop(i)
            i -= 1

    return code


def _greedy_fallback(q, n, radius, start_time, deadline):
    """Fallback greedy construction for general parameters."""
    total_words = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute balls
    balls = []
    for w in all_words:
        ball = [word_to_idx[w]]
        for pos in range(n):
            orig = w[pos]
            for new_val in range(q):
                if new_val != orig:
                    new_w = list(w)
                    new_w[pos] = new_val
                    ball.append(word_to_idx[tuple(new_w)])
        balls.append(ball)
    
    uncovered = np.ones(total_words, dtype=np.int32)
    code = []
    
    while uncovered.sum() > 0 and time.time() < deadline - 5.0:
        best_idx = -1
        best_count = -1
        for idx in range(total_words):
            if uncovered[idx] == 0:
                continue
            b = balls[idx]
            count = uncovered[b].sum()
            if count > best_count:
                best_count = count
                best_idx = idx
        
        if best_idx == -1:
            break
        
        code.append(all_words[best_idx])
        b = balls[best_idx]
        uncovered[b] = 0
    
    return code
# EVOLVE-BLOCK-END
