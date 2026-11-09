"""Greedy set-cover based construction for ternary covering codes K_3(8,1)."""
import numpy as np
import random


def construct(q, n, radius):
    # For this task: q=3, n=8, radius=1
    # Total words: 3^8 = 6561
    N = q ** n
    # Precompute the index for each word and its ball (word + all single-coordinate variants)
    # We'll use arrays for speed.
    
    # Precompute all words as tuples
    # Instead of storing all words, we can compute index from word on the fly,
    # but for ball computation we need to generate neighbors.
    
    # Let's precompute the ball for each index as a list of indices.
    # ball[i] = list of indices j such that hamming distance between word_i and word_j <= radius
    
    # First, generate all words
    # words[i] = the i-th word in base-q representation
    # We can compute neighbors by changing each coordinate.
    
    # For efficiency, precompute the "digit changes" offsets:
    # For position p, changing digit d to d' changes the index by (d' - d) * q**p
    
    # Let's precompute powers
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q
    
    # Precompute for each index, the list of indices at distance exactly 1
    # dist1_neighbors[i] = list of indices at hamming distance 1 from i
    # Then ball[i] = [i] + dist1_neighbors[i]
    
    # To compute dist1_neighbors for index i:
    # For each position p (0 to n-1):
    #   current_digit = (i // powers[p]) % q
    #   for new_digit in 0..q-1, new_digit != current_digit:
    #       neighbor = i + (new_digit - current_digit) * powers[p]
    
    # Let's precompute this for all i.
    # N = 6561, n = 8, q = 3. For each i, 8*2 = 16 neighbors. Total ~105k entries.
    
    dist1 = []
    for i in range(N):
        neighbors = []
        for p in range(n):
            cur = (i // powers[p]) % q
            base = i - cur * powers[p]
            for nd in range(q):
                if nd != cur:
                    nb = base + nd * powers[p]
                    neighbors.append(nb)
        dist1.append(neighbors)
    
    # Now greedy set cover:
    # uncovered = set of all indices
    # For each step, find the index that covers the most uncovered indices.
    # coverage(i) = len({i} ∪ dist1[i] ∩ uncovered)
    
    # For speed, we can maintain a count array: for each index j, 
    # how many uncovered words are in ball(j)? No, that's not quite right.
    # We want for each candidate i: |ball(i) ∩ uncovered|
    
    # Let's use numpy arrays for speed.
    uncovered = np.ones(N, dtype=bool)
    code = []
    
    # Precompute ball as a list of arrays for fast intersection
    # Actually, let's just iterate.
    
    # For each iteration:
    #   For each candidate i, compute cov = uncovered[i] + sum(uncovered[j] for j in dist1[i])
    #   Pick max
    
    # To speed up, we can use numpy:
    # cov[i] = uncovered[i] + np.sum(uncovered[dist1[i]])
    
    # But dist1[i] is a list. Let's convert to numpy arrays.
    
    dist1_arrays = [np.array(neb, dtype=np.int32) for neb in dist1]
    
    while np.any(uncovered):
        # Compute coverage for all candidates
        # cov[i] = int(uncovered[i]) + np.sum(uncovered[dist1_arrays[i]])
        # This is 6561 iterations per step, each with a small array sum.
        
        # Optimize: use a loop with local variables
        best_cov = -1
        best_i = -1
        unc = uncovered  # local ref
        for i in range(N):
            cov = 1 if unc[i] else 0
            if i < len(dist1_arrays):
                nb = dist1_arrays[i]
                cov += np.sum(unc[nb])
            if cov > best_cov:
                best_cov = cov
                best_i = i
            elif cov == best_cov and i < best_i:
                best_i = i
        
        if best_i < 0:
            break
        
        code.append(best_i)
        # Remove ball(best_i) from uncovered
        uncovered[best_i] = False
        nb = dist1_arrays[best_i]
        uncovered[nb] = False
    
    # Convert indices back to words
    result = []
    for idx in code:
        word = []
        rem = idx
        for p in range(n):
            digit = rem % q
            word.append(digit)
            rem //= q
        word.reverse()
        result.append(word)
    
    return result
