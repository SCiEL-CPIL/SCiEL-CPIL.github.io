"""Improved program for ternary covering code K_3(6,1)."""
# EVOLVE-BLOCK-START
import numpy as np
import itertools

def construct(q, n, radius):
    """
    Constructs a covering code using Greedy Set Cover and Greedy Removal.
    """
    total_words = q ** n
    
    # Function to convert index to tuple of digits
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n):
            word[n-1-i] = idx % q
            idx //= q
        return tuple(word)
        
    # Function to convert word to index
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx

    # Precompute neighbors for each word
    # neighbors[i] is a list of indices of words within distance <= radius of word i
    neighbors = []
    for i in range(total_words):
        w = idx_to_word(i)
        nbrs = [i]
        if radius == 1:
            for pos in range(n):
                for val in range(q):
                    if val != w[pos]:
                        new_w = list(w)
                        new_w[pos] = val
                        nbrs.append(word_to_idx(new_w))
        else:
            # General case: generate all words and check distance
            for j in range(total_words):
                wj = idx_to_word(j)
                dist = sum(1 for k in range(n) if w[k] != wj[k])
                if dist <= radius:
                    nbrs.append(j)
        neighbors.append(nbrs)
        
    # Greedy Set Cover
    uncovered = set(range(total_words))
    codewords = []
    
    while uncovered:
        best_c = -1
        max_new = 0
        for c in range(total_words):
            new_cov = 0
            for nbr in neighbors[c]:
                if nbr in uncovered:
                    new_cov += 1
            if new_cov > max_new:
                max_new = new_cov
                best_c = c
                
        if best_c == -1:
            break
            
        codewords.append(best_c)
        nbr_set = set(neighbors[best_c])
        uncovered.difference_update(nbr_set)
        
    # Greedy Removal using coverage counts
    coverage_count = np.zeros(total_words, dtype=int)
    for c in codewords:
        for nbr in neighbors[c]:
            coverage_count[nbr] += 1
            
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(codewords):
            c = codewords[i]
            removable = True
            for nbr in neighbors[c]:
                if coverage_count[nbr] == 1:
                    removable = False
                    break
            if removable:
                codewords.pop(i)
                for nbr in neighbors[c]:
                    coverage_count[nbr] -= 1
                changed = True
            else:
                i += 1
                
    # Convert indices back to lists of integers
    result = [list(idx_to_word(c)) for c in codewords]
    
    return result

# EVOLVE-BLOCK-END
