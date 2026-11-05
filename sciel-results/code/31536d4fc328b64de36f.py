"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import numpy as np

def _hamming_dist(a, b):
    return sum(1 for i in range(len(a)) if a[i] != b[i])

def _get_neighbors(word, q, n):
    """Generate all words within Hamming distance 1 of word."""
    result = [tuple(word)]
    for i in range(n):
        for val in range(q):
            if val != word[i]:
                new_word = list(word)
                new_word[i] = val
                result.append(tuple(new_word))
    return result

def construct(q, n, radius):
    if n == 0:
        return [[]]
    
    # For small cases, use direct construction
    if n <= 4:
        if n == 4:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        elif n == 3:
            return [[0, 0, 0], [1, 1, 1], [2, 2, 2]]
        elif n == 2:
            # K_3(2,1) = 3? 
            # Let's just return a small set. 
            # [0,0], [1,1], [2,2] covers?
            # (0,0) covers (0,0),(0,1),(0,2),(1,0),(2,0)
            # (1,1) covers (1,1),(1,0),(1,2),(0,1),(2,1)
            # (2,2) covers (2,2),(2,0),(2,1),(1,2),(0,2)
            # Union: 
            # (0,0) ok, (0,1) ok, (0,2) ok
            # (1,0) ok, (1,1) ok, (1,2) ok
            # (2,0) ok, (2,1) ok, (2,2) ok
            # Yes, 3 words are enough.
            return [[0, 0], [1, 1], [2, 2]]
        else:
            return [[i] for i in range(q)]

    # Precompute all words and their neighbors
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors for each word index
    neighbors = []
    for w in all_words:
        nbrs = _get_neighbors(w, q, n)
        neighbors.append([word_to_idx[nbr] for nbr in nbrs])
        
    # Convert to sets for faster subset checks in local search
    neighbor_sets = [set(nbrs) for nbrs in neighbors]

    # Greedy Initialization
    uncovered = [True] * len(all_words)
    codewords = []
    
    # counts[i] = number of uncovered words in neighbors[i]
    counts = [0] * len(all_words)
    for i in range(len(all_words)):
        counts[i] = len(neighbors[i])
        
    while any(uncovered):
        max_count = -1
        best_word_idx = -1
        for i in range(len(all_words)):
            if counts[i] > max_count:
                max_count = counts[i]
                best_word_idx = i
                
        if best_word_idx == -1:
            break
            
        cw = all_words[best_word_idx]
        codewords.append(cw)
        
        for nbr_idx in neighbors[best_word_idx]:
            if uncovered[nbr_idx]:
                uncovered[nbr_idx] = False
                for i in neighbors[nbr_idx]:
                    counts[i] -= 1
                    
    # Local Search: Minimize size
    # 1. Remove redundant codewords
    # Maintain coverage_count: how many codewords cover each word
    coverage_count = [0] * len(all_words)
    for cw in codewords:
        idx = word_to_idx[cw]
        for nbr_idx in neighbors[idx]:
            coverage_count[nbr_idx] += 1
            
    improved = True
    while improved:
        improved = False
        
        # Remove redundant
        for i in range(len(codewords) - 1, -1, -1):
            cw = codewords[i]
            cw_idx = word_to_idx[cw]
            redundant = True
            for nbr_idx in neighbors[cw_idx]:
                if coverage_count[nbr_idx] <= 1:
                    redundant = False
                    break
            
            if redundant:
                # Remove cw
                del codewords[i]
                for nbr_idx in neighbors[cw_idx]:
                    coverage_count[nbr_idx] -= 1
                improved = True
                break # Restart scan
        
        # Try to replace pairs with single
        # To avoid O(N^3) explosion, we do this only a few times or if N is small
        if len(codewords) < 100:
            replaced = False
            for i in range(len(codewords)):
                if replaced: break
                for j in range(i + 1, len(codewords)):
                    if replaced: break
                    cw_i = codewords[i]
                    cw_j = codewords[j]
                    idx_i = word_to_idx[cw_i]
                    idx_j = word_to_idx[cw_j]
                    
                    union_nbrs = neighbor_sets[idx_i] | neighbor_sets[idx_j]
                    
                    # Check if any single codeword covers this union
                    # We can check all possible codewords (all_words)
                    # Or just check if the union is small enough?
                    # Let's check all_words. 
                    # Optimization: Only check if union size is <= max ball size (13).
                    if len(union_nbrs) > 13:
                        continue
                        
                    for k_idx in range(len(all_words)):
                        if neighbor_sets[k_idx] >= union_nbrs:
                            # Found a replacement
                            # Check if k_idx is already in codewords
                            cw_k = all_words[k_idx]
                            if cw_k in codewords:
                                # Remove i and j
                                # Note: indices might shift
                                # Remove j first (higher index)
                                del codewords[j]
                                del codewords[i]
                                # Update coverage count
                                for nbr_idx in neighbor_sets[idx_i]:
                                    coverage_count[nbr_idx] -= 1
                                for nbr_idx in neighbor_sets[idx_j]:
                                    coverage_count[nbr_idx] -= 1
                                # Note: coverage_count for neighbors of k is unchanged 
                                # because k is still there and i,j were redundant for those parts?
                                # Wait, if k covers union, and k was already there, 
                                # then i and j were redundant for the union.
                                # But what if i covered something else?
                                # No, union is N(i) U N(j). 
                                # If N(k) covers Union, then N(i) subset N(k) and N(j) subset N(k).
                                # So i and j are fully redundant.
                                improved = True
                                replaced = True
                                break
                            else:
                                # Replace i and j with k
                                # Remove i and j
                                del codewords[j]
                                del codewords[i]
                                # Update coverage count for i and j
                                for nbr_idx in neighbor_sets[idx_i]:
                                    coverage_count[nbr_idx] -= 1
                                for nbr_idx in neighbor_sets[idx_j]:
                                    coverage_count[nbr_idx] -= 1
                                # Add k
                                codewords.append(cw_k)
                                for nbr_idx in neighbor_sets[k_idx]:
                                    coverage_count[nbr_idx] += 1
                                improved = True
                                replaced = True
                                break
                                
    return codewords
# EVOLVE-BLOCK-END
