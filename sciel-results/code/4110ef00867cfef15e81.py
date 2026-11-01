"""Rewritten program for ternary covering code K_3(7,1) using greedy set cover + local search."""
import time
import random
import numpy as np
from itertools import product

def _hamming_dist(a, b):
    """Hamming distance between two tuples/lists of same length."""
    return sum(x != y for x, y in zip(a, b))

def _neighbors(word, q, n, radius=1):
    """Generate all words within Hamming distance 1 of word."""
    result = [list(word)]
    for pos in range(n):
        for val in range(q):
            if val != word[pos]:
                new_word = list(word)
                new_word[pos] = val
                result.append(new_word)
    return result

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # For the specific case of q=3, n=7, radius=1
    if q == 3 and n == 7 and radius == 1:
        return _solve_ternary_7_1(start_time, deadline)
    
    # Fallback: direct sum construction for general case
    return _direct_sum_construct(q, n, radius)

def _direct_sum_construct(q, n, radius):
    """Direct sum construction as fallback."""
    if n < 4:
        # Small case: just use all words or a simple construction
        if n == 0:
            return [[]]
        if n == 1:
            return [[i] for i in range(q)]
        if n == 2:
            return [[a, b] for a in range(q) for b in range(q)]
        if n == 3:
            return [[a, b, (a + b) % q] for a in range(q) for b in range(q)]
    
    # For q=3, use tetracode-based construction
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        result = []
        for h in tetracode:
            for tail in product(range(3), repeat=n - 4):
                result.append(h + list(tail))
        return result
    else:
        # General fallback: all words
        return [list(w) for w in product(range(q), repeat=n)]

def _solve_ternary_7_1(start_time, deadline):
    """Solve K_3(7,1) using greedy set cover + local search."""
    q, n, radius = 3, 7, 1
    
    # Precompute all words and their neighbors
    all_words = [list(w) for w in product(range(q), repeat=n)]
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    num_words = len(all_words)
    
    # Precompute neighbors for each word
    neighbors = []
    for w in all_words:
        nb = [w]
        for pos in range(n):
            for val in range(q):
                if val != w[pos]:
                    nw = w[:]
                    nw[pos] = val
                    nb.append(nw)
        neighbors.append(nb)
    
    # Convert to index-based for speed
    neighbors_idx = []
    for nb_list in neighbors:
        neighbors_idx.append([word_to_idx[tuple(nb)] for nb in nb_list])
    
    # Greedy set cover
    uncovered = set(range(num_words))
    codewords = []
    
    while uncovered and time.time() < deadline:
        # Find the word that covers the most uncovered words
        best_word = None
        best_count = -1
        
        # Sample some candidates to speed up
        candidates = list(uncovered)
        random.shuffle(candidates)
        candidates = candidates[:min(len(candidates), 500)]
        
        for w_idx in candidates:
            count = sum(1 for nb in neighbors_idx[w_idx] if nb in uncovered)
            if count > best_count:
                best_count = count
                best_word = w_idx
        
        if best_word is None:
            break
        
        codewords.append(all_words[best_word])
        for nb in neighbors_idx[best_word]:
            uncovered.discard(nb)
        
        # Check time
        if time.time() > deadline - 30:
            break
    
    if not uncovered:
        # We have a valid solution, try to improve
        codewords = _local_search(codewords, neighbors_idx, word_to_idx, all_words, deadline)
        return codewords
    
    # If greedy didn't finish, fall back to direct sum
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    result = []
    for h in tetracode:
        for tail in product(range(3), repeat=n - 4):
            result.append(h + list(tail))
    return result

def _local_search(codewords, neighbors_idx, word_to_idx, all_words, deadline):
    """Try to reduce the number of codewords via local search."""
    num_words = len(all_words)
    n = len(all_words[0])
    
    # Build coverage count: for each word, how many codewords cover it
    coverage = [0] * num_words
    codeword_indices = [word_to_idx[tuple(cw)] for cw in codewords]
    
    for cw_idx in codeword_indices:
        for nb in neighbors_idx[cw_idx]:
            coverage[nb] += 1
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try removing each codeword
        for i in range(len(codeword_indices)):
            if time.time() > deadline:
                break
            
            cw_idx = codeword_indices[i]
            
            # Check if removing this codeword leaves any word uncovered
            removable = True
            for nb in neighbors_idx[cw_idx]:
                if coverage[nb] == 1:
                    removable = False
                    break
            
            if removable:
                # Remove it
                for nb in neighbors_idx[cw_idx]:
                    coverage[nb] -= 1
                codeword_indices.pop(i)
                codewords.pop(i)
                improved = True
                break
        
        # Try swapping: replace one codeword with another that covers more uncovered
        if not improved:
            # Try to find a codeword that can be replaced
            for i in range(len(codeword_indices)):
                if time.time() > deadline:
                    break
                
                cw_idx = codeword_indices[i]
                
                # Find words that would become uncovered if we remove this
                critical_words = [nb for nb in neighbors_idx[cw_idx] if coverage[nb] == 1]
                
                if not critical_words:
                    continue
                
                # Try to find a replacement codeword that covers all critical words
                # and doesn't make things worse
                best_replacement = None
                best_gain = 0
                
                # Sample candidate replacements
                candidates = list(range(num_words))
                random.shuffle(candidates)
                candidates = candidates[:min(len(candidates), 200)]
                
                for cand_idx in candidates:
                    if cand_idx in codeword_indices:
                        continue
                    
                    # Check if this candidate covers all critical words
                    cand_neighbors = set(neighbors_idx[cand_idx])
                    if not all(cw in cand_neighbors for cw in critical_words):
                        continue
                    
                    # Calculate gain: how many words are covered by replacement
                    # that are not covered by current
                    gain = 0
                    for nb in neighbors_idx[cand_idx]:
                        if coverage[nb] == 0:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_replacement = cand_idx
                
                if best_replacement is not None and best_gain > 0:
                    # Do the swap
                    old_idx = codeword_indices[i]
                    for nb in neighbors_idx[old_idx]:
                        coverage[nb] -= 1
                    for nb in neighbors_idx[best_replacement]:
                        coverage[nb] += 1
                    codeword_indices[i] = best_replacement
                    codewords[i] = all_words[best_replacement]
                    improved = True
                    break
    
    return codewords
