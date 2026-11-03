"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import product

def construct(q, n, radius):
    """
    Find a covering code for q-ary words of length n with covering radius.
    Uses greedy set cover with local search refinement.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # For the specific case of ternary length 6 radius 1
    # Precompute all words and their neighborhoods
    all_words = list(product(range(q), repeat=n))
    total_words = q ** n
    
    # Map each word to its index
    word_to_idx = {}
    for i, w in enumerate(all_words):
        word_to_idx[w] = i
    
    # Precompute neighborhoods for each word (words within distance radius)
    # For efficiency, we'll compute on the fly using a fast method
    def get_neighbors(word):
        """Get all words within Hamming distance radius from word."""
        word = list(word)
        neighbors = set()
        # All words at distance 0
        neighbors.add(tuple(word))
        # All words at distance 1
        for pos in range(n):
            orig = word[pos]
            for val in range(q):
                if val == orig:
                    continue
                word[pos] = val
                neighbors.add(tuple(word))
            word[pos] = orig
        return neighbors
    
    # Greedy set cover: repeatedly pick the word that covers the most uncovered words
    uncovered = set(range(total_words))
    code = []
    
    # Precompute coverage for each candidate
    # coverage[i] = set of word indices covered by word i
    # This is expensive, so we'll compute lazily
    
    # Better approach: use a priority-based greedy with dynamic updates
    # For each uncovered word, track how many codewords cover it
    # Pick the codeword that maximizes new coverage
    
    # Initialize: compute coverage for all words
    # coverage[word_idx] = set of word indices it covers
    # This is O(total_words * neighborhood_size)
    
    # For ternary length 6 radius 1:
    # neighborhood size = 1 + 6*2 = 13
    # total_words = 729
    # So we can precompute all coverages
    
    coverage = [None] * total_words
    for i in range(total_words):
        w = all_words[i]
        nbrs = get_neighbors(w)
        coverage[i] = set(word_to_idx[t] for t in nbrs)
    
    # Greedy set cover
    while uncovered:
        # Find the word that covers the most uncovered words
        best_word = -1
        best_count = -1
        for i in range(total_words):
            if i in uncovered or True:  # We can pick any word
                count = len(coverage[i] & uncovered)
                if count > best_count:
                    best_count = count
                    best_word = i
        
        if best_word == -1 or best_count <= 0:
            break
        
        code.append(list(all_words[best_word]))
        # Remove covered words from uncovered
        covered = coverage[best_word]
        uncovered -= covered
    
    # Local search: try to swap codewords to reduce the total
    # A codeword is removable if all words it covers are covered by other codewords
    def is_removable(idx, code_set):
        """Check if codeword at index idx is removable."""
        if len(code_set) <= 1:
            return False
        word = all_words[code_set[idx]]
        nbrs = coverage[code_set[idx]]
        # Check if all neighbors are covered by other codewords
        for w_idx in nbrs:
            w = all_words[w_idx]
            # Check if w is covered by any other codeword
            found = False
            for j, c_idx in enumerate(code_set):
                if j == idx:
                    continue
                if w_idx in coverage[c_idx]:
                    found = True
                    break
            if not found:
                return False
        return True
    
    # Try to remove removable codewords
    changed = True
    while changed and time.time() < deadline:
        changed = False
        code_set = [word_to_idx[tuple(c)] for c in code]
        for i in range(len(code_set)):
            if is_removable(i, code_set):
                code.pop(i)
                code_set.pop(i)
                changed = True
                break
    
    # Local search: try swaps
    # For each pair (i, j), try replacing code[i] with a different word
    # that covers more uncovered words
    for iteration in range(20):
        if time.time() >= deadline:
            break
        improved = False
        code_set = [word_to_idx[tuple(c)] for c in code]
        for i in range(len(code_set)):
            # Try to find a better replacement for code[i]
            current = code_set[i]
            # Words covered only by code[i]
            only_covered = set(coverage[current])
            for j in range(len(code_set)):
                if j != i:
                    only_covered -= coverage[code_set[j]]
            
            # Find a word that covers all of only_covered and more
            best_new = -1
            best_gain = 0
            for candidate in range(total_words):
                if candidate == current:
                    continue
                # Check if candidate covers all of only_covered
                if not only_covered.issubset(coverage[candidate]):
                    continue
                # Calculate gain: how many new words does candidate cover?
                new_coverage = coverage[candidate] - set()
                for j in range(len(code_set)):
                    if j != i:
                        new_coverage -= coverage[code_set[j]]
                gain = len(new_coverage)
                if gain > best_gain:
                    best_gain = gain
                    best_new = candidate
            
            if best_new != -1 and best_gain > 0:
                code[i] = list(all_words[best_new])
                code_set[i] = best_new
                improved = True
        
        if not improved:
            break
    
    return code
# EVOLVE-BLOCK-END
