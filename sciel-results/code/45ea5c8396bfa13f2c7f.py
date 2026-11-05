"""Covering code construction for K_3(7,1) using greedy + ILS."""
import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Precompute all words as integers 0..q^n-1
    total_words = q ** n
    
    # Convert word index to tuple
    def idx_to_tuple(idx):
        digits = []
        for _ in range(n):
            digits.append(idx % q)
            idx //= q
        return tuple(reversed(digits))
    
    # Precompute neighbors for each word (within Hamming distance radius)
    # For q=3, n=7, radius=1: each word has 1 + 6*2 = 13 neighbors
    # We'll use a bitset approach for efficiency
    
    # Generate all neighbors for each word
    neighbors = []
    for idx in range(total_words):
        word = idx_to_tuple(idx)
        nbrs = set()
        nbrs.add(idx)
        for pos in range(n):
            for val in range(q):
                if val == word[pos]:
                    continue
                new_word = list(word)
                new_word[pos] = val
                # Convert back to index
                new_idx = 0
                for d in new_word:
                    new_idx = new_idx * q + d
                nbrs.add(new_idx)
        neighbors.append(nbrs)
    
    # Greedy construction
    def greedy_construct():
        uncovered = set(range(total_words))
        codewords = []
        
        while uncovered:
            # Find the word that covers the most uncovered words
            best_word = None
            best_count = -1
            # Sample a subset for speed, or iterate all
            for word in uncovered:
                count = len(neighbors[word] & uncovered)
                if count > best_count:
                    best_count = count
                    best_word = word
                if best_count == len(uncovered):
                    break
            
            codewords.append(best_word)
            uncovered -= neighbors[best_word]
        
        return [idx_to_tuple(c) for c in codewords]
    
    # Initial greedy solution
    best_codewords = greedy_construct()
    best_size = len(best_codewords)
    
    # ILS: Iterated Local Search
    def try_remove(codewords):
        """Try to remove redundant codewords."""
        codewords = list(codewords)
        improved = True
        while improved:
            improved = False
            for i in range(len(codewords)):
                # Check if removing codewords[i] still covers everything
                remaining = codewords[:i] + codewords[i+1:]
                # Check coverage
                covered = set()
                for cw in remaining:
                    covered.update(neighbors[cw])
                if len(covered) == total_words:
                    codewords = remaining
                    improved = True
                    break
        return codewords
    
    def perturb(codewords, num_replacements):
        """Randomly replace some codewords with random words."""
        codewords = list(codewords)
        for _ in range(num_replacements):
            idx = random.randint(0, len(codewords) - 1)
            codewords[idx] = random.randint(0, total_words - 1)
        return codewords
    
    def repair(codewords):
        """Add codewords until all words are covered."""
        codewords = list(codewords)
        covered = set()
        for cw in codewords:
            covered.update(neighbors[cw])
        
        while len(covered) < total_words:
            uncovered = set(range(total_words)) - covered
            # Greedily add the word covering most uncovered
            best_word = None
            best_count = -1
            for word in uncovered:
                count = len(neighbors[word] & uncovered)
                if count > best_count:
                    best_count = count
                    best_word = word
            codewords.append(best_word)
            covered.update(neighbors[best_word])
        
        return codewords
    
    # Run ILS
    for iteration in range(500):
        if time.time() > deadline:
            break
        
        # Perturb
        num_repl = random.randint(1, 3)
        perturbed = perturb(best_codewords, num_repl)
        
        # Repair
        repaired = repair(perturbed)
        
        # Try to remove redundant
        improved = try_remove(repaired)
        
        if len(improved) < best_size:
            best_size = len(improved)
            best_codewords = improved
    
    # Final cleanup
    best_codewords = try_remove(best_codewords)
    
    # Convert to list of lists
    result = [list(idx_to_tuple(c)) for c in best_codewords]
    return result
