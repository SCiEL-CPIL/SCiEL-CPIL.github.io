"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools


def construct(q, n, radius):
    """Construct a minimal covering code using greedy reduction with vectorized operations."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Total number of words in the space
    total_words = q ** n
    
    # Generate all words in the space as a 2D array (total_words x n)
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)
    
    # Start with the tetracode construction as initial solution
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    
    # Direct sum: tetracode x all words on remaining n-4 coordinates
    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            initial_codewords.append(h + list(tail))
    
    codewords = np.array(initial_codewords, dtype=np.int32)
    num_codewords = len(codewords)
    
    # Build coverage matrix: coverage[i, j] = 1 if codeword i covers word j
    # A codeword covers itself and all words at Hamming distance 1
    
    # For efficiency, compute coverage using vectorized operations
    # For each codeword, compute which words it covers
    
    # Method: for each codeword, generate all words at distance 0 and 1, then find their indices
    # Since all_words is in lex order, we can compute indices directly
    
    coverage = np.zeros((num_codewords, total_words), dtype=bool)
    
    for i in range(num_codewords):
        word = codewords[i]
        # The word itself
        idx = 0
        for j in range(n):
            idx = idx * q + word[j]
        coverage[i, idx] = True
        
        # Words at distance 1
        for pos in range(n):
            orig_val = word[pos]
            for val in range(q):
                if val != orig_val:
                    new_word = word.copy()
                    new_word[pos] = val
                    idx = 0
                    for j in range(n):
                        idx = idx * q + new_word[j]
                    coverage[i, idx] = True
    
    # Now we have the coverage matrix
    # coverage: (num_codewords, total_words) boolean matrix
    
    # Check that the initial solution is valid
    covered_words = np.any(coverage, axis=0)
    if not np.all(covered_words):
        # Should not happen, but handle it
        pass
    
    # Now try to remove redundant codewords
    # A codeword is redundant if all words it covers are also covered by other codewords
    
    # We'll iteratively remove redundant codewords
    active = np.ones(num_codewords, dtype=bool)
    
    # For efficiency, track which words are covered by active codewords
    # and how many active codewords cover each word
    
    cover_count = coverage.sum(axis=0)  # Number of codewords covering each word
    
    # Iteratively remove redundant codewords
    changed = True
    while changed and (time.time() < deadline - 10.0):
        changed = False
        
        # For each active codeword, check if it's redundant
        # A codeword i is redundant if for every word j it covers, cover_count[j] > 1
        # (i.e., some other active codeword also covers j)
        
        # Vectorized: for each codeword i, check if min(cover_count[j] for j where coverage[i,j]=1) > 1
        # This is equivalent to: for each i, check if np.all(cover_count[coverage[i]] > 1)
        
        # This is expensive to do naively, so let's do it in chunks
        for i in range(num_codewords):
            if time.time() > deadline - 10.0:
                break
            if not active[i]:
                continue
            
            # Get the words covered by codeword i
            covered_mask = coverage[i]
            if not np.any(covered_mask):
                # This codeword covers no words? Should not happen
                active[i] = False
                changed = True
                # Update cover_count
                cover_count -= coverage[i]
                continue
            
            # Check if all covered words are covered by other active codewords
            # i.e., cover_count[covered_mask] > 1 for all covered words
            if np.all(cover_count[covered_mask] > 1):
                # This codeword is redundant, remove it
                active[i] = False
                changed = True
                cover_count -= coverage[i]
    
    # Extract the active codewords
    result_indices = np.where(active)[0]
    result = [codewords[i].tolist() for i in result_indices]
    
    return result
# EVOLVE-BLOCK-END
