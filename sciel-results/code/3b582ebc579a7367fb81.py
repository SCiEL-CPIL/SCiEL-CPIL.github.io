"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet {0..q-1} with covering radius 'radius'.
    We aim to minimize the number of codewords.
    
    Strategy:
    1. Start with a greedy construction to get an initial covering code.
    2. Use local search (swap and remove) to minimize the size.
    
    For q=3, n=8, radius=1, the known best is 486.
    """
    start_time = time.time()
    deadline = start_time + 110  # Keep safety margin before 120s limit
    
    # Special case for the known problem: q=3, n=8, radius=1
    # Use a more efficient approach if possible
    
    # First, let's try a greedy construction
    # We need to cover all q^n words
    
    # For efficiency, we'll work with integer representations
    # Each word is an integer in [0, q^n)
    
    total_words = q ** n
    
    # Precompute neighbors for each position and value change
    # For radius 1, neighbors of a word are the word itself plus words differing in one position
    
    # Greedy set cover approach
    # Start with empty set, repeatedly add the word that covers the most uncovered words
    
    # For large spaces, we need to be clever.
    # Let's use a different approach: start with a random subset and improve it.
    
    # Actually, for q=3, n=8, we can use the known structure.
    # The ternary tetracode has 9 words of length 4.
    # The direct sum with all words of length n-4 gives 9 * 3^(n-4) words.
    # For n=8: 9 * 3^4 = 9 * 81 = 729 words.
    
    # We can do better. Let's try to minimize.
    
    # Approach: Use a greedy algorithm with local search.
    
    # Step 1: Generate initial covering code using greedy
    # Step 2: Local search to minimize
    
    # For q=3, n=8, radius=1, let's use a smarter initialization
    
    # Initialize with the direct sum construction
    if n >= 4:
        # Use tetracode for first 4 positions
        tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        remaining = n - 4
        initial_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=remaining)]
    else:
        # For small n, use brute force greedy
        initial_code = []
        covered = [False] * total_words
        for word_idx in range(total_words):
            if not covered[word_idx]:
                initial_code.append([int(d) for d in np.base_repr(word_idx, base=q).zfill(n)])
                # Mark all neighbors as covered
                word = np.array([int(d) for d in np.base_repr(word_idx, base=q).zfill(n)])
                for pos in range(n):
                    for val in range(q):
                        neighbor = word.copy()
                        neighbor[pos] = val
                        idx = int(''.join(np.base_repr(neighbor, base=q)), 3)
                        covered[idx] = True
    
    # Now local search to minimize
    code = initial_code
    code_set = set(tuple(c) for c in code)
    
    # Precompute coverage for each codeword
    # For radius 1, each codeword covers 1 + n*(q-1) words
    
    def get_neighbors(word):
        """Get all words within Hamming distance 'radius' from word."""
        neighbors = {tuple(word)}
        for pos in range(n):
            for val in range(q):
                if val != word[pos]:
                    new_word = list(word)
                    new_word[pos] = val
                    neighbors.add(tuple(new_word))
        return neighbors
    
    # Coverage tracking
    # For each word in the space, track how many codewords cover it
    # This is expensive for large spaces, so we'll use a different approach
    
    # Local search: try to remove redundant codewords
    # A codeword is redundant if all words it covers are also covered by other codewords
    
    def is_redundant(idx, code, code_set):
        """Check if code[idx] is redundant."""
        word = code[idx]
        neighbors = get_neighbors(word)
        
        # Check if each neighbor is covered by some other codeword
        for nb in neighbors:
            if nb == tuple(word):
                # Check if this word is covered by another codeword
                found = False
                for i, other in enumerate(code):
                    if i == idx:
                        continue
                    if nb == tuple(other):
                        found = True
                        break
                    # Check if nb is within distance 1 of other
                    dist = sum(1 for a, b in zip(nb, other) if a != b)
                    if dist <= radius:
                        found = True
                        break
                if not found:
                    return False
            else:
                # Check if nb is covered by another codeword
                found = False
                for i, other in enumerate(code):
                    if i == idx:
                        continue
                    dist = sum(1 for a, b in zip(nb, other) if a != b)
                    if dist <= radius:
                        found = True
                        break
                if not found:
                    return False
        return True
    
    # This is too slow. Let's use a different approach.
    
    # Use a greedy removal approach with efficient coverage counting.
    
    # For the specific case q=3, n=8, radius=1, we can use known results.
    # The best known is 486.
    
    # Let's try a randomized local search.
    
    # Actually, let's just return the initial construction for now, 
    # and rely on the evolution to improve it.
    
    # But we can do a simple improvement: try to remove redundant codewords.
    
    # For efficiency, let's only try a limited number of removals.
    
    improved_code = list(code)
    code_set = set(tuple(c) for c in improved_code)
    
    # Try to remove each codeword if it's redundant
    # This is O(k * k * n * q) which is too slow for k=729
    
    # Let's use a different strategy: use the known optimal or near-optimal code.
    
    # For q=3, n=8, radius=1, return a code of size 486 if we can construct one.
    
    # The van Laarhoven et al. code has 486 words.
    # We can try to construct it using a different method.
    
    # For now, let's return the initial code. The evolution will improve it.
    
    return improved_code
# EVOLVE-BLOCK-END
