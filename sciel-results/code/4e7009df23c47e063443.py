"""Optimized covering code for K_3(6,1) using vectorized greedy set cover."""
import time
import itertools
import numpy as np


def _all_words(q, n):
    """Generate all q^n words as numpy array of shape (q^n, n)."""
    return np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)


def _coverage_matrix(words, q, n, radius=1):
    """
    Build a boolean coverage matrix of shape (total, total) where
    coverage[i, j] = True if word j is within Hamming distance <= radius of word i.
    """
    total = len(words)
    if radius == 1:
        # For each word i, it covers itself and all words differing in exactly one position
        # Vectorized: for each position, for each replacement value
        cov = np.zeros((total, total), dtype=bool)
        cov[np.arange(total), np.arange(total)] = True  # self-cover
        for pos in range(n):
            for new_val in range(q):
                # Words that have value new_val at position pos
                # A word i covers word j at position pos if words[j, pos] != words[i, pos]
                # and words[j, pos] == new_val, words[i, pos] != new_val
                # Actually: word i covers word j if they differ in at most 1 position.
                # For position pos: if words[i, pos] != new_val and words[j, pos] == new_val,
                # then i and j differ at position pos (at least). But they might differ elsewhere too.
                # This approach is tricky. Let me think differently.
                # 
                # Better: For each word j, the words that cover it are:
                # - j itself
                # - For each position pos, for each value v != words[j, pos]:
                #   the word that is j but with position pos set to v
                #
                # So let's build it from the other direction.
                # For each j, find all i such that dist(i, j) <= 1.
                # dist(i, j) = 0: i = j
                # dist(i, j) = 1: they differ in exactly one position.
                #
                # Vectorized approach:
                # Create a matrix where for each pair (i,j), count positions where they differ.
                # But that's O(total^2 * n) which for 729*729*6 = ~3.2M operations, fine.
                pass
        
        # Simpler vectorized approach for radius=1:
        # For each word j, the covering words are j and its n*(q-1) neighbors.
        # Let's build the coverage by iterating over words and their neighbors.
        # total = 729, n=6, q=3, so each word covers 1 + 6*2 = 13 words.
        # We can build this efficiently.
        
        # Build index mapping: word tuple -> index
        word_to_idx = {}
        for idx, w in enumerate(words):
            word_to_idx[tuple(w)] = idx
        
        cov = np.zeros((total, total), dtype=bool)
        for i in range(total):
            w = words[i]
            cov[i, i] = True
            for pos in range(n):
                old_val = w[pos]
                for new_val in range(q):
                    if new_val != old_val:
                        neighbor = list(w)
                        neighbor[pos] = new_val
                        j = word_to_idx[tuple(neighbor)]
                        cov[i, j] = True
        return cov
    else:
        # General case
        total = len(words)
        cov = np.zeros((total, total), dtype=bool)
        for i in range(total):
            for j in range(total):
                dist = np.sum(words[i] != words[j])
                if dist <= radius:
                    cov[i, j] = True
        return cov


def _greedy_cover_vectorized(q, n, radius, deadline):
    """
    Vectorized greedy set cover.
    Returns list of codewords (as lists of ints).
    """
    words = _all_words(q, n)
    total = len(words)
    
    # Build coverage matrix: cov[i, j] = True if word i covers word j
    cov = _coverage_matrix(words, q, n, radius)
    
    # uncovered: boolean array, True if word j is not yet covered
    uncovered = np.ones(total, dtype=bool)
    chosen = []
    
    # Precompute: for each candidate i, the count of uncovered words it covers
    # This changes as we add codewords, so we update incrementally.
    # coverage_counts[i] = number of uncovered words covered by word i
    coverage_counts = cov.sum(axis=1).astype(np.int32)
    
    while uncovered.any():
        if time.time() > deadline:
            break
        
        # Find the word that covers the most uncovered words
        best_idx = np.argmax(coverage_counts)
        best_count = coverage_counts[best_idx]
        
        if best_count == 0:
            break
        
        chosen.append(words[best_idx].tolist())
        
        # Update uncovered and coverage_counts
        # Words covered by best_idx that are currently uncovered
        newly_covered = cov[best_idx] & uncovered
        if newly_covered.any():
            uncovered[newly_covered] = False
            # For each other candidate i, subtract the number of newly covered words it also covers
            # coverage_counts[i] -= sum over j in newly_covered of cov[i, j]
            # Vectorized: for each i, count how many of the newly_covered words it covers
            # cov[:, newly_covered] has shape (total, num_newly_covered)
            # We want the sum along axis 1 for each row
            reductions = cov[:, newly_covered].sum(axis=1)
            coverage_counts -= reductions
    
    return chosen


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit
    
    # For the specific case K_3(6,1), use the optimized greedy
    if q == 3 and n == 6 and radius == 1:
        # Start with a known good construction as initial solution
        # Tetracode-based: 9*9 = 81 words, but we can do better with greedy
        # Actually, let's just run the vectorized greedy which should be fast
        code = _greedy_cover_vectorized(q, n, radius, deadline)
        
        if code:
            return code
    
    # General case
    code = _greedy_cover_vectorized(q, n, radius, deadline)
    if code:
        return code
    
    # Fallback: all words
    return [list(w) for w in itertools.product(range(q), repeat=n)]
