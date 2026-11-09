"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np

def _all_words(q, n):
    """Generate all q^n words as integers for efficient processing."""
    return list(range(q ** n))

def _int_to_word(val, q, n):
    """Convert integer to word (list of digits)."""
    word = [0] * n
    for i in range(n):
        word[i] = val % q
        val //= q
    return word

def _word_to_int(word, q):
    """Convert word to integer."""
    val = 0
    for digit in reversed(word):
        val = val * q + digit
    return val

def _hamming_dist(a, b):
    """Compute Hamming distance between two words."""
    return sum(1 for x, y in zip(a, b) if x != y)

def _neighbors(word, q, n, radius=1):
    """Generate all words within Hamming distance <= radius."""
    if radius == 0:
        return [word]
    if radius == 1:
        neighbors = [word]
        for pos in range(n):
            for val in range(q):
                if val != word[pos]:
                    new_word = word.copy()
                    new_word[pos] = val
                    neighbors.append(new_word)
        return neighbors
    # For larger radii, use recursive generation
    result = set()
    for r in range(radius + 1):
        for positions in itertools.combinations(range(n), r):
            for values in itertools.product(range(q), repeat=r):
                new_word = word.copy()
                for pos, val in zip(positions, values):
                    if val != word[pos]:
                        new_word[pos] = val
                    else:
                        break
                else:
                    result.append(tuple(new_word))
    return [list(w) for w in result]

def _greedy_set_cover(all_words, q, n, radius, deadline):
    """Greedy set cover algorithm for covering code."""
    # Represent words as integers for efficiency
    word_to_idx = {}
    idx_to_word = {}
    for i, w in enumerate(all_words):
        word_to_idx[tuple(w)] = i
        idx_to_word[i] = w
    
    # Precompute coverage sets
    coverage = {}
    for i, w in enumerate(all_words):
        neighbors = _neighbors(w, q, n, radius)
        covered = []
        for nb in neighbors:
            if tuple(nb) in word_to_idx:
                covered.append(word_to_idx[tuple(nb)])
        coverage[i] = covered
    
    # Greedy: repeatedly pick the word that covers the most uncovered words
    uncovered = set(range(len(all_words)))
    selected = []
    
    while uncovered:
        if time.time() > deadline - 5:  # Safety margin
            break
        
        best_idx = -1
        best_covered = None
        best_count = 0
        
        for i in range(len(all_words)):
            if i in selected:
                continue
            covered = coverage[i]
            count = sum(1 for c in covered if c in uncovered)
            if count > best_count:
                best_count = count
                best_idx = i
                best_covered = covered
        
        if best_idx == -1 or best_count == 0:
            break
        
        selected.append(best_idx)
        for c in best_covered:
            if c in uncovered:
                uncovered.remove(c)
    
    return [idx_to_word[i] for i in selected]

def _local_search(codewords, all_words, q, n, radius, deadline):
    """Try to remove codewords that are still covered by others."""
    if time.time() > deadline - 5:
        return codewords
    
    word_to_idx = {}
    for i, w in enumerate(all_words):
        word_to_idx[tuple(w)] = i
    
    coverage = {}
    for i, w in enumerate(all_words):
        neighbors = _neighbors(w, q, n, radius)
        covered = []
        for nb in neighbors:
            if tuple(nb) in word_to_idx:
                covered.append(word_to_idx[tuple(nb)])
        coverage[i] = covered
    
    # Try to remove each codeword and check if coverage is maintained
    for i in range(len(codewords)):
        if time.time() > deadline - 5:
            break
        
        # Check if removing codeword i still covers everything
        remaining = [j for j in range(len(codewords)) if j != i]
        covered = set()
        for j in remaining:
            covered.update(coverage[j])
        
        if len(covered) == len(all_words):
            codewords.pop(i)
            i -= 1  # Re-check same index
    
    return codewords

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # 5 second safety margin
    
    # Start with the tetracode direct sum construction
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # For small n, try greedy set cover
    if q ** n <= 100000:  # Only for manageable sizes
        all_words = [list(w) for w in itertools.product(range(q), repeat=n)]
        
        # Try greedy approach
        greedy_result = _greedy_set_cover(all_words, q, n, radius, deadline)
        
        if len(greedy_result) < len(initial_codewords):
            # Try local search to improve further
            improved = _local_search(greedy_result, all_words, q, n, radius, deadline)
            return improved
        else:
            return initial_codewords
    else:
        # For large spaces, just use the tetracode construction
        return initial_codewords
# EVOLVE-BLOCK-END
