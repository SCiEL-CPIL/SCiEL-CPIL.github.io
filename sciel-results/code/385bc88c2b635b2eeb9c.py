"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def construct(q, n, radius):
    """Construct a covering code of length n over alphabet {0..q-1} with covering radius.
    
    Uses greedy set cover with removal-based optimization for the ternary case.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback: direct sum construction for other parameters
        # For general case, use a simple greedy approach
        return _greedy_construct_general(q, n, radius, deadline)
    
    # Specialized for K_3(6,1)
    return _construct_k3_6_1(deadline)


def _greedy_construct_general(q, n, radius, deadline):
    """General greedy set cover for covering codes."""
    all_words = list(itertools.product(range(q), repeat=n))
    total = q ** n
    
    # Precompute neighborhoods
    def hamming_dist(a, b):
        return sum(1 for i in range(n) if a[i] != b[i])
    
    # Greedy: pick codeword that covers most uncovered words
    uncovered = set(all_words)
    codewords = []
    
    while uncovered and time.time() < deadline - 5:
        best_word = None
        best_count = -1
        for w in uncovered:
            count = 0
            for d in range(radius + 1):
                # Count words at distance d from w that are uncovered
                # This is expensive, so use sampling for large spaces
                pass
            # Simplified: just count neighbors
            for v in uncovered:
                if hamming_dist(w, v) <= radius:
                    count += 1
            if count > best_count:
                best_count = count
                best_word = w
        
        if best_word is None:
            break
        
        codewords.append(best_word)
        # Remove covered words
        to_remove = set()
        for v in uncovered:
            if hamming_dist(best_word, v) <= radius:
                to_remove.add(v)
        uncovered -= to_remove
    
    return [list(w) for w in codewords]


def _construct_k3_6_1(deadline):
    """Specialized construction for K_3(6,1) using greedy + removal optimization."""
    
    # Start with the direct sum construction (81 codewords)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=2)]
    
    # Precompute all 729 words and their indices
    all_words = list(itertools.product(range(3), repeat=6))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Build coverage: for each codeword, which words does it cover?
    # A word is covered if Hamming distance <= 1
    def covers(cw, w):
        diff = sum(1 for i in range(6) if cw[i] != w[i])
        return diff <= 1
    
    # Coverage matrix: cover_matrix[i][j] = True if codeword i covers word j
    # But this is 81 x 729, manageable
    coverage = []
    for cw in codewords:
        cov = set()
        for w in all_words:
            if covers(cw, w):
                cov.add(w)
        coverage.append(cov)
    
    # Greedy removal: try to remove codewords that don't leave any word uncovered
    # First, compute for each word, how many codewords cover it
    word_cover_count = {w: 0 for w in all_words}
    word_coverers = {w: [] for w in all_words}
    
    for i, cw in enumerate(codewords):
        for w in coverage[i]:
            word_cover_count[w] += 1
            word_coverers[w].append(i)
    
    # Try to remove codewords in order of "least essential"
    # A codeword is removable if every word it covers is also covered by another codeword
    removed = [False] * len(codewords)
    
    def is_removable(i):
        if removed[i]:
            return False
        for w in coverage[i]:
            # Check if w is covered by some other non-removed codeword
            for j in word_coverers[w]:
                if j != i and not removed[j]:
                    break
            else:
                return False
        return True
    
    # Repeatedly remove removable codewords
    changed = True
    while changed and time.time() < deadline - 10:
        changed = False
        for i in range(len(codewords)):
            if not removed[i] and is_removable(i):
                removed[i] = True
                changed = True
    
    # Now try swap-based local search: replace one codeword with another to enable more removals
    # This is more complex, so we'll do a simpler improvement:
    # Try adding new codewords that cover many currently singly-covered words
    
    # Compute current coverage after removals
    active_indices = [i for i in range(len(codewords)) if not removed[i]]
    current_cover_count = {w: 0 for w in all_words}
    for i in active_indices:
        for w in coverage[i]:
            current_cover_count[w] += 1
    
    # Find words that are covered by exactly 1 codeword
    singly_covered = [w for w in all_words if current_cover_count[w] == 1]
    
    # Try to add codewords that cover many singly-covered words
    # But we want to MINIMIZE, so instead try to find replacements
    
    # Better approach: use a greedy algorithm to find a smaller covering code from scratch
    # using the structure of the problem
    
    # For now, return the result of removal
    result = [codewords[i] for i in active_indices]
    
    # Verify coverage
    def is_covered(w, cws):
        for cw in cws:
            if covers(cw, w):
                return True
        return False
    
    # If not fully covered (shouldn't happen), add back removed codewords
    if not all(is_covered(w, result) for w in all_words):
        # Fallback: return all codewords
        return codewords
    
    return result
# EVOLVE-BLOCK-END
