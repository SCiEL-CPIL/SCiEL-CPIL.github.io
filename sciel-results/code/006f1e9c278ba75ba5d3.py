"""Covering code K_3(6,1) via greedy set-cover with NumPy distance computation."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools


def _hamming_distances(codewords, all_words):
    """Compute Hamming distances from each codeword to all words.
    codewords: (M, n) int array
    all_words: (729, n) int array
    Returns: (M, 729) int array of distances.
    """
    # Broadcast: (M, 1, n) vs (1, 729, n)
    diff = (codewords[:, np.newaxis, :] != all_words[np.newaxis, :, :])
    return diff.sum(axis=2)


def _greedy_cover(q, n, radius, deadline):
    """Greedy set-cover: repeatedly pick codeword covering most uncovered words."""
    # Generate all words
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)
    total = len(all_words)
    
    covered = np.zeros(total, dtype=bool)
    codewords = []
    
    # Precompute all possible words as candidate codewords
    # We'll consider each word as a potential codeword
    # For efficiency, compute which words each candidate covers
    
    # Coverage: for each candidate word w, the set of words within Hamming distance radius
    # Precompute coverage masks for all 729 candidates
    # coverage[c, i] = 1 if word i is within radius of candidate c
    
    # Compute pairwise Hamming distances between all 729 words
    # This is 729x729x6 operations - doable with NumPy
    diff = (all_words[:, np.newaxis, :] != all_words[np.newaxis, :, :])
    dist_matrix = diff.sum(axis=2)  # (729, 729)
    
    # coverage_mask[c] = indices of words within radius of candidate c
    coverage_mask = (dist_matrix <= radius)  # (729, 729) boolean
    
    # Precompute coverage counts for each candidate
    # uncovered_count[c] = number of currently uncovered words within radius of c
    
    while not covered.all():
        if time.time() > deadline:
            break
        
        # For each candidate, count how many uncovered words it covers
        # uncovered_count[c] = sum over i of (not covered[i]) AND coverage_mask[c, i]
        # = sum of coverage_mask[c, :] * (not covered)
        uncovered_mask = ~covered
        uncovered_counts = (coverage_mask & uncovered_mask[np.newaxis, :]).sum(axis=1)
        
        # Pick candidate with max uncovered coverage
        best = np.argmax(uncovered_counts)
        if uncovered_counts[best] == 0:
            break  # No improvement possible
        
        # Add this codeword
        codeword = all_words[best]
        codewords.append(codeword.tolist())
        
        # Update covered
        covered[coverage_mask[best, :]] = True
    
    return codewords


def _local_search(codewords, q, n, radius, deadline):
    """Try to remove codewords that are redundant, then try swaps."""
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)
    total = len(all_words)
    
    # Compute coverage for current codewords
    cw = np.array(codewords, dtype=np.int32)
    
    # For each codeword, compute which words it uniquely covers
    # A codeword is redundant if all words it covers are also covered by others
    
    while time.time() < deadline:
        improved = False
        m = len(codewords)
        if m == 0:
            break
        
        cw = np.array(codewords, dtype=np.int32)
        
        # Compute coverage for each codeword
        diff = (cw[:, np.newaxis, :] != all_words[np.newaxis, :, :])
        dist = diff.sum(axis=2)  # (m, 729)
        cov = (dist <= radius)  # (m, 729)
        
        # Union of all coverages
        union = cov.any(axis=0)
        
        # For each codeword, check if removing it still covers everything
        for i in range(m):
            if time.time() > deadline:
                break
            # Check if all words covered by codeword i are also covered by others
            others_cov = cov[:i, :].any(axis=0) | cov[i+1:, :].any(axis=0)
            # Words that would become uncovered
            would_uncover = cov[i, :] & ~others_cov
            if not would_uncover.any():
                # Codeword i is redundant, remove it
                codewords.pop(i)
                improved = True
                break
        
        if not improved:
            # Try swap: replace one codeword with another to potentially enable further removals
            # For now, just break if no removals
            break
    
    return codewords


def construct(q, n, radius):
    # Deadline: 145s (5s safety margin from 150s limit)
    start_time = time.time()
    deadline = start_time + 145.0
    
    if q == 3 and n == 6 and radius == 1:
        # Specialized path for the main task
        codewords = _greedy_cover(q, n, radius, deadline)
        
        # Local search to remove redundant codewords
        if time.time() < deadline - 5.0:
            codewords = _local_search(codewords, q, n, radius, deadline - 5.0)
        
        return codewords
    else:
        # Fallback: product construction
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Brute force for small n
            all_words = list(itertools.product(range(q), repeat=n))
            return all_words
# EVOLVE-BLOCK-END
