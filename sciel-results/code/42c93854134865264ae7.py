"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def _build_cover_structure(q, n, radius, codewords):
    """
    Build data structures for fast coverage checking.
    Returns:
        cover_count: array of shape (q**n,) counting how many codewords cover each word
        codeword_indices: list of indices
    """
    total_words = q ** n
    cover_count = np.zeros(total_words, dtype=np.int32)
    
    # Precompute all neighbors for each codeword
    # For radius 1, neighbors are the word itself plus all words at Hamming distance 1
    # Number of neighbors = 1 + n*(q-1)
    
    # For efficiency, we'll compute the index of each neighbor
    # Index of a word (w0, w1, ..., w_{n-1}) in base-q is sum(wi * q**(n-1-i))
    
    powers = [q ** (n - 1 - i) for i in range(n)]
    
    for cw in codewords:
        base_idx = 0
        for i in range(n):
            base_idx += cw[i] * powers[i]
        cover_count[base_idx] += 1
        
        # Add all neighbors at Hamming distance 1
        for pos in range(n):
            old_val = cw[pos]
            for new_val in range(q):
                if new_val == old_val:
                    continue
                neighbor_idx = base_idx + (new_val - old_val) * powers[pos]
                cover_count[neighbor_idx] += 1
    
    return cover_count


def _is_cover(cover_count):
    """Check if all words are covered (cover_count >= 1 for all)."""
    return np.all(cover_count >= 1)


def _greedy_cover(q, n, radius, time_limit):
    """
    Greedy construction: repeatedly pick the word that covers the most uncovered words.
    """
    start_time = time.time()
    total_words = q ** n
    powers = [q ** (n - 1 - i) for i in range(n)]
    
    # Track uncovered words
    uncovered = np.ones(total_words, dtype=bool)
    uncovered_count = total_words
    codewords = []
    
    # Precompute neighbor lists for all words? That's too much memory for large spaces.
    # Instead, for each candidate word, compute how many uncovered words it covers.
    
    # For radius 1, a word covers itself and all words at Hamming distance 1.
    # We need to efficiently find the word that maximizes coverage of uncovered words.
    
    # Strategy: maintain a count of how many times each word is "needed" (i.e., how many
    # uncovered words it would cover). Use a priority approach.
    
    # For small enough spaces, we can be more aggressive.
    # For q=3, n=8, total_words = 6561, which is manageable.
    
    # Precompute for each word, its neighbors (including itself) as indices
    # This is 6561 * (1 + 8*2) = 6561 * 17 = ~111k entries, very manageable.
    
    neighbors = {}
    for idx in range(total_words):
        # Convert index to word
        word = []
        temp = idx
        for i in range(n):
            word.append(temp % q)
            temp //= q
        word.reverse()
        
        neigh = [idx]
        for pos in range(n):
            old_val = word[pos]
            for new_val in range(q):
                if new_val == old_val:
                    continue
                neighbor_idx = idx + (new_val - old_val) * powers[pos]
                neigh.append(neighbor_idx)
        neighbors[idx] = neigh
    
    # Now greedily pick words
    while uncovered_count > 0:
        if time.time() - start_time > time_limit:
            break
        
        best_idx = -1
        best_coverage = 0
        
        for idx in range(total_words):
            if not uncovered[idx]:
                continue
            # Count how many uncovered words this word covers
            coverage = 0
            for neighbor in neighbors[idx]:
                if uncovered[neighbor]:
                    coverage += 1
            if coverage > best_coverage:
                best_coverage = coverage
                best_idx = idx
        
        if best_idx == -1:
            break
        
        # Add this word
        word = []
        temp = best_idx
        for i in range(n):
            word.append(temp % q)
            temp //= q
        word.reverse()
        codewords.append(word)
        
        # Update uncovered
        for neighbor in neighbors[best_idx]:
            if uncovered[neighbor]:
                uncovered[neighbor] = False
                uncovered_count -= 1
    
    return codewords


def _local_search(q, n, radius, codewords, time_limit):
    """
    Local search: try to remove codewords and swap to improve coverage.
    """
    start_time = time.time()
    total_words = q ** n
    powers = [q ** (n - 1 - i) for i in range(n)]
    
    # Precompute neighbors for all words
    neighbors = {}
    for idx in range(total_words):
        word = []
        temp = idx
        for i in range(n):
            word.append(temp % q)
            temp //= q
        word.reverse()
        
        neigh = [idx]
        for pos in range(n):
            old_val = word[pos]
            for new_val in range(q):
                if new_val == old_val:
                    continue
                neighbor_idx = idx + (new_val - old_val) * powers[pos]
                neigh.append(neighbor_idx)
        neighbors[idx] = neigh
    
    # Build initial cover count
    cover_count = np.zeros(total_words, dtype=np.int32)
    cw_indices = []
    for cw in codewords:
        idx = 0
        for i in range(n):
            idx += cw[i] * powers[i]
        cw_indices.append(idx)
        for neighbor in neighbors[idx]:
            cover_count[neighbor] += 1
    
    # Try to remove codewords
    improved = True
    while improved:
        if time.time() - start_time > time_limit:
            break
        improved = False
        
        # Find a codeword that can be removed
        for i in range(len(cw_indices) - 1, -1, -1):
            idx = cw_indices[i]
            # Check if removing this codeword leaves all words covered
            # A word is still covered if cover_count[neighbor] >= 2 for all neighbors
            can_remove = True
            for neighbor in neighbors[idx]:
                if cover_count[neighbor] < 2:
                    can_remove = False
                    break
            
            if can_remove:
                # Remove it
                for neighbor in neighbors[idx]:
                    cover_count[neighbor] -= 1
                cw_indices.pop(i)
                codewords.pop(i)
                improved = True
                break
    
    # Try swaps: remove one, add another
    for _ in range(100):  # Limit iterations
        if time.time() - start_time > time_limit:
            break
        
        # Find a codeword whose removal would cause uncovered words
        # Then try to find a replacement that covers those uncovered words
        for i in range(len(cw_indices)):
            idx = cw_indices[i]
            # Find words that would become uncovered if we remove this
            would_uncover = []
            for neighbor in neighbors[idx]:
                if cover_count[neighbor] == 1:
                    would_uncover.append(neighbor)
            
            if not would_uncover:
                # Can remove without replacement
                for neighbor in neighbors[idx]:
                    cover_count[neighbor] -= 1
                cw_indices.pop(i)
                codewords.pop(i)
                break
            
            # Try to find a replacement
            best_replace = -1
            best_new_coverage = 0
            
            # Sample some candidates
            candidates = random.sample(range(total_words), min(500, total_words))
            for cand_idx in candidates:
                if cand_idx in cw_indices:
                    continue
                # Count how many of would_uncover this candidate covers
                new_coverage = 0
                for wu in would_uncover:
                    if wu in neighbors[cand_idx]:
                        new_coverage += 1
                if new_coverage > best_new_coverage:
                    best_new_coverage = new_coverage
                    best_replace = cand_idx
            
            if best_replace >= 0 and best_new_coverage == len(would_uncover):
                # Do the swap
                for neighbor in neighbors[idx]:
                    cover_count[neighbor] -= 1
                for neighbor in neighbors[best_replace]:
                    cover_count[neighbor] += 1
                cw_indices[i] = best_replace
                # Update codewords list
                word = []
                temp = best_replace
                for j in range(n):
                    word.append(temp % q)
                    temp //= q
                word.reverse()
                codewords[i] = word
                break
    
    return codewords


def construct(q, n, radius):
    """
    Construct a covering code for the given parameters.
    """
    start_time = time.time()
    total_time_limit = 110  # Leave margin before 120s kill
    
    # Start with the tetracode-based construction
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    if n >= 4:
        import itertools
        codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    else:
        # Fallback: brute force for small n
        codewords = [list(word) for word in itertools.product(range(q), repeat=n)]
    
    # Phase 1: Greedy improvement
    time_for_greedy = min(30, total_time_limit - (time.time() - start_time))
    if time_for_greedy > 5:
        greedy_result = _greedy_cover(q, n, radius, time_for_greedy)
        if len(greedy_result) < len(codewords):
            codewords = greedy_result
    
    # Phase 2: Local search
    time_elapsed = time.time() - start_time
    time_for_ls = min(60, total_time_limit - time_elapsed)
    if time_for_ls > 5:
        codewords = _local_search(q, n, radius, codewords, time_for_ls)
    
    return codewords
# EVOLVE-BLOCK-END
