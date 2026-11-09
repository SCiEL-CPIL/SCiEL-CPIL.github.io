"""
Ternary covering code K_3(6,1) construction.
Uses a greedy set-cover algorithm with efficient bitset-based coverage tracking.
Falls back to the direct-sum construction (81 words) if time runs out.
"""
import time
import numpy as np

def _direct_sum_construction(q, n, radius):
    """Fallback: direct sum of ternary tetracode with all words on remaining coords."""
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    import itertools
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]


def _neighbors_of(word, q, n):
    """Generate all words within Hamming distance 1 of `word` (including itself)."""
    neighbors = [list(word)]
    for i in range(n):
        for v in range(q):
            if v != word[i]:
                w = list(word)
                w[i] = v
                neighbors.append(w)
    return neighbors


def _word_to_int(word, q, n):
    """Convert a word (list of ints) to a unique integer index."""
    idx = 0
    for i, v in enumerate(word):
        idx = idx * q + v
    return idx


def _int_to_word(idx, q, n):
    """Convert integer index back to word."""
    word = [0] * n
    for i in range(n - 1, -1, -1):
        word[i] = idx % q
        idx //= q
    return word


def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet {0..q-1} with covering radius `radius`.
    Returns a list of codewords (each a list of n integers).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s (evaluator kills at 150s)
    
    total_words = q ** n
    
    # For the specific case q=3, n=6, radius=1:
    # total_words = 729
    # Each codeword covers 1 + n*(q-1) = 1 + 6*2 = 13 words
    # Lower bound: ceil(729/13) = 57 (but best known is 73)
    
    # Generate all words
    import itertools
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {}
    for i, w in enumerate(all_words):
        word_to_idx[tuple(w)] = i
    
    # Precompute neighbor sets for each word (as frozensets of indices)
    # For q=3, n=6: 729 words, each with 13 neighbors -> manageable
    neighbor_sets = []
    for w in all_words:
        wt = tuple(w)
        nbrs = set()
        # Self
        nbrs.add(word_to_idx[wt])
        # Single changes
        for i in range(n):
            for v in range(q):
                if v != w[i]:
                    nw = list(w)
                    nw[i] = v
                    nbrs.add(word_to_idx[tuple(nw)])
        neighbor_sets.append(frozenset(nbrs))
    
    # Greedy set cover
    uncovered = set(range(total_words))
    codewords = []
    
    # Precompute for each potential codeword, which uncovered words it covers
    # We'll track this dynamically
    
    # Initial candidate scores: for each word, how many uncovered words does it cover?
    # Initially all 729 are uncovered, so score = len(neighbor_sets[i]) = 13 for all
    # But we need to be smarter: use the structure.
    
    # Greedy: repeatedly pick the codeword that covers the most uncovered words
    # To speed up, we can use the fact that all codewords cover the same number initially.
    # But as we add codewords, the marginal coverage changes.
    
    # For efficiency, let's use a different approach:
    # Start with a good initial set, then try to improve by removing redundant codewords.
    
    # Strategy 1: Start with the direct-sum construction (81 words), then try to reduce.
    initial_code = _direct_sum_construction(q, n, radius)
    
    # Check if initial_code is valid
    covered = set()
    for cw in initial_code:
        cwt = tuple(cw)
        idx = word_to_idx.get(cwt)
        if idx is not None:
            covered.update(neighbor_sets[idx])
    
    if len(covered) == total_words:
        # Try to remove redundant codewords
        code_set = set(tuple(cw) for cw in initial_code)
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for cw_tuple in list(code_set):
                # Try removing this codeword
                if len(covered) <= total_words:  # still valid
                    # Compute what this codeword uniquely covers
                    idx = word_to_idx[cw_tuple]
                    # Find words covered only by this codeword
                    unique = set()
                    for nbr_idx in neighbor_sets[idx]:
                        # Check if nbr_idx is covered by any other codeword
                        is_covered_by_other = False
                        for other in code_set:
                            if other == cw_tuple:
                                continue
                            other_idx = word_to_idx[other]
                            if nbr_idx in neighbor_sets[other_idx]:
                                is_covered_by_other = True
                                break
                        if not is_covered_by_other:
                            unique.add(nbr_idx)
                    
                    if not unique:
                        # Safe to remove
                        code_set.remove(cw_tuple)
                        changed = True
                        break  # Restart iteration since coverage changed
    
        # Convert back to list
        result = [list(cw) for cw in code_set]
        # Verify
        covered = set()
        for cw in result:
            idx = word_to_idx[tuple(cw)]
            covered.update(neighbor_sets[idx])
        if len(covered) == total_words:
            return result
    
    # Strategy 2: Greedy set cover from scratch
    # Reset uncovered
    uncovered = set(range(total_words))
    codewords = []
    
    # For each candidate, compute how many currently-uncovered words it covers
    # Use numpy for speed where possible
    
    # Precompute neighbor index arrays
    neighbor_indices = np.array([list(ns) for ns in neighbor_sets])  # shape (729, 13)
    
    while uncovered and time.time() < deadline:
        # Find the codeword that covers the most uncovered words
        best_idx = -1
        best_count = 0
        for i in range(total_words):
            count = 0
            for nbr in neighbor_indices[i]:
                if nbr in uncovered:
                    count += 1
            if count > best_count:
                best_count = count
                best_idx = i
            if time.time() > deadline - 5:
                break
        if best_idx < 0 or best_count == 0:
            break
        
        codewords.append(list(all_words[best_idx]))
        for nbr in neighbor_indices[best_idx]:
            uncovered.discard(nbr)
    
    if not uncovered:
        return codewords
    
    # If greedy didn't finish in time, fall back to direct sum
    return _direct_sum_construction(q, n, radius)
