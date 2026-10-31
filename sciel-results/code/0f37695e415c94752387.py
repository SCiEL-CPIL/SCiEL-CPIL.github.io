"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    if n == 0:
        return [[]]

    # For q=3, radius=1, use tetracode-based construction
    if q == 3 and radius == 1:
        if n == 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return tetracode
        
        # Start with direct sum construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
        # For n >= 5, apply local search to improve
        if n >= 5:
            code = _local_search_covering(q, n, radius, code, deadline)
        
        return code
    else:
        # General fallback: greedy construction
        return _greedy_covering(q, n, radius, deadline)


def _get_neighbors(c, q, n):
    """Get all words within Hamming distance 1 of c."""
    neighbors = [tuple(c)]
    for i in range(n):
        orig = c[i]
        for val in range(q):
            if val == orig:
                continue
            new_c = c[:]
            new_c[i] = val
            neighbors.append(tuple(new_c))
    return neighbors


def _build_coverage(code, q, n):
    """Build coverage count for all words in the space."""
    # Map each word to an index
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    coverage = np.zeros(len(all_words), dtype=np.int32)
    
    for c in code:
        ct = tuple(c)
        for w in _get_neighbors(ct, q, n):
            coverage[word_to_idx[w]] += 1
    
    return coverage, word_to_idx, all_words


def _local_search_covering(q, n, radius, code, deadline):
    """Local search to minimize code size while maintaining coverage."""
    if time.time() > deadline:
        return code
    
    # Build initial coverage
    coverage, word_to_idx, all_words = _build_coverage(code, q, n)
    
    # Precompute neighbors for all codewords
    def get_neighbors_tuple(ct):
        neighbors = [ct]
        for i in range(n):
            orig = ct[i]
            for val in range(q):
                if val == orig:
                    continue
                lst = list(ct)
                lst[i] = val
                neighbors.append(tuple(lst))
        return neighbors
    
    # Iterate removal
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for idx in range(len(code)):
            if time.time() > deadline:
                break
            ct = tuple(code[idx])
            neighbors = get_neighbors_tuple(ct)
            
            # Check if removing this codeword leaves all neighbors covered
            can_remove = True
            for w in neighbors:
                if coverage[word_to_idx[w]] <= 1:
                    can_remove = False
                    break
            
            if can_remove:
                # Remove codeword
                for w in neighbors:
                    coverage[word_to_idx[w]] -= 1
                code.pop(idx)
                changed = True
                break  # Restart scan since indices changed
    
    # Try swap-based local search to escape local optima
    # Try replacing one codeword with another to reduce size or maintain size with better structure
    random.seed(42)
    all_words_set = set(all_words)
    
    for iteration in range(50):  # Limited iterations
        if time.time() > deadline:
            break
        
        # Try to find a codeword that can be removed
        removable = []
        for idx in range(len(code)):
            ct = tuple(code[idx])
            neighbors = get_neighbors_tuple(ct)
            can_remove = True
            for w in neighbors:
                if coverage[word_to_idx[w]] <= 1:
                    can_remove = False
                    break
            if can_remove:
                removable.append(idx)
        
        if removable:
            # Remove the first removable codeword
            idx = removable[0]
            ct = tuple(code[idx])
            neighbors = get_neighbors_tuple(ct)
            for w in neighbors:
                coverage[word_to_idx[w]] -= 1
            code.pop(idx)
            continue
        
        # Try swap: replace codeword i with a random word that covers more
        if len(code) > 0:
            i = random.randrange(len(code))
            ci = tuple(code[i])
            neighbors_i = get_neighbors_tuple(ci)
            
            # Find words that are currently under-covered or try to improve
            # Try a random candidate word
            for attempt in range(20):
                if time.time() > deadline:
                    break
                candidate = random.choice(all_words)
                if candidate == ci:
                    continue
                
                neighbors_cand = get_neighbors_tuple(candidate)
                
                # Check: after removing ci and adding candidate, is coverage maintained?
                # A word w is safe if coverage[w] - (1 if w in neighbors_i else 0) + (1 if w in neighbors_cand else 0) >= 1
                
                # Quick check: words only covered by ci must also be covered by candidate
                only_covered_by_ci = []
                for w in neighbors_i:
                    if coverage[word_to_idx[w]] == 1:
                        only_covered_by_ci.append(w)
                
                valid = True
                for w in only_covered_by_ci:
                    if w not in neighbors_cand:
                        valid = False
                        break
                
                if valid:
                    # Perform swap
                    for w in neighbors_i:
                        coverage[word_to_idx[w]] -= 1
                    for w in neighbors_cand:
                        coverage[word_to_idx[w]] += 1
                    code[i] = list(candidate)
                    break
    
    # Final removal pass
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for idx in range(len(code)):
            if time.time() > deadline:
                break
            ct = tuple(code[idx])
            neighbors = get_neighbors_tuple(ct)
            can_remove = True
            for w in neighbors:
                if coverage[word_to_idx[w]] <= 1:
                    can_remove = False
                    break
            if can_remove:
                for w in neighbors:
                    coverage[word_to_idx[w]] -= 1
                code.pop(idx)
                changed = True
                break
    
    return code


def _greedy_covering(q, n, radius, deadline):
    """Greedy covering code construction."""
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    covered = set()
    code = []
    
    # Precompute neighborhoods for all words (expensive for large n)
    # For small n this is fine
    if len(all_words) <= 10000:
        neighborhoods = {}
        for w in all_words:
            neighborhoods[w] = _get_neighbors(w, q, n)
    
    while len(covered) < len(all_words) and time.time() < deadline:
        best_word = None
        best_gain = 0
        
        for w in all_words:
            if w in covered:
                continue
            if len(all_words) <= 10000:
                gain = len(set(neighborhoods[w]) - covered)
            else:
                gain = len(set(_get_neighbors(w, q, n)) - covered)
            
            if gain > best_gain:
                best_gain = gain
                best_word = w
        
        if best_word is None:
            break
        
        code.append(list(best_word))
        if len(all_words) <= 10000:
            covered.update(neighborhoods[best_word])
        else:
            covered.update(_get_neighbors(best_word, q, n))
    
    return code
# EVOLVE-BLOCK-END
