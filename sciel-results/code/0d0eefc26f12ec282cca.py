# EVOLVE-BLOCK-START
import time
import random
import itertools
import math

import numpy as np

# Precompute the ternary tetracode (9 words of length 4, perfect covering radius 1)
_TETRA = None

def _get_tetracode():
    global _TETRA
    if _TETRA is None:
        _TETRA = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    return _TETRA


def _base_seed(q, n, radius):
    """Direct sum of tetracode with all words on remaining coords."""
    tetracode = _get_tetracode()
    result = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            result.append(h + list(tail))
    return result


def _hamming_dist(a, b):
    return sum(x != y for x, y in zip(a, b))


def _all_neighbors(word, q, n):
    """Yield the word and all words at Hamming distance 1."""
    yield word
    for i in range(n):
        for v in range(q):
            if v == word[i]:
                continue
            nb = list(word)
            nb[i] = v
            yield tuple(nb)


def _is_covering(code, q, n, radius=1):
    """Check if code covers all of {0..q-1}^n within radius."""
    if not code:
        return False
    code_set = set(tuple(c) for c in code)
    # For radius 1, a word is covered if it is in code_set or differs in exactly 1 position
    # We can check by iterating all words
    for word in itertools.product(range(q), repeat=n):
        if word in code_set:
            continue
        # Check distance 1
        covered = False
        for i in range(n):
            for v in range(q):
                if v == word[i]:
                    continue
                nb = list(word)
                nb[i] = v
                if tuple(nb) in code_set:
                    covered = True
                    break
            if covered:
                break
        if not covered:
            return False
    return True


def _coverage_count(code, q, n, radius=1):
    """Count how many words are covered by the code (for radius 1)."""
    if not code:
        return 0
    code_set = set(tuple(c) for c in code)
    covered = set()
    for c in code:
        for nb in _all_neighbors(list(c), q, n):
            covered.add(nb)
    return len(covered)


def _uncovered_words(code, q, n, radius=1):
    """Return list of words not covered by code."""
    if not code:
        return list(itertools.product(range(q), repeat=n))
    code_set = set(tuple(c) for c in code)
    covered = set()
    for c in code:
        for nb in _all_neighbors(list(c), q, n):
            covered.add(nb)
    return [w for w in itertools.product(range(q), repeat=n) if w not in covered]


def _greedy_construct(q, n, radius, deadline, rng):
    """Greedy set-cover style construction with local search."""
    # Start with a random subset of the base seed
    base = _base_seed(q, n, radius)
    rng.shuffle(base)
    
    # Start with a small random subset
    initial_size = max(1, len(base) // 4)
    code = list(base[:initial_size])
    
    best_code = list(code)
    best_size = len(code)
    
    # Greedily add codewords that cover the most uncovered words
    uncovered = _uncovered_words(code, q, n, radius)
    
    while uncovered and time.time() < deadline:
        # Find the codeword from remaining base that covers the most uncovered words
        remaining = [c for c in base if c not in code]
        if not remaining:
            break
        
        best_gain = 0
        best_cw = None
        for cw in remaining:
            # Count how many uncovered words are within radius
            gain = 0
            for nb in _all_neighbors(list(cw), q, n):
                if nb in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_cw = cw
            if best_gain == len(uncovered):
                break
        
        if best_cw is None or best_gain == 0:
            break
        
        code.append(best_cw)
        uncovered = _uncovered_words(code, q, n, radius)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)
    
    # Try to remove redundant codewords
    if time.time() < deadline:
        for i in range(len(code)):
            if time.time() > deadline:
                break
            trial = code[:i] + code[i+1:]
            if _is_covering(trial, q, n, radius):
                code = trial
                i = -1  # restart from beginning since indices shifted
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = list(code)
    
    return best_code


def _local_search(code, q, n, radius, deadline, rng):
    """Try to improve the code by swapping codewords."""
    code = list(code)
    best_code = list(code)
    best_size = len(code)
    
    base = _base_seed(q, n, radius)
    code_set = set(tuple(c) for c in code)
    
    attempts = 0
    while time.time() < deadline and attempts < 5000:
        attempts += 1
        # Pick a random codeword to potentially remove
        i = rng.randrange(len(code))
        removed = code[i]
        trial = code[:i] + code[i+1:]
        
        if not _is_covering(trial, q, n, radius):
            # Need to add something back; try adding a random codeword
            j = rng.randrange(len(base))
            candidate = base[j]
            if tuple(candidate) in code_set:
                continue
            trial2 = trial + [candidate]
            if len(trial2) < len(code):
                if _is_covering(trial2, q, n, radius):
                    code = trial2
                    code_set = set(tuple(c) for c in code)
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = list(code)
        else:
            code = trial
            code_set = set(tuple(c) for c in code)
            if len(code) < best_size:
                best_size = len(code)
                best_code = list(code)
    
    return best_code


def construct(q, n, radius):
    """Construct a covering code for {0..q-1}^n with given radius."""
    start_time = time.time()
    deadline = start_time + 100.0  # Leave 20s safety margin
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback: use the direct sum approach
        return _base_seed(q, n, radius)
    
    rng = random.Random(42)
    
    # Get the base seed (2187 words)
    base = _base_seed(q, n, radius)
    
    # Run multiple greedy constructions with different random seeds
    best_code = None
    best_size = len(base)
    
    for seed in range(10):
        if time.time() > deadline:
            break
        rng_seed = 42 + seed * 1000
        rng_i = random.Random(rng_seed)
        
        # Create a shuffled copy
        shuffled = list(base)
        rng_i.shuffle(shuffled)
        
        # Greedy construction
        code = []
        initial_size = max(1, len(shuffled) // 4)
        code = list(shuffled[:initial_size])
        
        uncovered = _uncovered_words(code, q, n, radius)
        
        while uncovered and time.time() < deadline:
            remaining = [c for c in shuffled if c not in code]
            if not remaining:
                break
            
            best_gain = 0
            best_cw = None
            for cw in remaining:
                gain = 0
                for nb in _all_neighbors(list(cw), q, n):
                    if nb in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_cw = cw
                if best_gain == len(uncovered):
                    break
            
            if best_cw is None or best_gain == 0:
                break
            
            code.append(best_cw)
            uncovered = _uncovered_words(code, q, n, radius)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)
        
        # Local search to remove redundancies
        if time.time() < deadline:
            improved = _local_search(code, q, n, radius, deadline, rng_i)
            if len(improved) < best_size:
                best_size = len(improved)
                best_code = list(improved)
    
    # Final redundancy removal pass
    if best_code is not None and time.time() < deadline:
        code = list(best_code)
        for i in range(len(code)):
            if time.time() > deadline:
                break
            trial = code[:i] + code[i+1:]
            if _is_covering(trial, q, n, radius):
                code = trial
                i = -1
        
        if len(code) < best_size:
            best_code = list(code)
    
    if best_code is None:
        best_code = base
    
    return best_code
# EVOLVE-BLOCK-END
