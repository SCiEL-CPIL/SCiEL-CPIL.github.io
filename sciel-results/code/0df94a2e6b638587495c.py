"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import product

# Precompute the tetracode
_TETRACODE = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
_TETRA_SET = set(tuple(c) for c in _TETRACODE)

def _build_initial_code(q, n, radius):
    """Build a valid covering code using the tetracode product construction."""
    if n < 4:
        # For small n, just enumerate everything
        return [list(p) for p in product(range(q), repeat=n)]
    
    code = []
    for head in _TETRACODE:
        for tail in product(range(q), repeat=n - 4):
            code.append(head + list(tail))
    return code

def _hamming_distance(a, b):
    return sum(x != y for x, y in zip(a, b))

def _is_covering(code, q, n, radius):
    """Check if code covers all words within radius."""
    if not code:
        return True
    code_set = set(tuple(c) for c in code)
    for word in product(range(q), repeat=n):
        found = False
        for c in code_set:
            if _hamming_distance(word, c) <= radius:
                found = True
                break
        if not found:
            return False
    return True

def _greedy_cover(code, q, n, radius):
    """Greedy: keep only codewords that cover at least one word not covered by others."""
    if not code:
        return code
    code_list = list(code)
    m = len(code_list)
    if m == 0:
        return code_list
    
    # For each codeword, count how many words it uniquely covers
    # A word is uniquely covered by c if no other codeword covers it
    # This is expensive, so we use a heuristic: keep codewords in order of "importance"
    
    # Simple approach: iteratively try to remove each codeword
    result = list(code_list)
    changed = True
    while changed:
        changed = False
        for i in range(len(result)):
            candidate = result[:i] + result[i+1:]
            if _is_covering(candidate, q, n, radius):
                result = candidate
                changed = True
                break
    return result

def _local_search(code, q, n, radius, time_limit=10):
    """Local search: try swapping one codeword for another to reduce size."""
    if not code:
        return code
    code_list = list(code)
    best = list(code_list)
    start = time.time()
    
    # Precompute all words
    all_words = list(product(range(q), repeat=n))
    
    # Try removing each codeword
    for i in range(len(best)):
        if time.time() - start > time_limit:
            break
        candidate = best[:i] + best[i+1:]
        if _is_covering(candidate, q, n, radius):
            best = candidate
    
    # Try adding one codeword and removing two (if that helps)
    # This is more complex, skip for now
    
    return best

def _random_perturbation(code, q, n, radius, num_perturb=5):
    """Randomly perturb the code and try to repair it."""
    if not code:
        return code
    code_list = list(code)
    best = list(code_list)
    
    for _ in range(num_perturb):
        # Remove a random codeword
        if len(best) > 1:
            idx = random.randint(0, len(best) - 1)
            candidate = best[:idx] + best[idx+1:]
            # Try to repair by adding a random word
            if not _is_covering(candidate, q, n, radius):
                # Find uncovered words
                uncovered = []
                for w in product(range(q), repeat=n):
                    if not any(_hamming_distance(w, c) <= radius for c in candidate):
                        uncovered.append(w)
                if uncovered:
                    # Add a random uncovered word
                    new_word = random.choice(uncovered)
                    candidate.append(list(new_word))
            if len(candidate) < len(best) and _is_covering(candidate, q, n, radius):
                best = candidate
    
    return best

def construct(q, n, radius):
    """Construct a covering code of length n over alphabet size q with covering radius radius."""
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Build initial code using tetracode product
    code = _build_initial_code(q, n, radius)
    
    # Greedy reduction
    code = _greedy_cover(code, q, n, radius)
    
    # Local search
    if time.time() < deadline - 20:
        code = _local_search(code, q, n, radius, time_limit=min(15, deadline - time.time() - 10))
    
    # Random perturbation
    if time.time() < deadline - 10:
        code = _random_perturbation(code, q, n, radius, num_perturb=3)
    
    # Final greedy pass
    if time.time() < deadline - 5:
        code = _greedy_cover(code, q, n, radius)
    
    return code
# EVOLVE-BLOCK-END
