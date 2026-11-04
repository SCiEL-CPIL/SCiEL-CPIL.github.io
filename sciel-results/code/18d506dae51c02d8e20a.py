"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random


def _generate_sphere(word, q, n, radius):
    """Generate all words within Hamming distance `radius` of `word`."""
    # For radius 1, we can be very efficient
    if radius == 0:
        return [word]
    if radius == 1:
        result = [word[:]]
        for i in range(n):
            for v in range(q):
                if v != word[i]:
                    w = word[:]
                    w[i] = v
                    result.append(w)
        return result
    # General case (not expected for this task)
    result = [word[:]]
    for dist in range(1, radius + 1):
        new_words = []
        for w in result:
            for i in range(n):
                for v in range(q):
                    if v != w[i]:
                        nw = w[:]
                        nw[i] = v
                        new_words.append(nw)
        # Deduplicate
        seen = set(tuple(w) for w in result)
        for nw in new_words:
            t = tuple(nw)
            if t not in seen:
                seen.add(t)
                result.append(nw)
    return result


def _word_to_idx(word, q, n):
    """Convert a word to a linear index in [0, q^n)."""
    idx = 0
    for v in word:
        idx = idx * q + v
    return idx


def _idx_to_word(idx, q, n):
    """Convert a linear index back to a word."""
    word = [0] * n
    for i in range(n - 1, -1, -1):
        word[i] = idx % q
        idx //= q
    return word


def _compute_covered_set(codewords, q, n, radius):
    """Return a set of indices of all words covered by the given codewords."""
    covered = set()
    for cw in codewords:
        for w in _generate_sphere(cw, q, n, radius):
            covered.add(_word_to_idx(w, q, n))
    return covered


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Start with the direct sum construction (upper bound)
    # For q=3, n=8, radius=1: use tetracode direct sum
    if q == 3 and n == 8 and radius == 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    else:
        # Fallback: greedy construction
        codewords = []
        uncovered = set(range(q ** n))
        while uncovered:
            # Pick a random uncovered word as center
            idx = random.choice(list(uncovered))
            word = _idx_to_word(idx, q, n)
            sphere = _generate_sphere(word, q, n, radius)
            for w in sphere:
                uncovered.discard(_word_to_idx(w, q, n))
            codewords.append(word)
    
    # Compute initial coverage
    total_space = q ** n
    covered = _compute_covered_set(codewords, q, n, radius)
    
    # Local search: try to remove codewords that are redundant
    # A codeword is redundant if all words in its sphere are covered by other codewords
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        # Try removing each codeword
        for i in range(len(codewords) - 1, -1, -1):
            if time.time() > deadline - 5.0:
                break
            # Check if codeword i is redundant
            # Compute coverage without codeword i
            remaining = codewords[:i] + codewords[i+1:]
            covered_without_i = _compute_covered_set(remaining, q, n, radius)
            if len(covered_without_i) == len(covered):
                # Codeword i is redundant
                codewords = remaining
                covered = covered_without_i
                improved = True
    
    # Try swaps: replace a codeword with a neighbor to increase coverage
    for _ in range(50):  # Limit number of swap attempts
        if time.time() > deadline - 10.0:
            break
        if not codewords:
            break
        i = random.randint(0, len(codewords) - 1)
        old_cw = codewords[i]
        # Try moving to a random neighbor
        pos = random.randint(0, n - 1)
        new_val = random.randint(0, q - 1)
        if new_val == old_cw[pos]:
            continue
        new_cw = old_cw[:]
        new_cw[pos] = new_val
        
        # Check if this swap improves coverage
        remaining = [cw for j, cw in enumerate(codewords) if j != i]
        covered_before = len(covered)
        covered_after = _compute_covered_set(remaining + [new_cw], q, n, radius)
        if len(covered_after) > covered_before:
            codewords[i] = new_cw
            covered = covered_after
            improved = True
    
    return codewords
# EVOLVE-BLOCK-END
