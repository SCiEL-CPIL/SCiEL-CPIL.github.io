"""Improved covering code construction using greedy local search with swaps."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def _hamming_dist(a, b):
    """Compute Hamming distance between two tuples."""
    return sum(1 for x, y in zip(a, b) if x != y)


def _ball_radius1(cw, q, n):
    """Generate all words within Hamming distance 1 of cw."""
    cw = tuple(cw)
    result = {cw}
    for pos in range(n):
        for val in range(q):
            if val != cw[pos]:
                new_word = list(cw)
                new_word[pos] = val
                result.add(tuple(new_word))
    return result


def _initial_solution(q, n, radius):
    """Generate an initial covering code solution."""
    if q == 3 and n == 6 and radius == 1:
        # Start with a structured initial solution
        # Use the projected tetracode: 9 * 3^(6-3) = 243 words
        full = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        projected = [[w[0], w[1], w[2]] for w in full]
        rows = []
        for p in projected:
            for tail in itertools.product(range(3), repeat=n - 3):
                rows.append(list(p) + list(tail))
        return rows
    else:
        # Generic construction
        m = (q + radius) // (radius + 1)
        base = [[a, b, (a + b) % m, (a + 2 * b) % m] for a in range(m) for b in range(m)]
        rows = []
        if n >= 4:
            for h in base:
                for tail in itertools.product(range(q), repeat=n - 4):
                    rows.append(list(h) + list(tail))
        else:
            for w in itertools.product(range(q), repeat=n):
                rows.append(list(w))
        return rows


def _coverage_map(codewords, q, n, radius):
    """Build a map from each word to the list of codeword indices that cover it."""
    # For efficiency, use a dictionary mapping word -> list of indices
    coverage = {}
    for i, cw in enumerate(codewords):
        for word in _ball_radius1(cw, q, n):
            if word not in coverage:
                coverage[word] = []
            coverage[word].append(i)
    return coverage


def _is_redundant(cw_idx, codewords, coverage, q, n, radius):
    """Check if codeword at index cw_idx is redundant (all its neighbors covered by others)."""
    cw = tuple(codewords[cw_idx])
    for word in _ball_radius1(cw, q, n):
        # Check if this word is covered by any other codeword
        covering_indices = coverage.get(word, [])
        if covering_indices.count(cw_idx) == len(covering_indices):
            # Only this codeword covers this word
            return False
    return True


def _try_remove(codewords, coverage, q, n, radius, cw_idx):
    """Try to remove codeword at cw_idx if it's redundant. Returns (success, new_codewords, new_coverage)."""
    if not _is_redundant(cw_idx, codewords, coverage, q, n, radius):
        return False, codewords, coverage
    
    # Remove the codeword
    new_codewords = codewords[:cw_idx] + codewords[cw_idx+1:]
    
    # Rebuild coverage (expensive but necessary for correctness)
    new_coverage = _coverage_map(new_codewords, q, n, radius)
    
    return True, new_codewords, new_coverage


def _try_swap(codewords, coverage, q, n, radius, cw_idx, new_word):
    """Try to replace codeword at cw_idx with new_word. Returns (success, new_codewords, new_coverage)."""
    # Check if new_word is already in codewords
    if new_word in [tuple(cw) for cw in codewords]:
        return False, codewords, coverage
    
    # Check if the replacement maintains coverage
    # The new word must cover all words that the old word uniquely covers
    old_cw = tuple(codewords[cw_idx])
    old_ball = _ball_radius1(old_cw, q, n)
    new_ball = _ball_radius1(new_word, q, n)
    
    # Words that old_cw covers but new_word doesn't must be covered by other codewords
    for word in old_ball:
        if word not in new_ball:
            covering_indices = coverage.get(word, [])
            # Remove cw_idx from covering indices
            others = [i for i in covering_indices if i != cw_idx]
            if not others:
                # This word would become uncovered
                return False, codewords, coverage
    
    # Perform the swap
    new_codewords = codewords[:]
    new_codewords[cw_idx] = list(new_word)
    
    # Rebuild coverage
    new_coverage = _coverage_map(new_codewords, q, n, radius)
    
    return True, new_codewords, new_coverage


def _greedy_optimize(codewords, q, n, radius, deadline, max_iter=10000):
    """Greedy optimization: try to remove or swap codewords to reduce count."""
    start_time = time.time()
    codewords = [list(cw) for cw in codewords]
    
    if len(codewords) < 50:
        return codewords
    
    coverage = _coverage_map(codewords, q, n, radius)
    improved = True
    iterations = 0
    
    while improved and time.time() < deadline - 5 and iterations < max_iter:
        improved = False
        iterations += 1
        
        # Try to remove redundant codewords
        for i in range(len(codewords) - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            success, codewords, coverage = _try_remove(codewords, coverage, q, n, radius, i)
            if success:
                improved = True
                break
        
        if improved:
            continue
        
        # Try swaps: replace a codeword with a nearby word that might allow further removal
        if len(codewords) > 80:
            # Randomly try some swaps
            for _ in range(10):
                if time.time() > deadline - 5:
                    break
                idx = random.randint(0, len(codewords) - 1)
                # Generate a random nearby word (distance 1 or 2)
                base = codewords[idx][:]
                if random.random() < 0.5:
                    # Distance 1
                    pos = random.randint(0, n - 1)
                    val = random.randint(0, q - 1)
                    if val != base[pos]:
                        base[pos] = val
                else:
                    # Distance 2
                    if n >= 2:
                        pos1, pos2 = random.sample(range(n), 2)
                        val1 = random.randint(0, q - 1)
                        val2 = random.randint(0, q - 1)
                        if val1 != base[pos1] and val2 != base[pos2]:
                            base[pos1] = val1
                            base[pos2] = val2
                
                success, codewords, coverage = _try_swap(codewords, coverage, q, n, radius, idx, tuple(base))
                if success:
                    improved = True
                    break
    
    return codewords


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Generate initial solution
    codewords = _initial_solution(q, n, radius)
    
    # Optimize
    if time.time() < deadline - 10:
        codewords = _greedy_optimize(codewords, q, n, radius, deadline)
    
    return codewords
# EVOLVE-BLOCK-END
