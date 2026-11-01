"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time


def _hamming_dist(a, b):
    d = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            d += 1
            if d > 1:
                return d
    return d


def _is_redundant(cw, codewords, idx, n, q, radius):
    """Check if removing codeword at idx still leaves all its covered words covered by others."""
    # For each word within radius of cw, check if some other codeword covers it
    # We only need to check words that are at distance exactly radius from cw
    # (words at distance < radius from cw are also within radius of cw, but might
    # only be covered by cw if no other codeword is within radius)
    # Actually, we need to check ALL words within radius of cw.
    # A word w is within radius of cw. It's still covered if some other cw' has dist(w, cw') <= radius.
    
    # Generate all words within Hamming distance `radius` of cw
    # For radius=1, n=6: the word itself + 6*(q-1) neighbors = 1 + 12 = 13 words
    # But we can be smarter: only check words at distance exactly radius from cw,
    # because words at distance < radius from cw are "interior" and more likely covered.
    # However, to be safe, check all words within radius.
    
    # For efficiency with radius=1: check the word itself and its neighbors
    # Actually, the word cw itself: if we remove cw, is cw still covered by another codeword?
    # Yes if some other cw' has dist(cw, cw') <= radius.
    
    # Let's just check all words in the ball around cw
    # For radius=1: cw + neighbors
    
    # First check cw itself
    for j, other in enumerate(codewords):
        if j == idx:
            continue
        if _hamming_dist(cw, other) <= radius:
            break
    else:
        # cw itself is not covered by any other codeword, so it's essential
        return False
    
    # Now check neighbors of cw (words at distance exactly 1)
    for pos in range(n):
        orig = cw[pos]
        for new_val in range(q):
            if new_val == orig:
                continue
            # Form neighbor word
            neighbor = list(cw)
            neighbor[pos] = new_val
            # Check if neighbor is covered by some other codeword
            covered = False
            for j, other in enumerate(codewords):
                if j == idx:
                    continue
                if _hamming_dist(neighbor, other) <= radius:
                    covered = True
                    break
            if not covered:
                return False
    return True


def _greedy_reduce(codewords, q, n, radius, time_limit):
    """Remove redundant codewords from a valid covering code."""
    start_time = time.time()
    codewords = [list(cw) for cw in codewords]
    changed = True
    while changed:
        if time.time() - start_time > time_limit:
            break
        changed = False
        # Try to remove each codeword
        for i in range(len(codewords)):
            if time.time() - start_time > time_limit:
                break
            if _is_redundant(codewords[i], codewords, i, n, q, radius):
                codewords.pop(i)
                changed = True
                break  # Restart scan from beginning after removal
    return codewords


def construct(q, n, radius):
    # Deadline guard: leave margin for the evaluator's 120s limit
    deadline = time.time() + 110.0
    
    # Direct sum of the ternary tetracode (9 words of length 4, a perfect code of covering
    # radius 1) with every word on the other n - 4 coordinates: 9 * 3**(n - 4) words.
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Greedy reduction: remove redundant codewords
    remaining = deadline - time.time()
    if remaining > 5.0:
        codewords = _greedy_reduce(codewords, q, n, radius, remaining - 5.0)
    
    return codewords
# EVOLVE-BLOCK-END
