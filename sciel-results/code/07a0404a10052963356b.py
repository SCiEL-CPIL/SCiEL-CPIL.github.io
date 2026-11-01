"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random


def construct(q, n, radius):
    """
    Construct a covering code for the football pool problem.
    For the specific case q=3, n=6, radius=1, we aim to find a code with 73 words.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    # If it's not the specific known case, use a generic greedy approach
    if not (q == 3 and n == 6 and radius == 1):
        return _generic_greedy(q, n, radius, deadline)

    # For the specific case, use a specialized approach
    return _football_pool_6_1(deadline)


def _generic_greedy(q, n, radius, deadline):
    """Generic greedy covering code construction."""
    all_words = list(itertools.product(range(q), repeat=n))
    remaining = set(all_words)
    codewords = []

    while remaining and time.time() < deadline:
        # Find the word that covers the most remaining words
        best_word = None
        best_count = -1
        # Sample a subset if the space is too large
        sample_size = min(len(remaining), 10000)
        sample = random.sample(list(remaining), sample_size)
        for w in sample:
            count = 0
            for r in remaining:
                if _hamming_dist(w, r) <= radius:
                    count += 1
            if count > best_count:
                best_count = count
                best_word = w

        if best_word is None:
            break

        codewords.append(list(best_word))
        # Remove covered words
        new_remaining = set()
        for r in remaining:
            if _hamming_dist(best_word, r) > radius:
                new_remaining.add(r)
        remaining = new_remaining

    return codewords


def _hamming_dist(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


def _football_pool_6_1(deadline):
    """
    Specialized solver for the football pool problem (q=3, n=6, radius=1).
    Uses a combination of initial construction and local search to minimize codewords.
    """
    # Start with a known good initial solution (tetracode direct sum gives 81 words)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=2)]

    # Verify it's a valid covering
    if not _is_covering(codewords, 3, 6, 1):
        # This shouldn't happen, but just in case
        pass

    # Try to reduce the number of codewords using local search
    codewords = _reduce_codewords(codewords, 3, 6, 1, deadline)

    return codewords


def _is_covering(codewords, q, n, radius):
    """Check if the codewords form a valid covering code."""
    for word in itertools.product(range(q), repeat=n):
        for cw in codewords:
            if _hamming_dist(word, cw) <= radius:
                break
        else:
            return False
    return True


def _reduce_codewords(codewords, q, n, radius, deadline):
    """
    Try to reduce the number of codewords by removing redundant ones
    and using local search (ILS-style perturbation).
    """
    codewords = list(codewords)
    best_count = len(codewords)
    best_codewords = list(codewords)

    # First pass: remove redundant codewords
    codewords = _remove_redundant(codewords, q, n, radius)
    best_count = len(codewords)
    best_codewords = list(codewords)

    # ILS loop: perturb and reduce
    rng = random.Random(42)
    while time.time() < deadline - 5:  # Leave 5s margin
        # Perturbation: randomly modify a few codewords
        perturbed = list(best_codewords)
        num_perturb = rng.randint(1, 3)
        for _ in range(num_perturb):
            idx = rng.randrange(len(perturbed))
            # Change one coordinate
            coord = rng.randrange(n)
            old_val = perturbed[idx][coord]
            new_val = rng.choice([v for v in range(q) if v != old_val])
            perturbed[idx][coord] = new_val

        # Reduce the perturbed set
        reduced = _remove_redundant(perturbed, q, n, radius)

        # Check if it's still a valid covering
        if _is_covering(reduced, q, n, radius) and len(reduced) < best_count:
            best_count = len(reduced)
            best_codewords = reduced

    return best_codewords


def _remove_redundant(codewords, q, n, radius):
    """Remove codewords that don't cover any unique words."""
    codewords = list(codewords)
    if not codewords:
        return codewords

    # For each codeword, check if removing it still leaves a valid covering
    changed = True
    while changed:
        changed = False
        for i in range(len(codewords)):
            if time.time() > deadline:
                break
            # Try removing codeword i
            candidate = [c for j, c in enumerate(codewords) if j != i]
            if _is_covering(candidate, q, n, radius):
                codewords = candidate
                changed = True
                break
    return codewords


# EVOLVE-BLOCK-END
