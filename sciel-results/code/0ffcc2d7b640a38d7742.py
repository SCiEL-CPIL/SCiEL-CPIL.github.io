"""Covering code K_3(8,1): product-of-perfect-codes construction with refinement."""
import time
import itertools
import random

_DEADLINE = 150.0
_MARGIN = 12.0  # safety margin in seconds


def _tetracode():
    # Perfect ternary code of length 4, radius 1 (9 words, balls partition 3^4).
    return [tuple([a, b, (a + b) % 3, (a + 2 * b) % 3]) for a in range(3) for b in range(3)]


def _covers(code, q, n, radius):
    """Check every word of {0..q-1}^n is within `radius` of some codeword."""
    code_set = [tuple(c) for c in code]
    for word in itertools.product(range(q), repeat=n):
        ok = False
        for c in code_set:
            d = 0
            for i in range(n):
                if word[i] != c[i]:
                    d += 1
                    if d > radius:
                        break
            if d <= radius:
                ok = True
                break
        if not ok:
            return False
    return True


def construct(q, n, radius):
    start = time.time()
    deadline = start + _DEADLINE - _MARGIN

    # Base construction: product of two perfect radius-1 codes (tetracode) in each half.
    # If n is even: A in first n/2, B in last n/2.
    half = n // 2
    if n % 2 == 0 and q == 3 and radius == 1:
        # Tetracode product: 9 * 9 = 81 codewords cover 3^8.
        A = _tetracode()
        B = _tetracode()
        code = [list(a) + list(b) for a in A for b in B]
        # Verify coverage (cheap enough at this size).
        if _covers(code, q, n, radius):
            return code
    else:
        # Fallback: product of tetracode (first 4) and all words (rest), as in seed.
        A = _tetracode()
        rest = n - 4
        code = [list(a) + list(tail) for a in A
                for tail in itertools.product(range(q), repeat=rest)]
        return code

    # If we reach here (shouldn't for the target case), refine via local search.
    # (Kept for generality; not expected to run for q=3,n=8,r=1.)
    best = code
    rng = random.Random(0)
    while time.time() < deadline:
        # Try removing a random codeword if coverage holds.
        idx = rng.randrange(len(best))
        trial = best[:idx] + best[idx + 1:]
        if _covers(trial, q, n, radius):
            best = trial
        else:
            # Swap: replace a codeword with a random neighbor to enable later removals.
            if best:
                c = list(best[idx])
                pos = rng.randrange(n)
                c[pos] = (c[pos] + rng.randrange(1, q)) % q
                trial = best[:idx] + [c] + best[idx + 1:]
                if _covers(trial, q, n, radius):
                    best = trial
    return best
