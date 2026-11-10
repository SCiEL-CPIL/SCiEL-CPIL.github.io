# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Furstenberg-Sarkozy problem (modular formulation). Find a subset of Z/mZ of maximum
    size that does NOT contain any two elements differing by a nonzero perfect k-th power
    modulo m. Here m = 205 and k = 2 (squares), i.e. forbidden differences are the nonzero
    quadratic residues modulo 205.

    Returns:
        subset: np.ndarray of shape (s,), dtype int, a list of distinct residues in
            {0, ..., 204} with no two differing by a nonzero square modulo 205.
    """
    m = 205
    k = 2
    deadline = time.time() + 580  # Safety margin before 600s limit

    # Forbidden differences: the nonzero k-th power residues modulo m.
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, k, m))
    forbidden.discard(0)

    # Precompute for each element, which other elements are forbidden with it
    # forbidden_with[i] = set of j such that (i-j) % m or (j-i) % m is in forbidden
    forbidden_with = [set() for _ in range(m)]
    for i in range(m):
        for f in forbidden:
            j1 = (i + f) % m
            j2 = (i - f) % m
            forbidden_with[i].add(j1)
            forbidden_with[i].add(j2)

    def greedy(order):
        """Greedy: process candidates in given order, add if compatible with all selected."""
        selected = []
        for candidate in order:
            ok = True
            for s in selected:
                if candidate in forbidden_with[s]:
                    ok = False
                    break
            if ok:
                selected.append(candidate)
        return selected

    # Start with deterministic greedy in natural order
    best = greedy(list(range(m)))
    best_size = len(best)

    # Try multiple random orderings
    rng = random.Random(42)
    attempts = 0
    while time.time() < deadline:
        order = list(range(m))
        rng.shuffle(order)
        result = greedy(order)
        if len(result) > best_size:
            best = result
            best_size = len(result)
        attempts += 1
        # Also try reverse and other simple permutations occasionally
        if attempts % 10 == 0:
            result = greedy(list(range(m-1, -1, -1)))
            if len(result) > best_size:
                best = result
                best_size = len(result)

    return np.array(best, dtype=np.int64)


# EVOLVE-BLOCK-END
