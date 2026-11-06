"""Suite task: maximize a set of distinct positive integers <=192 (<=48)
such that no selected integer divides the sum of any 3 distinct OTHER selected integers."""
import time
import random
from itertools import combinations

DEADLINE = 80.0          # explicit wall-clock guard (evaluator kills at 90s)
N = 192
MAXK = 48
RNG = random.Random(20240517)


def _valid_add(S, x):
    """Incremental O(|S|^2) check: is S ∪ {x} valid?
    Only the new element x and the existing elements of S need re-checking
    (S was already valid before adding x)."""
    # (1) x must not divide the sum of any 3 distinct elements of S
    for a, b, c in combinations(S, 3):
        if (a + b + c) % x == 0:
            return False
    # (2) each existing element a must not divide the sum of any 3 distinct
    #     elements of S ∪ {x} that are all != a.  A triple can contain x or not.
    m = len(S)
    for i in range(m):
        a = S[i]
        # triples that include x: (x, S[j], S[k]) with j<k, j,k != i
        for j in range(m):
            if j == i:
                continue
            vj = S[j]
            for k in range(j + 1, m):
                if k == i:
                    continue
                if (x + vj + S[k]) % a == 0:
                    return False
        # triples that do not include x: unchanged from before x was added,
        # so they are already known-good for a.  (No need to re-check.)
    return True


def _is_valid(S):
    """Full O(n^3) validity check of set S."""
    for a in S:
        others = [v for v in S if v != a]
        for t in combinations(others, 3):
            if sum(t) % a == 0:
                return False
    return True


def _greedy(order, S, seen):
    """Try to add elements of `order` (not in S) to S if valid."""
    for x in order:
        if x in seen or x in S:
            continue
        if len(S) >= MAXK:
            break
        if _valid_add(S, x):
            S.append(x)
            seen.add(x)


def _local_search(S, deadline):
    """1- and 2-exchange local search to grow/fix S."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        S2 = list(S)
        seen = set(S2)
        # 1-exchange: remove a, add b
        for i in range(len(S2)):
            if time.time() >= deadline:
                break
            a = S2[i]
            for b in range(N, 0, -1):
                if b in seen:
                    continue
                cand = S2[:i] + S2[i + 1:] + [b]
                if _is_valid(cand):
                    S2 = cand
                    seen = set(S2)
                    improved = True
                    break
        # 2-exchange: remove a, b ; add c, d
        if not improved and len(S2) >= 2:
            for i in range(len(S2)):
                if time.time() >= deadline:
                    break
                for j in range(i + 1, len(S2)):
                    base = [v for k, v in enumerate(S2) if k != i and k != j]
                    for c in range(N, 0, -1):
                        if c in base:
                            continue
                        for d in range(c + 1, N + 1):
                            if d in base:
                                continue
                            cand = base + [c, d]
                            if _is_valid(cand):
                                S2 = cand
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        S = S2
    return S


def construct():
    start = time.time()
    deadline = start + DEADLINE
    best = []

    # --- Phase 1: deterministic greedy seeds from many orderings ---
    orders = [
        range(N, 0, -1),                 # descending (original seed behaviour)
        range(1, N + 1),                 # ascending
        range(N, 0, -1, 2),              # descending odds
        range(1, N + 1, 2),              # ascending odds
        range(N, 0, -1, 3),              # descending multiples-of-3
        range(2, N + 1, 2),              # ascending evens
        sorted(range(1, N + 1), key=lambda v: (v % 2, -v)),
    ]
    for order in orders:
        if time.time() >= deadline:
            break
        S = []
        seen = set()
        _greedy(order, S, seen)
        if len(S) > len(best):
            best = list(S)

    # --- Phase 2: randomized greedy seeds ---
    rng = RNG
    while time.time() < deadline - 5.0:
        order = list(range(1, N + 1))
        rng.shuffle(order)
        # bias: interleave high and low to diversify
        S = []
        seen = set()
        _greedy(order, S, seen)
        if len(S) > len(best):
            best = list(S)

    # --- Phase 3: local search from the best seed (and a few others) ---
    seeds = [best]
    # add a couple of fresh greedy seeds as alternate local-search starts
    for _ in range(3):
        if time.time() >= deadline:
            break
        order = list(range(1, N + 1))
        rng.shuffle(order)
        S = []
        seen = set()
        _greedy(order, S, seen)
        seeds.append(S)

    for seed in seeds:
        if time.time() >= deadline - 1.0:
            break
        S = _local_search(seed, deadline)
        if len(S) > len(best):
            best = list(S)

    # --- Final safety: verify and trim to a guaranteed-valid subset ---
    if time.time() < deadline - 0.5:
        if not _is_valid(best):
            # rebuild a valid subset from best in descending order
            fixed = []
            seen = set()
            for x in sorted(best, reverse=True):
                if x in seen:
                    continue
                if _valid_add(fixed, x):
                    fixed.append(x)
                    seen.add(x)
            best = fixed

    return best
