"""Large integer sets avoiding equal-gcd triples.

Strategy: Precompute GCD matrix. Use a conflict-based greedy construction
with multiple random restarts and local search (swap moves) to maximize
set size while maintaining the no-equal-gcd-triple constraint.
"""
# EVOLVE-BLOCK-START
import math
import random
import time
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = 85.0  # safety margin under the 90s wall-clock kill


def _bad_triple(a, b, c, gcd_mat):
    """True if the three pairwise gcds of a,b,c are all equal."""
    ga = gcd_mat[a][b]
    gb = gcd_mat[a][c]
    gc = gcd_mat[b][c]
    return ga == gb == gc


def _is_valid(S, gcd_mat):
    """True if no three distinct elements of S form an equal-gcd triple."""
    for a, b, c in combinations(S, 3):
        if _bad_triple(a, b, c, gcd_mat):
            return False
    return True


def _conflict_count(S, x, gcd_mat):
    """Count how many pairs (a,b) in S would form a bad triple with x."""
    cnt = 0
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            if _bad_triple(S[i], S[j], x, gcd_mat):
                cnt += 1
    return cnt


def _conflict_pairs(S, x, gcd_mat):
    """Return list of pairs (a,b) in S that form a bad triple with x."""
    pairs = []
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            if _bad_triple(S[i], S[j], x, gcd_mat):
                pairs.append((S[i], S[j]))
    return pairs


def _repair(S, gcd_mat):
    """Remove elements to make S valid."""
    S = list(S)
    changed = True
    while changed and (time.time() - _START_TIME) < _DEADLINE - 2.0:
        changed = False
        for a, b, c in combinations(S, 3):
            if _bad_triple(a, b, c, gcd_mat):
                # Remove the element with highest conflict count
                from collections import Counter
                cnt = Counter()
                for aa, bb, cc in combinations(S, 3):
                    if _bad_triple(aa, bb, cc, gcd_mat):
                        cnt[aa] += 1
                        cnt[bb] += 1
                        cnt[cc] += 1
                if cnt:
                    v = cnt.most_common(1)[0][0]
                    S = [x for x in S if x != v]
                    changed = True
                    break
    return S


def construct():
    """Construct a large subset of [1,128] with no three distinct elements
    whose three pairwise gcds are all equal. Returns the best set found."""
    random.seed(20240611)
    full = list(range(1, 129))
    n = 128

    # Precompute GCD matrix (indexed 1..128)
    gcd_mat = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            g = math.gcd(i, j)
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g

    def within_deadline():
        return (time.time() - _START_TIME) < _DEADLINE - 1.5

    best_set = []
    best_size = 0

    # Strategy 1: Greedy by 2-adic valuation classes
    # Numbers with same 2-adic valuation: a = 2^v * a' where a' odd
    # Group by v
    groups = {}
    for x in full:
        v = 0
        t = x
        while t % 2 == 0:
            t //= 2
            v += 1
        groups.setdefault(v, []).append(x)

    # Base: take all odds (v=0)
    base = list(groups.get(0, []))

    # Try adding from higher valuation groups
    def greedy_from_base(base, gcd_mat):
        S = list(base)
        for v in sorted(groups.keys()):
            if v == 0:
                continue
            for x in groups[v]:
                if x in S:
                    continue
                if _conflict_count(S, x, gcd_mat) == 0:
                    S.append(x)
        return S

    S1 = greedy_from_base(base, gcd_mat)
    if len(S1) > best_size:
        best_size = len(S1)
        best_set = S1

    # Strategy 2: Multiple random restarts with greedy + local search
    for restart in range(30):
        if not within_deadline():
            break

        # Random greedy: shuffle full, add if no conflict
        shuffled = list(full)
        random.shuffle(shuffled)
        S = []
        for x in shuffled:
            if not within_deadline():
                break
            if _conflict_count(S, x, gcd_mat) == 0:
                S.append(x)

        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)

        # Local search: try swaps to improve
        S = list(S)
        improved = True
        while improved and within_deadline():
            improved = False
            # Try adding any missing element
            for x in full:
                if not within_deadline():
                    break
                if x in S:
                    continue
                if _conflict_count(S, x, gcd_mat) == 0:
                    S.append(x)
                    improved = True
                    break
            # Try swap: remove one, add one
            if not improved:
                for x in full:
                    if not within_deadline():
                        break
                    if x in S:
                        continue
                    # Find elements whose removal would allow x
                    for y in list(S):
                        S2 = [z for z in S if z != y]
                        if _conflict_count(S2, x, gcd_mat) == 0:
                            S2.append(x)
                            if len(S2) > len(S):
                                S = S2
                                improved = True
                                break
                    if improved:
                        break

        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)

    # Strategy 3: Start from structural sets
    # All numbers with odd part = 1 (powers of 2) + all odds
    # Or: numbers that are 1 mod 4, 3 mod 4, etc.
    structural_sets = [
        [x for x in full if x % 2 == 1],  # all odds
        [x for x in full if x % 4 == 1],  # 1 mod 4
        [x for x in full if x % 4 == 3],  # 3 mod 4
        [x for x in full if x % 2 == 1 or x == 128],
        [x for x in full if x % 2 == 1 or x % 4 == 0],
    ]

    for base_set in structural_sets:
        if not within_deadline():
            break
        S = greedy_from_base(base_set, gcd_mat)
        # Local search
        improved = True
        while improved and within_deadline():
            improved = False
            for x in full:
                if not within_deadline():
                    break
                if x in S:
                    continue
                if _conflict_count(S, x, gcd_mat) == 0:
                    S.append(x)
                    improved = True
                    break
            if not improved:
                for x in full:
                    if not within_deadline():
                        break
                    if x in S:
                        continue
                    for y in list(S):
                        S2 = [z for z in S if z != y]
                        if _conflict_count(S2, x, gcd_mat) == 0:
                            S2.append(x)
                            if len(S2) > len(S):
                                S = S2
                                improved = True
                                break
                    if improved:
                        break
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)

    # Final validation and repair
    if not _is_valid(best_set, gcd_mat):
        best_set = _repair(best_set, gcd_mat)

    if not best_set:
        return [1]

    return best_set

# EVOLVE-BLOCK-END
