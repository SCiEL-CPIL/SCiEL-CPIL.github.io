"""Combined: conflict-pair precomputation + multi-greedy + enhanced local search."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

_TIME_LIMIT = 84.0

# Precompute gcd matrix for all pairs in [1, 128]
_N = 128
_gcd_matrix = [[0] * (_N + 1) for _ in range(_N + 1)]
for _i in range(1, _N + 1):
    for _j in range(_i, _N + 1):
        _g = gcd(_i, _j)
        _gcd_matrix[_i][_j] = _g
        _gcd_matrix[_j][_i] = _g

# Precompute conflict_pairs_for_x[x] = set of pairs (u,v) with u<v
# such that {u,v,x} is a forbidden triple (all three pairwise gcds equal)
# This enables O(min(|conflicts|, |selected_pairs|)) validity checks.
_conflict_pairs_for_x = [set() for _ in range(_N + 1)]
for _a in range(1, _N + 1):
    for _b in range(_a + 1, _N + 1):
        _g_ab = _gcd_matrix[_a][_b]
        for _c in range(_b + 1, _N + 1):
            if _gcd_matrix[_a][_c] == _g_ab and _gcd_matrix[_b][_c] == _g_ab:
                # (a, b, c) is a forbidden triple
                _conflict_pairs_for_x[_a].add((_b, _c))
                _conflict_pairs_for_x[_b].add((_a, _c))
                _conflict_pairs_for_x[_c].add((_a, _b))


def _can_add_pairs(selected_pairs, x):
    """Check if x can be added given the set of selected pairs.
    Returns True if no forbidden pair is present in selected_pairs."""
    conflicts = _conflict_pairs_for_x[x]
    if not conflicts:
        return True
    if len(conflicts) < len(selected_pairs):
        for p in conflicts:
            if p in selected_pairs:
                return False
    else:
        for p in selected_pairs:
            if p in conflicts:
                return False
    return True


def _greedy(order, deadline):
    """Greedy: iterate order, add if valid using pair-set check."""
    S = []
    Sset = set()
    selected_pairs = set()
    for x in order:
        if time.time() > deadline:
            break
        if x in Sset:
            continue
        if _can_add_pairs(selected_pairs, x):
            S.append(x)
            Sset.add(x)
            for u in S[:-1]:
                pair = (min(u, x), max(u, x))
                selected_pairs.add(pair)
    return S


def _greedy_fill(S, Sset, universe, deadline):
    """Greedily add all possible elements. Returns (S, Sset, selected_pairs, num_added)."""
    selected_pairs = set()
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            selected_pairs.add((min(S[i], S[j]), max(S[i], S[j])))
    num_added = 0
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for x in universe:
            if time.time() > deadline:
                break
            if x in Sset:
                continue
            if _can_add_pairs(selected_pairs, x):
                S.append(x)
                Sset.add(x)
                for u in S[:-1]:
                    pair = (min(u, x), max(u, x))
                    selected_pairs.add(pair)
                num_added += 1
                improved = True
    return S, Sset, selected_pairs, num_added


def _rebuild_pairs(S):
    """Rebuild the set of pairs from list S."""
    pairs = set()
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            pairs.add((min(S[i], S[j]), max(S[i], S[j])))
    return pairs


def _local_search(S, deadline, rng):
    """Enhanced local search using pair-set validity checks."""
    S = list(S)
    Sset = set(S)
    universe = list(range(1, _N + 1))
    d = deadline - 2  # safety margin

    # Build initial pairs
    selected_pairs = _rebuild_pairs(S)

    # Phase 1: add all possible
    S, Sset, selected_pairs, _ = _greedy_fill(S, Sset, universe, d)

    # Phase 2: 1-swap local search with greedy fill
    if time.time() < d - 4:
        for _iter in range(50):
            if time.time() > d - 4:
                break
            progress = False
            for idx in range(len(S)):
                if time.time() > d - 4:
                    break
                removed = S[idx]
                S2 = S[:idx] + S[idx + 1:]
                S2set = set(S2)
                best_gain = 0
                best_x = None
                best_added = None
                for x in universe:
                    if x in S2set:
                        continue
                    if _can_add_pairs(selected_pairs, x):
                        # x is valid in S (not S2, but S2 = S \ {removed})
                        # Actually we need to check x against S2's pairs
                        pass
                # Simpler: try removing each element, then greedy fill
                S2_pairs = _rebuild_pairs(S2)
                S3 = S2[:]
                S3set = set(S3)
                S3_pairs = set(S2_pairs)
                for x in universe:
                    if time.time() > d - 4:
                        break
                    if x in S3set:
                        continue
                    if _can_add_pairs(S3_pairs, x):
                        S3.append(x)
                        S3set.add(x)
                        for u in S3[:-1]:
                            pair = (min(u, x), max(u, x))
                            S3_pairs.add(pair)
                if len(S3) > len(S):
                    S = S3
                    Sset = S3set
                    selected_pairs = S3_pairs
                    progress = True
                    break
            if not progress:
                break

    # Phase 3: add all possible again
    S, Sset, selected_pairs, _ = _greedy_fill(S, Sset, universe, d)

    # Phase 4: 2-remove local search
    if time.time() < d - 3:
        for _iter in range(20):
            if time.time() > d - 3:
                break
            progress = False
            n = len(S)
            for i in range(n):
                if time.time() > d - 3:
                    break
                for j in range(i + 1, n):
                    if time.time() > d - 3:
                        break
                    S2 = S[:i] + S[i+1:j] + S[j+1:]
                    S2set = set(S2)
                    S2_pairs = _rebuild_pairs(S2)
                    S3 = S2[:]
                    S3set = set(S3)
                    S3_pairs = set(S2_pairs)
                    for x in universe:
                        if time.time() > d - 3:
                            break
                        if x in S3set:
                            continue
                        if _can_add_pairs(S3_pairs, x):
                            S3.append(x)
                            S3set.add(x)
                            for u in S3[:-1]:
                                pair = (min(u, x), max(u, x))
                                S3_pairs.add(pair)
                    if len(S3) > len(S):
                        S = S3
                        Sset = S3set
                        selected_pairs = S3_pairs
                        progress = True
                        break
                if progress:
                    break
            if not progress:
                break

    # Phase 5: add all possible
    S, Sset, selected_pairs, _ = _greedy_fill(S, Sset, universe, d)

    # Phase 6: Kick strategy - randomly remove 1-2 elements, then greedy refill
    if time.time() < d - 2:
        for _kick in range(20):
            if time.time() > d - 2:
                break
            S_kick = S[:]
            S_kick_set = set(S_kick)
            n_remove = 1
            if len(S) > 50 and rng.random() < 0.5:
                n_remove = 2
            to_remove = rng.sample(range(len(S_kick)), n_remove)
            for idx in sorted(to_remove, reverse=True):
                S_kick_set.discard(S_kick[idx])
                S_kick.pop(idx)
            S_kick_pairs = _rebuild_pairs(S_kick)
            S_kick, S_kick_set, S_kick_pairs, added = _greedy_fill(
                S_kick, S_kick_set, universe, d)
            if len(S_kick) > len(S):
                S = S_kick
                Sset = S_kick_set
                selected_pairs = S_kick_pairs

    # Phase 7: Conflict-count guided multi-swap search
    if time.time() < d - 2:
        for k_remove in [1, 2, 3]:
            if time.time() > d - 2:
                break
            n = len(S)
            if n < k_remove + 1:
                continue
            n_trials = min(30, max(5, n - k_remove))
            for _trial in range(n_trials):
                if time.time() > d - 2:
                    break
                remove_idx = set(rng.sample(range(n), k_remove))
                S2 = [S[i] for i in range(n) if i not in remove_idx]
                S2set = set(S2)
                S2_pairs = _rebuild_pairs(S2)
                S3 = S2[:]
                S3set = set(S3)
                S3_pairs = set(S2_pairs)
                for x in universe:
                    if time.time() > d - 2:
                        break
                    if x in S3set:
                        continue
                    if _can_add_pairs(S3_pairs, x):
                        S3.append(x)
                        S3set.add(x)
                        for u in S3[:-1]:
                            pair = (min(u, x), max(u, x))
                            S3_pairs.add(pair)
                if len(S3) > len(S):
                    S = S3
                    Sset = S3set
                    selected_pairs = S3_pairs
                    n = len(S)

    # Phase 8: Final exhaustive polish
    if time.time() < d - 1:
        polish_improved = True
        while polish_improved:
            if time.time() > d - 1:
                break
            polish_improved = False
            for idx in range(len(S)):
                if time.time() > d - 1:
                    break
                S2 = S[:idx] + S[idx + 1:]
                S2set = set(S2)
                S2_pairs = _rebuild_pairs(S2)
                S2, S2set, S2_pairs, added = _greedy_fill(S2, S2set, universe, d)
                if len(S2) > len(S):
                    S = S2
                    Sset = S2set
                    selected_pairs = S2_pairs
                    polish_improved = True
                    break

    # Phase 9: Branch-and-bound style search
    if time.time() < d - 1:
        current_size = len(S)
        for x in universe:
            if time.time() > d - 1:
                break
            if x in Sset:
                continue
            # Find which elements in S conflict with x
            conflicts = []
            for i, a in enumerate(S):
                ga = _gcd_matrix[a][x]
                for j in range(i + 1, len(S)):
                    b = S[j]
                    if _gcd_matrix[b][x] == ga and _gcd_matrix[a][b] == ga:
                        if a not in conflicts:
                            conflicts.append(a)
                        if b not in conflicts:
                            conflicts.append(b)
            if len(conflicts) <= 2:
                S2 = [a for a in S if a not in set(conflicts)]
                S2set = set(S2)
                S2_pairs = _rebuild_pairs(S2)
                if _can_add_pairs(S2_pairs, x):
                    S2.append(x)
                    S2set.add(x)
                    for u in S2[:-1]:
                        pair = (min(u, x), max(u, x))
                        S2_pairs.add(pair)
                    S2, S2set, S2_pairs, _ = _greedy_fill(S2, S2set, universe, d)
                    if len(S2) > current_size:
                        S = S2
                        Sset = S2set
                        selected_pairs = S2_pairs
                        current_size = len(S)

    # Phase 10: final add
    S, Sset, selected_pairs, _ = _greedy_fill(S, Sset, universe, d)

    return S


def _num_distinct_prime_factors(x):
    """Count distinct prime factors of x."""
    if x <= 1:
        return 0
    count = 0
    d = 2
    n = x
    while d * d <= n:
        if n % d == 0:
            count += 1
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        count += 1
    return count


def _multi_greedy_local(deadline):
    rng = random.Random(42)
    universe = list(range(1, _N + 1))
    best = []
    gm = _gcd_matrix

    # Precompute factor counts
    pf = [0] * (_N + 1)
    for x in range(1, _N + 1):
        pf[x] = _num_distinct_prime_factors(x)

    # Precompute conflict degree for each number
    conflict_degree = [0] * (_N + 1)
    for x in range(1, _N + 1):
        conflict_degree[x] = len(_conflict_pairs_for_x[x])

    # Generate diverse orderings
    orders = [
        list(range(1, _N + 1)),
        list(range(_N, 0, -1)),
        [x for x in range(1, _N + 1) if x % 2 == 1] + [x for x in range(1, _N + 1) if x % 2 == 0],
        [x for x in range(1, _N + 1) if x % 2 == 0] + [x for x in range(1, _N + 1) if x % 2 == 1],
        sorted(range(1, _N + 1), key=lambda x: -x.bit_count()),
        sorted(range(1, _N + 1), key=lambda x: x.bit_count()),
        [x for x in range(1, _N + 1) if x % 3 != 0] + [x for x in range(1, _N + 1) if x % 3 == 0],
        [x for x in range(1, _N + 1) if x % 3 == 0] + [x for x in range(1, _N + 1) if x % 3 != 0],
        sorted(range(1, _N + 1), key=lambda x: (x % 2, x)),
        sorted(range(1, _N + 1), key=lambda x: (-x.bit_count(), x)),
        sorted(range(1, _N + 1), key=lambda x: pf[x]),
        sorted(range(1, _N + 1), key=lambda x: -pf[x]),
        sorted(range(1, _N + 1), key=lambda x: (x & (x - 1) == 0, x)),
        sorted(range(1, _N + 1), key=lambda x: (x.bit_count(), x)),
        sorted(range(1, _N + 1), key=lambda x: conflict_degree[x]),
        sorted(range(1, _N + 1), key=lambda x: -conflict_degree[x]),
    ]
    # Add random orderings
    for _ in range(20):
        o = universe[:]
        rng.shuffle(o)
        orders.append(o)

    # Run greedy on each ordering
    candidates = []
    for order in orders:
        if time.time() > deadline - 25:
            break
        cand = _greedy(order, deadline)
        candidates.append(cand)
        if len(cand) > len(best):
            best = cand

    # Local search on top candidates
    candidates.sort(key=len, reverse=True)
    for cand in candidates[:12]:
        if time.time() > deadline - 5:
            break
        improved = _local_search(cand, deadline, rng)
        if len(improved) > len(best):
            best = improved

    # Additional local search passes on best with different random seeds
    if time.time() < deadline - 3:
        for seed in [123, 456, 789, 101, 202]:
            if time.time() > deadline - 3:
                break
            rng2 = random.Random(seed)
            improved = _local_search(best[:], deadline, rng2)
            if len(improved) > len(best):
                best = improved

    return best


def construct():
    global _TIME_LIMIT
    deadline = time.time() + _TIME_LIMIT
    try:
        result = _multi_greedy_local(deadline)
    except Exception:
        # Fallback: simple ascending greedy with pair-set check
        result = []
        result_set = set()
        selected_pairs = set()
        for x in range(1, _N + 1):
            if _can_add_pairs(selected_pairs, x):
                result.append(x)
                result_set.add(x)
                for u in result[:-1]:
                    pair = (min(u, x), max(u, x))
                    selected_pairs.add(pair)
    if not result:
        result = [1]
    return result
# EVOLVE-BLOCK-END
