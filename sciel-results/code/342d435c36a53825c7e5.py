"""Combined: precomputed gcd matrix + multi-greedy + conflict-count local search."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

_TIME_LIMIT = 84.0
_DEADLINE = None

# Precompute gcd matrix for all pairs in [1, 128]
_N = 128
_gcd_matrix = [[0] * (_N + 1) for _ in range(_N + 1)]
for _i in range(1, _N + 1):
    for _j in range(_i, _N + 1):
        _g = gcd(_i, _j)
        _gcd_matrix[_i][_j] = _g
        _gcd_matrix[_j][_i] = _g


def _can_add(S, x):
    """Check if adding x to S keeps the set triple-free using precomputed gcd matrix."""
    n = len(S)
    if n < 2:
        return True
    # Collect gcds of x with each element in S
    gx = [0] * n
    for i in range(n):
        gx[i] = _gcd_matrix[S[i]][x]
    for i in range(n):
        ga = gx[i]
        for j in range(i + 1, n):
            if gx[j] == ga and _gcd_matrix[S[i]][S[j]] == ga:
                return False
    return True


def _can_add_fast(S_list, S_set, x, gcd_mat):
    """Faster check using set membership and direct matrix access."""
    n = len(S_list)
    if n < 2:
        return True
    for i in range(n):
        a = S_list[i]
        ga = gcd_mat[a][x]
        for j in range(i + 1, n):
            b = S_list[j]
            if gcd_mat[b][x] == ga and gcd_mat[a][b] == ga:
                return False
    return True


def _count_conflicts_fast(S, gcd_mat):
    """Count bad triples efficiently."""
    n = len(S)
    count = 0
    for i in range(n):
        a = S[i]
        for j in range(i + 1, n):
            b = S[j]
            g = gcd_mat[a][b]
            for k in range(j + 1, n):
                c = S[k]
                if gcd_mat[a][c] == g and gcd_mat[b][c] == g:
                    count += 1
    return count


def _count_conflicts(S):
    """Count the number of bad triples in S (triples where all 3 pairwise gcds are equal)."""
    n = len(S)
    count = 0
    for i in range(n):
        a = S[i]
        for j in range(i + 1, n):
            b = S[j]
            g_ab = _gcd_matrix[a][b]
            for k in range(j + 1, n):
                c = S[k]
                if _gcd_matrix[a][c] == g_ab and _gcd_matrix[b][c] == g_ab:
                    count += 1
    return count


def _is_valid(S):
    """Full validation check."""
    S_list = list(S)
    for i in range(len(S_list)):
        a = S_list[i]
        for j in range(i + 1, len(S_list)):
            b = S_list[j]
            g_ab = _gcd_matrix[a][b]
            for k in range(j + 1, len(S_list)):
                c = S_list[k]
                if _gcd_matrix[a][c] == g_ab and _gcd_matrix[b][c] == g_ab:
                    return False
    return True


def _greedy(order, deadline):
    """Greedy: iterate order, add if valid."""
    S = []
    Sset = set()
    gm = _gcd_matrix
    for x in order:
        if time.time() > deadline:
            break
        if x in Sset:
            continue
        if _can_add_fast(S, Sset, x, gm):
            S.append(x)
            Sset.add(x)
    return S


def _greedy_fill(S, Sset, universe, deadline):
    """Greedily add all possible elements. Returns (S, Sset, num_added)."""
    num_added = 0
    improved = True
    gm = _gcd_matrix
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for x in universe:
            if time.time() > deadline:
                break
            if x in Sset:
                continue
            if _can_add_fast(S, Sset, x, gm):
                S.append(x)
                Sset.add(x)
                num_added += 1
                improved = True
    return S, Sset, num_added


def _local_search(S, deadline, rng):
    """Enhanced local search with conflict-count guided swaps and branch-and-bound."""
    S = list(S)
    Sset = set(S)
    universe = list(range(1, 129))
    d = deadline - 2  # safety margin
    gm = _gcd_matrix

    # Phase 1: add all possible
    S, Sset, _ = _greedy_fill(S, Sset, universe, d)

    # Phase 2: 1-swap local search with greedy fill
    if time.time() < d - 4:
        for _iter in range(30):
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
                    if _can_add_fast(S2, S2set, x, gm):
                        S3 = S2 + [x]
                        S3set = set(S3)
                        added = []
                        for y in universe:
                            if y in S3set:
                                continue
                            if _can_add_fast(S3, S3set, y, gm):
                                S3.append(y)
                                S3set.add(y)
                                added.append(y)
                        gain = len(added)
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                            best_added = added
                            if gain >= 2:
                                break
                if best_x is not None and best_gain > 0:
                    S = S2 + [best_x]
                    Sset = set(S)
                    for y in best_added:
                        S.append(y)
                        Sset.add(y)
                    progress = True
                    break
            if not progress:
                break

    # Phase 3: add all possible again
    S, Sset, _ = _greedy_fill(S, Sset, universe, d)

    # Phase 4: 2-swap local search (remove 2, add 2+)
    if time.time() < d - 3:
        for _iter in range(10):
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
                    best_gain = 0
                    best_x = None
                    best_added = None
                    for x in universe:
                        if time.time() > d - 3:
                            break
                        if x in S2set:
                            continue
                        if _can_add_fast(S2, S2set, x, gm):
                            S3 = S2 + [x]
                            S3set = set(S3)
                            added = []
                            for y in universe:
                                if y in S3set:
                                    continue
                                if _can_add_fast(S3, S3set, y, gm):
                                    S3.append(y)
                                    S3set.add(y)
                                    added.append(y)
                                    if len(added) >= 2:
                                        break
                            gain = len(added)
                            if gain > best_gain:
                                best_gain = gain
                                best_x = x
                                best_added = added
                                if gain >= 2:
                                    break
                    if best_x is not None and best_gain > 0:
                        S = S2 + [best_x]
                        Sset = set(S)
                        for y in best_added:
                            S.append(y)
                            Sset.add(y)
                        progress = True
                        break
                if progress:
                    break
            if not progress:
                break

    # Phase 5: add all possible
    S, Sset, _ = _greedy_fill(S, Sset, universe, d)

    # Phase 6: Kick strategy - randomly remove 1-2 elements, then greedy refill
    if time.time() < d - 2:
        for _kick in range(10):
            if time.time() > d - 2:
                break
            S_kick = S[:]
            S_kick_set = set(S_kick)
            # Remove 1 or 2 random elements
            n_remove = 1 if len(S) > 40 else 1
            if len(S) > 50 and rng.random() < 0.5:
                n_remove = 2
            to_remove = rng.sample(range(len(S_kick)), n_remove)
            for idx in sorted(to_remove, reverse=True):
                S_kick_set.discard(S_kick[idx])
                S_kick.pop(idx)
            # Greedy refill
            S_kick, S_kick_set, added = _greedy_fill(S_kick, S_kick_set, universe, d)
            if len(S_kick) > len(S):
                S = S_kick
                Sset = S_kick_set

    # Phase 7: Conflict-count guided multi-swap search
    # Try removing k elements and adding back more
    if time.time() < d - 2:
        for k_remove in [1, 2, 3]:
            if time.time() > d - 2:
                break
            n = len(S)
            if n < k_remove + 1:
                continue
            # Sample a few random removal sets
            n_trials = min(20, max(5, n - k_remove))
            for _trial in range(n_trials):
                if time.time() > d - 2:
                    break
                remove_idx = sorted(rng.sample(range(n), k_remove))
                S2 = [S[i] for i in range(n) if i not in set(remove_idx)]
                S2set = set(S2)
                # Try to add back elements greedily
                S3 = S2[:]
                S3set = set(S3)
                for x in universe:
                    if time.time() > d - 2:
                        break
                    if x in S3set:
                        continue
                    if _can_add_fast(S3, S3set, x, gm):
                        S3.append(x)
                        S3set.add(x)
                if len(S3) > len(S):
                    S = S3
                    Sset = S3set
                    n = len(S)  # update n for next trial

    # Phase 8: Final exhaustive polish - try removing each element and refilling
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
                S2, S2set, added = _greedy_fill(S2, S2set, universe, d)
                if len(S2) > len(S):
                    S = S2
                    Sset = S2set
                    polish_improved = True
                    break

    # Phase 9: Branch-and-bound style search using conflict structure
    # For each element not in S, check if it can be added after removing minimal conflicts
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
                ga = gm[a][x]
                for j in range(i + 1, len(S)):
                    b = S[j]
                    if gm[b][x] == ga and gm[a][b] == ga:
                        # a, b, x form a bad triple
                        if a not in conflicts:
                            conflicts.append(a)
                        if b not in conflicts:
                            conflicts.append(b)
            # If we remove conflicts, can we add x and possibly more?
            if len(conflicts) <= 2:
                S2 = [a for a in S if a not in set(conflicts)]
                S2set = set(S2)
                if _can_add_fast(S2, S2set, x, gm):
                    S2.append(x)
                    S2set.add(x)
                    S2, S2set, _ = _greedy_fill(S2, S2set, universe, d)
                    if len(S2) > current_size:
                        S = S2
                        Sset = S2set
                        current_size = len(S)

    # Phase 10: final add
    S, Sset, _ = _greedy_fill(S, Sset, universe, d)

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
    universe = list(range(1, 129))
    best = []
    gm = _gcd_matrix

    # Precompute factor counts
    pf = [0] * 129
    for x in range(1, 129):
        pf[x] = _num_distinct_prime_factors(x)

    # Precompute conflict degree for each number
    # conflict_degree[x] = number of pairs (a,b) in universe\{x} such that gcd(a,b)=gcd(a,x)=gcd(b,x)
    conflict_degree = [0] * 129
    for x in range(1, 129):
        cnt = 0
        for a in range(1, 129):
            if a == x:
                continue
            ga = gm[a][x]
            for b in range(a + 1, 129):
                if b == x:
                    continue
                if gm[b][x] == ga and gm[a][b] == ga:
                    cnt += 1
        conflict_degree[x] = cnt

    # Generate diverse orderings
    orders = [
        list(range(1, 129)),
        list(range(128, 0, -1)),
        [x for x in range(1, 129) if x % 2 == 1] + [x for x in range(1, 129) if x % 2 == 0],
        [x for x in range(1, 129) if x % 2 == 0] + [x for x in range(1, 129) if x % 2 == 1],
        sorted(range(1, 129), key=lambda x: -x.bit_count()),
        sorted(range(1, 129), key=lambda x: x.bit_count()),
        [x for x in range(1, 129) if x % 3 != 0] + [x for x in range(1, 129) if x % 3 == 0],
        [x for x in range(1, 129) if x % 3 == 0] + [x for x in range(1, 129) if x % 3 != 0],
        sorted(range(1, 129), key=lambda x: (x % 2, x)),
        sorted(range(1, 129), key=lambda x: (-x.bit_count(), x)),
        # Numbers with fewer distinct prime factors first (less constrained)
        sorted(range(1, 129), key=lambda x: pf[x]),
        # Numbers with more distinct prime factors first
        sorted(range(1, 129), key=lambda x: -pf[x]),
        # Powers of 2 first, then others
        sorted(range(1, 129), key=lambda x: (x & (x-1) == 0, x)),
        # Coprime pairs preferred: sort by number of divisors
        sorted(range(1, 129), key=lambda x: (x.bit_count(), x)),
        # Sort by conflict degree (least constrained first)
        sorted(range(1, 129), key=lambda x: conflict_degree[x]),
        # Sort by conflict degree (most constrained first - try to place them early)
        sorted(range(1, 129), key=lambda x: -conflict_degree[x]),
    ]
    # Add random orderings
    for _ in range(20):
        o = universe[:]
        rng.shuffle(o)
        orders.append(o)

    # Run greedy on each ordering
    candidates = []
    for order in orders:
        if time.time() > deadline - 20:
            break
        cand = _greedy(order, deadline)
        candidates.append(cand)
        if len(cand) > len(best):
            best = cand

    # Local search on top candidates
    candidates.sort(key=len, reverse=True)
    for cand in candidates[:10]:
        if time.time() > deadline - 5:
            break
        improved = _local_search(cand, deadline, rng)
        if len(improved) > len(best):
            best = improved

    # Additional local search passes on best with different random seeds
    if time.time() < deadline - 3:
        for seed in [123, 456, 789]:
            if time.time() > deadline - 3:
                break
            rng2 = random.Random(seed)
            improved = _local_search(best[:], deadline, rng2)
            if len(improved) > len(best):
                best = improved

    return best


def construct():
    global _DEADLINE
    _DEADLINE = time.time() + _TIME_LIMIT
    try:
        result = _multi_greedy_local(_DEADLINE)
    except Exception:
        # Fallback: simple ascending greedy with precomputed matrix
        result = []
        for x in range(1, 129):
            ok = True
            for i in range(len(result)):
                a = result[i]
                ga = _gcd_matrix[a][x]
                for j in range(i + 1, len(result)):
                    b = result[j]
                    if ga == _gcd_matrix[b][x] == _gcd_matrix[a][b]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                result.append(x)
    if not result:
        result = [1]
    return result
# EVOLVE-BLOCK-END
