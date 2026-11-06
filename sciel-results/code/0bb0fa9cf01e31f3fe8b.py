"""Improved: gcd-diversity greedy + conflict-targeted local search + simulated annealing."""
import time
import random
from math import gcd

_TIME_LIMIT = 85.0
_N = 128

# Precompute gcd matrix
_gcd_matrix = [[0] * (_N + 1) for _ in range(_N + 1)]
for _i in range(1, _N + 1):
    for _j in range(_i, _N + 1):
        _g = gcd(_i, _j)
        _gcd_matrix[_i][_j] = _g
        _gcd_matrix[_j][_i] = _g


def _can_add(S, x, Sset=None):
    """Check if adding x to S keeps the set triple-free."""
    n = len(S)
    if n < 2:
        return True
    # Collect gcds of x with each element in S
    gx = [_gcd_matrix[S[i]][x] for i in range(n)]
    for i in range(n):
        ga = gx[i]
        if ga == 1:
            continue  # gcd=1 can't form bad triple with another gcd=1 unless both are 1
        for j in range(i + 1, n):
            if gx[j] == ga and _gcd_matrix[S[i]][S[j]] == ga:
                return False
    return True


def _count_conflicts(S):
    """Count the number of bad triples in S."""
    n = len(S)
    count = 0
    for i in range(n):
        a = S[i]
        for j in range(i + 1, n):
            b = S[j]
            g_ab = _gcd_matrix[a][b]
            if g_ab == 1:
                continue
            for k in range(j + 1, n):
                c = S[k]
                if _gcd_matrix[a][c] == g_ab and _gcd_matrix[b][c] == g_ab:
                    count += 1
    return count


def _greedy(order, deadline):
    """Greedy: iterate order, add if valid."""
    S = []
    Sset = set()
    for x in order:
        if time.time() > deadline:
            break
        if x in Sset:
            continue
        if _can_add(S, x, Sset):
            S.append(x)
            Sset.add(x)
    return S


def _greedy_fill(S, Sset, universe, deadline):
    """Greedily add all possible elements repeatedly."""
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
            if _can_add(S, x, Sset):
                S.append(x)
                Sset.add(x)
                num_added += 1
                improved = True
    return S, Sset, num_added


def _local_search(S, deadline, rng):
    """Conflict-targeted local search."""
    S = list(S)
    Sset = set(S)
    universe = list(range(1, 129))
    d = deadline - 3

    # Phase 1: greedy fill
    S, Sset, _ = _greedy_fill(S, Sset, universe, d)

    # Phase 2: 1-swap with greedy refill
    if time.time() < d - 8:
        for _iter in range(50):
            if time.time() > d - 8:
                break
            progress = False
            for idx in range(len(S)):
                if time.time() > d - 8:
                    break
                removed = S[idx]
                S2 = S[:idx] + S[idx + 1:]
                S2set = set(S2)
                best_gain = 0
                best_x = None
                best_S3 = None
                for x in universe:
                    if time.time() > d - 8:
                        break
                    if x in S2set:
                        continue
                    if _can_add(S2, x, S2set):
                        S3 = S2 + [x]
                        S3set = set(S3)
                        # Greedy refill
                        improved = True
                        while improved:
                            if time.time() > d - 8:
                                break
                            improved = False
                            for y in universe:
                                if y in S3set:
                                    continue
                                if _can_add(S3, y, S3set):
                                    S3.append(y)
                                    S3set.add(y)
                                    improved = True
                        gain = len(S3) - len(S)
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                            best_S3 = S3
                            if gain >= 3:
                                break
                if best_x is not None and best_gain > 0:
                    S = best_S3
                    Sset = set(S)
                    progress = True
                    break
            if not progress:
                break

    # Phase 3: 2-swap
    if time.time() < d - 5:
        for _iter in range(15):
            if time.time() > d - 5:
                break
            progress = False
            n = len(S)
            for i in range(n):
                if time.time() > d - 5:
                    break
                for j in range(i + 1, n):
                    if time.time() > d - 5:
                        break
                    S2 = S[:i] + S[i+1:j] + S[j+1:]
                    S2set = set(S2)
                    best_gain = 0
                    best_S3 = None
                    for x in universe:
                        if time.time() > d - 5:
                            break
                        if x in S2set:
                            continue
                        if _can_add(S2, x, S2set):
                            S3 = S2 + [x]
                            S3set = set(S3)
                            improved = True
                            while improved:
                                if time.time() > d - 5:
                                    break
                                improved = False
                                for y in universe:
                                    if y in S3set:
                                        continue
                                    if _can_add(S3, y, S3set):
                                        S3.append(y)
                                        S3set.add(y)
                                        improved = True
                            gain = len(S3) - len(S)
                            if gain > best_gain:
                                best_gain = gain
                                best_S3 = S3
                                if gain >= 3:
                                    break
                    if best_S3 is not None and best_gain > 0:
                        S = best_S3
                        Sset = set(S)
                        progress = True
                        break
                if progress:
                    break
            if not progress:
                break

    # Phase 4: Simulated annealing
    if time.time() < d - 2:
        T = 5.0
        cooling = 0.995
        S_sa = S[:]
        S_sa_set = set(S_sa)
        for step in range(2000):
            if time.time() > d - 2:
                break
            if T < 0.1:
                break
            move_type = rng.random()
            if move_type < 0.3:
                # Add a random element
                candidates = [x for x in universe if x not in S_sa_set]
                if not candidates:
                    continue
                x = rng.choice(candidates)
                if _can_add(S_sa, x, S_sa_set):
                    S_sa.append(x)
                    S_sa_set.add(x)
                else:
                    # Accept worse with probability
                    conf = _count_conflicts(S_sa)
                    if rng.random() < min(1.0, T / max(1, conf)):
                        S_sa.append(x)
                        S_sa_set.add(x)
            elif move_type < 0.6:
                # Remove a random element
                if len(S_sa) > 10:
                    idx = rng.randrange(len(S_sa))
                    removed = S_sa.pop(idx)
                    S_sa_set.discard(removed)
            else:
                # Swap: remove one, add another
                if len(S_sa) > 10:
                    idx = rng.randrange(len(S_sa))
                    removed = S_sa[idx]
                    candidates = [x for x in universe if x not in S_sa_set]
                    if candidates:
                        x = rng.choice(candidates)
                        S_sa[idx] = x
                        S_sa_set.discard(removed)
                        S_sa_set.add(x)
            T *= cooling
            if len(S_sa) > len(S):
                S = S_sa[:]
                Sset = set(S_sa)

    # Phase 5: Final greedy fill
    S, Sset, _ = _greedy_fill(S, Sset, universe, d)

    # Phase 6: Polish - try removing each element and refilling
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
                S2, S2set, _ = _greedy_fill(S2, S2set, universe, d)
                if len(S2) > len(S):
                    S = S2
                    Sset = S2set
                    polish_improved = True
                    break

    return S


def _gcd_diversity_score(x, S):
    """Score x by how diverse its gcds are with elements in S.
    Higher = more diverse = safer to add."""
    if not S:
        return 0
    gcds = set()
    for a in S:
        gcds.add(_gcd_matrix[x][a])
    return len(gcds)


def _multi_greedy_local(deadline):
    rng = random.Random(12345)
    universe = list(range(1, 129))
    best = []

    # Generate diverse orderings
    orders = [
        list(range(1, 129)),
        list(range(128, 0, -1)),
        # Primes first
        [x for x in range(1, 129) if all(x % p != 0 for p in range(2, int(x**0.5)+1)) and x > 1] +
        [x for x in range(1, 129) if not (all(x % p != 0 for p in range(2, int(x**0.5)+1)) and x > 1)],
        # Odds first
        [x for x in range(1, 129) if x % 2 == 1] + [x for x in range(1, 129) if x % 2 == 0],
        # Evens first
        [x for x in range(1, 129) if x % 2 == 0] + [x for x in range(1, 129) if x % 2 == 1],
        # Numbers with few divisors first (more likely to be coprime)
        sorted(range(1, 129), key=lambda x: sum(1 for d in range(1, x+1) if x % d == 0)),
        # Numbers with many divisors first
        sorted(range(1, 129), key=lambda x: -sum(1 for d in range(1, x+1) if x % d == 0)),
        # Large prime powers first
        sorted(range(1, 129), key=lambda x: (x & (x-1) == 0, -x.bit_count(), x)),
        # Small numbers first
        list(range(1, 129)),
        # Alternating small/large
        [i for i in range(1, 129, 2)] + [i for i in range(2, 129, 2)],
    ]
    # Add random orderings
    for _ in range(30):
        o = universe[:]
        rng.shuffle(o)
        orders.append(o)

    # Run greedy on each ordering
    candidates = []
    for order in orders:
        if time.time() > deadline - 15:
            break
        cand = _greedy(order, deadline)
        candidates.append(cand)
        if len(cand) > len(best):
            best = cand[:]

    # Also try adaptive greedy: at each step, pick the element with highest gcd diversity
    for _ in range(10):
        if time.time() > deadline - 15:
            break
        S = []
        Sset = set()
        remaining = set(range(1, 129))
        while remaining:
            if time.time() > deadline - 15:
                break
            best_x = None
            best_score = -1
            for x in remaining:
                if _can_add(S, x, Sset):
                    score = _gcd_diversity_score(x, S) + rng.random() * 0.1
                    if score > best_score:
                        best_score = score
                        best_x = x
            if best_x is None:
                break
            S.append(best_x)
            Sset.add(best_x)
            remaining.discard(best_x)
        if len(S) > len(best):
            best = S[:]
        candidates.append(S)

    # Local search on top candidates
    candidates.sort(key=len, reverse=True)
    for cand in candidates[:8]:
        if time.time() > deadline - 5:
            break
        improved = _local_search(cand, deadline, rng)
        if len(improved) > len(best):
            best = improved

    return best


def construct():
    try:
        result = _multi_greedy_local(time.time() + _TIME_LIMIT)
    except Exception:
        # Fallback: simple ascending greedy
        result = []
        Sset = set()
        for x in range(1, 129):
            if _can_add(result, x, Sset):
                result.append(x)
                Sset.add(x)
    if not result:
        result = [1]
    return result
