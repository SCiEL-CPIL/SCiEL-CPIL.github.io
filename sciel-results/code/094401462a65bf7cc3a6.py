"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
import math

# EVOLVE-BLOCK-START

_N = 12
_RANGE = 129
_ALL = list(range(_RANGE))
_ALL_SET = set(_RANGE)


def _is_sidon(subset):
    """True if all unordered pair sums (i<=j, incl. diagonals) are unique."""
    seen = set()
    n = len(subset)
    for i in range(n):
        a = subset[i]
        for j in range(i, n):
            s = a + subset[j]
            if s in seen:
                return False
            seen.add(s)
    return True


def _max_sidon_size(s):
    """Exact largest Sidon subset size via branch-and-bound.

    Valid upper bound: len(chosen) + len(remaining). Exact feasibility checked
    by recomputing pair sums of the partial set. Fast for n=12.
    """
    s = list(s)
    n = len(s)
    best = [0]

    def backtrack(remaining, chosen, sums):
        # optimistic bound
        if len(chosen) + len(remaining) <= best[0]:
            return
        if not remaining:
            if len(chosen) > best[0]:
                best[0] = len(chosen)
            return
        v = remaining[0]
        rest = remaining[1:]
        # include v if its sums with chosen are all new
        ok = True
        new_sums = set(sums)
        for c in chosen:
            sm = s[v] + s[c]
            if sm in new_sums:
                ok = False
                break
            new_sums.add(sm)
        if ok:
            sm = 2 * s[v]
            if sm in new_sums:
                ok = False
            else:
                new_sums.add(sm)
        if ok:
            backtrack(rest, chosen + [v], new_sums)
        # exclude v
        backtrack(rest, chosen, sums)

    backtrack(list(range(n - 1, -1, -1)), [], set())
    return best[0]


def _evaluate(s):
    """Return the largest Sidon subset size for set s (exact)."""
    return _max_sidon_size(sorted(s))


def _collision_count(subset):
    """Number of duplicate pair sums in subset (0 = fully Sidon)."""
    sums = {}
    conflicts = 0
    n = len(subset)
    for i in range(n):
        a = subset[i]
        for j in range(i, n):
            sm = a + subset[j]
            if sm in sums:
                conflicts += 1
            else:
                sums[sm] = 1
    return conflicts


def _soft_objective(s):
    """Smooth surrogate: soft-min of collision counts over all subsets (size>=2).

    Low soft-min => some subset is very collision-free (bad for us), so the
    annealer minimizes it. A hard penalty for any fully-Sidon subset reinforces
    forcing collisions. log-sum-exp for numerical stability.
    """
    s = list(s)
    n = len(s)
    T = 0.5
    vals = []
    for size in range(2, n + 1):
        for combo in itertools.combinations(range(n), size):
            vals.append(_collision_count([s[i] for i in combo]))
    m = min(vals)
    tot = 0.0
    for v in vals:
        tot += math.exp(-(v - m) / T)
    softmin = m - T * math.log(tot)
    penalty = 10.0 * sum(1 for v in vals if v == 0)
    return softmin + penalty


def _random_candidate():
    return sorted(random.sample(range(_RANGE), _N))


def _graph_seed():
    """Greedy set-cover style: pick 12 most collision-prone vertices."""
    pair_count = [0] * (2 * _RANGE)
    for a in range(_RANGE):
        for b in range(a + 1, _RANGE):
            pair_count[a + b] += 1
    score = []
    for x in range(_RANGE):
        sc = pair_count[x]
        if 2 * x < len(pair_count):
            sc += pair_count[2 * x]
        score.append(sc)
    order = sorted(range(_RANGE), key=lambda x: (-score[x], x))
    return sorted(order[:_N])


def _local_search(current, deadline, max_iter=500):
    """Hill-climbing on exact Sidon size; swap one element for another."""
    current = sorted(current)
    best_score = _evaluate(current)
    best_set = list(current)
    in_set = set(current)
    iter_count = 0
    while time.time() < deadline and iter_count < max_iter:
        iter_count += 1
        improved = False
        positions = list(range(_N))
        random.shuffle(positions)
        for i in positions:
            if time.time() > deadline:
                break
            candidates = list(_ALL_SET - in_set)
            random.shuffle(candidates)
            for new_val in candidates[:60]:
                candidate = list(current)
                candidate[i] = new_val
                candidate = sorted(candidate)
                score = _evaluate(candidate)
                if score < best_score:
                    best_score = score
                    best_set = candidate
                    in_set.discard(current[i])
                    in_set.add(new_val)
                    current = best_set
                    improved = True
                    break
            if improved or time.time() > deadline:
                break
        if not improved:
            break
    return best_set, best_score


def _local_search_limited(current, deadline):
    return _local_search(current, deadline, max_iter=120)


def _simulated_annealing(current, deadline, max_iter=2000):
    """Annealing on the smooth surrogate; track best by exact Sidon size."""
    current = sorted(current)
    best_set = list(current)
    best_score = _evaluate(current)
    in_set = set(current)
    temp = 3.0
    cooling = 0.997
    cur_obj = _soft_objective(current)
    iter_count = 0
    while time.time() < deadline and iter_count < max_iter:
        iter_count += 1
        if time.time() > deadline:
            break
        i = random.randint(0, _N - 1)
        old_val = current[i]
        new_val = random.choice(list(_ALL_SET - in_set))
        candidate = list(current)
        candidate[i] = new_val
        candidate = sorted(candidate)
        new_obj = _soft_objective(candidate)
        delta = new_obj - cur_obj
        if delta <= 0 or random.random() < math.exp(-delta / temp):
            in_set.discard(old_val)
            in_set.add(new_val)
            current = candidate
            cur_obj = new_obj
            new_score = _evaluate(candidate)
            if new_score < best_score:
                best_score = new_score
                best_set = list(current)
        temp *= cooling
    return best_set, best_score


def _polish(current, deadline):
    """High-precision polish: exhaustive single-swap over ALL replacements."""
    current = sorted(current)
    best_score = _evaluate(current)
    best_set = list(current)
    in_set = set(current)
    for i in range(_N):
        for new_val in _ALL_SET - in_set:
            candidate = list(current)
            candidate[i] = new_val
            candidate = sorted(candidate)
            score = _evaluate(candidate)
            if score < best_score:
                best_score = score
                best_set = candidate
                in_set.discard(current[i])
                in_set.add(new_val)
                current = best_set
    return best_set, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s limit

    random.seed(42)

    best_set = list(range(_N))
    best_score = _evaluate(best_set)

    # Seed with graph-based collision-prone set
    gset = _graph_seed()
    gscore = _evaluate(gset)
    if gscore < best_score:
        best_score = gscore
        best_set = gset

    restarts = 0
    while time.time() < deadline:
        restarts += 1
        candidate = _random_candidate()
        score = _evaluate(candidate)
        if score < best_score:
            best_score = score
            best_set = candidate
            best_set, best_score = _local_search_limited(best_set, deadline)
        if restarts % 3 == 0 and time.time() < deadline:
            sa_set, sa_score = _simulated_annealing(best_set, deadline, max_iter=400)
            if sa_score < best_score:
                best_score = sa_score
                best_set = sa_set

    if time.time() < deadline:
        best_set, best_score = _polish(best_set, deadline)

    return sorted(best_set)

# EVOLVE-BLOCK-END
