import math
import itertools
import time
import random

def score_set(U):
    if not U:
        return 0
    U = list(set(U))
    if 0 not in U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a+b)
            diffs.add(a-b)
    size_sum = len(sums)
    size_diff = len(diffs)
    if size_sum == 0:
        return 0
    ratio = size_diff / size_sum
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 0
    return 1 + math.log(ratio) / denom

def _is_golomb(U):
    """Check if U is a Golomb ruler (all pairwise differences are unique)."""
    diffs = set()
    n = len(U)
    for i in range(n):
        for j in range(i + 1, n):
            d = U[j] - U[i]
            if d in diffs:
                return False
            diffs.add(d)
    return True

def local_search(U, max_val, time_limit, allow_swaps=True):
    """Try to improve U by replacing elements one at a time, and optionally 2-swaps."""
    U = sorted(set(U))
    if 0 not in U:
        return U
    best = U
    best_score = score_set(best)
    start = time.time()
    rng = random.Random(12345)
    
    improved = True
    while improved and (time.time() - start) < time_limit:
        improved = False
        # Single element moves
        for i in range(1, len(best)):  # Don't replace 0
            orig = best[i]
            # Try a wider range of new values
            candidates = []
            for new_val in range(1, max_val + 1):
                if new_val not in best:
                    candidates.append(new_val)
            rng.shuffle(candidates)
            for new_val in candidates:
                if (time.time() - start) >= time_limit:
                    break
                trial = list(best)
                trial[i] = new_val
                trial.sort()
                s = score_set(trial)
                if s > best_score + 1e-12:
                    best = trial
                    best_score = s
                    improved = True
        # 2-swap moves (exchange two non-zero elements)
        if allow_swaps and len(best) >= 3 and (time.time() - start) < time_limit:
            for i in range(1, len(best)):
                for j in range(i+1, len(best)):
                    if (time.time() - start) >= time_limit:
                        break
                    trial = list(best)
                    trial[i], trial[j] = trial[j], trial[i]
                    # Also try shifting one of the swapped elements
                    for delta in [-2, -1, 1, 2]:
                        nv = trial[j] + delta
                        if nv < 1 or nv in best:
                            continue
                        trial2 = list(trial)
                        trial2[j] = nv
                        trial2.sort()
                        s = score_set(trial2)
                        if s > best_score + 1e-12:
                            best = trial2
                            best_score = s
                            improved = True
                            break
    return best

def greedy_construct(max_size, max_val, time_limit):
    """Greedy construction: build set to maximize diff/sum ratio."""
    U = [0]
    start = time.time()
    rng = random.Random(777)
    for _ in range(max_size - 1):
        if (time.time() - start) >= time_limit:
            break
        best_candidate = None
        best_ratio = -1
        # Try candidate values
        candidates = list(range(1, max_val + 1))
        rng.shuffle(candidates)
        for cand in candidates:
            if cand in U:
                continue
            trial = sorted(U + [cand])
            sums = set()
            diffs = set()
            for a in trial:
                for b in trial:
                    sums.add(a+b)
                    diffs.add(a-b)
            ratio = len(diffs) / max(1, len(sums))
            # Prefer higher ratio, and penalize large max
            penalty = math.log(2 * max(trial) + 1) / 100
            score = ratio - penalty
            if score > best_ratio:
                best_ratio = score
                best_candidate = cand
        if best_candidate is not None:
            U.append(best_candidate)
        else:
            break
    return sorted(U)

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin from 150s

    best_set = [0, 1, 3]
    best_score = score_set(best_set)

    # Known good Golomb rulers as starting candidates
    golomb_rulers = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 18, 23, 25],
        [0, 1, 4, 10, 18, 23, 29, 34],
        [0, 1, 4, 10, 18, 23, 29, 37, 44],
    ]
    
    # Evaluate Golomb rulers and pick best starting point
    for gr in golomb_rulers:
        s = score_set(gr)
        if s > best_score:
            best_score = s
            best_set = gr

    # Phase 1: Exhaustive search for small sets
    configs = [
        (3, 50),
        (4, 40),
        (5, 30),
        (6, 20),
        (7, 15),
        (8, 12),
    ]

    for n, lim in configs:
        if time.time() > deadline:
            break
        count = 0
        for comb in itertools.combinations(range(1, lim+1), n-1):
            if time.time() > deadline:
                break
            U = [0] + list(comb)
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = U
            count += 1

    # Phase 2: Local search on the best found set (wider range + swaps)
    if time.time() < deadline - 5:
        max_val = max(best_set) + 25
        improved = local_search(best_set, max_val, deadline - time.time() - 3, allow_swaps=True)
        s = score_set(improved)
        if s > best_score:
            best_score = s
            best_set = improved

    # Phase 3: Greedy construction + local search
    if time.time() < deadline - 10:
        for attempt in range(3):
            if time.time() > deadline - 5:
                break
            U = greedy_construct(8, 40, 3.0)
            if U:
                s = score_set(U)
                if s > best_score - 0.01:
                    max_val = max(U) + 20
                    improved = local_search(U, max_val, min(3.0, deadline - time.time() - 3), allow_swaps=True)
                    s2 = score_set(improved)
                    if s2 > best_score:
                        best_score = s2
                        best_set = improved

    # Phase 4: Random initial candidates and local search
    if time.time() < deadline - 10:
        random.seed(42)
        for attempt in range(40):
            if time.time() > deadline - 5:
                break
            n = random.randint(4, 8)
            lim = random.randint(10, 35)
            U = [0] + sorted(random.sample(range(1, lim+1), min(n-1, lim)))
            s = score_set(U)
            if s > best_score - 0.01:  # Only polish if reasonably close
                max_val = max(U) + 15
                improved = local_search(U, max_val, min(2.5, deadline - time.time() - 3), allow_swaps=True)
                s2 = score_set(improved)
                if s2 > best_score:
                    best_score = s2
                    best_set = improved

    # Phase 5: Final polish with even wider search
    if time.time() < deadline - 2:
        max_val = max(best_set) + 30
        improved = local_search(best_set, max_val, deadline - time.time() - 1, allow_swaps=True)
        s = score_set(improved)
        if s > best_score:
            best_score = s
            best_set = improved

    return best_set
