"""Search for 20 distinct positive integers <= 10000 minimizing max(|A+A|,|A*A|)."""
import time
import random
import itertools
from collections import defaultdict

_DEADLINE = time.time() + 80.0  # safety margin under 90s kill

def _sumset_size(A):
    s = set()
    n = len(A)
    for i in range(n):
        for j in range(i, n):
            s.add(A[i] + A[j])
    return len(s)

def _prodset_size(A):
    s = set()
    n = len(A)
    for i in range(n):
        for j in range(i, n):
            s.add(A[i] * A[j])
    return len(s)

def _objective(A):
    return max(_sumset_size(A), _prodset_size(A))

def _random_candidate():
    return sorted(random.sample(range(1, 10001), 20))

def _smooth_candidate():
    # Build from 2^a * 3^b * 5^c <= 10000, pick 20 distinct
    pool = set()
    for a in range(14):
        p2 = 1 << a
        if p2 > 10000:
            break
        for b in range(10):
            p3 = p2 * (3 ** b)
            if p3 > 10000:
                break
            for c in range(7):
                v = p3 * (5 ** c)
                if v <= 10000:
                    pool.add(v)
    pool = sorted(pool)
    if len(pool) < 20:
        return _random_candidate()
    # Try to pick a spread of smooth numbers
    candidates = random.sample(pool, min(40, len(pool)))
    best = None
    best_obj = float('inf')
    for _ in range(20):
        sub = random.sample(candidates, 20)
        A = sorted(sub)
        o = _objective(A)
        if o < best_obj:
            best_obj = o
            best = A
    return best

def _local_search(A, time_limit):
    A = sorted(A)
    best = list(A)
    best_obj = _objective(best)
    start = time.time()
    n = len(best)
    stall = 0
    step = 3
    while time.time() < time_limit:
        improved = False
        # Try single-element replacement
        for i in range(n):
            if time.time() > time_limit:
                break
            # Try replacing best[i] with nearby values
            lo = max(1, best[i] - step)
            hi = min(10000, best[i] + step)
            # Try a few random values in range plus structured ones
            trials = set()
            for d in range(-step, step + 1, max(1, step // 5)):
                v = best[i] + d
                if 1 <= v <= 10000:
                    trials.add(v)
            # Add some randoms
            for _ in range(8):
                trials.add(random.randint(lo, hi))
            for v in trials:
                if v in best:
                    continue
                newA = best[:]
                newA[i] = v
                newA.sort()
                o = _objective(newA)
                if o < best_obj:
                    best = newA
                    best_obj = o
                    improved = True
                    break
            if improved:
                break
        if not improved:
            stall += 1
            if stall > 5:
                step = min(step + 2, 200)
                # Occasional larger jump
                idx = random.randrange(n)
                best[idx] = random.randint(1, 10000)
                best.sort()
                best_obj = _objective(best)
                stall = 0
            elif stall > 15:
                break
        else:
            stall = 0
    return best, best_obj

def construct():
    random.seed(42)
    start = time.time()
    deadline = start + 78.0  # keep 12s safety margin

    best_A = None
    best_obj = float('inf')

    # Initial diverse candidates
    inits = []
    # Arithmetic progression (minimizes sumset to 39)
    inits.append(list(range(1, 21)))
    # Smooth numbers
    inits.append(_smooth_candidate())
    # Random
    for _ in range(5):
        inits.append(_random_candidate())
    # Geometric-ish
    base = 2
    inits.append(sorted(set(int(base ** (i * 0.5)) for i in range(40)) and list({max(1, int(base ** (i * 0.5))) for i in range(40)})))
    # Ensure distinct and valid
    cleaned = []
    for cand in inits:
        cand = sorted(set(int(x) for x in cand))
        cand = [x for x in cand if 1 <= x <= 10000]
        if len(cand) == 20:
            cleaned.append(cand)
        else:
            # pad
            cand = list(cand)
            used = set(cand)
            while len(cand) < 20:
                v = random.randint(1, 10000)
                if v not in used:
                    cand.append(v)
                    used.add(v)
            cand = sorted(cand)
            cleaned.append(cand)
    inits = cleaned

    # Evaluate and pick best start
    for cand in inits:
        o = _objective(cand)
        if o < best_obj:
            best_obj = o
            best_A = cand

    # Local search from best
    if time.time() < deadline:
        best_A, best_obj = _local_search(best_A, deadline)

    # Multi-start: run more local searches from random inits while time permits
    budget = 10  # seconds per local search attempt
    while time.time() + budget < deadline - 5.0:
        cand = _random_candidate()
        # Quick evaluate; if promising, polish
        o0 = _objective(cand)
        # Always try local search with limited budget
        t_end = min(deadline - 5.0, time.time() + budget)
        res, o = _local_search(cand, t_end)
        if o < best_obj:
            best_obj = o
            best_A = res

    # Final polish on the best with remaining time
    if time.time() < deadline - 2.0:
        best_A, best_obj = _local_search(best_A, deadline - 2.0)

    # Guarantee validity: 20 distinct positive ints <= 10000
    best_A = sorted(set(int(x) for x in best_A))
    best_A = [x for x in best_A if 1 <= x <= 10000]
    if len(best_A) != 20:
        # Fallback to safe set
        best_A = list(range(1, 21))

    return best_A
