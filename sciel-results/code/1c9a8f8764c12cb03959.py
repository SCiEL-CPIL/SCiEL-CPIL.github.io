import time
import random

def _count_progression_diffs(S):
    S_set = set(S)
    diffs = set()
    for a in S:
        for b in S:
            d = b - a
            if d > 0:
                if a + 2 * d in S_set:
                    diffs.add(d)
    return len(diffs)

def construct():
    start_time = time.time()
    deadline = start_time + 142.0
    random.seed(7)
    N = 32
    MAXV = 512
    e = 2.718281828459045

    current = set(random.sample(range(1, MAXV + 1), N))
    cur_score = _count_progression_diffs(current)
    best = set(current)
    best_score = cur_score

    # Phase 1: Simulated annealing
    T = 40.0
    T_min = 0.05
    cool = 0.9997
    while time.time() < deadline - 90.0 and T > T_min:
        elem = random.choice(list(current))
        new_elem = random.randint(1, MAXV)
        if new_elem in current:
            continue
        current.discard(elem)
        current.add(new_elem)
        ns = _count_progression_diffs(current)
        d = ns - cur_score
        if d >= 0 or random.random() < e ** (d / max(T, 1e-9)):
            cur_score = ns
            if ns > best_score:
                best_score = ns
                best = set(current)
        else:
            current.discard(new_elem)
            current.add(elem)
        T *= cool

    # Phase 2: Greedy hill climb with restarts
    while time.time() < deadline - 2.0:
        improved = False
        blist = list(best)
        for elem in blist:
            if time.time() > deadline - 2.0:
                break
            best_cand = None
            best_cand_score = best_score
            for _ in range(25):
                cand = random.randint(1, MAXV)
                if cand in best:
                    continue
                test = set(best)
                test.discard(elem)
                test.add(cand)
                sc = _count_progression_diffs(test)
                if sc > best_cand_score:
                    best_cand_score = sc
                    best_cand = cand
            if best_cand is not None:
                best.discard(elem)
                best.add(best_cand)
                best_score = best_cand_score
                improved = True
        if not improved:
            elem = random.choice(list(best))
            cand = random.randint(1, MAXV)
            if cand not in best:
                best.discard(elem)
                best.add(cand)
                best_score = _count_progression_diffs(best)

    return list(best)
