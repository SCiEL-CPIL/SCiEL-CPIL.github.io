"""Combined approach: fast bitset scoring + structured/greedy init + SA + local search."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _score_mask(mask):
    """
    Count distinct d > 0 such that there exists a with a, a+d, a+2d all in set.
    mask: bit k set means value k is in the set (k from 1..512).
    """
    count = 0
    for d in range(1, 256):
        if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
            count += 1
    return count


def _score_set(s):
    mask = 0
    for x in s:
        mask |= (1 << x)
    return _score_mask(mask)


def _greedy_init():
    """
    Greedy construction: start with empty set, add elements that maximize score increment.
    """
    s = set()
    mask = 0
    best_score = 0
    candidates = list(range(1, 513))
    random.shuffle(candidates)

    for _ in range(32):
        best_elem = None
        best_new_score = best_score
        for c in candidates:
            if c in s:
                continue
            new_mask = mask | (1 << c)
            new_score = _score_mask(new_mask)
            if new_score > best_new_score:
                best_new_score = new_score
                best_elem = c
        if best_elem is not None:
            s.add(best_elem)
            mask |= (1 << best_elem)
            best_score = best_new_score
        else:
            if candidates:
                c = candidates.pop()
                s.add(c)
                mask |= (1 << c)
                best_score = _score_mask(mask)
    return s, best_score


def _structured_init():
    """
    Try several structured initializations: intervals, arithmetic progressions, etc.
    """
    best_s = set()
    best_score = -1

    # Interval
    for start in range(1, 513 - 32 + 1):
        s = set(range(start, start + 32))
        sc = _score_set(s)
        if sc > best_score:
            best_score = sc
            best_s = set(s)

    # Arithmetic progression with various steps
    for step in range(1, 20):
        for start in range(1, 513 - 31 * step + 1):
            s = set(start + i * step for i in range(32))
            if max(s) > 512:
                continue
            sc = _score_set(s)
            if sc > best_score:
                best_score = sc
                best_s = set(s)

    # Random structured: mix of two intervals
    for _ in range(50):
        s = set()
        # First half from one interval, second half from another
        start1 = random.randint(1, 200)
        start2 = random.randint(250, 480)
        for i in range(16):
            s.add(start1 + i * 2)
        for i in range(16):
            s.add(start2 + i * 2)
        if len(s) != 32 or max(s) > 512:
            continue
        sc = _score_set(s)
        if sc > best_score:
            best_score = sc
            best_s = set(s)

    return best_s, best_score


def _simulated_annealing(s, initial_score, deadline):
    """
    Simulated annealing to improve the set.
    """
    current = set(s)
    current_score = initial_score
    best = set(current)
    best_score = current_score

    T_start = 5.0
    T_end = 0.01
    alpha = 0.9995

    T = T_start
    while T > T_end:
        if time.time() > deadline:
            break

        if len(current) != 32:
            break

        to_remove = random.choice(list(current))
        to_add = random.randint(1, 512)
        if to_add in current:
            continue

        new_set = set(current)
        new_set.remove(to_remove)
        new_set.add(to_add)
        new_score = _score_set(new_set)

        if new_score > current_score:
            current = new_set
            current_score = new_score
            if current_score > best_score:
                best = set(current)
                best_score = current_score
        else:
            delta = new_score - current_score
            if delta < 0:
                prob = np.exp(delta / T)
                if random.random() < prob:
                    current = new_set
                    current_score = new_score

        T *= alpha

    return best, best_score


def _local_search(s, initial_score, deadline):
    """
    Hill climbing: try all single swaps, accept improvements.
    """
    current = set(s)
    current_score = initial_score
    best = set(current)
    best_score = current_score

    improved = True
    while improved and time.time() < deadline:
        improved = False
        for a in list(current):
            if time.time() > deadline:
                break
            for b in range(1, 513):
                if b in current:
                    continue
                new_set = set(current)
                new_set.remove(a)
                new_set.add(b)
                new_score = _score_set(new_set)
                if new_score > current_score:
                    current = new_set
                    current_score = new_score
                    improved = True
                    break
            if improved:
                break

        if improved:
            best = set(current)
            best_score = current_score

    return best, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin

    random.seed(42)
    np.random.seed(42)

    # Phase 1: Multiple initializations
    s1, score1 = _greedy_init()
    s2, score2 = _structured_init()

    if score1 >= score2:
        best_s, best_score = set(s1), score1
    else:
        best_s, best_score = set(s2), score2

    # Phase 2: Simulated annealing
    if time.time() < deadline - 30:
        best_s, best_score = _simulated_annealing(best_s, best_score, deadline - 30)

    # Phase 3: Local search (hill climbing)
    if time.time() < deadline:
        best_s, best_score = _local_search(best_s, best_score, deadline)

    return sorted(list(best_s))

# EVOLVE-BLOCK-END
