import time
import random
import numpy as np

_DEADLINE = 85.0  # seconds wall-clock safety margin

def _any_triple_divisible(arr, d):
    """Return True if any 3 distinct elements of arr sum to a multiple of d."""
    m = len(arr)
    if m < 3:
        return False
    # suffix residue sets: res_set[i] = set of arr[k] % d for k in range(i, m)
    suffix = [None] * (m + 1)
    suffix[m] = frozenset()
    for i in range(m - 1, -1, -1):
        suffix[i] = suffix[i + 1] | {arr[i] % d}
    for i in range(m - 2):
        ai = arr[i]
        for j in range(i + 1, m - 1):
            need = (-ai - arr[j]) % d
            if need in suffix[j + 1]:
                return True
    return False


def _can_add(S, x):
    """S is a list/set of selected ints (not containing x). Check if adding x keeps validity."""
    S_list = list(S)
    m = len(S_list)
    # 1) For a = x: any triple of S_list sum divisible by x?
    if _any_triple_divisible(S_list, x):
        return False
    # 2) For each a in S_list: triples {x, u, v} with u,v in S_list\{a}
    #    need (x+u+v) % a != 0 for all u<v.
    #    i.e. no pair (u,v) in S_list\{a} with (u+v)%a == (-x)%a.
    for a in S_list:
        others = [v for v in S_list if v != a]
        need = (-x) % a
        # check if any pair in others sums to need mod a
        seen = set()
        for v in others:
            r = v % a
            if (need - r) % a in seen:
                return False
            seen.add(r)
    return True


def _is_valid(S):
    """Full validation of set S."""
    S_list = list(S)
    m = len(S_list)
    for a in S_list:
        others = [v for v in S_list if v != a]
        if _any_triple_divisible(others, a):
            return False
    return True


def _greedy(order, deadline):
    """Greedy add in given order, respecting deadline."""
    selected = []
    sel_set = set()
    for x in order:
        if time.time() > deadline:
            break
        if _can_add(sel_set, x):
            selected.append(x)
            sel_set.add(x)
    return selected


def _local_search_improve(best_set, deadline):
    """
    Local search on best_set (a set of ints).
    Tries to increase cardinality by:
      1. Direct adds.
      2. Remove one, add two (net +1).
    Returns improved set.
    """
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False

        # 1) Direct adds
        for x in range(192, 0, -1):
            if time.time() > deadline - 2.0:
                break
            if x not in best_set:
                if _can_add(best_set, x):
                    best_set.add(x)
                    improved = True

        # 2) Remove one, add two (net +1)
        # Try removing each element, then try to add two new ones.
        for a in list(best_set):
            if time.time() > deadline - 2.0:
                break
            trial = best_set - {a}
            # Try to add first new element
            for x in range(192, 0, -1):
                if time.time() > deadline - 2.0:
                    break
                if x in trial:
                    continue
                if _can_add(trial, x):
                    trial2 = trial | {x}
                    # Try to add second new element
                    for y in range(192, 0, -1):
                        if time.time() > deadline - 2.0:
                            break
                        if y in trial2:
                            continue
                        if _can_add(trial2, y):
                            # Net +1
                            best_set = trial2 | {y}
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

    return best_set


def construct():
    start = time.time()
    deadline = start + _DEADLINE
    n = 192
    best = set()

    # Seed 1: descending (matches original, known good)
    cand = _greedy(range(n, 0, -1), deadline)
    if len(cand) > len(best):
        best = set(cand)
    if len(best) == 48:
        if _is_valid(best):
            return list(best)

    # Seed 2: ascending
    if time.time() < deadline:
        cand = _greedy(range(1, n + 1), deadline)
        if len(cand) > len(best):
            best = set(cand)
        if len(best) == 48:
            if _is_valid(best):
                return list(best)

    # Seed 3: randomized greedy runs
    rng = random.Random(42)
    random_deadline = start + 30.0
    while time.time() < random_deadline:
        if time.time() > deadline:
            break
        order = list(range(1, n + 1))
        rng.shuffle(order)
        cand = _greedy(order, deadline)
        if len(cand) > len(best):
            best = set(cand)
        if len(best) == 48:
            if _is_valid(best):
                return list(best)
            break

    # Local search on the best found set
    if time.time() < deadline - 5.0:
        best = _local_search_improve(best, deadline)
        if len(best) == 48:
            if _is_valid(best):
                return list(best)

    result = list(best)
    if _is_valid(result):
        return result
    fb = _greedy(range(n, 0, -1), start + _DEADLINE)
    if _is_valid(fb):
        return fb
    return [n]
