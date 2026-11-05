"""Construct a 12-state binary deterministic synchronizing automaton with a long reset word."""
import time
import random
from collections import deque

N = 12
FULL_MASK = (1 << N) - 1

def _apply_mask(a, mask):
    """Apply transformation a to bitmask, return new bitmask."""
    result = 0
    m = mask
    while m:
        low = m & (-m)
        i = low.bit_length() - 1
        result |= (1 << a[i])
        m ^= low
    return result

def _threshold(a, b):
    """BFS on subset automaton using bitmasks for speed. Returns -1 if not synchronizing."""
    start = FULL_MASK
    if start & (start - 1) == 0:  # single bit
        return 0
    visited = {start: 0}
    q = deque([start])
    while q:
        curr = q.popleft()
        d = visited[curr]
        if curr & (curr - 1) == 0:  # single state
            return d
        na = _apply_mask(a, curr)
        if na not in visited:
            visited[na] = d + 1
            q.append(na)
        nb = _apply_mask(b, curr)
        if nb not in visited:
            visited[nb] = d + 1
            q.append(nb)
    return -1

def _is_synchronizing(a, b):
    """Quick check if automaton synchronizes. Returns threshold or -1."""
    return _threshold(a, b)

def construct():
    """Construct a 12-state binary deterministic synchronizing automaton with long reset word."""
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin below 60s wall-clock limit

    best_a = None
    best_b = None
    best_thr = 0

    a_cycle = [(i + 1) % N for i in range(N)]
    a_rev = [(i - 1) % N for i in range(N)]

    def try_automaton(a, b):
        nonlocal best_a, best_b, best_thr
        thr = _threshold(a, b)
        if thr > 0 and thr > best_thr:
            best_a, best_b, best_thr = a[:], b[:], thr

    # Construction A: a = cycle, b = [0,1,2,...,10,10] (pinwheel)
    b_mono = list(range(N-1)) + [N-2]
    try_automaton(a_cycle, b_mono)
    try_automaton(a_rev, b_mono)

    if time.time() > deadline:
        return best_a, best_b

    # Construction B: a = cycle, b = identity with one defect
    for defect_pos in range(N):
        for defect_target in range(N):
            if defect_target == defect_pos:
                continue
            b_id = list(range(N))
            b_id[defect_pos] = defect_target
            try_automaton(a_cycle, b_id)
            try_automaton(a_rev, b_id)
            if time.time() > deadline:
                return best_a, best_b

    if time.time() > deadline:
        return best_a, best_b

    # Construction C: Systematic two-defect search on identity base
    # b = identity with two positions modified
    for p1 in range(N):
        for p2 in range(p1+1, N):
            for t1 in range(N):
                if t1 == p1:
                    continue
                for t2 in range(N):
                    if t2 == p2:
                        continue
                    if time.time() > deadline:
                        return best_a, best_b
                    b_two_def = list(range(N))
                    b_two_def[p1] = t1
                    b_two_def[p2] = t2
                    try_automaton(a_cycle, b_two_def)

    if time.time() > deadline:
        return best_a, best_b

    # Construction D: One-defect perturbations of monotone base
    base_b = list(range(N-1)) + [N-2]
    for i in range(N):
        for new_val in range(N):
            if new_val == base_b[i]:
                continue
            if time.time() > deadline:
                return best_a, best_b
            b_test = base_b[:]
            b_test[i] = new_val
            try_automaton(a_cycle, b_test)
            try_automaton(a_rev, b_test)

    if time.time() > deadline:
        return best_a, best_b

    # Construction E: Structured "shift with gap" family
    # b(i) = (i - k) % N for most, with a gap/defect
    for k in range(1, N):
        for defect_pos in range(N):
            for defect_target in range(N):
                if time.time() > deadline:
                    return best_a, best_b
                b_shift = [((i - k) % N) for i in range(N)]
                b_shift[defect_pos] = defect_target
                try_automaton(a_cycle, b_shift)

    if time.time() > deadline:
        return best_a, best_b

    # Construction F: Random search with a = cycle (more trials)
    rng = random.Random(12345)
    for trial in range(500):
        if time.time() > deadline:
            break
        b_rand = [rng.randint(0, N-1) for _ in range(N)]
        try_automaton(a_cycle, b_rand)

    if time.time() > deadline:
        return best_a, best_b

    # Construction G: Random permutations for a, random b
    for trial in range(200):
        if time.time() > deadline:
            break
        a_perm = list(range(N))
        rng.shuffle(a_perm)
        b_rand = [rng.randint(0, N-1) for _ in range(N)]
        try_automaton(a_perm, b_rand)

    if time.time() > deadline:
        return best_a, best_b

    # Construction H: Local improvement - mutate best found
    if best_a is not None and best_b is not None and best_thr > 0:
        for _ in range(500):
            if time.time() > deadline:
                break
            b_mut = best_b[:]
            pos = rng.randint(0, N-1)
            b_mut[pos] = rng.randint(0, N-1)
            try_automaton(best_a, b_mut)

    if time.time() > deadline:
        return best_a, best_b

    # Fallback: ensure we have a valid synchronizing automaton
    if best_a is None or best_b is None or best_thr <= 0:
        best_a = [(i + 1) % N for i in range(N)]
        best_b = [0] * N
        best_thr = _threshold(best_a, best_b)

    return best_a, best_b
