import time
import random
import sys
import math
from collections import deque

N = 12
FULL_MASK = (1 << N) - 1
DEADLINE_SECONDS = 85.0  # safety margin before 90s kill


def _build_subset_trans(trans):
    """Precompute subset transitions for one letter using dynamic programming."""
    size = 1 << N
    next_subset = [0] * size
    for mask in range(1, size):
        lsb = mask & (-mask)
        i = lsb.bit_length() - 1
        rest = mask ^ lsb
        next_subset[mask] = next_subset[rest] | (1 << trans[i])
    return next_subset


def compute_threshold(trans_a, trans_b):
    """
    Computes the length of the shortest reset word for the automaton defined by
    trans_a and trans_b. Returns 0 if not synchronizing.
    Uses BFS on the subset automaton with precomputed transitions.
    """
    next_a = _build_subset_trans(trans_a)
    next_b = _build_subset_trans(trans_b)

    size = 1 << N
    dist = [-1] * size
    dist[FULL_MASK] = 0
    queue = deque([FULL_MASK])

    while queue:
        cur = queue.popleft()
        if cur & (cur - 1) == 0:
            return dist[cur]

        nxt_a = next_a[cur]
        if dist[nxt_a] == -1:
            dist[nxt_a] = dist[cur] + 1
            queue.append(nxt_a)

        nxt_b = next_b[cur]
        if dist[nxt_b] == -1:
            dist[nxt_b] = dist[cur] + 1
            queue.append(nxt_b)

    return 0


def _pinwheel_automaton():
    """
    Returns the Pinwheel automaton.
    a: cycle (i -> i+1 mod N)
    b: 0 -> 0, i -> i-1 for i > 0
    """
    a = [(i + 1) % N for i in range(N)]
    b = [0] + [i - 1 for i in range(1, N)]
    return a, b


def _hardcoded_best():
    """Return a known strong automaton for N=12."""
    a = [(i + 1) % N for i in range(N)]
    b = [11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    return a, b


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    random.seed(42)

    best_a = None
    best_b = None
    best_score = 0

    # 1) Evaluate Pinwheel
    pa, pb = _pinwheel_automaton()
    ps = compute_threshold(pa, pb)
    if ps > best_score:
        best_score = ps
        best_a = pa[:]
        best_b = pb[:]

    # 2) Evaluate Hardcoded
    ha, hb = _hardcoded_best()
    hs = compute_threshold(ha, hb)
    if hs > best_score:
        best_score = hs
        best_a = ha[:]
        best_b = hb[:]

    # 3) Simulated Annealing
    if best_a is None:
        # Fallback
        best_a = list(range(N))
        best_b = list(range(N))
        best_score = compute_threshold(best_a, best_b)

    cur_a = best_a[:]
    cur_b = best_b[:]
    cur_score = best_score

    # Energy: -score if sync, large penalty if not
    def get_energy(a, b):
        s = compute_threshold(a, b)
        if s == 0:
            return 10000.0
        return -float(s)

    cur_energy = get_energy(cur_a, cur_b)
    
    T = 10.0
    cooling_factor = 0.999
    
    while time.time() < deadline - 5.0:
        # Generate neighbor
        if random.random() < 0.5:
            state = random.randint(0, N - 1)
            old_val = cur_a[state]
            new_val = random.randint(0, N - 1)
            if new_val == old_val:
                continue
            cur_a[state] = new_val
            is_a_move = True
        else:
            state = random.randint(0, N - 1)
            old_val = cur_b[state]
            new_val = random.randint(0, N - 1)
            if new_val == old_val:
                continue
            cur_b[state] = new_val
            is_a_move = False

        new_energy = get_energy(cur_a, cur_b)
        # We need the score to update best. 
        # get_energy returns energy. 
        # If new_energy < 10000, score = -new_energy.
        new_score = int(-new_energy) if new_energy < 10000 else 0

        delta_e = new_energy - cur_energy
        
        # Accept if better (lower energy) or with probability exp(-delta_e/T)
        if delta_e < 0 or (T > 0 and random.random() < math.exp(-delta_e / T)):
            cur_energy = new_energy
            cur_score = new_score
            
            if cur_score > best_score:
                best_score = cur_score
                best_a = cur_a[:]
                best_b = cur_b[:]
        else:
            # Revert
            if is_a_move:
                cur_a[state] = old_val
            else:
                cur_b[state] = old_val

        # Cool down
        T *= cooling_factor
        if T < 0.01:
            T = 0.01

    # Ensure valid return
    if best_a is None or best_score == 0:
        a = [(i + 1) % N for i in range(N)]
        b = [0] * N
        return [a, b]

    return [best_a, best_b]
