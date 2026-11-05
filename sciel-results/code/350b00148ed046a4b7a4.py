"""Construct a 12-state binary deterministic synchronizing automaton with a long reset word."""
import time
import random
from collections import deque

N = 12
_DEADLINE = 55.0  # safety margin below the 60s wall-clock kill
_start = time.time()


def _elapsed():
    return time.time() - _start


def _build_trans_table(a, b):
    """Precompute transition tables for all 2^N subsets for letters a and b."""
    # For each subset (as bitmask), precompute the image under a and b
    trans_a = [0] * (1 << N)
    trans_b = [0] * (1 << N)
    for mask in range(1 << N):
        ta = 0
        tb = 0
        m = mask
        while m:
            low = m & (-m)
            i = low.bit_length() - 1
            ta |= (1 << a[i])
            tb |= (1 << b[i])
            m ^= low
        trans_a[mask] = ta
        trans_b[mask] = tb
    return trans_a, trans_b


def _threshold_fast(a, b):
    """BFS on subset automaton using precomputed tables. Returns -1 if not synchronizing."""
    trans_a, trans_b = _build_trans_table(a, b)
    start_mask = (1 << N) - 1
    visited = {start_mask: 0}
    q = deque([start_mask])
    
    while q:
        cur = q.popleft()
        d = visited[cur]
        # Check if single state: power of 2
        if cur & (cur - 1) == 0:
            return d
        
        next_a = trans_a[cur]
        if next_a not in visited:
            visited[next_a] = d + 1
            q.append(next_a)
            
        next_b = trans_b[cur]
        if next_b not in visited:
            visited[next_b] = d + 1
            q.append(next_b)
            
    return -1


def _threshold(a, b):
    """BFS on subset automaton to find shortest reset word length.
    Returns -1 if not synchronizing. Uses bitmask representation for speed."""
    start_mask = (1 << N) - 1
    if start_mask & (start_mask - 1) == 0:
        return 0
    visited = {start_mask: 0}
    q = deque([start_mask])
    
    while q:
        cur = q.popleft()
        d = visited[cur]
        # Check if single state: power of 2
        if cur & (cur - 1) == 0:
            return d
        
        # Next state under 'a'
        next_a = 0
        for i in range(N):
            if cur & (1 << i):
                next_a |= (1 << a[i])
        if next_a not in visited:
            visited[next_a] = d + 1
            q.append(next_a)
            
        # Next state under 'b'
        next_b = 0
        for i in range(N):
            if cur & (1 << i):
                next_b |= (1 << b[i])
        if next_b not in visited:
            visited[next_b] = d + 1
            q.append(next_b)
            
    return -1


def construct():
    """Return two length-N lists of destination states in 0..N-1.

    Must define a complete binary deterministic automaton that synchronizes.
    Score is the exact shortest reset-word length, maximized.
    """
    best_thresh = -1
    best_a = None
    best_b = None
    rng = random.Random(12345)

    # A known-good fallback (cycle + sink) guarantees synchronization.
    fallback_a = [(i + 1) % N for i in range(N)]
    fallback_b = [0] * N
    fb_thr = _threshold(fallback_a, fallback_b)
    if fb_thr > best_thresh:
        best_thresh = fb_thr
        best_a = fallback_a[:]
        best_b = fallback_b[:]

    # ---- Structured search over a = cycle, b = perturbed monotone ----
    a_cycle = [(i + 1) % N for i in range(N)]
    
    # Construction A: b = [0,1,2,...,10,10]
    b_mono = list(range(N-1)) + [N-2]
    thr = _threshold(a_cycle, b_mono)
    if thr > best_thresh:
        best_thresh = thr
        best_a = a_cycle[:]
        best_b = b_mono[:]

    # Construction B: a = cycle, b = identity except b(0) = 1
    b_id = list(range(N))
    b_id[0] = 1
    thr = _threshold(a_cycle, b_id)
    if thr > best_thresh:
        best_thresh = thr
        best_a = a_cycle[:]
        best_b = b_id[:]

    # Construction C: a = cycle, b = [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    b_shift = [0, 0] + list(range(1, N-1))
    thr = _threshold(a_cycle, b_shift)
    if thr > best_thresh:
        best_thresh = thr
        best_a = a_cycle[:]
        best_b = b_shift[:]

    # Construction D: Try all "one defect" perturbations of the monotone b
    base_b = list(range(N-1)) + [N-2]
    for i in range(N):
        for new_val in range(N):
            if new_val == base_b[i]:
                continue
            if _elapsed() > _DEADLINE:
                break
            b_test = base_b[:]
            b_test[i] = new_val
            thr = _threshold(a_cycle, b_test)
            if thr > best_thresh:
                best_thresh = thr
                best_a = a_cycle[:]
                best_b = b_test[:]

    # Construction E: Random search with a = cycle
    for trial in range(200):
        if _elapsed() > _DEADLINE:
            break
        b_rand = [rng.randint(0, N-1) for _ in range(N)]
        thr = _threshold(a_cycle, b_rand)
        if thr > best_thresh:
            best_thresh = thr
            best_a = a_cycle[:]
            best_b = b_rand[:]

    # Construction F: Random permutations for a, random b
    for trial in range(100):
        if _elapsed() > _DEADLINE:
            break
        a_perm = list(range(N))
        rng.shuffle(a_perm)
        b_rand = [rng.randint(0, N-1) for _ in range(N)]
        thr = _threshold(a_perm, b_rand)
        if thr > best_thresh:
            best_thresh = thr
            best_a = a_perm[:]
            best_b = b_rand[:]

    # Construction G: a = cycle in reverse direction
    a_rev = [(i - 1) % N for i in range(N)]
    b_mono2 = list(range(N-1)) + [N-2]
    thr = _threshold(a_rev, b_mono2)
    if thr > best_thresh:
        best_thresh = thr
        best_a = a_rev[:]
        best_b = b_mono2[:]

    # Construction H: Try a "bottleneck" construction
    # a = cycle, b maps most states forward but one state maps to a "sink"
    for sink in range(N):
        if _elapsed() > _DEADLINE:
            break
        b_sink = [(i + 1) % N for i in range(N)]
        b_sink[sink] = sink  # one state is fixed
        thr = _threshold(a_cycle, b_sink)
        if thr > best_thresh:
            best_thresh = thr
            best_a = a_cycle[:]
            best_b = b_sink[:]

    # Construction I: Two-step bottleneck
    # a = cycle, b maps two adjacent states to the same value
    for i in range(N):
        if _elapsed() > _DEADLINE:
            break
        b_two = [(i + 1) % N for i in range(N)]
        b_two[i] = (i + 2) % N  # two states map to same
        thr = _threshold(a_cycle, b_two)
        if thr > best_thresh:
            best_thresh = thr
            best_a = a_cycle[:]
            best_b = b_two[:]

    # Construction J: a = cycle, b = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    b_three = [0, 0, 0] + list(range(1, N-1))
    thr = _threshold(a_cycle, b_three)
    if thr > best_thresh:
        best_thresh = thr
        best_a = a_cycle[:]
        best_b = b_three[:]

    # Construction K: a = cycle, b = [0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    b_dup = [0, 1, 1] + list(range(2, N-1))
    thr = _threshold(a_cycle, b_dup)
    if thr > best_thresh:
        best_thresh = thr
        best_a = a_cycle[:]
        best_b = b_dup[:]

    # ---- Large-scale random restart local search using fast threshold ----
    if best_a is not None and best_b is not None:
        # Phase 1: Many random restarts with local search
        for restart in range(50):
            if _elapsed() > _DEADLINE:
                break
            
            # Start from a random automaton or from current best with perturbation
            if rng.random() < 0.3:
                # Random start
                cur_a = [rng.randint(0, N-1) for _ in range(N)]
                cur_b = [rng.randint(0, N-1) for _ in range(N)]
            else:
                # Perturb best
                cur_a = best_a[:]
                cur_b = best_b[:]
                # Perturb 1-3 transitions
                n_pert = rng.randint(1, 3)
                for _ in range(n_pert):
                    if rng.random() < 0.5:
                        cur_a[rng.randrange(N)] = rng.randrange(N)
                    else:
                        cur_b[rng.randrange(N)] = rng.randrange(N)
            
            cur_thr = _threshold_fast(cur_a, cur_b)
            if cur_thr < 0:
                continue
            
            # Local search from this starting point
            for step in range(200):
                if _elapsed() > _DEADLINE:
                    break
                
                # Perturb one transition
                new_a = cur_a[:]
                new_b = cur_b[:]
                if rng.random() < 0.5:
                    new_a[rng.randrange(N)] = rng.randrange(N)
                else:
                    new_b[rng.randrange(N)] = rng.randrange(N)
                
                new_thr = _threshold_fast(new_a, new_b)
                
                if new_thr > cur_thr:
                    cur_a = new_a
                    cur_b = new_b
                    cur_thr = new_thr
                    if cur_thr > best_thresh:
                        best_thresh = cur_thr
                        best_a = cur_a[:]
                        best_b = cur_b[:]
                elif new_thr > 0 and rng.random() < 0.15:
                    # Simulated annealing: accept worse with some probability
                    cur_a = new_a
                    cur_b = new_b
                    cur_thr = new_thr

        # Phase 2: Focused local search from best solution
        if best_a is not None and best_b is not None:
            cur_a = best_a[:]
            cur_b = best_b[:]
            cur_thr = best_thresh
            for step in range(500):
                if _elapsed() > _DEADLINE:
                    break
                
                new_a = cur_a[:]
                new_b = cur_b[:]
                if rng.random() < 0.5:
                    new_a[rng.randrange(N)] = rng.randrange(N)
                else:
                    new_b[rng.randrange(N)] = rng.randrange(N)
                
                new_thr = _threshold_fast(new_a, new_b)
                
                if new_thr > cur_thr:
                    cur_a = new_a
                    cur_b = new_b
                    cur_thr = new_thr
                    if cur_thr > best_thresh:
                        best_thresh = cur_thr
                        best_a = cur_a[:]
                        best_b = cur_b[:]
                elif new_thr > 0 and rng.random() < 0.1:
                    cur_a = new_a
                    cur_b = new_b
                    cur_thr = new_thr

    # Fallback: ensure we have a valid synchronizing automaton
    if best_a is None or best_b is None or best_thresh <= 0:
        best_a = fallback_a
        best_b = fallback_b
        best_thresh = _threshold(best_a, best_b)

    return best_a, best_b
