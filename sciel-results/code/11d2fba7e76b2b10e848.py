import time
import random
import sys
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


def _hardcoded_best():
    """Return the standard Ping-Pong automaton for N=12."""
    # a: cycle i -> i+1
    a = [(i + 1) % N for i in range(N)]
    # b: 0 -> 0, i -> i-1 for i > 0
    b = [0] + [i - 1 for i in range(1, N)]
    return a, b


def _try_ping_pong_variants(deadline):
    """Try variants of the Ping-Pong automaton."""
    candidates = []
    a_cycle = [(i + 1) % N for i in range(N)]
    # Standard Ping-Pong
    b_std = [0] + [i - 1 for i in range(1, N)]
    candidates.append((a_cycle[:], b_std[:]))
    
    # Variants: b(0) = k, b(i) = i-1 for i>0
    for k in range(N):
        b_var = [k] + [i - 1 for i in range(1, N)]
        candidates.append((a_cycle[:], b_var[:]))
        
    # Variants: a is cycle, b is "shift down" with different sink
    # e.g. b(0)=0, b(i)=i-1 is standard.
    # What if a is reverse cycle?
    a_rev = [(i - 1) % N for i in range(N)]
    candidates.append((a_rev[:], b_std[:]))
    
    return candidates


def _random_start():
    """Generate a random automaton biased toward synchronizability."""
    # Bias: make one letter have a "sink-ish" behaviour
    a = [random.randint(0, N - 1) for _ in range(N)]
    b = [random.randint(0, N - 1) for _ in range(N)]
    # Ensure at least one letter is a permutation to help synchronisation
    if random.random() < 0.7:
        perm = list(range(N))
        random.shuffle(perm)
        a = perm[:]
    return a, b


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    random.seed(42)

    best_a = None
    best_b = None
    best_score = 0

    # 1) Hardcoded strong start (Ping-Pong)
    ha, hb = _hardcoded_best()
    hs = compute_threshold(ha, hb)
    if hs > best_score:
        best_score = hs
        best_a = ha[:]
        best_b = hb[:]

    # 2) Ping-Pong variants
    for pa, pb in _try_ping_pong_variants(deadline):
        if time.time() > deadline - 10.0:
            break
        s = compute_threshold(pa, pb)
        if s > best_score:
            best_score = s
            best_a = pa[:]
            best_b = pb[:]

    # 3) Random search with hill climbing
    remaining = deadline - time.time()
    n_starts = max(1, int(remaining // 6))
    for _ in range(n_starts):
        if time.time() > deadline - 12.0:
            break
        a, b = _random_start()
        s = compute_threshold(a, b)
        if s == 0:
            continue
        if s > best_score:
            best_score = s
            best_a = a[:]
            best_b = b[:]
            # Continue with random starts; hill-climb later if time

    # 4) Hill-climb from the best found
    if best_a is not None and time.time() < deadline - 10.0:
        cur_a = best_a[:]
        cur_b = best_b[:]
        cur_score = best_score
        no_improve = 0
        while time.time() < deadline - 5.0 and no_improve < 200:
            # Pick a random transition in a or b
            if random.random() < 0.5:
                state = random.randint(0, N - 1)
                old = cur_a[state]
                new = random.randint(0, N - 1)
                if new == old:
                    continue
                cur_a[state] = new
                s = compute_threshold(cur_a, cur_b)
                if s > cur_score:
                    cur_score = s
                    no_improve = 0
                    if cur_score > best_score:
                        best_score = cur_score
                        best_a = cur_a[:]
                        best_b = cur_b[:]
                else:
                    cur_a[state] = old
                    no_improve += 1
            else:
                state = random.randint(0, N - 1)
                old = cur_b[state]
                new = random.randint(0, N - 1)
                if new == old:
                    continue
                cur_b[state] = new
                s = compute_threshold(cur_a, cur_b)
                if s > cur_score:
                    cur_score = s
                    no_improve = 0
                    if cur_score > best_score:
                        best_score = cur_score
                        best_a = cur_a[:]
                        best_b = cur_b[:]
                else:
                    cur_b[state] = old
                    no_improve += 1

    # Ensure valid return
    if best_a is None or best_score == 0:
        a = [(i + 1) % N for i in range(N)]
        b = [0] * N
        return [a, b]

    return [best_a, best_b]
