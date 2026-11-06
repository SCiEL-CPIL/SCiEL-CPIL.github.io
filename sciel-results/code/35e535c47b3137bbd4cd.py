import time
import random
import sys
from collections import deque

N = 12
FULL_MASK = (1 << N) - 1
DEADLINE_SECONDS = 82.0  # safety margin before 90s kill


def _build_subset_trans(trans):
    """Precompute subset transitions for one letter using DP."""
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
    Length of the shortest reset word. 0 if not synchronizing.
    BFS on the subset automaton.
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


def _pinwheel(sink):
    """
    Pinwheel skeleton: a is the cycle i -> i+1 (mod N),
    b maps 0 -> sink, and i -> i-1 for i > 0.
    """
    a = [(i + 1) % N for i in range(N)]
    b = [sink] + [i - 1 for i in range(1, N)]
    return a, b


def _apply_perm(perm, trans):
    """
    Relabel states by perm: new_trans[perm[i]] = perm[trans[i]].
    This is the isomorphic relabeling of the automaton.
    """
    new = [0] * N
    for i in range(N):
        new[perm[i]] = perm[trans[i]]
    return new


def _pinwheel_relabel(perm, sink):
    """Pinwheel with a global state relabeling perm and a chosen sink."""
    a0, b0 = _pinwheel(sink)
    return _apply_perm(perm, a0), _apply_perm(perm, b0)


def _crossover(a1, b1, a2, b2):
    """
    Uniform crossover on the transition tables, with a bias to preserve
    the cycle structure of 'a' (a is more structurally important).
    """
    a = [0] * N
    b = [0] * N
    for i in range(N):
        if random.random() < 0.5:
            a[i] = a1[i]
            b[i] = b1[i]
        else:
            a[i] = a2[i]
            b[i] = b2[i]
    return a, b


def _mutate(a, b, rate):
    """
    Mutation: with probability rate per state, reassign that state's target
    for a or b. Bias toward keeping 'a' a permutation-like/cycle structure
    by occasionally swapping two of 'a's targets instead of a full reassign.
    """
    a = a[:]
    b = b[:]
    for i in range(N):
        if random.random() < rate:
            if random.random() < 0.5:
                a[i] = random.randint(0, N - 1)
            else:
                b[i] = random.randint(0, N - 1)
    # Occasionally swap two targets in 'a' to preserve bijectivity tendency
    if random.random() < 0.3:
        i, j = random.sample(range(N), 2)
        a[i], a[j] = a[j], a[i]
    if random.random() < 0.3:
        i, j = random.sample(range(N), 2)
        b[i], b[j] = b[j], b[i]
    return a, b


def _seed_population():
    """
    Build an initial population of structurally-valid pinwheel variants:
    varying sink and global relabeling. This seeds the GA in the
    high-threshold region of the space.
    """
    pop = []
    seen = set()
    # All sinks with a few relabelings each
    for sink in range(N):
        # identity relabeling
        perm = list(range(N))
        a, b = _pinwheel_relabel(perm, sink)
        key = (tuple(a), tuple(b))
        if key not in seen:
            seen.add(key)
            pop.append((a, b))
    # Add some random relabelings of the standard pinwheel
    for _ in range(20):
        perm = list(range(N))
        random.shuffle(perm)
        sink = random.randint(0, N - 1)
        a, b = _pinwheel_relabel(perm, sink)
        key = (tuple(a), tuple(b))
        if key not in seen:
            seen.add(key)
            pop.append((a, b))
    return pop


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    random.seed(12345)

    best_a = None
    best_b = None
    best_score = 0

    def _consider(a, b):
        nonlocal best_a, best_b, best_score
        s = compute_threshold(a, b)
        if s > best_score:
            best_score = s
            best_a = a[:]
            best_b = b[:]
        return s

    # 1) Evaluate the seed population (structural pinwheel variants)
    pop = _seed_population()
    scored = []
    for a, b in pop:
        if time.time() > deadline - 30.0:
            break
        s = _consider(a, b)
        scored.append((s, a[:], b[:]))
    scored.sort(key=lambda x: -x[0])

    # 2) Genetic algorithm: selection, crossover, mutation
    POP_SIZE = 12
    if time.time() < deadline - 12.0 and len(scored) >= 2:
        # Build initial GA population from the top scored + fresh variants
        ga_pop = []
        for s, a, b in scored[:POP_SIZE]:
            ga_pop.append((a[:], b[:]))
        while len(ga_pop) < POP_SIZE:
            # add a fresh pinwheel variant
            perm = list(range(N))
            random.shuffle(perm)
            sink = random.randint(0, N - 1)
            a, b = _pinwheel_relabel(perm, sink)
            ga_pop.append((a, b))

        gen = 0
        while time.time() < deadline - 10.0:
            gen += 1
            # Score current GA population
            ga_scored = []
            for a, b in ga_pop:
                s = _consider(a, b)
                ga_scored.append((s, a[:], b[:]))
            ga_scored.sort(key=lambda x: -x[0])

            # Build next generation
            next_pop = []
            # Elitism: keep the best
            next_pop.append((ga_scored[0][1][:], ga_scored[0][2][:]))
            while len(next_pop) < POP_SIZE:
                # Tournament selection of two parents
                p1 = random.choice(ga_scored[:max(2, len(ga_scored) // 2)])
                p2 = random.choice(ga_scored[:max(2, len(ga_scored) // 2)])
                child_a, child_b = _crossover(p1[1], p1[2], p2[1], p2[2])
                # Mutation rate anneals down slightly over generations
                rate = max(0.05, 0.25 - 0.01 * gen)
                child_a, child_b = _mutate(child_a, child_b, rate)
                next_pop.append((child_a, child_b))
            ga_pop = next_pop

    # 3) Local hill-climb from the best found (short, targeted)
    if best_a is not None and time.time() < deadline - 5.0:
        cur_a = best_a[:]
        cur_b = best_b[:]
        cur_score = best_score
        no_improve = 0
        while time.time() < deadline - 3.0 and no_improve < 80:
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
