import time
import random

_START = None
_DEADLINE = 85.0  # safety margin under the 90s kill

def _elapsed():
    return time.time() - _START

def construct():
    global _START
    _START = time.time()
    random.seed(98765)
    N = 180

    # Forbidden triples: a < b < c with (b-a)(c-a) = a^2  <=>  1/a = 1/b + 1/c
    triples = []
    vertex_to_triples = [[] for _ in range(N + 1)]

    for a in range(1, N + 1):
        a2 = a * a
        for x in range(1, int(a2 ** 0.5) + 1):
            if a2 % x == 0:
                y = a2 // x
                if x < y:
                    b = a + x
                    c = a + y
                    if c <= N:
                        idx = len(triples)
                        triples.append((a, b, c))
                        for v in (a, b, c):
                            vertex_to_triples[v].append(idx)

    T = len(triples)
    triple_count = [0] * T
    in_set = [False] * (N + 1)

    def add(v):
        in_set[v] = True
        for ti in vertex_to_triples[v]:
            triple_count[ti] += 1
            if triple_count[ti] == 3:
                return False
        return True

    def remove(v):
        in_set[v] = False
        for ti in vertex_to_triples[v]:
            triple_count[ti] -= 1

    def can_add(v):
        for ti in vertex_to_triples[v]:
            if triple_count[ti] == 2:
                return False
        return True

    def size():
        return sum(in_set)

    def reset():
        for ti in range(T):
            triple_count[ti] = 0
        for v in range(1, N + 1):
            in_set[v] = False

    best = []
    best_size = 0

    def record():
        nonlocal best, best_size
        s = size()
        if s > best_size:
            best_size = s
            best = [v for v in range(1, N + 1) if in_set[v]]

    # Strong initial solution: the tail [90,180]
    reset()
    for v in range(90, N + 1):
        add(v)
    record()

    deg = [len(vertex_to_triples[v]) for v in range(N + 1)]

    def greedy_from(order):
        reset()
        for v in order:
            if can_add(v):
                add(v)
        record()

    orders = [
        list(range(1, N + 1)),
        sorted(range(1, N + 1), key=lambda v: deg[v]),
        sorted(range(1, N + 1), key=lambda v: -deg[v]),
    ]
    for _ in range(30):
        o = list(range(1, N + 1))
        random.shuffle(o)
        orders.append(o)

    for o in orders:
        if _elapsed() > _DEADLINE - 8:
            break
        greedy_from(o)

    # Local search from the best solution found so far
    reset()
    for v in best:
        add(v)

    while _elapsed() < _DEADLINE - 8:
        progress = False
        # Completion pass: add every vertex that can still be added
        for v in range(1, N + 1):
            if _elapsed() > _DEADLINE - 8:
                break
            if not in_set[v] and can_add(v):
                add(v)
                progress = True
        if progress:
            record()
        # Swap moves: remove one in-set vertex, try to add one outside
        for _ in range(3000):
            if _elapsed() > _DEADLINE - 8:
                break
            cur = [v for v in range(1, N + 1) if in_set[v]]
            if not cur:
                break
            rem = random.choice(cur)
            addv = random.randint(1, N)
            if addv == rem or in_set[addv]:
                continue
            remove(rem)
            if can_add(addv):
                add(addv)
                record()
                progress = True
            else:
                add(rem)  # revert (safe: state was valid before removal)

    # Final completion
    for v in range(1, N + 1):
        if not in_set[v] and can_add(v):
            add(v)
    record()

    if not best:
        best = list(range(90, N + 1))
    return best
