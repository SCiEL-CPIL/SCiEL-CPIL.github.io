"""Bidirectional Warnsdorff greedy snake with randomized restarts in Q_9."""
import time
import random

N = 9
FULL = (1 << N) - 1
MASKS = [1 << b for b in range(N)]


def _popcount(x):
    return x.bit_count()


def _bidirectional(start, rng, deadline, noise):
    """Grow a snake bidirectionally from a single start vertex.

    At each step extend the end that currently has fewer legal moves
    (Warnsdorff), choosing among legal moves the one that leaves the new
    end most flexible.  Returns the vertex list (induced path).
    """
    path = [start]
    used = {start}
    best = path  # track best (here only one path is built per call)

    while time.time() < deadline:
        # candidate moves at the front and the back
        front = path[0]
        back = path[-1]

        front_moves = []
        for m in MASKS:
            v = front ^ m
            if v in used:
                continue
            # must not be adjacent to any path vertex except the new front (path[0])
            ok = True
            for p in path[1:]:
                if _popcount(v ^ p) == 1:
                    ok = False
                    break
            if ok:
                front_moves.append(v)

        back_moves = []
        for m in MASKS:
            v = back ^ m
            if v in used:
                continue
            ok = True
            for p in path[:-1]:
                if _popcount(v ^ p) == 1:
                    ok = False
                    break
            if ok:
                back_moves.append(v)

        if not front_moves and not back_moves:
            break

        # Warnsdorff: extend the end with fewer options
        if len(front_moves) <= len(back_moves):
            # choose a front move that keeps the new front flexible
            scored = []
            for v in front_moves:
                # flexibility of new front = number of legal moves it would have
                flex = 0
                for m in MASKS:
                    w = v ^ m
                    if w in used:
                        continue
                    ok = True
                    for p in path[1:]:
                        if _popcount(w ^ p) == 1:
                            ok = False
                            break
                    if ok:
                        flex += 1
                scored.append((flex, v))
            # prefer low flexibility (Warnsdorff on the move too), with noise
            scored.sort(key=lambda t: t[0])
            k = min(len(scored), max(1, int(noise * len(scored))))
            _, v = scored[rng.randrange(k)]
            path.insert(0, v)
            used.add(v)
        else:
            scored = []
            for v in back_moves:
                flex = 0
                for m in MASKS:
                    w = v ^ m
                    if w in used:
                        continue
                    ok = True
                    for p in path[:-1]:
                        if _popcount(w ^ p) == 1:
                            ok = False
                            break
                    if ok:
                        flex += 1
                scored.append((flex, v))
            scored.sort(key=lambda t: t[0])
            k = min(len(scored), max(1, int(noise * len(scored))))
            _, v = scored[rng.randrange(k)]
            path.append(v)
            used.add(v)

    return path


def _is_valid(path):
    """Verify the induced-path constraints; return True if valid."""
    n = len(path)
    s = set(path)
    if len(s) != n:
        return False
    for i in range(n):
        v = path[i]
        if v < 0 or v >= (1 << N):
            return False
    for i in range(n - 1):
        if _popcount(path[i] ^ path[i + 1]) != 1:
            return False
    # non-consecutive must differ in at least two bits
    for i in range(n):
        for j in range(i + 2, n):
            if _popcount(path[i] ^ path[j]) == 1:
                return False
    return True


def construct():
    t0 = time.time()
    deadline = t0 + 135.0  # safety margin before the 150 s kill
    best = None
    best_len = 0

    # deterministic seed-like start first (vertex 0) for reproducibility floor
    rng = random.Random(12345)

    while time.time() < deadline:
        # leave a little time for the final validation
        if time.time() > deadline - 2.0:
            break
        start = rng.randrange(1 << N)
        noise = rng.random() * 0.4
        p = _bidirectional(start, rng, deadline, noise)
        if len(p) > best_len:
            if _is_valid(p):
                best = p
                best_len = len(p)

    if best is None:
        # fallback: the simple greedy snake
        path, used = [0], {0}
        while True:
            cur = path[-1]
            nxt = None
            for b in range(N):
                v = cur ^ (1 << b)
                if v in used or any((v ^ q).bit_count() == 1 for q in path[:-1]):
                    continue
                nxt = v
                break
            if nxt is None:
                break
            path.append(nxt)
            used.add(nxt)
        best = path

    return best
