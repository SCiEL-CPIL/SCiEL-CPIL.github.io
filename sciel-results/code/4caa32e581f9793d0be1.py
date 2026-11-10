import random
import time

_START = time.time()
_DEADLINE = 150.0 - 3.0   # safety margin


def _edge_index(i, j):
    # 0-based index for pair (i,j), 0<=i<j<16, in lexicographic order
    return i * 15 - (i * (i + 1)) // 2 + (j - i - 1)


def _build_k4s():
    # For each K4, store its 6 edge indices
    k4s = []
    for a in range(16):
        for b in range(a + 1, 16):
            for c in range(b + 1, 16):
                for d in range(c + 1, 16):
                    k4s.append((
                        _edge_index(a, b),
                        _edge_index(a, c),
                        _edge_index(a, d),
                        _edge_index(b, c),
                        _edge_index(b, d),
                        _edge_index(c, d),
                    ))
    return k4s


_K4S = _build_k4s()


def _colour_counts(colours):
    cnt = [0] * 120
    for c in colours:
        cnt[c] += 1
    return cnt


def _try_merge(colours, cnt, a, b):
    # Return new colours list if merging colour b into a keeps validity, else None.
    # Merge only if b is "cheaper" or equal count to keep palette shrinking.
    if cnt[b] > cnt[a]:
        return None
    new = list(colours)
    for idx, c in enumerate(new):
        if c == b:
            new[idx] = a
    # Validate all K4s
    for e in _K4S:
        s = {new[e[0]], new[e[1]], new[e[2]], new[e[3]], new[e[4]], new[e[5]]}
        if len(s) < 5:
            return None
    newcnt = [0] * 120
    for c in new:
        newcnt[c] += 1
    return new, newcnt


def _num_colours(cnt):
    return sum(1 for x in cnt if x > 0)


def construct():
    # Start from the known valid 17-colour solution
    colours = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    cnt = _colour_counts(colours)
    best = list(colours)
    best_cnt = _num_colours(cnt)

    rng = random.Random(12345)
    used = [c for c in range(120) if cnt[c] > 0]

    # Deterministic greedy merge: repeatedly try to merge the least-used colour
    # into another colour, preferring to keep the palette small.
    def greedy(colours, cnt):
        colours = list(colours)
        cnt = list(cnt)
        nc = _num_colours(cnt)
        progress = True
        while progress:
            progress = False
            used = [c for c in range(120) if cnt[c] > 0]
            used.sort(key=lambda c: (cnt[c], c))
            for b in used:
                if cnt[b] == 0:
                    continue
                # try merging b into each other used colour
                for a in used:
                    if a == b:
                        continue
                    res = _try_merge(colours, cnt, a, b)
                    if res is not None:
                        colours, cnt = res
                        nc -= 1
                        progress = True
                        break
                if progress:
                    break
        return colours, cnt, nc

    colours, cnt, nc = greedy(colours, cnt)
    if nc < best_cnt:
        best = list(colours)
        best_cnt = nc

    # Local search / hill climbing with random restarts to escape local minima
    def local_search(colours, cnt, iters=20000):
        colours = list(colours)
        cnt = list(cnt)
        nc = _num_colours(cnt)
        for _ in range(iters):
            if time.time() > _DEADLINE:
                break
            used = [c for c in range(120) if cnt[c] > 0]
            if len(used) <= 1:
                break
            # pick a random colour to try to eliminate (prefer least used)
            b = used[rng.randrange(len(used))]
            if cnt[b] == 0:
                continue
            # pick a random target
            targets = [a for a in used if a != b]
            a = targets[rng.randrange(len(targets))]
            res = _try_merge(colours, cnt, a, b)
            if res is not None:
                colours, cnt = res
                nc -= 1
        return colours, cnt, nc

    # Try several random perturbations + local search
    for restart in range(200):
        if time.time() > _DEADLINE:
            break
        # Start from best, apply a few random single-edge colour changes to perturb
        cur = list(best)
        curcnt = _colour_counts(cur)
        # small perturbation: change a few edges to random colours
        for _ in range(rng.randint(1, 4)):
            idx = rng.randrange(120)
            cur[idx] = rng.randrange(120)
        # validate; if invalid, skip
        ok = True
        for e in _K4S:
            s = {cur[e[0]], cur[e[1]], cur[e[2]], cur[e[3]], cur[e[4]], cur[e[5]]}
            if len(s) < 5:
                ok = False
                break
        if not ok:
            continue
        curcnt = _colour_counts(cur)
        # local search
        cur, curcnt, nc = local_search(cur, curcnt, iters=5000)
        if nc < best_cnt:
            best = list(cur)
            best_cnt = nc
            # try greedy from here too
            gcol, gcnt, gnc = greedy(best, _colour_counts(best))
            if gnc < best_cnt:
                best = list(gcol)
                best_cnt = gnc

    return best
