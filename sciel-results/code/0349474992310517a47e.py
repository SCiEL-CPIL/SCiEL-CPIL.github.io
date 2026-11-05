import random, time, itertools

N = 16
M = N * (N - 1) // 2
MAXC = 119

edges = []
edge_idx = {}
for i in range(N):
    for j in range(i + 1, N):
        edge_idx[(i, j)] = len(edges)
        edges.append((i, j))

cycles = []
for quad in itertools.combinations(range(N), 4):
    a, b, c, d = quad
    cycles.append((edge_idx[(a, b)], edge_idx[(b, c)], edge_idx[(c, d)], edge_idx[(d, a)]))
    cycles.append((edge_idx[(a, b)], edge_idx[(b, d)], edge_idx[(d, c)], edge_idx[(c, a)]))
    cycles.append((edge_idx[(a, c)], edge_idx[(c, b)], edge_idx[(b, d)], edge_idx[(d, a)]))
NC = len(cycles)
edge_to_cycles = [[] for _ in range(M)]
for ci, cyc in enumerate(cycles):
    for e in cyc:
        edge_to_cycles[e].append(ci)


def init_state(colors):
    cycle_counts = [{} for _ in range(NC)]
    cycle_distinct = [0] * NC
    for ci, cyc in enumerate(cycles):
        d = cycle_counts[ci]
        for e in cyc:
            col = colors[e]
            d[col] = d.get(col, 0) + 1
        cycle_distinct[ci] = len(d)
    usage = {}
    for col in colors:
        usage[col] = usage.get(col, 0) + 1
    return cycle_counts, cycle_distinct, usage


def try_change(e, newcolor, colors, cycle_counts, cycle_distinct, usage):
    old = colors[e]
    if old == newcolor:
        return True
    ok = True
    for ci in edge_to_cycles[e]:
        d = cycle_counts[ci]
        d[old] -= 1
        if d[old] == 0:
            del d[old]
        d[newcolor] = d.get(newcolor, 0) + 1
        nd = len(d)
        cycle_distinct[ci] = nd
        if nd > 3:
            ok = False
    if ok:
        colors[e] = newcolor
        usage[old] -= 1
        if usage[old] == 0:
            del usage[old]
        usage[newcolor] = usage.get(newcolor, 0) + 1
        return True
    for ci in edge_to_cycles[e]:
        d = cycle_counts[ci]
        d[newcolor] -= 1
        if d[newcolor] == 0:
            del d[newcolor]
        d[old] = d.get(old, 0) + 1
        cycle_distinct[ci] = len(d)
    return False


def undo_change(e, cur, prev, colors, cycle_counts, cycle_distinct, usage):
    for ci in edge_to_cycles[e]:
        d = cycle_counts[ci]
        d[cur] -= 1
        if d[cur] == 0:
            del d[cur]
        d[prev] = d.get(prev, 0) + 1
        cycle_distinct[ci] = len(d)
    colors[e] = prev
    usage[cur] -= 1
    if usage[cur] == 0:
        del usage[cur]
    usage[prev] = usage.get(prev, 0) + 1


def full_valid(colors):
    for cyc in cycles:
        if len(set(colors[e] for e in cyc)) > 3:
            return False
    return True


def search_one(rng, deadline, init_colors):
    colors = init_colors[:]
    cycle_counts, cycle_distinct, usage = init_state(colors)
    max_color = max(colors) if colors else 0
    stuck = 0
    while time.time() < deadline:
        if max_color >= MAXC:
            break
        e = rng.randrange(M)
        col = colors[e]
        if usage.get(col, 0) < 2:
            stuck += 1
            if stuck > 300:
                break
            continue
        f = max_color + 1
        if try_change(e, f, colors, cycle_counts, cycle_distinct, usage):
            max_color = f
            stuck = 0
            continue
        # 2-edge repair: fresh color on e plus recolor a conflicting cycle-mate
        repaired = False
        for ci in edge_to_cycles[e]:
            cyc = cycles[ci]
            others = [x for x in cyc if x != e]
            ocols = [colors[x] for x in others]
            targets = set([f]) | set(ocols)
            for ox in others:
                cur = colors[ox]
                for t in targets:
                    if t == cur:
                        continue
                    if try_change(e, f, colors, cycle_counts, cycle_distinct, usage):
                        if try_change(ox, t, colors, cycle_counts, cycle_distinct, usage):
                            max_color = max(max_color, f)
                            repaired = True
                            break
                        else:
                            undo_change(e, f, col, colors, cycle_counts, cycle_distinct, usage)
                            break
                    else:
                        break
                if repaired:
                    break
            if repaired:
                break
        if repaired:
            stuck = 0
        else:
            stuck += 1
            if stuck > 300:
                break
    return colors, len(usage)


def construct():
    start = time.time()
    deadline = start + 75.0
    rng = random.Random(12345)
    best = None
    best_score = -1

    ci = [0] * M
    for (i, j), e in edge_idx.items():
        ci[e] = i
    cj = [0] * M
    for (i, j), e in edge_idx.items():
        cj[e] = j
    seeds = [[0] * M, ci, cj, [rng.randrange(4) for _ in range(M)]]

    attempt = 0
    while time.time() < deadline - 1.0:
        seed = seeds[attempt % len(seeds)]
        if not full_valid(seed):
            seed = ci
        local_deadline = min(deadline - 0.5, time.time() + 2.0)
        colors, score = search_one(rng, local_deadline, seed)
        if full_valid(colors) and score > best_score:
            best_score = score
            best = colors
        attempt += 1

    if best is None or not full_valid(best):
        best = ci
    return list(best)
