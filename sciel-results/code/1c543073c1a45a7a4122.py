import time
import random

N = 40
LENGTHS = list(range(3, N + 1))

def _cycle_edges(c):
    m = len(c)
    es = set()
    for i in range(m):
        u = c[i]; v = c[(i + 1) % m]
        a, b = (u, v) if u < v else (v, u)
        es.add((a, b))
    return es

def _candidate_pool(L, rng):
    """Diverse pool of simple cycles of length L on vertices 0..39."""
    cands = []
    # Base: [0,1,...,L-1]
    cands.append(list(range(L)))
    # [0, L-1, ..., 1]
    cands.append([0] + list(range(L - 1, 0, -1)))
    # [0, 39, 38, ..., k] for L >= 22
    if L >= 22:
        k = 41 - L
        cands.append([0] + list(range(N - 1, k - 1, -1)))
    # [0, 1, ..., L-2, L-1] with vertex 0 fixed, rest shuffled
    for _ in range(12):
        rest = list(range(1, L))
        rng.shuffle(rest)
        cands.append([0] + rest)
    # Random full permutations
    for _ in range(6):
        perm = list(range(L))
        rng.shuffle(perm)
        cands.append(perm)
    # Deduplicate (as cycles, consider rotation but keep as-is for simplicity)
    seen = set()
    out = []
    for c in cands:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out

def construct():
    start = time.time()
    DEADLINE = 72.0
    rng = random.Random(7)

    # Precompute candidate pools with edge sets
    pools = {}
    for L in LENGTHS:
        cands = _candidate_pool(L, rng)
        pools[L] = [(c, _cycle_edges(c)) for c in cands]

    # Greedy assignment: pick cycle adding fewest new edges
    assigned = {}
    edge_union = set()
    for L in LENGTHS:
        if (time.time() - start) > DEADLINE:
            break
        best_fe = None
        best_c = None
        best_add = None
        for c, fe in pools[L]:
            add = len(fe - edge_union)
            if best_add is None or add < best_add:
                best_add = add
                best_fe = fe
                best_c = c
            if add == 0:
                break
        assigned[L] = (best_c, best_fe)
        edge_union |= best_fe

    # Local search: iteratively reassign lengths to reduce union
    improved = True
    while improved and (time.time() - start) < DEADLINE - 3.0:
        improved = False
        for L in LENGTHS:
            if (time.time() - start) > DEADLINE - 3.0:
                break
            cur_c, cur_fe = assigned[L]
            best_fe = cur_fe
            best_c = cur_c
            for c, fe in pools[L]:
                if c == cur_c:
                    continue
                new_union = (edge_union - cur_fe) | fe
                if len(new_union) < len(edge_union):
                    best_fe = fe
                    best_c = c
            if best_fe is not cur_fe:
                edge_union = (edge_union - cur_fe) | best_fe
                assigned[L] = (best_c, best_fe)
                improved = True

    # Collect needed edges
    needed = set()
    for L in LENGTHS:
        needed |= assigned[L][1]
    edges = list(needed)
    cycles = [assigned[L][0] for L in LENGTHS]

    # Validation with fallback
    eset = set(tuple(sorted(e)) for e in edges)
    valid = True
    for L, c in zip(LENGTHS, cycles):
        if len(set(c)) != len(c):
            valid = False
            break
        m = len(c)
        for i in range(m):
            u = c[i]; v = c[(i + 1) % m]
            a, b = (u, v) if u < v else (v, u)
            if (a, b) not in eset:
                valid = False
                break
        if not valid:
            break

    if not valid:
        # Fallback: known-good hub-and-spoke
        edges = [(i, (i + 1) % N) for i in range(N)] + [(0, k) for k in range(2, 21)]
        cycles = []
        for L2 in LENGTHS:
            if L2 == N:
                cycles.append(list(range(N)))
            elif L2 <= 21:
                cycles.append(list(range(L2)))
            else:
                k2 = 41 - L2
                cycles.append([0] + list(range(N - 1, k2 - 1, -1)))

    return {'edges': edges, 'cycles': cycles}
