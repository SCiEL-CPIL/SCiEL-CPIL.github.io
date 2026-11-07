import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

N = P['n']
ALL_EDGES = [(min(a, b), max(a, b)) for a, b in P['edges']]


def _greedy():
    """Greedy star partition: repeatedly take the vertex with most unassigned edges."""
    n = N
    unassigned = set(ALL_EDGES)
    incident = defaultdict(set)
    for e in ALL_EDGES:
        incident[e[0]].add(e)
        incident[e[1]].add(e)
    bicliques = []
    while unassigned:
        best_v = -1
        best_c = -1
        for v in range(n):
            c = 0
            for e in incident[v]:
                if e in unassigned:
                    c += 1
            if c > best_c:
                best_c = c
                best_v = v
        if best_v < 0 or best_c == 0:
            break
        others = []
        for e in incident[best_v]:
            if e in unassigned:
                other = e[1] if e[0] == best_v else e[0]
                others.append(other)
                unassigned.remove(e)
        if others:
            bicliques.append(([best_v], others))
    return bicliques


def _rebuild(bicliques):
    n = N
    L = [set(b[0]) for b in bicliques]
    R = [set(b[1]) for b in bicliques]
    edges = [set() for _ in bicliques]
    adj = [set() for _ in range(n)]
    for i, b in enumerate(bicliques):
        for u in b[0]:
            for v in b[1]:
                e = (u, v) if u < v else (v, u)
                edges[i].add(e)
                adj[u].add(v)
                adj[v].add(u)
    return L, R, edges, adj


def _can_merge(i, j, L, R, edges):
    if edges[i] & edges[j]:
        return False
    if (L[i] & L[j]) or (L[i] & R[j]) or (R[i] & L[j]) or (R[i] & R[j]):
        return False
    return True


def _reassign(i, j, e, L, R, edges, adj):
    if i == j:
        return False
    u, v = e
    if (u in L[i] and v in L[i]) or (u in R[i] and v in R[i]):
        return False
    if (u in L[j] and v in L[j]) or (u in R[j] and v in R[j]):
        return False
    edges[i].remove(e)
    if not edges[i]:
        return False
    edges[j].add(e)
    return True


def _move(i, v, side, L, R, edges, adj):
    # side: 0=left, 1=right ; move v from side to the other side of biclique i
    if side == 0:
        L[i].discard(v)
        R[i].add(v)
    else:
        R[i].discard(v)
        L[i].add(v)
    for w in adj[v]:
        if w == v:
            continue
        if (w in L[i]) != (w in R[i]):
            continue
        if (w in L[i] and v in L[i]) or (w in R[i] and v in R[i]):
            return False
    return True


def _local(bicliques, rng, deadline):
    L, R, edges, adj = _rebuild(bicliques)
    n = N
    while time.time() < deadline:
        improved = False
        # 1) merges
        m = len(bicliques)
        for i in range(m):
            if i >= len(bicliques):
                break
            for j in range(i + 1, len(bicliques)):
                if _can_merge(i, j, L, R, edges):
                    L[i] |= L[j]
                    R[i] |= R[j]
                    edges[i] |= edges[j]
                    del L[j], R[j], edges[j]
                    del bicliques[j]
                    improved = True
                    break
            if improved:
                break
        if improved:
            continue
        # 2) reassignments
        for i in range(len(bicliques)):
            for j in range(len(bicliques)):
                if i == j:
                    continue
                if len(edges[i]) <= len(edges[j]):
                    continue
                for e in list(edges[i]):
                    if _reassign(i, j, e, L, R, edges, adj):
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # 3) vertex moves (only accept if a merge then becomes possible)
        m = len(bicliques)
        moved = False
        for i in range(m):
            if not L[i] and not R[i]:
                continue
            for side in (0, 1):
                side_set = L[i] if side == 0 else R[i]
                for v in list(side_set):
                    ok = _move(i, v, side, L, R, edges, adj)
                    if not ok:
                        continue
                    merged = False
                    for a in range(m):
                        if a == i:
                            continue
                        for b in range(a + 1, m):
                            if b == i:
                                continue
                            if _can_merge(a, b, L, R, edges):
                                L[a] |= L[b]
                                R[a] |= R[b]
                                edges[a] |= edges[b]
                                del L[b], R[b], edges[b]
                                del bicliques[b]
                                merged = True
                                break
                        if merged:
                            break
                    if merged:
                        moved = True
                        improved = True
                        break
                    else:
                        _move(i, v, 1 - side, L, R, edges, adj)
                if moved:
                    break
            if moved:
                break
        if improved:
            continue
        # 4) split (escape local minima): split largest biclique into two stars
        if len(bicliques) > 1:
            idx = max(range(len(bicliques)), key=lambda k: len(L[k]) + len(R[k]))
            verts = list(L[idx]) + list(R[idx])
            if len(verts) >= 2:
                rng.shuffle(verts)
                mid = len(verts) // 2
                newA = verts[:mid]
                newB = verts[mid:]
                eA = set()
                eB = set()
                for u in newA:
                    for v in newB:
                        e = (u, v) if u < v else (v, u)
                        if e in edges[idx]:
                            eA.add(e)
                for e in edges[idx]:
                    if e not in eA:
                        eB.add(e)
                if eA and eB:
                    L[idx] = set(newA)
                    R[idx] = set()
                    edges[idx] = eA
                    L.append(set(newB))
                    R.append(set())
                    edges.append(eB)
                    bicliques.append((newB, []))
                    improved = True
        if not improved:
            break
    # final merge pass
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(bicliques)):
            for j in range(i + 1, len(bicliques)):
                if _can_merge(i, j, L, R, edges):
                    L[i] |= L[j]
                    R[i] |= R[j]
                    edges[i] |= edges[j]
                    del L[j], R[j], edges[j]
                    del bicliques[j]
                    changed = True
                    break
            if changed:
                break
    out = []
    for i in range(len(bicliques)):
        out.append(([int(x) for x in L[i]], [int(x) for x in R[i]]))
    return out


def construct():
    t0 = time.time()
    deadline = t0 + 55.0
    rng = random.Random(12345)
    bicliques = _greedy()
    bicliques = _local(bicliques, rng, deadline)
    # quick validity repair: drop empty sides
    cleaned = []
    for b in bicliques:
        l = [int(x) for x in b[0]]
        r = [int(x) for x in b[1]]
        if l and r:
            cleaned.append([l, r])
    return cleaned
# EVOLVE-BLOCK-END
