"""Fresh design: triangle-free diameter-2 graph (complement of a 2-(16,5,1) packing)."""
import time

_N = 16


def _edge_essential(sets, u, v, edge_set):
    # sets[u] is a frozenset of neighbors. Removing (u,v) disconnects iff no
    # length-2 path u-w-v remains, i.e. (sets[u]-{v}) & (sets[v]-{u}) is empty.
    a = sets[u] - frozenset((v,))
    b = sets[v] - frozenset((u,))
    return len(a & b) == 0


def _diameter_at_most_2(sets):
    for i in range(_N):
        s = sets[i]
        for j in range(_N):
            if j == i or j in s:
                continue
            # need a common neighbor
            if len(sets[i] & sets[j]) == 0:
                return False
    return True


def _build_base():
    # Singer difference set on Z_15
    D = {0, 1, 3, 5, 9}
    sets = [set() for _ in range(_N)]
    # cycle edges on vertices 0..14
    for i in range(15):
        j = (i + 1) % 15
        sets[i].add(j)
        sets[j].add(i)
    # hub = 15 connected to difference-set vertices
    for d in D:
        sets[15].add(d)
        sets[d].add(15)
    return sets


def _edge_list(sets):
    out = []
    for u in range(_N):
        for v in sets[u]:
            if u < v:
                out.append([u, v])
    return out


def construct():
    deadline = time.time() + 80.0  # safety margin under 90s kill

    sets = _build_base()
    edge_set = set()
    for u in range(_N):
        for v in sets[u]:
            if u < v:
                edge_set.add((u, v))

    # ensure base is valid
    if not _diameter_at_most_2(sets):
        return _edge_list(sets)
    for (u, v) in list(edge_set):
        if not _edge_essential(sets, u, v, edge_set):
            # remove a non-essential edge to restore the property
            sets[u].discard(v)
            sets[v].discard(u)
            edge_set.discard((u, v))

    best = len(edge_set)
    best_sets = [frozenset(s) for s in sets]

    # local search: try adding each non-edge, keep it if validity preserved
    all_pairs = [(u, v) for u in range(_N) for v in range(u + 1, _N)]
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for (u, v) in all_pairs:
            if time.time() > deadline:
                break
            if (u, v) in edge_set:
                continue
            # candidate: add (u,v); must keep triangle-free and diam<=2
            # triangle-free check: no common neighbor of u and v
            if len(sets[u] & sets[v]) > 0:
                continue
            # add tentatively
            sets[u].add(v)
            sets[v].add(u)
            edge_set.add((u, v))
            ok = _diameter_at_most_2(sets)
            if ok:
                # check every edge still essential (only edges incident to u or v
                # can change essentiality, but do full check for safety on small n)
                ok = all(_edge_essential(sets, a, b, edge_set) for (a, b) in edge_set)
            if ok:
                best = len(edge_set)
                best_sets = [frozenset(s) for s in sets]
                changed = True
            else:
                # revert
                sets[u].discard(v)
                sets[v].discard(u)
                edge_set.discard((u, v))

    # return from best snapshot
    out = []
    for u in range(_N):
        for v in best_sets[u]:
            if u < v:
                out.append([int(u), int(v)])
    return out
