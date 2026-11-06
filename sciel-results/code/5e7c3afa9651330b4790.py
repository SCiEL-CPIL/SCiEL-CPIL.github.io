"""C4-free edge set on Q7 via randomized greedy with precomputed witnesses."""
import time
import random

D = 7
N = 1 << D

def _build():
    edges = []
    edge_idx = {}
    for u in range(N):
        for i in range(D):
            v = u ^ (1 << i)
            if u < v:
                idx = len(edges)
                edges.append((u, v, i))
                edge_idx[(u, v)] = idx
    n = len(edges)
    # For each edge, the set of witness indices (the other 3 edges of each face it belongs to)
    # A C4 is completed when this edge AND all 3 witnesses of some face are all chosen.
    # So: adding edge idx is forbidden if for some face, all 3 other edges are already chosen.
    # We store for each edge, the list of "witness triples" - each triple is a set of 3 edge indices.
    witness_triples = [[] for _ in range(n)]
    for idx, (u, v, i) in enumerate(edges):
        for j in range(D):
            if j == i:
                continue
            c = u ^ (1 << j)
            e = v ^ (1 << j)
            triple = []
            for a, b in ((u, c), (v, e), (c, e)):
                if a > b:
                    a, b = b, a
                widx = edge_idx[(a, b)]
                triple.append(widx)
            witness_triples[idx].append(triple)
    return edges, witness_triples

_edges, _witness_triples = _build()
_ne = len(_edges)

def _greedy(order):
    chosen = [False] * _ne
    count = 0
    for idx in order:
        # Check each face: are all 3 other edges already chosen?
        blocked = False
        for triple in _witness_triples[idx]:
            if chosen[triple[0]] and chosen[triple[1]] and chosen[triple[2]]:
                blocked = True
                break
        if not blocked:
            chosen[idx] = True
            count += 1
    return chosen, count

def construct():
    start = time.time()
    deadline = start + 110.0

    best_chosen = None
    best_count = 0

    # Deterministic baseline
    order = list(range(_ne))
    chosen, cnt = _greedy(order)
    best_count = cnt
    best_chosen = chosen[:]

    # Randomized restarts
    rng = random.Random(12345)
    while time.time() < deadline:
        order = list(range(_ne))
        rng.shuffle(order)
        chosen, cnt = _greedy(order)
        if cnt > best_count:
            best_count = cnt
            best_chosen = chosen[:]

    return [[_edges[i][0], _edges[i][1]] for i in range(_ne) if best_chosen[i]]
