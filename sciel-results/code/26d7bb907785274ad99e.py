"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random as _random
from collections import defaultdict

def _bron_kerbosch(adj, R, P, X, cliques):
    if not P and not X:
        if len(R) >= 2:
            cliques.append(frozenset(R))
        return
    PX = P | X
    if PX:
        pivot = max(PX, key=lambda v: len(adj[v] & P))
        candidates = P - adj[pivot]
    else:
        candidates = P
    for v in list(candidates):
        _bron_kerbosch(adj, R | {v}, P & adj[v], X & adj[v], cliques)
        P.discard(v)
        X.add(v)

def _enumerate_maximal_cliques(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    cliques = []
    _bron_kerbosch(adj, frozenset(), frozenset(range(n)), frozenset(), cliques)
    return [c for c in cliques if len(c) >= 2]

def _greedy_set_cover(cliques, n):
    vertex_cliques = defaultdict(list)
    for i, c in enumerate(cliques):
        for v in c:
            vertex_cliques[v].append(i)
    
    uncovered = set(range(len(cliques)))
    selected = set()
    
    while uncovered:
        best_v = None
        best_count = -1
        for v in range(n):
            if v in selected:
                continue
            cnt = sum(1 for ci in vertex_cliques[v] if ci in uncovered)
            if cnt > best_count:
                best_count = cnt
                best_v = v
        if best_v is None or best_count == 0:
            break
        selected.add(best_v)
        for ci in vertex_cliques[best_v]:
            uncovered.discard(ci)
    
    return selected

def _is_valid(selected, cliques):
    for c in cliques:
        if not selected & c:
            return False
    return True

def _local_search(selected, cliques, n, max_iters=200):
    selected = set(selected)
    
    # Phase 1: Remove redundant vertices
    changed = True
    while changed:
        changed = False
        for v in list(selected):
            trial = selected - {v}
            if _is_valid(trial, cliques):
                selected = trial
                changed = True
    
    # Phase 2: Try swaps to reduce size
    for _ in range(max_iters):
        improved = False
        for v in list(selected):
            trial = selected - {v}
            uncovered = [c for c in cliques if not trial & c]
            if not uncovered:
                selected = trial
                improved = True
                break
            # Find a vertex that covers all uncovered cliques
            for w in range(n):
                if w in trial:
                    continue
                if all(w in c for c in uncovered):
                    selected = trial | {w}
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return selected

def _solve_instance(n, p, seed, deadline):
    rng = _random.Random(seed)
    edges = [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]
    
    cliques = _enumerate_maximal_cliques(n, edges)
    
    if not cliques:
        return []
    
    selected = _greedy_set_cover(cliques, n)
    
    if time.time() < deadline - 3:
        selected = _local_search(selected, cliques, n)
    
    return sorted(selected)

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            results.append([])
            continue
        sol = _solve_instance(n, p, seed, deadline)
        results.append(sol)
    
    return results
# EVOLVE-BLOCK-END
