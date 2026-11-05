"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_DEADLINE = 140.0  # safety margin before 150s kill
_START = None

def _elapsed():
    global _START
    if _START is None:
        _START = time.time()
    return time.time() - _START

def _instance(n, p, seed):
    rng = random.Random(seed)
    edges = [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]
    return n, edges

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _maximal_cliques(n, adj):
    # Bron-Kerbosch with pivot, iterative to avoid recursion limits
    cliques = []
    def expand(R, P, X):
        if not P and not X:
            cliques.append(frozenset(R))
            return
        # choose pivot u in P ∪ X maximizing |P ∩ N(u)|
        PX = P | X
        if not PX:
            return
        best_u = max(PX, key=lambda u: len(P & adj[u]))
        candidates = P - adj[best_u]
        for v in list(candidates):
            expand(R | {v}, P & adj[v], X & adj[v])
            P = P - {v}
            X = X | {v}
    expand(set(), set(range(n)), set())
    return cliques

def _greedy_set_cover(cliques, n):
    # returns set of vertices hitting all cliques
    if not cliques:
        return set()
    # map vertex -> set of clique indices it covers
    vcliques = [set() for _ in range(n)]
    for ci, c in enumerate(cliques):
        for v in c:
            vcliques[v].add(ci)
    uncovered = set(range(len(cliques)))
    chosen = set()
    while uncovered:
        best_v = -1
        best_gain = 0
        for v in range(n):
            if v in chosen:
                continue
            gain = len(vcliques[v] & uncovered)
            if gain > best_gain:
                best_gain = gain
                best_v = v
        if best_v < 0 or best_gain == 0:
            # should not happen since each clique has >=1 vertex
            break
        chosen.add(best_v)
        uncovered -= vcliques[best_v]
    return chosen

def _covers_all(chosen, cliques):
    for c in cliques:
        if not (chosen & c):
            return False
    return True

def _local_search(chosen, cliques, n, time_budget):
    # Improve via removals and swaps. Returns best set found.
    best = set(chosen)
    best_size = len(best)
    # Precompute: for each clique, the set of chosen vertices in it (for quick check)
    def clique_cover_count(c):
        return len(best & c)
    deadline = _elapsed() + time_budget
    # Removal pass: try removing each vertex
    improved = True
    while improved:
        if _elapsed() > deadline:
            break
        improved = False
        for v in list(best):
            if _elapsed() > deadline:
                break
            # check if removing v still covers all cliques
            ok = True
            for c in cliques:
                if v in c and len((best & c)) <= 1:
                    ok = False
                    break
            if ok:
                best.discard(v)
                best_size -= 1
                improved = True
    # Swap pass: try replacing one chosen vertex with one non-chosen
    improved = True
    while improved:
        if _elapsed() > deadline:
            break
        improved = False
        non_chosen = [v for v in range(n) if v not in best]
        for v_out in list(best):
            if _elapsed() > deadline:
                break
            for v_in in non_chosen:
                if _elapsed() > deadline:
                    break
                # simulate
                trial = set(best)
                trial.discard(v_out)
                trial.add(v_in)
                if _covers_all(trial, cliques) and len(trial) < best_size:
                    best = trial
                    best_size = len(trial)
                    improved = True
                    non_chosen = [v for v in range(n) if v not in best]
                    break
    return best

def _solve_instance(n, p, seed, time_budget):
    n, edges = _instance(n, p, seed)
    adj = _build_adj(n, edges)
    cliques = _maximal_cliques(n, adj)
    cliques = [c for c in cliques if len(c) >= 2]
    if not cliques:
        return []
    # Greedy initial
    chosen = _greedy_set_cover(cliques, n)
    # Local search polish
    chosen = _local_search(chosen, cliques, n, time_budget)
    return sorted(chosen)

def construct():
    global _START
    _START = time.time()
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    results = []
    # Allocate time: give each instance a share, leaving margin
    total_budget = _DEADLINE - 5.0  # 5s margin
    per_instance = total_budget / len(instances)
    for i, (n, p, seed) in enumerate(instances):
        remaining = _DEADLINE - _elapsed()
        if remaining <= 1.0:
            # Not enough time; use greedy only
            n2, edges = _instance(n, p, seed)
            adj = _build_adj(n2, edges)
            cliques = [c for c in _maximal_cliques(n2, adj) if len(c) >= 2]
            if cliques:
                chosen = _greedy_set_cover(cliques, n2)
                results.append(sorted(chosen))
            else:
                results.append([])
            continue
        # Use min(per_instance, remaining - 2) as budget for local search
        ts_budget = min(per_instance, remaining - 2.0)
        results.append(_solve_instance(n, p, seed, ts_budget))
    return results
# EVOLVE-BLOCK-END
