"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _maximal_sets(adj):
    n = P['n']
    # maximal cliques
    cliques = []
    def expand_clique(cand, curr):
        if not cand:
            if curr:
                cliques.append(frozenset(curr))
            return
        for v in list(cand):
            ncand = cand & adj[v]
            expand_clique(ncand, curr | {v})
            cand.remove(v)
    expand_clique(set(range(n)), frozenset())
    # maximal independent sets = maximal cliques in complement
    comp = [set(range(n)) - adj[v] - {v} for v in range(n)]
    indep = []
    def expand_indep(cand, curr):
        if not cand:
            if curr:
                indep.append(frozenset(curr))
            return
        for v in list(cand):
            ncand = cand & comp[v]
            expand_indep(ncand, curr | {v})
            cand.remove(v)
    expand_indep(set(range(n)), frozenset())
    # Deduplicate
    seen = set()
    result = []
    for s in cliques + indep:
        if s not in seen:
            seen.add(s)
            result.append(s)
    return result

def _greedy_cover(adj, sets, deadline):
    n = P['n']
    remaining = set(range(n))
    chosen = []
    while remaining:
        best = None
        best_cov = -1
        for s in sets:
            if not remaining:
                break
            cov = len(s & remaining)
            if cov > best_cov:
                best_cov = cov
                best = s
        if best is None:
            break
        chosen.append(best)
        remaining -= best
    return chosen

def _exact_cover(adj, sets, deadline):
    n = P['n']
    # Precompute for each vertex, which sets contain it
    vertex_sets = [set() for _ in range(n)]
    for i, s in enumerate(sets):
        for v in s:
            vertex_sets[v].add(i)
    
    # Sort sets by size descending for better pruning
    order = sorted(range(len(sets)), key=lambda i: -len(sets[i]))
    ordered_sets = [sets[i] for i in order]
    ordered_vertex_sets = [set() for _ in range(n)]
    for i, si in enumerate(order):
        for v in sets[si]:
            ordered_vertex_sets[v].add(i)
    
    best_solution = None
    best_k = n + 1
    
    def search(remaining, chosen, depth, k_limit):
        nonlocal best_solution, best_k
        if time.time() > deadline:
            return
        if not remaining:
            if depth < best_k:
                best_k = depth
                best_solution = list(chosen)
            return
        if depth >= best_k:
            return
        if depth >= k_limit:
            return
        # Pick vertex with fewest candidate sets
        min_count = len(remaining)
        target = None
        for v in remaining:
            c = len(ordered_vertex_sets[v])
            if c < min_count:
                min_count = c
                target = v
        if min_count == 0:
            return
        for si in sorted(ordered_vertex_sets[target], key=lambda i: -len(ordered_sets[i])):
            s = ordered_sets[si]
            new_remaining = remaining - s
            chosen.append(si)
            search(new_remaining, chosen, depth + 1, k_limit)
            chosen.pop()
    
    # Try increasing k
    for k in range(1, n + 1):
        search(set(range(n)), [], 0, k)
        if best_solution is not None:
            break
    return best_solution

def construct():
    start = time.time()
    deadline = start + 80.0  # safety margin
    
    adj = _build_adj()
    sets = _maximal_sets(adj)
    
    # Try exact cover first
    solution = _exact_cover(adj, sets, deadline)
    
    if solution is None or time.time() > deadline - 2.0:
        # Fallback to greedy
        solution = _greedy_cover(adj, sets, deadline)
    
    # Convert to list of lists of ints
    result = []
    for si in solution:
        part = [int(v) for v in sorted(sets[si])]
        result.append(part)
    
    return result
# EVOLVE-BLOCK-END
