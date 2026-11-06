"""C4-free graph construction with α < 11 via greedy + backtracking."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _deadline_ok(t0, margin=10.0):
    return (time.time() - t0) < (150.0 - margin)

def _has_c4(edges, n):
    """Return True if the graph has a C4."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    for u in range(n):
        for v in adj[u]:
            if u < v:
                common = adj[u] & adj[v]
                if len(common) >= 2:
                    return True
    return False

def _alpha(edges, n):
    """Compute independence number via greedy + branch & bound."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    # Greedy lower bound
    remaining = set(range(n))
    size = 0
    while remaining:
        # Pick vertex with minimum degree in remaining
        min_v = min(remaining, key=lambda x: len(adj[x] & remaining))
        size += 1
        remaining -= adj[min_v]
        remaining.discard(min_v)
    # Upper bound via greedy coloring of complement
    comp_adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u+1, n):
            if v not in adj[u]:
                comp_adj[u].add(v)
                comp_adj[v].add(u)
    # Greedy coloring
    colors = {}
    order = sorted(range(n), key=lambda x: -len(comp_adj[x]))
    for v in order:
        used = set()
        for nb in comp_adj[v]:
            if nb in colors:
                used.add(colors[nb])
        c = 0
        while c in used:
            c += 1
        colors[v] = c
    upper = len(set(colors.values()))
    # If greedy lower == upper, that's the answer
    if size >= upper:
        return size
    # Branch and bound
    best = size
    def bb(candidates, current):
        nonlocal best
        if len(candidates) + current <= best:
            return
        if not candidates:
            if current > best:
                best = current
            return
        # Pick vertex
        v = max(candidates, key=lambda x: len(candidates - adj[x]))
        # Include v
        new_cands = candidates - adj[v] - {v}
        bb(new_cands, current + 1)
        # Exclude v
        candidates.discard(v)
        bb(candidates, current)
        candidates.add(v)
    bb(set(range(n)), 0)
    return best

def _greedy_construct(n, t0, seed):
    """Greedy edge addition to maximize edges while maintaining C4-free."""
    random.seed(seed)
    adj = [set() for _ in range(n)]
    edges = []
    # Try all edges in random order, add if no C4 created
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_pairs)
    for u, v in all_pairs:
        if not _deadline_ok(t0):
            break
        # Check: adding (u,v) creates C4 iff u,v have >= 2 common neighbors
        common = adj[u] & adj[v]
        if len(common) < 2:
            adj[u].add(v)
            adj[v].add(u)
            edges.append([u, v])
    return edges, n

def _local_search(edges, n, t0, max_iter=200):
    """Try to reduce independence number via edge swaps."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    best_alpha = _alpha(edges, n)
    current_edges = list(edges)
    
    for _ in range(max_iter):
        if not _deadline_ok(t0):
            break
        # Try random edge swap: remove one, add another
        if not current_edges:
            break
        rem_idx = random.randint(0, len(current_edges)-1)
        ru, rv = current_edges[rem_idx]
        
        # Find a candidate edge to add
        candidates = []
        for u in range(n):
            for v in range(u+1, n):
                if (u, v) not in current_edges and (v, u) not in current_edges:
                    if v not in adj[u]:  # not already edge
                        common = adj[u] & adj[v]
                        if len(common) < 2:
                            candidates.append((u, v))
        if not candidates:
            break
        au, av = random.choice(candidates)
        
        # Check: removing (ru,rv) and adding (au,av) maintains C4-free
        # Temporarily remove
        adj[ru].discard(rv)
        adj[rv].discard(ru)
        # Check new edge
        common_new = adj[au] & adj[av]
        if len(common_new) < 2:
            adj[au].add(av)
            adj[av].add(au)
            new_edges = [e for i, e in enumerate(current_edges) if i != rem_idx]
            new_edges.append([au, av])
            a = _alpha(new_edges, n)
            if a < best_alpha:
                best_alpha = a
                current_edges = new_edges
                adj[ru].add(rv)
                adj[rv].add(ru)
            else:
                # Revert
                adj[au].discard(av)
                adj[av].discard(au)
                adj[ru].add(rv)
                adj[rv].add(ru)
        else:
            adj[ru].add(rv)
            adj[rv].add(ru)
    
    return current_edges, best_alpha

def construct():
    t0 = time.time()
    
    # Start with 30-vertex disjoint triangles (known valid, α=10)
    base_edges = []
    for t in range(10):
        a, b, c = 3*t, 3*t+1, 3*t+2
        base_edges += [[a, b], [a, c], [b, c]]
    
    best_n = 30
    best_edges = base_edges
    
    # Try to find larger graphs
    for n in range(31, 44):
        if not _deadline_ok(t0, margin=20.0):
            break
        # Multiple random restarts
        for seed in range(20):
            if not _deadline_ok(t0, margin=15.0):
                break
            edges, nn = _greedy_construct(n, t0, seed)
            if len(edges) == 0:
                continue
            # Quick C4 check
            if _has_c4(edges, n):
                continue
            a = _alpha(edges, n)
            if a < 11:
                # Valid! Try local search to improve
                edges2, a2 = _local_search(edges, n, t0, max_iter=50)
                if a2 < 11 and a2 <= a:
                    best_n = n
                    best_edges = edges2
                    break
                elif a < 11:
                    best_n = n
                    best_edges = edges
                    break
    
    if not _deadline_ok(t0, margin=5.0):
        # Return best found
        pass
    
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
