"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

sys.setrecursionlimit(1000000)

_DEADLINE = time.time() + 130.0  # safety margin before the 150s kill


def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _build_adj(edges):
    adj = {}
    for (u, v) in edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    for k in list(adj.keys()):
        adj[k] = list(adj[k])
    return adj


def _greedy(adj, edges, rng, time_budget):
    """Repeatedly extract a long simple path using greedy extension."""
    remaining = set(edges)
    paths = []
    while remaining:
        if time.time() > time_budget:
            return None
        
        # Calculate current degrees in the remaining graph
        deg = {v: 0 for v in adj}
        for (u, v) in remaining:
            deg[u] += 1
            deg[v] += 1
            
        # Candidate start vertices: endpoints of remaining edges
        cand = set()
        for (u, v) in remaining:
            cand.add(u)
            cand.add(v)
            
        # Prefer low-degree starts to maximize path length potential
        cand_list = list(cand)
        cand_list.sort(key=lambda x: deg[x])
        
        # Try a few random low-degree starts to find a good path
        k = min(len(cand_list), 10)
        starts = cand_list[:k]
        rng.shuffle(starts)

        best_path = None
        best_len = -1
        
        for s in starts:
            # Greedy extension from s
            current_path = [s]
            visited = {s}
            current_node = s
            
            while True:
                # Find neighbors of current_node that are in remaining edges and not visited
                nxts = []
                for nb in adj[current_node]:
                    e = (min(current_node, nb), max(current_node, nb))
                    if e in remaining and nb not in visited:
                        nxts.append(nb)
                        
                if not nxts:
                    break
                
                # Choose the neighbor with the lowest degree (in remaining graph)
                # This is a heuristic to avoid isolating vertices prematurely
                nxts.sort(key=lambda x: deg[x])
                
                # Add some randomness to avoid getting stuck in local minima
                # Pick from the top 3 lowest degree options
                top_k = min(3, len(nxts))
                chosen = rng.choice(nxts[:top_k])
                
                # Extend path
                e = (min(current_node, chosen), max(current_node, chosen))
                remaining.discard(e)
                current_path.append(chosen)
                visited.add(chosen)
                current_node = chosen
                
            if len(current_path) > best_len:
                best_len = len(current_path)
                best_path = current_path
                
        if best_path is None or len(best_path) < 2:
            # Fallback: take any single edge
            e = next(iter(remaining))
            best_path = list(e)
            remaining.discard(e)
        else:
            # The edges were already discarded during extension
            pass
            
        paths.append(best_path)
        
    return paths


def _try_decompose(edges, adj, rng, time_budget):
    paths = _greedy(adj, edges, rng, time_budget)
    if paths is None:
        return None
    # validate
    used = set()
    for p in paths:
        if len(p) < 1:
            return None
        seen = set(p)
        if len(seen) != len(p):
            return None  # not simple
        for i in range(len(p) - 1):
            a, b = p[i], p[i + 1]
            e = (min(a, b), max(a, b))
            if e not in edges:
                return None
            if e in used:
                return None
            used.add(e)
    if used != set(edges):
        return None
    return paths


def construct():
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    instances = [_instance(*s) for s in specs]
    adjs = [_build_adj(e) for e in instances]

    results = [None] * 3
    # per-instance time budget
    # The new greedy is very fast, so we can afford more restarts to find better solutions
    per_budget = 10.0
    total_budget = time.time() + per_budget * 3

    for idx in range(3):
        edges = instances[idx]
        adj = adjs[idx]
        best = None
        best_count = len(edges)  # trivial bound
        t_start = time.time()
        limit = min(t_start + per_budget, total_budget - 5.0)
        
        # restart with different seeds
        attempt = 0
        while time.time() < limit:
            rng = random.Random(1000 + attempt)
            # We need a fresh copy of edges for each attempt
            paths = _try_decompose(edges, adj, rng, limit)
            if paths is not None:
                cnt = len(paths)
                if cnt < best_count:
                    best_count = cnt
                    best = paths
            attempt += 1
            
        if best is None:
            # fallback: every edge as its own path
            best = [list(e) for e in edges]
        results[idx] = best

    return results
# EVOLVE-BLOCK-END
