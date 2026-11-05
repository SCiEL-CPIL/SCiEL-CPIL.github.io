"""Decompose three fixed Erdős-Gallai-style graphs into few cycles + edges."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _decompose(edges, n, time_limit=12.0):
    """
    Greedy + local search: repeatedly extract a longest cycle, then try to
    merge adjacent single-edge pieces into cycles when possible.
    """
    start = time.time()
    # Build adjacency as sets for fast membership
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Track which edges are still available
    used_edges = set()
    pieces = []  # list of cycles (list of vertices) or single edges [u,v]
    
    def find_long_cycle():
        """Find a cycle using DFS with backtracking, preferring long paths."""
        # Try from each vertex
        for start_v in range(n):
            if not adj[start_v]:
                continue
            # DFS to find a path back to start_v
            best_cycle = None
            def dfs(v, path, visited_edges):
                nonlocal best_cycle
                if time.time() - start > time_limit * 0.5:
                    return
                if len(path) > (len(best_cycle) if best_cycle else 3):
                    # Check if we can close
                    if start_v in adj[v] and (min(start_v,v), max(start_v,v)) not in visited_edges:
                        cand = path[:]
                        if best_cycle is None or len(cand) > len(best_cycle):
                            best_cycle = cand[:]
                for nb in list(adj[v]):
                    e = tuple(sorted((v, nb)))
                    if e in visited_edges:
                        continue
                    if nb == start_v and len(path) >= 3:
                        # Found a cycle
                        cand = path[:]
                        if best_cycle is None or len(cand) > len(best_cycle):
                            best_cycle = cand[:]
                        continue
                    if nb in path:
                        continue
                    visited_edges.add(e)
                    path.append(nb)
                    dfs(nb, path, visited_edges)
                    path.pop()
                    visited_edges.discard(e)
            dfs(start_v, [start_v], set())
            if best_cycle:
                return best_cycle
        return None
    
    # Phase 1: Extract cycles greedily
    remaining = set(edges)
    while remaining:
        if time.time() - start > time_limit * 0.7:
            break
        # Build temp adjacency from remaining
        temp_adj = defaultdict(set)
        for u, v in remaining:
            temp_adj[u].add(v)
            temp_adj[v].add(u)
        
        # Find a cycle in the remaining graph
        cycle = None
        for start_v in range(n):
            if not temp_adj[start_v]:
                continue
            # Simple DFS for any cycle through start_v
            def dfs_cycle(v, path, visited):
                for nb in temp_adj[v]:
                    if nb == start_v and len(path) >= 3:
                        return path[:]
                    if nb not in visited and nb not in path:
                        visited.add(nb)
                        path.append(nb)
                        res = dfs_cycle(nb, path, visited)
                        if res:
                            return res
                        path.pop()
                        visited.discard(nb)
                return None
            cycle = dfs_cycle(start_v, [start_v], {start_v})
            if cycle:
                break
        
        if cycle and len(cycle) >= 3:
            # Remove cycle edges
            for i in range(len(cycle)):
                a, b = cycle[i], cycle[(i+1) % len(cycle)]
                e = tuple(sorted((a, b)))
                remaining.discard(e)
            pieces.append(cycle)
        else:
            # No cycle found; take a single edge
            if not remaining:
                break
            e = next(iter(remaining))
            remaining.discard(e)
            pieces.append(list(e))
    
    # Phase 2: Try to merge single edges into cycles
    # Map single-edge pieces by their edge
    single_map = {}
    for i, p in enumerate(pieces):
        if len(p) == 2:
            e = tuple(sorted(p))
            single_map[e] = i
    
    # Try to form cycles from consecutive single edges
    changed = True
    while changed and time.time() - start < time_limit:
        changed = False
        new_pieces = []
        used_in_merge = set()
        for i, p in enumerate(pieces):
            if i in used_in_merge:
                continue
            if len(p) == 2:
                e = tuple(sorted(p))
                if e in single_map and single_map[e] == i:
                    # Try to extend: look for another single edge sharing a vertex
                    u, v = p
                    # Check if v connects to another single edge
                    for nb in adj[v]:
                        e2 = tuple(sorted((v, nb)))
                        if e2 in single_map and e2 != e:
                            j = single_map[e2]
                            if j in used_in_merge:
                                continue
                            # Check if nb connects back to u via a third edge or path
                            # For a 3-cycle, need edge (u, nb)
                            e3 = tuple(sorted((u, nb)))
                            if e3 in single_map and e3 != e and e3 != e2:
                                k = single_map[e3]
                                if k in used_in_merge:
                                    continue
                                # Found a triangle!
                                used_in_merge.update([i, j, k])
                                new_pieces.append([u, v, nb])
                                del single_map[e]
                                del single_map[e2]
                                del single_map[e3]
                                changed = True
                                break
        if changed:
            pieces = [p for i, p in enumerate(pieces) if i not in used_in_merge] + new_pieces
    
    # Phase 3: Final pass - convert any remaining single edges that form cycles
    # (already handled above, but do one more sweep)
    
    return pieces

def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    for n, p, seed in specs:
        edges = _instance(n, p, seed)
        pieces = _decompose(edges, n, time_limit=38.0)
        results.append(pieces)
    return results
# EVOLVE-BLOCK-END
