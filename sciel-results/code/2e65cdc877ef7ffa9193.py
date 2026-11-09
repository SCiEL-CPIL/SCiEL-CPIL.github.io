import time
import random
import sys

def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _decompose_optimized(edges, deadline):
    if not edges:
        return []
    
    # Build adjacency
    adj = {u: set() for u in range(30)} # Max n is 30
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    all_edges_set = set(tuple(sorted(e)) for e in edges)
    total_edges = len(all_edges_set)
    
    # 1. Initial Greedy Decomposition
    # We use a slightly smarter greedy: start from highest degree, prefer neighbors with lower degree (Wade's heuristic for trails, but we need paths)
    # Actually, for paths, we want to avoid getting stuck.
    # Let's stick to the previous greedy but maybe with a better start selection.
    
    used = set()
    paths = []
    avail = {u: set(v for v in adj[u]) for u in adj}
    
    def extract_path(start):
        path = [start]
        visited = {start}
        cur = start
        while True:
            candidates = [nb for nb in avail[cur] if nb not in visited]
            if not candidates:
                break
            # Heuristic: prefer neighbors with fewer options (to leave more options for others?)
            # Or prefer neighbors with more options?
            # Standard DFS for longest path often benefits from visiting high-degree nodes later or earlier.
            # Let's try: sort by degree ascending (least connected first) to keep high-degree nodes for later?
            # Or descending?
            # Let's try descending (most connected first) to try to traverse the core early.
            candidates.sort(key=lambda x: (-len(avail[x]), x))
            nxt = candidates[0]
            e = tuple(sorted((cur, nxt)))
            used.add(e)
            avail[cur].discard(nxt)
            avail[nxt].discard(cur)
            path.append(nxt)
            visited.add(nxt)
            cur = nxt
        return path

    while used != all_edges_set:
        if time.time() > deadline:
            break
        active_vertices = [v for v in avail if avail[v]]
        if not active_vertices:
            break
        start = max(active_vertices, key=lambda v: (len(avail[v]), -v))
        path = extract_path(start)
        if len(path) >= 2:
            paths.append(path)
        else:
            # Force single edge
            for v in active_vertices:
                if avail[v]:
                    nb = next(iter(avail[v]))
                    e = tuple(sorted((v, nb)))
                    if e not in used:
                        paths.append([v, nb])
                        used.add(e)
                        avail[v].discard(nb)
                        avail[nb].discard(v)
                        break

    # 2. Local Search / Hill Climbing
    # Goal: Minimize number of paths.
    # Moves:
    # A. Merge two paths if an edge connects an endpoint of P1 to an endpoint of P2.
    #    If P1 ends at u and P2 starts at v, and (u,v) is an edge, we can merge them into one path.
    #    This reduces path count by 1.
    # B. Rotate path: Reverse path. (Doesn't change count, but might enable merges)
    # C. Split path? (Increases count, avoid)
    # D. Swap endpoints?
    
    # To make merging efficient, we need to quickly find if two paths can be merged.
    # We maintain a set of "endpoints" for each path.
    # A path P = [v0, v1, ..., vk]. Endpoints are v0 and vk.
    # If we have path A ending at u and path B starting at v, and edge (u,v) exists, we can merge A+B (if A ends at u and B starts at v, resulting path is A + B).
    # Note: We must ensure the merged path is simple. Since A and B are edge-disjoint and vertex-disjoint?
    # Wait, paths in a decomposition are edge-disjoint, but NOT necessarily vertex-disjoint.
    # If they share a vertex, merging might create a cycle or a non-simple path.
    # The problem requires paths to be SIMPLE (no repeated vertices).
    # So we can only merge if the resulting sequence of vertices has no repeats.
    
    # This makes local search harder.
    # However, if the graph is sparse, vertex collisions are less likely but still possible.
    
    # Let's implement a simple local search:
    # 1. Try to merge any pair of paths.
    # 2. If merge is valid (simple), perform it.
    # 3. Repeat until no more merges possible or time limit.
    
    # Optimization:
    # Instead of checking all pairs O(N^2), we can index paths by their endpoints.
    # end_to_paths[u] = list of indices of paths that have u as an endpoint.
    
    # Let's rebuild the structure for local search.
    # paths: list of lists of vertices.
    
    # Check validity of merge:
    def is_valid_merge(p1, p2):
        # p1 and p2 are lists of vertices.
        # We can merge p1 + p2 if p1[-1] == p2[0] is NOT the case (that would be concatenation, but we need an EDGE between them).
        # The edge must be (p1[-1], p2[0]).
        # Let u = p1[-1], v = p2[0].
        # If (u, v) is in all_edges_set, we can form new_path = p1 + p2.
        # We must check if new_path is simple.
        # i.e., len(set(new_path)) == len(new_path).
        # Since p1 and p2 are simple, we just need to check if any vertex in p1 is in p2.
        # If they share any vertex, the merge is invalid (unless the shared vertex is the connection point? No, if p1 has x and p2 has x, and x is not the connection point, then x appears twice).
        # If the connection point is shared? e.g. p1 ends at u, p2 starts at u? No, edge is (u,v).
        # If p1 ends at u and p2 starts at v.
        # If u is in p2 (and u != v), then u appears in p1 (as last) and in p2 (as some internal or end). Duplicate.
        # If v is in p1 (and v != u), then v appears in p2 (as first) and in p1 (as some internal or start). Duplicate.
        # So we just need to check for intersection of vertex sets.
        
        u = p1[-1]
        v = p2[0]
        e = tuple(sorted((u, v)))
        if e not in all_edges_set:
            return False
        
        set1 = set(p1)
        set2 = set(p2)
        if set1 & set2:
            return False
        return True

    # We also need to consider reversing paths to find merges.
    # If p1 ends at u, we can reverse p1 to end at p1[0].
    
    # Strategy:
    # Maintain a list of paths.
    # Loop:
    #   Found_merge = False
    #   For i in range(len(paths)):
    #     For j in range(i+1, len(paths)):
    #       Try merging paths[i] + paths[j]
    #       Try merging paths[i] + reversed(paths[j])
    #       Try merging reversed(paths[i]) + paths[j]
    #       Try merging reversed(paths[i]) + reversed(paths[j])
    #       If valid, merge, update paths, Found_merge = True, break
    #   If not Found_merge: break
    
    # This is O(N^2 * L) per iteration, where L is path length.
    # N is number of paths (small, ~10-20).
    # This should be fast enough.
    
    # To speed up, we can use the endpoint index.
    # But N is small, so O(N^2) is fine.
    
    # Let's add a "cooling" or randomization if stuck?
    # For now, just greedy merge.
    
    # We also need to ensure we don't exceed time.
    
    while True:
        if time.time() > deadline:
            break
        merged = False
        n_paths = len(paths)
        for i in range(n_paths):
            if time.time() > deadline:
                break
            for j in range(i + 1, n_paths):
                p1 = paths[i]
                p2 = paths[j]
                
                candidates = [
                    (p1, p2),
                    (p1, p2[::-1]),
                    (p1[::-1], p2),
                    (p1[::-1], p2[::-1])
                ]
                
                for c1, c2 in candidates:
                    if is_valid_merge(c1, c2):
                        new_path = c1 + c2
                        # Remove i and j, add new_path
                        # Note: indices change
                        paths.pop(j)
                        paths.pop(i)
                        paths.append(new_path)
                        merged = True
                        break
                if merged:
                    break
            if merged:
                break
        if not merged:
            break
            
    return paths

def construct():
    deadline = time.time() + 145.0  # Safety margin
    
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    
    for spec in specs:
        edges = _instance(*spec)
        paths = _decompose_optimized(edges, deadline)
        results.append(paths)
        
        # Verify and fix
        edge_set = set(tuple(sorted(e)) for e in edges)
        covered = set()
        for p in results[-1]:
            for j in range(len(p) - 1):
                e = tuple(sorted((p[j], p[j+1])))
                covered.add(e)
        
        # Check for invalid paths (non-simple or non-edge)
        # The local search ensures simplicity.
        # The greedy ensures edges.
        # But let's double check.
        
        # Add missing edges
        missing = edge_set - covered
        for e in missing:
            results[-1].append(list(e))
            
        # If we ran out of time, the last graph might be incomplete.
        # But we break the loop if time > deadline.
        # So we should handle the case where we didn't process all specs.
        if time.time() > deadline:
             # Fill remaining specs with trivial decomposition?
             # Or just return what we have?
             # The contract says "three lists".
             # If we break early, results has fewer than 3 items.
             # We must return 3 items.
             for rem_spec in specs[len(results):]:
                 rem_edges = _instance(*rem_spec)
                 # Trivial: each edge is a path
                 rem_paths = [list(e) for e in rem_edges]
                 results.append(rem_paths)
             break

    return results
