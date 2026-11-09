import time
import random
from collections import defaultdict
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

def construct():
    start_time = time.time()
    DEADLINE = 140.0

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_best_cycle(adj, rng):
        # Keep the existing greedy cycle finder as a fallback/improvement step
        candidates = [u for u, neighbors in adj.items() if len(neighbors) > 1]
        if not candidates:
            return None

        best_cycle = None
        best_len = 0

        candidates.sort(key=lambda x: len(adj[x]), reverse=True)
        starts = candidates[:min(8, len(candidates))]
        if len(candidates) > 8:
            starts += rng.sample(candidates, min(8, len(candidates) - 8))
        starts = list(set(starts))
        rng.shuffle(starts)

        for start in starts:
            color = {}
            parent = {}
            path = []
            path_index = {}
            local_best_cycle = None
            local_best_len = 0

            stack = [(start, iter(adj[start]))]
            color[start] = 1
            parent[start] = -1
            path.append(start)
            path_index[start] = 0

            while stack:
                u, neighbors_iter = stack[-1]
                found_next = False
                for w in neighbors_iter:
                    if w == parent[u]:
                        continue
                    if color.get(w, 0) == 0:
                        color[w] = 1
                        parent[w] = u
                        path.append(w)
                        path_index[w] = len(path) - 1
                        stack.append((w, iter(adj[w])))
                        found_next = True
                        break
                    elif color.get(w, 0) == 1:
                        idx = path_index.get(w)
                        if idx is not None:
                            cyc = path[idx:]
                            if len(cyc) > local_best_len:
                                local_best_len = len(cyc)
                                local_best_cycle = list(cyc)
                if not found_next:
                    stack.pop()
                    color[u] = 2
                    if path and path[-1] == u:
                        path.pop()
                        if path_index:
                            del path_index[u]

            if local_best_cycle and local_best_len > best_len:
                best_len = local_best_len
                best_cycle = local_best_cycle
            if best_len >= 25:
                break

        return best_cycle

    def decompose(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        n = max(adj.keys()) + 1 if adj else 0
        
        # Phase 1: Maximum 2-Matching via Bipartite Matching
        # Build bipartite graph
        # Left nodes: 0..n-1, Right nodes: n..2n-1
        # Edge (u, v) in G -> (u, n+v) and (v, n+u) in B
        
        row_ind = []
        col_ind = []
        for u, v in edges:
            row_ind.append(u)
            col_ind.append(n + v)
            row_ind.append(v)
            col_ind.append(n + u)
            
        if not row_ind:
            return [[u, v] for u, v in edges]

        # Create sparse matrix
        data = np.ones(len(row_ind))
        bip_graph = csr_matrix((data, (row_ind, col_ind)), shape=(n, 2*n))
        
        # Find maximum matching
        # scipy returns matching for left and right
        # If scipy version is old, this might not exist or have different name
        try:
            matching = maximum_bipartite_matching(bip_graph)
            # matching is an array of length 2*n
            # matching[i] is the node matched to i, or -1
            # We need to extract edges
            
            selected_edges = set()
            for u in range(n):
                v = matching[u]
                if v != -1:
                    # u is matched to v
                    # v is in the right part, so original node is v - n
                    orig_v = v - n
                    selected_edges.add((min(u, orig_v), max(u, orig_v)))
        except Exception:
            selected_edges = set()
            
        # Extract cycles from selected_edges
        # Build directed graph
        dir_adj = defaultdict(list)
        for u, v in selected_edges:
            # Determine direction? 
            # The matching gives specific directions.
            # Let's reconstruct directions from the matching array
            pass
        
        # Reconstruct directions properly
        dir_adj = defaultdict(list)
        try:
            for u in range(n):
                v = matching[u]
                if v != -1:
                    orig_v = v - n
                    dir_adj[u].append(orig_v)
        except:
            pass
            
        cycles = []
        visited = set()
        for u in range(n):
            if u not in visited and dir_adj[u]:
                # Find cycle starting from u
                # Since it's a 2-matching, each node has out-degree 1 and in-degree 1 (if in matching)
                # So we can just follow the path
                path = []
                curr = u
                while curr not in visited:
                    visited.add(curr)
                    path.append(curr)
                    if not dir_adj[curr]:
                        break
                    curr = dir_adj[curr][0]
                if curr in path:
                    # Found a cycle
                    idx = path.index(curr)
                    cycle = path[idx:]
                    if len(cycle) >= 3:
                        cycles.append(cycle)
                    # Mark path nodes as processed for this component?
                    # Actually, we just need the cycles.
        
        # Phase 2: Close paths
        # Identify paths in the matching
        # A path is a component that is not a cycle.
        # We can try to close them if an edge exists between endpoints.
        
        # Let's identify all components
        comp_adj = defaultdict(list)
        for u in range(n):
            if dir_adj[u]:
                v = dir_adj[u][0]
                comp_adj[u].append(v)
                # Also need reverse for undirected traversal? 
                # For closing, we just need the endpoints.
                
        # Find paths
        # In-degree and out-degree
        in_deg = {u: 0 for u in range(n)}
        out_deg = {u: 0 for u in range(n)}
        for u in range(n):
            if dir_adj[u]:
                out_deg[u] = 1
                in_deg[dir_adj[u][0]] = 1
                
        paths = []
        visited_nodes = set()
        for u in range(n):
            if in_deg[u] == 0 and out_deg[u] == 1 and u not in visited_nodes:
                # Start of a path
                path = [u]
                curr = u
                visited_nodes.add(curr)
                while out_deg[curr] == 1:
                    curr = dir_adj[curr][0]
                    path.append(curr)
                    visited_nodes.add(curr)
                if len(path) > 1:
                    paths.append(path)
                    
        # Try to close paths
        new_cycles = []
        for path in paths:
            start, end = path[0], path[-1]
            if (min(start, end), max(start, end)) in edges:
                # Check if this edge is already used in a cycle?
                # If it's in selected_edges, it's part of the matching.
                # If it's part of the matching, it's already in a cycle or another path.
                # If it connects start and end of *this* path, and it's in the matching,
                # then this path was actually a cycle?
                # No, if it's in the matching, it means the edge is "used".
                # If the edge (start, end) is in selected_edges, then the path is closed in the matching,
                # so it should have been detected as a cycle.
                # So if we found it as a path, the edge (start, end) is NOT in selected_edges.
                # So we can add it!
                # But we must remove the "openness" of the path.
                # The path edges are already in selected_edges.
                # Adding (start, end) makes it a cycle.
                # We add this cycle to our list.
                new_cycles.append(path)
                # We don't need to update selected_edges for the purpose of counting,
                # but we should mark the edge as used so we don't use it as a single edge later.
                selected_edges.add((min(start, end), max(start, end)))
                
        cycles.extend(new_cycles)
        
        # Phase 3: Greedy on remaining edges
        remaining_edges = [e for e in edges if e not in selected_edges]
        rem_adj = defaultdict(set)
        for u, v in remaining_edges:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
            
        rng = random.Random(123)
        greedy_cycles = []
        while True:
            if time.time() - start_time > DEADLINE - 10:
                break
            if not any(len(neigh) > 0 for neigh in rem_adj.values()):
                break
            cycle = find_best_cycle(rem_adj, rng)
            if cycle is None:
                break
            # Remove edges from rem_adj
            for j in range(len(cycle)):
                a, b = cycle[j], cycle[(j + 1) % len(cycle)]
                e = (min(a, b), max(a, b))
                if b in rem_adj[a]:
                    rem_adj[a].discard(b)
                if a in rem_adj[b]:
                    rem_adj[b].discard(a)
            greedy_cycles.append(cycle)
            
        all_cycles = cycles + greedy_cycles
        
        # Create pieces
        pieces = [list(c) for c in all_cycles]
        used_edges = set()
        for c in all_cycles:
            for j in range(len(c)):
                a, b = c[j], c[(j + 1) % len(c)]
                used_edges.add((min(a, b), max(a, b)))
                
        for e in edges:
            if e not in used_edges:
                pieces.append([e[0], e[1]])
                
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = instance(*spec)
        results.append(decompose(edges))
    return results
