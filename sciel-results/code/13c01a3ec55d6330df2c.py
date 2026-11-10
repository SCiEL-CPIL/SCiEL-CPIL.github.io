# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def _build_adj(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def _matching_path_decomposition(edges, max_time=None):
        """
        Use minimum-weight perfect matching on edge-adjacency to minimize path count.
        Each edge is a vertex in the matching graph. Two edges can be matched if they
        share a vertex and can be ordered to form a valid path segment.
        """
        if max_time is None:
            max_time = deadline - 5
        
        if not edges:
            return []
        
        edge_list = list(edges)
        m = len(edge_list)
        
        # Build edge index
        edge_idx = {e: i for i, e in enumerate(edge_list)}
        
        # Build adjacency: for each edge, which other edges can follow it
        # Edge (u,v) can be followed by (v,w) or (w,v) if w != u
        # We need to track orientation: for path construction, we need directed edges
        # Let's represent each edge as having two possible orientations
        
        # For matching: create 2*m nodes (one per oriented edge)
        # Node 2*i: edge i in orientation (edge_list[i][0], edge_list[i][1])
        # Node 2*i+1: edge i in orientation (edge_list[i][1], edge_list[i][0])
        
        # Actually, for path decomposition, we want to find a set of directed paths
        # that cover all edges. This is equivalent to finding a minimum path cover
        # in a directed graph where each edge appears exactly once.
        
        # Simpler approach: greedy with local search, but use matching-like pairing
        
        # Let's use a different structural approach:
        # 1. Find all odd-degree vertices (these must be path endpoints)
        # 2. Pair up edges greedily to form chains
        # 3. Use local search to improve
        
        adj = _build_adj(edges)
        deg = defaultdict(int)
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        
        odd_verts = [v for v in deg if deg[v] % 2 == 1]
        
        # Minimum number of paths is at least ceil(|odd|/2)
        # Try to achieve this bound
        
        # Strategy: iteratively find long paths starting from odd vertices
        remaining = set(edges)
        paths = []
        
        while remaining and time.time() < max_time:
            # Compute degrees in remaining
            rem_deg = defaultdict(int)
            for u, v in remaining:
                rem_deg[u] += 1
                rem_deg[v] += 1
            
            # Prefer starting from odd-degree vertices in remaining
            starts = [v for v in rem_deg if rem_deg[v] % 2 == 1]
            if not starts:
                starts = list(rem_deg.keys())
            
            # Try multiple starts, pick best path
            best_path = None
            best_len = 0
            
            for start in starts[:15]:
                # DFS-based longest path with backtracking (limited)
                path = _find_long_path_dfs(start, remaining, adj, max_depth=25)
                if len(path) > best_len:
                    best_len = len(path)
                    best_path = path
            
            if best_path is None or len(best_path) < 2:
                # Single edge
                e = next(iter(remaining))
                best_path = list(e)
            
            paths.append(best_path)
            for i in range(len(best_path) - 1):
                e = tuple(sorted((best_path[i], best_path[i+1])))
                remaining.discard(e)
        
        # Fallback
        if remaining:
            for e in remaining:
                paths.append(list(e))
        
        # Local search: try to merge paths
        paths = _merge_paths(paths, edges, max_time=max_time - 2)
        
        return paths

    def _find_long_path_dfs(start, remaining, adj, max_depth=25):
        """Find a long simple path using DFS with pruning."""
        best = [start]
        
        def dfs(path, visited, current):
            nonlocal best
            if len(path) > len(best):
                best = list(path)
            if len(path) >= max_depth:
                return
            
            # Sort candidates by degree (prefer low-degree to avoid dead ends)
            candidates = []
            for nb in adj[current]:
                if nb not in visited:
                    e = tuple(sorted((current, nb)))
                    if e in remaining:
                        # Count remaining unvisited neighbors
                        opts = sum(1 for x in adj[nb] if x not in visited and 
                                   tuple(sorted((nb, x))) in remaining)
                        candidates.append((opts, nb))
            
            candidates.sort(key=lambda x: x[0])
            
            for _, nb in candidates[:4]:  # Limit branching
                path.append(nb)
                visited.add(nb)
                dfs(path, visited, nb)
                path.pop()
                visited.discard(nb)
        
        dfs([start], {start}, start)
        return best

    def _merge_paths(paths, edges, max_time=None):
        """Try to merge paths by reconnecting at shared vertices."""
        if max_time is None:
            max_time = deadline - 3
        
        edge_set = set(edges)
        improved = True
        
        while improved and time.time() < max_time:
            improved = False
            for i in range(len(paths)):
                for j in range(i + 1, len(paths)):
                    if time.time() > max_time:
                        break
                    pi, pj = paths[i], paths[j]
                    
                    # Try all endpoint combinations
                    for vi in [pi[0], pi[-1]]:
                        for vj in [pj[0], pj[-1]]:
                            if vi == vj:
                                # Try merging
                                new_path = None
                                if pi[-1] == vj:
                                    new_path = pi + pj[1:]
                                elif pi[0] == vj:
                                    new_path = [pj[0]] + pi + pj[1:] if pj[0] == vi else None
                                    if new_path is None:
                                        new_path = pi + pj[1:]
                                elif pj[-1] == vi:
                                    new_path = pj + pi[1:]
                                elif pj[0] == vi:
                                    new_path = pj + pi[1:]
                                
                                if new_path and len(set(new_path)) == len(new_path):
                                    # Check all edges exist
                                    valid = all(
                                        tuple(sorted((new_path[k], new_path[k+1]))) in edge_set
                                        for k in range(len(new_path) - 1)
                                    )
                                    if valid:
                                        paths[i] = new_path
                                        paths.pop(j)
                                        improved = True
                                        break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        
        return paths

    # Process each graph
    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    for spec in specs:
        edges = instance(*spec)
        
        if time.time() < deadline - 5:
            paths = _matching_path_decomposition(edges, max_time=deadline - 5)
        else:
            paths = [list(e) for e in edges]
        
        results.append(paths)
    
    return results
# EVOLVE-BLOCK-END
