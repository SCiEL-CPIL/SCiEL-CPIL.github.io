"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque
from random import Random

def construct():
    # Deadline guard: aim to finish well before 150s wall-clock.
    t0 = time.time()
    DEADLINE = 140.0  # safety margin

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_into_paths(edges, n):
        """
        Decompose edges into simple paths, minimizing count.
        Uses a flow-based / matching-based approach:
        1. Compute degree of each vertex.
        2. Odd-degree vertices must be endpoints of paths.
        3. We want to pair up odd-degree vertices as path endpoints.
        4. The remaining edges form Eulerian trails between these endpoints.
        
        Strategy:
        - Let odd = vertices with odd degree.
        - Minimum number of paths >= len(odd) // 2.
        - We try to find a decomposition into exactly len(odd)//2 paths if possible.
        - If not, we add more paths.
        
        Approach:
        - Use a greedy + local search approach to find edge-disjoint simple paths covering all edges.
        """
        # Build adjacency with edge tracking
        adj = defaultdict(list)  # adj[u] = list of (v, edge_id)
        edge_list = list(edges)
        for eid, (u, v) in enumerate(edge_list):
            adj[u].append((v, eid))
            adj[v].append((u, eid))
        
        m = len(edge_list)
        
        # Degree of each vertex
        deg = [0] * n
        for u, v in edge_list:
            deg[u] += 1
            deg[v] += 1
        
        odd_vertices = [v for v in range(n) if deg[v] % 2 == 1]
        min_paths = len(odd_vertices) // 2
        if min_paths == 0:
            min_paths = 1 if m > 0 else 0
        
        # We'll use a heuristic: repeatedly find a simple path using as many unused edges as possible
        # This is similar to a greedy trail decomposition but ensuring simplicity.
        
        # Better approach: Use the fact that we can decompose into trails, then split trails at repeated vertices.
        # But we need simple paths. Let's use a different strategy.
        
        # Strategy: 
        # 1. Find a set of edge-disjoint simple paths covering all edges.
        # 2. Start with each edge as a path.
        # 3. Merge paths when they share an endpoint and the merge preserves simplicity.
        # 4. Repeat until no more merges are possible.
        
        # This is a local search approach. Let's implement it.
        
        # Represent each path as a list of vertices. Track which edges are used.
        # used_edges[eid] = True if edge eid is in some path.
        # paths = list of paths, each path is a list of vertices.
        # path_edges[pid] = set of edge ids in path pid.
        
        paths = []
        path_edges = []
        used_edges = [False] * m
        
        # Initialize: each edge is its own path
        for eid, (u, v) in enumerate(edge_list):
            paths.append([u, v])
            path_edges.append({eid})
            used_edges[eid] = True
        
        # Build a map from vertex to list of path indices that have that vertex as an endpoint
        # For merging, we need paths that share an endpoint.
        
        def get_endpoints(path):
            return path[0], path[-1]
        
        def can_merge(p1, p2, shared_vertex):
            """Check if paths p1 and p2 can be merged at shared_vertex to form a simple path."""
            # Determine which end of each path is at shared_vertex
            ep1a, ep1b = get_endpoints(paths[p1])
            ep2a, ep2b = get_endpoints(paths[p2])
            
            # Find the orientation
            if ep1a == shared_vertex and ep2a == shared_vertex:
                new_path = list(reversed(paths[p1]))[1:] + paths[p2]
            elif ep1a == shared_vertex and ep2b == shared_vertex:
                new_path = list(reversed(paths[p1]))[1:] + list(reversed(paths[p2]))
            elif ep1b == shared_vertex and ep2a == shared_vertex:
                new_path = paths[p1][1:] + paths[p2]
            elif ep1b == shared_vertex and ep2b == shared_vertex:
                new_path = paths[p1][1:] + list(reversed(paths[p2]))
            else:
                return False, None
            
            # Check simplicity: all vertices in new_path must be distinct
            if len(set(new_path)) != len(new_path):
                return False, None
            return True, new_path
        
        # Merge loop
        changed = True
        while changed:
            changed = False
            # Build adjacency: for each vertex, which paths have it as endpoint
            vertex_to_paths = defaultdict(list)
            for pid, path in enumerate(paths):
                u, v = get_endpoints(path)
                vertex_to_paths[u].append(pid)
                vertex_to_paths[v].append(pid)
            
            # Try to merge pairs
            merged = [False] * len(paths)
            for v in vertex_to_paths:
                pids = vertex_to_paths[v]
                if len(pids) < 2:
                    continue
                # Try all pairs
                i = 0
                while i < len(pids):
                    if merged[pids[i]]:
                        i += 1
                        continue
                    j = i + 1
                    while j < len(pids):
                        if merged[pids[j]]:
                            j += 1
                            continue
                        ok, new_path = can_merge(pids[i], pids[j], v)
                        if ok:
                            # Merge: replace pids[i] with new_path, remove pids[j]
                            new_eids = path_edges[pids[i]] | path_edges[pids[j]]
                            paths[pids[i]] = new_path
                            path_edges[pids[i]] = new_eids
                            # Mark pids[j] for removal
                            merged[pids[j]] = True
                            changed = True
                            # Rebuild vertex_to_paths for this vertex? Just break and restart
                            break
                        j += 1
                    if merged[pids[i]]:
                        i += 1
                        continue
                    i += 1
                    if changed:
                        break
                if changed:
                    break
            
            if changed:
                # Remove merged paths
                new_paths = []
                new_path_edges = []
                for pid in range(len(paths)):
                    if not merged[pid]:
                        new_paths.append(paths[pid])
                        new_path_edges.append(path_edges[pid])
                paths = new_paths
                path_edges = new_path_edges
        
        return paths
    
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() - t0 > DEADLINE:
            # Fallback: return per-edge paths for remaining
            for remaining_spec in spec[2:]:
                pass
            # Just return what we have plus fallbacks
            break
        edges = instance(*spec)
        n = spec[0]
        paths = decompose_into_paths(edges, n)
        results.append(paths)
    
    # If we ran out of time, add fallbacks
    while len(results) < 3:
        idx = len(results)
        spec = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))[idx]
        edges = instance(*spec)
        results.append([[list(e)] for e in edges])
    
    return results
# EVOLVE-BLOCK-END
