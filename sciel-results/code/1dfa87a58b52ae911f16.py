import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def edge_key(u, v):
        return (min(u, v), max(u, v))

    def build_adj(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def validate_cover(paths, edges):
        """Validate that paths are simple, use graph edges, are edge-disjoint, and cover all edges."""
        edge_set = set(edge_key(u, v) for u, v in edges)
        used_edges = set()
        
        for path in paths:
            # Check simplicity
            if len(path) != len(set(path)):
                return False
            # Check edges exist and are not reused
            for i in range(len(path) - 1):
                e = edge_key(path[i], path[i + 1])
                if e not in edge_set:
                    return False
                if e in used_edges:
                    return False
                used_edges.add(e)
        
        # Check all edges are covered
        return used_edges == edge_set

    def eulerian_decompose(edges, time_limit):
        """Decompose an Eulerian graph (all even degrees) into cycles."""
        if not edges:
            return []
        
        adj = build_adj(edges)
        remaining = set(edge_key(u, v) for u, v in edges)
        cycles = []
        
        while remaining:
            if time.time() > time_limit:
                break
            
            # Find a vertex with remaining edges
            start = None
            for v in adj:
                if adj[v]:
                    start = v
                    break
            if start is None:
                break
            
            # Trace a cycle
            cycle = [start]
            current = start
            visited_edges = set()
            
            while True:
                next_v = None
                for neighbor in sorted(adj[current]):
                    e = edge_key(current, neighbor)
                    if e in remaining and e not in visited_edges:
                        next_v = neighbor
                        break
                
                if next_v is None:
                    break
                
                cycle.append(next_v)
                e = edge_key(current, next_v)
                visited_edges.add(e)
                remaining.discard(e)
                adj[current].discard(next_v)
                adj[next_v].discard(current)
                current = next_v
            
            if len(cycle) > 1:
                # Remove the last vertex if it equals the first (for cycle representation)
                if cycle[-1] == cycle[0]:
                    cycle = cycle[:-1]
                cycles.append(cycle)
        
        return cycles

    def greedy_trail_cover(edges, rng, time_limit):
        """
        Greedy approach: repeatedly find the longest simple path in the remaining graph,
        prioritizing paths that start/end at odd-degree vertices.
        """
        if not edges:
            return []
        
        adj = build_adj(edges)
        remaining = set(edge_key(u, v) for u, v in edges)
        paths = []
        
        # Precompute odd vertices for guidance
        odd_verts = set(v for v in adj if len(adj[v]) % 2 != 0)
        
        while remaining:
            if time.time() > time_limit:
                break
            
            # Choose start vertex
            if odd_verts:
                start = rng.choice(list(odd_verts))
            else:
                candidates = [v for v in adj if adj[v]]
                if not candidates:
                    break
                start = rng.choice(candidates)
            
            # Try to extend path from start
            best_path = [start]
            best_visited = {start}
            
            # DFS to find longest simple path
            def dfs_extend(current, path, visited, remaining_edges, adj_local):
                if time.time() > time_limit:
                    return path
                
                best_local = path
                
                for neighbor in sorted(adj_local[current], key=lambda x: (-len(adj_local[x]), x)):
                    e = edge_key(current, neighbor)
                    if e in remaining_edges and neighbor not in visited:
                        new_path = path + [neighbor]
                        new_visited = visited | {neighbor}
                        
                        # Recurse
                        result = dfs_extend(neighbor, new_path, new_visited, remaining_edges, adj_local)
                        if len(result) > len(best_local):
                            best_local = result
                        
                        # Limit depth for performance
                        if len(best_local) >= 15:
                            break
                
                return best_local
            
            path = dfs_extend(start, [start], {start}, remaining, adj)
            
            if len(path) > 1:
                paths.append(path)
                # Remove used edges
                for i in range(len(path) - 1):
                    e = edge_key(path[i], path[i + 1])
                    remaining.discard(e)
                    adj[path[i]].discard(path[i + 1])
                    adj[path[i + 1]].discard(path[i])
                # Update odd vertices
                for v in path:
                    if v in odd_verts:
                        odd_verts.discard(v)
                    if len(adj[v]) % 2 != 0:
                        odd_verts.add(v)
        
        return paths

    def two_phase_decompose(edges, time_limit):
        """
        Two-phase approach:
        1. Add dummy edges to make all degrees even, decompose into cycles
        2. Remove dummy edges, split cycles at those points to get paths
        """
        if not edges:
            return []
        
        adj = build_adj(edges)
        odd_verts = [v for v in adj if len(adj[v]) % 2 != 0]
        n_odd = len(odd_verts)
        
        # Pair up odd vertices and add dummy edges
        dummy_edges = []
        if n_odd > 0:
            # Simple pairing
            for i in range(0, n_odd - 1, 2):
                dummy_edges.append((odd_verts[i], odd_verts[i + 1]))
        
        all_edges = list(edges) + dummy_edges
        adj2 = build_adj(all_edges)
        
        # Decompose into cycles
        remaining = set(edge_key(u, v) for u, v in all_edges)
        cycles = []
        
        while remaining:
            if time.time() > time_limit:
                break
            
            start = None
            for v in adj2:
                if adj2[v]:
                    start = v
                    break
            if start is None:
                break
            
            cycle = [start]
            current = start
            
            while True:
                next_v = None
                for neighbor in sorted(adj2[current]):
                    e = edge_key(current, neighbor)
                    if e in remaining:
                        next_v = neighbor
                        break
                
                if next_v is None:
                    break
                
                cycle.append(next_v)
                e = edge_key(current, next_v)
                remaining.discard(e)
                adj2[current].discard(next_v)
                adj2[next_v].discard(current)
                current = next_v
            
            if len(cycle) > 1:
                if cycle[-1] == cycle[0]:
                    cycle = cycle[:-1]
                cycles.append(cycle)
        
        # Now split cycles at dummy edge positions
        dummy_set = set(edge_key(u, v) for u, v in dummy_edges)
        paths = []
        
        for cycle in cycles:
            if len(cycle) < 2:
                continue
            
            # Find positions of dummy edges in the cycle
            dummy_positions = []
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                if edge_key(u, v) in dummy_set:
                    dummy_positions.append(i)
            
            if not dummy_positions:
                # Pure cycle, treat as a path by breaking at any point
                paths.append(cycle)
            else:
                # Split at dummy positions
                # Sort positions
                dummy_positions.sort()
                
                for j, pos in enumerate(dummy_positions):
                    start_idx = (pos + 1) % len(cycle)
                    end_idx = (dummy_positions[j + 1] if j + 1 < len(dummy_positions) else 0 + len(cycle)) % len(cycle)
                    
                    # Extract path from start_idx to end_idx
                    path = []
                    i = start_idx
                    while i != end_idx or not path:
                        path.append(cycle[i % len(cycle)])
                        i += 1
                        if len(path) > len(cycle) + 1:
                            break
                    
                    if len(path) >= 2:
                        # Check if this path only uses real edges
                        valid = True
                        for k in range(len(path) - 1):
                            if edge_key(path[k], path[k + 1]) in dummy_set:
                                valid = False
                                break
                        if valid:
                            paths.append(path)
        
        return paths

    def improved_greedy_with_lookahead(edges, rng, time_limit):
        """Greedy with lookahead: try multiple start choices and pick the one giving longest path."""
        if not edges:
            return []
        
        adj = build_adj(edges)
        remaining = set(edge_key(u, v) for u, v in edges)
        paths = []
        
        while remaining:
            if time.time() > time_limit:
                break
            
            # Get current odd vertices
            odd_verts = [v for v in adj if len(adj[v]) % 2 != 0]
            
            # Candidate starts: odd vertices first, then high-degree vertices
            if odd_verts:
                candidates = odd_verts[:10] + [v for v in adj if len(adj[v]) > 2][:5]
            else:
                candidates = sorted(adj.keys(), key=lambda x: -len(adj[x]))[:10]
            
            best_path = None
            best_len = 0
            
            for start in candidates[:8]:  # Limit starts tried
                if not adj[start]:
                    continue
                
                # DFS with depth limit
                def dfs(current, path, visited):
                    if time.time() > time_limit:
                        return path
                    
                    if len(path) >= 12:  # Depth limit
                        return path
                    
                    best_local = path
                    for neighbor in sorted(adj[current], key=lambda x: (-len(adj[x]), x)):
                        e = edge_key(current, neighbor)
                        if e in remaining and neighbor not in visited:
                            new_path = path + [neighbor]
                            new_visited = visited | {neighbor}
                            result = dfs(neighbor, new_path, new_visited)
                            if len(result) > len(best_local):
                                best_local = result
                    return best_local
                
                path = dfs(start, [start], {start})
                if len(path) > best_len:
                    best_len = len(path)
                    best_path = path
            
            if best_path and len(best_path) > 1:
                paths.append(best_path)
                for i in range(len(best_path) - 1):
                    e = edge_key(best_path[i], best_path[i + 1])
                    remaining.discard(e)
                    adj[best_path[i]].discard(best_path[i + 1])
                    adj[best_path[i + 1]].discard(best_path[i])
        
        return paths

    def merge_paths(paths, edges_set):
        """Try to merge two paths that share an endpoint, creating a longer simple path."""
        if len(paths) < 2:
            return paths
        
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(paths):
                j = i + 1
                while j < len(paths):
                    p1 = paths[i]
                    p2 = paths[j]
                    
                    # Check if p1 can be extended with p2 (p1 end == p2 start or p1 end == p2 end)
                    merged = None
                    
                    # p1 end connects to p2 start
                    if p1[-1] == p2[0]:
                        merged = p1 + p2[1:]
                        # Check simplicity
                        if len(merged) == len(set(merged)):
                            # Check all edges are valid
                            valid = True
                            for k in range(len(merged) - 1):
                                e = edge_key(merged[k], merged[k + 1])
                                if e not in edges_set:
                                    valid = False
                                    break
                            if valid:
                                paths.pop(j)
                                paths[i] = merged
                                changed = True
                                j = i + 1
                                continue
                            else:
                                j += 1
                                continue
                        else:
                            j += 1
                            continue
                    
                    # p2 end connects to p1 start
                    if p2[-1] == p1[0]:
                        merged = p2 + p1[1:]
                        if len(merged) == len(set(merged)):
                            valid = True
                            for k in range(len(merged) - 1):
                                e = edge_key(merged[k], merged[k + 1])
                                if e not in edges_set:
                                    valid = False
                                    break
                            if valid:
                                paths.pop(j)
                                paths[i] = merged
                                changed = True
                                j = i + 1
                                continue
                            else:
                                j += 1
                                continue
                        else:
                            j += 1
                            continue
                    
                    # p1 start connects to p2 end
                    if p1[0] == p2[-1]:
                        merged = p2 + p1[1:]
                        if len(merged) == len(set(merged)):
                            valid = True
                            for k in range(len(merged) - 1):
                                e = edge_key(merged[k], merged[k + 1])
                                if e not in edges_set:
                                    valid = False
                                    break
                            if valid:
                                paths.pop(j)
                                paths[i] = merged
                                changed = True
                                j = i + 1
                                continue
                            else:
                                j += 1
                                continue
                        else:
                            j += 1
                            continue
                    
                    # p1 end connects to p2 end (reverse p2)
                    if p1[-1] == p2[-1]:
                        merged = p1 + list(reversed(p2))[:-1]
                        if len(merged) == len(set(merged)):
                            valid = True
                            for k in range(len(merged) - 1):
                                e = edge_key(merged[k], merged[k + 1])
                                if e not in edges_set:
                                    valid = False
                                    break
                            if valid:
                                paths.pop(j)
                                paths[i] = merged
                                changed = True
                                j = i + 1
                                continue
                            else:
                                j += 1
                                continue
                        else:
                            j += 1
                            continue
                    
                    j += 1
                i += 1
        
        return paths

    def solve_graph(edges, time_limit):
        best_cover = None
        best_count = float('inf')
        
        edges_set = set(edge_key(u, v) for u, v in edges)
        
        # Method 1: Two-phase decomposition
        if time.time() < time_limit - 2:
            cover = two_phase_decompose(edges, time_limit - 2)
            if cover and validate_cover(cover, edges):
                cover = merge_paths(cover, edges_set)
                count = len(cover)
                if count < best_count:
                    best_count = count
                    best_cover = cover
        
        # Method 2: Improved greedy with multiple seeds
        for seed in range(40):
            if time.time() > time_limit - 2:
                break
            rng = random.Random(seed * 17 + 3)
            cover = improved_greedy_with_lookahead(edges, rng, time_limit - 2)
            if cover and validate_cover(cover, edges):
                cover = merge_paths(cover, edges_set)
                count = len(cover)
                if count < best_count:
                    best_count = count
                    best_cover = cover
        
        # Method 3: Basic greedy with multiple seeds
        for seed in range(30):
            if time.time() > time_limit - 1:
                break
            rng = random.Random(seed * 7 + 11)
            cover = greedy_trail_cover(edges, rng, time_limit - 1)
            if cover and validate_cover(cover, edges):
                cover = merge_paths(cover, edges_set)
                count = len(cover)
                if count < best_count:
                    best_count = count
                    best_cover = cover
        
        if best_cover is None:
            # Fallback
            best_cover = [[u, v] for u, v in edges]
        
        return best_cover

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    for spec in specs:
        elapsed = time.time() - start_time
        remaining_time = deadline - elapsed - 2.0
        
        if remaining_time < 5:
            edges = instance(*spec)
            results.append([[u, v] for u, v in edges])
            continue
        
        edges = instance(*spec)
        cover = solve_graph(edges, start_time + remaining_time)
        results.append(cover)
    
    return results
