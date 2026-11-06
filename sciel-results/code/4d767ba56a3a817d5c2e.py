"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
from collections import defaultdict, deque
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    from random import Random

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def get_adjacency(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def get_odd_vertices(edges, n):
        deg = defaultdict(int)
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        return [v for v in range(n) if deg[v] % 2 == 1]

    def find_longest_path(start, adj, remaining_edges, max_len=100):
        """Find a long simple path starting from start using only remaining edges."""
        best_path = [start]
        
        def dfs(node, path, used_edges, depth):
            nonlocal best_path
            if depth >= max_len:
                return
            if len(path) > len(best_path):
                best_path = path[:]
            
            # Sort neighbors by remaining degree (prefer high degree)
            neighbors = []
            for nb in adj[node]:
                e = tuple(sorted((node, nb)))
                if e in remaining_edges and e not in used_edges and nb not in path:
                    # Calculate remaining degree of nb
                    rem_deg = 0
                    for x in adj[nb]:
                        e2 = tuple(sorted((nb, x)))
                        if e2 in remaining_edges and e2 not in used_edges:
                            rem_deg += 1
                    neighbors.append((rem_deg, nb))
            
            neighbors.sort(reverse=True)
            
            for rem_deg, nb in neighbors:
                e = tuple(sorted((node, nb)))
                used_edges.add(e)
                path.append(nb)
                dfs(nb, path, used_edges, depth + 1)
                path.pop()
                used_edges.remove(e)
        
        dfs(start, [start], set(), 0)
        return best_path

    def decompose_graph(edges, adj, n, time_limit=40):
        """Decompose graph into minimum number of edge-disjoint paths."""
        start = time.time()
        
        odd_verts = get_odd_vertices(edges, n)
        min_paths = max(1, len(odd_verts) // 2)
        
        remaining = set(tuple(sorted(e)) for e in edges)
        paths = []
        
        # Strategy: greedily build paths, preferring to connect odd vertices
        # Track which odd vertices are still "free" (not used as endpoint)
        free_odd = set(odd_verts)
        
        while remaining and time.time() - start < time_limit:
            # Choose start vertex: prefer free odd vertex with highest remaining degree
            deg_in_rem = defaultdict(int)
            for u, v in remaining:
                deg_in_rem[u] += 1
                deg_in_rem[v] += 1
            
            # Priority: free odd > other odd > even, then by degree
            candidates = []
            for v in range(n):
                if deg_in_rem[v] == 0:
                    continue
                priority = 0
                if v in free_odd:
                    priority = 1000
                elif deg[v] % 2 == 1:
                    priority = 500
                candidates.append((priority + deg_in_rem[v], v))
            
            if not candidates:
                break
            
            candidates.sort(reverse=True)
            start_v = candidates[0][1]
            
            # Find a long path from start_v
            path = find_longest_path(start_v, adj, remaining)
            
            if len(path) < 2:
                # Single edge case
                for e in list(remaining):
                    if e[0] == start_v or e[1] == start_v:
                        other = e[0] if e[1] == start_v else e[1]
                        paths.append([start_v, other])
                        remaining.remove(e)
                        break
                continue
            
            # Remove edges from path
            for i in range(len(path) - 1):
                e = tuple(sorted((path[i], path[i+1])))
                if e in remaining:
                    remaining.remove(e)
            
            paths.append(path)
            
            # Update free_odd: endpoints of this path are no longer free
            if path[0] in free_odd:
                free_odd.remove(path[0])
            if path[-1] in free_odd:
                free_odd.remove(path[-1])
        
        # If edges remain, add them as single-edge paths
        for e in remaining:
            paths.append(list(e))
        
        return paths

    def merge_paths(paths, edges, adj, n, time_limit=20):
        """Try to merge paths to reduce count."""
        start = time.time()
        edge_set = set(tuple(sorted(e)) for e in edges)
        current = [list(p) for p in paths]
        
        improved = True
        while improved and time.time() - start < time_limit:
            improved = False
            for i in range(len(current)):
                for j in range(i + 1, len(current)):
                    p1, p2 = current[i], current[j]
                    
                    # Try all 4 endpoint combinations
                    for a, b, rev1, rev2 in [
                        (p1[-1], p2[0], False, False),
                        (p1[0], p2[-1], False, False),
                        (p1[-1], p2[-1], False, True),
                        (p1[0], p2[0], True, False),
                    ]:
                        e = tuple(sorted((a, b)))
                        if e not in edge_set:
                            continue
                        
                        # Build merged path
                        path1 = p1[:] if not rev1 else p1[::-1]
                        path2 = p2[:] if not rev2 else p2[::-1]
                        
                        # Connect: path1 ends at a, path2 starts at b (or vice versa)
                        if a == path1[-1] and b == path2[0]:
                            merged = path1 + path2[1:]
                        elif a == path1[0] and b == path2[-1]:
                            merged = path2 + path1[1:]
                        else:
                            continue
                        
                        # Check simplicity
                        if len(set(merged)) != len(merged):
                            continue
                        
                        # Check all edges valid
                        valid = True
                        for k in range(len(merged) - 1):
                            e2 = tuple(sorted((merged[k], merged[k+1])))
                            if e2 not in edge_set:
                                valid = False
                                break
                        if not valid:
                            continue
                        
                        # Verify full decomposition
                        new_paths = current[:i] + current[i+1:j] + current[j+1:]
                        new_paths.append(merged)
                        
                        used = set()
                        ok = True
                        for p in new_paths:
                            for k in range(len(p) - 1):
                                e3 = tuple(sorted((p[k], p[k+1])))
                                if e3 in used or e3 not in edge_set:
                                    ok = False
                                    break
                                used.add(e3)
                            if not ok:
                                break
                        if ok and used == edge_set:
                            current = new_paths
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        return current

    def verify_decomposition(paths, edges, n):
        """Verify that paths are valid: simple, use edges, edge-disjoint, cover all edges."""
        all_edges = set(tuple(sorted(e)) for e in edges)
        used_edges = set()
        
        for path in paths:
            if len(set(path)) != len(path):
                return False
            for i in range(len(path) - 1):
                e = tuple(sorted((path[i], path[i+1])))
                if e not in all_edges:
                    return False
                if e in used_edges:
                    return False
                used_edges.add(e)
        
        if used_edges != all_edges:
            return False
        
        return True

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    for spec in specs:
        n, p, seed = spec
        edges = instance(n, p, seed)
        adj = get_adjacency(edges)
        
        # Get decomposition
        paths = decompose_graph(edges, adj, n, time_limit=35)
        
        # Try to improve by merging
        if len(paths) > 1 and time.time() - start_time < deadline - 20:
            improved = merge_paths(paths, edges, adj, n, time_limit=15)
            if verify_decomposition(improved, edges, n) and len(improved) <= len(paths):
                paths = improved
        
        # Verify and fallback
        if not verify_decomposition(paths, edges, n):
            paths = [list(e) for e in edges]
        
        results.append(paths)
        
        if time.time() - start_time > deadline - 10:
            for _ in range(3 - len(results)):
                n2, p2, seed2 = specs[len(results)]
                edges2 = instance(n2, p2, seed2)
                results.append([list(e) for e in edges2])
            break
    
    return results
# EVOLVE-BLOCK-END
