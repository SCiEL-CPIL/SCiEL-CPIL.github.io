"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    deadline = time.time() + 145  # safety margin before 150s kill
    
    # Reproduce the instance generator from the verifier
    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def solve_graph(edges):
        """Find minimum path cover for a graph given its edges."""
        # Build adjacency
        adj = defaultdict(set)
        edge_set = set()
        for e in edges:
            u, v = e
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add(tuple(sorted(e)))
        
        # Start with each edge as a path
        paths = [[e[0], e[1]] for e in edges]
        
        # Track which edges are used in which path (for simplicity, we track by edge)
        # We'll work with paths as lists of vertices
        
        def is_simple(path):
            """Check if path is simple (no repeated vertices)."""
            return len(set(path)) == len(path)
        
        def path_edges(path):
            """Get set of edges in a path."""
            return {tuple(sorted((path[i], path[i+1]))) for i in range(len(path)-1)}
        
        def can_merge_at_endpoint(p1, p2):
            """
            Check if we can merge p1 and p2 at a shared endpoint.
            Returns the merged path if possible, else None.
            """
            # p1 endpoint at end: p1[-1]
            # p2 endpoint at start: p2[0]
            # Try: p1 + p2[1:] if p1[-1] == p2[0]
            # Try: p1[1:] + p2 if p1[0] == p2[-1]
            # Try: p1 + reversed(p2)[1:] if p1[-1] == p2[-1]
            # Try: reversed(p1)[1:] + p2 if p1[0] == p2[0]
            
            candidates = []
            
            # Case 1: p1[-1] == p2[0], merge as p1 + p2[1:]
            if p1[-1] == p2[0]:
                merged = p1 + p2[1:]
                if is_simple(merged):
                    candidates.append(merged)
            
            # Case 2: p1[0] == p2[-1], merge as p1[1:] + p2
            if p1[0] == p2[-1]:
                merged = p1[1:] + p2
                if is_simple(merged):
                    candidates.append(merged)
            
            # Case 3: p1[-1] == p2[-1], merge as p1 + reversed(p2)[1:]
            if p1[-1] == p2[-1]:
                merged = p1 + list(reversed(p2))[1:]
                if is_simple(merged):
                    candidates.append(merged)
            
            # Case 4: p1[0] == p2[0], merge as reversed(p1)[1:] + p2
            if p1[0] == p2[0]:
                merged = list(reversed(p1))[1:] + p2
                if is_simple(merged):
                    candidates.append(merged)
            
            if candidates:
                return candidates[0]
            return None
        
        def try_internal_reconnect(paths_list, i, j):
            """
            Try to split path i at some internal edge and reconnect with path j.
            This allows more aggressive merging.
            """
            p1 = paths_list[i]
            p2 = paths_list[j]
            
            # Try splitting p1 at position k (0 < k < len(p1)-1)
            # p1 becomes p1[:k+1] and p1[k+1:]
            # Then try to merge these with p2 in various ways
            
            for k in range(1, len(p1) - 1):
                part1 = p1[:k+1]  # starts at p1[0], ends at p1[k]
                part2 = p1[k+1:]  # starts at p1[k+1], ends at p1[-1]
                
                # Try merging part1 and p2
                merged12 = can_merge_at_endpoint(part1, p2)
                if merged12:
                    # Now we have merged12 and part2. Check if all edges are covered and simple
                    # merged12 uses edges from part1 and p2
                    # part2 uses edges from p1[k+1:]
                    # Total edges should equal original p1 + p2 edges
                    # Just check simplicity and edge validity
                    all_new = [merged12, part2]
                    if all(is_simple(p) for p in all_new):
                        # Check all edges are valid graph edges
                        for p in all_new:
                            for a, b in zip(p, p[1:]):
                                if (min(a,b), max(a,b)) not in edge_set:
                                    break
                            else:
                                continue
                            break
                        else:
                            return all_new
                
                # Try merging part2 and p2
                merged22 = can_merge_at_endpoint(part2, p2)
                if merged22:
                    all_new = [part1, merged22]
                    if all(is_simple(p) for p in all_new):
                        for p in all_new:
                            for a, b in zip(p, p[1:]):
                                if (min(a,b), max(a,b)) not in edge_set:
                                    break
                            else:
                                continue
                            break
                        else:
                            return all_new
            
            # Also try splitting p2
            for k in range(1, len(p2) - 1):
                part1 = p2[:k+1]
                part2 = p2[k+1:]
                
                merged11 = can_merge_at_endpoint(p1, part1)
                if merged11:
                    all_new = [merged11, part2]
                    if all(is_simple(p) for p in all_new):
                        for p in all_new:
                            for a, b in zip(p, p[1:]):
                                if (min(a,b), max(a,b)) not in edge_set:
                                    break
                            else:
                                continue
                            break
                        else:
                            return all_new
                
                merged12 = can_merge_at_endpoint(p1, part2)
                if merged12:
                    all_new = [merged12, part1]
                    if all(is_simple(p) for p in all_new):
                        for p in all_new:
                            for a, b in zip(p, p[1:]):
                                if (min(a,b), max(a,b)) not in edge_set:
                                    break
                            else:
                                continue
                            break
                        else:
                            return all_new
            
            return None
        
        # Local search: repeatedly try to merge paths
        improved = True
        while improved and time.time() < deadline:
            improved = False
            n_paths = len(paths)
            for i in range(n_paths):
                if time.time() > deadline:
                    break
                for j in range(i+1, n_paths):
                    if time.time() > deadline:
                        break
                    merged = can_merge_at_endpoint(paths[i], paths[j])
                    if merged:
                        # Replace paths[i] and paths[j] with merged
                        paths[i] = merged
                        paths.pop(j)
                        improved = True
                        break
                if improved:
                    break
            
            if improved:
                continue
            
            # Try internal reconnections
            for i in range(len(paths)):
                if time.time() > deadline:
                    break
                for j in range(i+1, len(paths)):
                    if time.time() > deadline:
                        break
                    result = try_internal_reconnect(paths, i, j)
                    if result:
                        # Replace paths[i] and paths[j] with result
                        paths[i] = result[0]
                        paths[j] = result[1]
                        improved = True
                        break
                if improved:
                    break
        
        return paths
    
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline:
            # Fallback: return remaining as single-edge paths
            edges = instance(*spec)
            results.append([[e[0], e[1]] for e in edges])
            continue
        edges = instance(*spec)
        paths = solve_graph(edges)
        results.append(paths)
    
    return results
# EVOLVE-BLOCK-END

