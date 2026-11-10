"""Rewritten program with stronger path decomposition algorithm."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s limit

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

    def _eulerian_decomposition(remaining, adj):
        """
        Decompose a graph where all vertices have even degree into edge-disjoint cycles.
        Returns list of cycles (each cycle is a list of vertices forming a cycle).
        """
        if not remaining:
            return []
        
        remaining = set(remaining)
        cycles = []
        
        while remaining:
            # Find a vertex with non-zero degree in remaining
            start = None
            for v in adj:
                if any(e in remaining for e in [(min(v, u), max(v, u)) for u in adj[v]]):
                    start = v
                    break
            if start is None:
                break
            
            # Find an Eulerian circuit starting from start
            circuit = _find_eulerian_circuit(start, remaining, adj)
            if circuit:
                cycles.append(circuit)
        
        return cycles

    def _find_eulerian_circuit(start, remaining, adj):
        """Find an Eulerian circuit in the subgraph defined by remaining edges."""
        remaining = set(remaining)
        if not remaining:
            return []
        
        # Hierholzer's algorithm
        stack = [start]
        circuit = []
        used_edges = set()
        
        while stack:
            v = stack[-1]
            # Find an unused edge from v
            found = False
            for u in list(adj[v]):
                e = (min(v, u), max(v, u))
                if e in remaining and e not in used_edges:
                    used_edges.add(e)
                    remaining.discard(e)
                    stack.append(u)
                    found = True
                    break
            if not found:
                circuit.append(stack.pop())
        
        circuit.reverse()
        # Remove duplicate start vertex at end if present
        if len(circuit) > 1 and circuit[0] == circuit[-1]:
            circuit = circuit[:-1]
        
        # Verify all edges were used
        if len(used_edges) != len(remaining) + len(used_edges) - len(used_edges):
            pass
        
        return circuit

    def _greedy_path(start, remaining, adj, strategy='narrow'):
        """
        Greedily extend a path from start.
        strategy: 'narrow' prefers neighbors with fewer options, 'wide' prefers more.
        """
        remaining = set(remaining)
        path = [start]
        visited = {start}
        current = start
        
        while True:
            candidates = []
            for nb in adj[current]:
                if nb not in visited:
                    e = (min(current, nb), max(current, nb))
                    if e in remaining:
                        # Count available neighbors of nb
                        opts = sum(1 for x in adj[nb] 
                                  if x not in visited and (min(nb, x), max(nb, x)) in remaining)
                        candidates.append((opts, nb))
            
            if not candidates:
                break
            
            if strategy == 'narrow':
                candidates.sort(key=lambda x: x[0])
            else:
                candidates.sort(key=lambda x: -x[0])
            
            next_v = candidates[0][1]
            path.append(next_v)
            visited.add(next_v)
            remaining.discard((min(current, next_v), max(current, next_v)))
            current = next_v
        
        return path, remaining

    def _path_decomposition(edges, max_time=None):
        """
        Decompose edges into edge-disjoint paths minimizing count.
        Strategy:
        1. Handle odd-degree vertices by pairing them with paths.
        2. Decompose remaining even-degree graph into cycles, then break cycles into paths.
        """
        if max_time is None:
            max_time = deadline - 5
        
        if not edges:
            return []
        
        edges = set(edges)
        adj = _build_adj(edges)
        
        # Compute degrees
        deg = defaultdict(int)
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        
        # Find odd-degree vertices
        odd_verts = sorted([v for v, d in deg.items() if d % 2 == 1])
        
        paths = []
        remaining = set(edges)
        
        # Strategy: For each odd vertex, start a path that tries to be as long as possible.
        # This is based on the idea that we need at least |odd|/2 paths.
        
        # First, let's try to find long paths starting from odd vertices
        odd_set = set(odd_verts)
        
        # Try to find paths starting from each odd vertex
        for start in odd_verts:
            if time.time() > max_time:
                break
            if start not in remaining and all((min(start, u), max(start, u)) not in remaining for u in adj[start]):
                continue
            
            best_path = None
            best_remaining = None
            
            # Try multiple strategies
            for strategy in ['narrow', 'wide']:
                path, rem = _greedy_path(start, remaining, adj, strategy)
                if best_path is None or len(path) > len(best_path):
                    best_path = path
                    best_remaining = rem
            
            # Also try a randomized approach
            for _ in range(5):
                path, rem = _greedy_path(start, remaining, adj, random.choice(['narrow', 'wide']))
                if best_path is None or len(path) > len(best_path):
                    best_path = path
                    best_remaining = rem
            
            if best_path and len(best_path) >= 2:
                paths.append(best_path)
                remaining = best_remaining
            elif best_path:
                paths.append(best_path)
        
        # Now handle the remaining edges
        # The remaining graph should have all even degrees (or be empty)
        if remaining:
            # Try to decompose into cycles and then break cycles
            cycles = _eulerian_decomposition(remaining, adj)
            
            if cycles:
                for cycle in cycles:
                    if len(cycle) >= 2:
                        # Break cycle into a path by removing one edge
                        # Just use the cycle as a path (remove last edge to make it a path)
                        if len(cycle) > 1:
                            paths.append(cycle)
                            for i in range(len(cycle) - 1):
                                remaining.discard((min(cycle[i], cycle[i+1]), max(cycle[i], cycle[i+1])))
            
            # Any remaining edges become individual paths
            for e in list(remaining):
                paths.append(list(e))
        
        # Remove any degenerate paths (single vertex)
        paths = [p for p in paths if len(p) >= 2]
        
        # Verify and fix: ensure all edges are covered exactly once
        used_edges = set()
        valid_paths = []
        for path in paths:
            path_edges = set()
            valid = True
            for i in range(len(path) - 1):
                e = (min(path[i], path[i+1]), max(path[i], path[i+1]))
                if e in used_edges or e not in edges:
                    valid = False
                    break
                path_edges.add(e)
            if valid:
                valid_paths.append(path)
                used_edges.update(path_edges)
        
        # Add any missing edges as single-edge paths
        missing = edges - used_edges
        for e in missing:
            valid_paths.append(list(e))
        
        return valid_paths

    def _improve_by_merging(paths, edges, max_time=None):
        """Try to merge paths to reduce count."""
        if max_time is None:
            max_time = deadline - 3
        
        edges = set(edges)
        improved = True
        while improved and time.time() < max_time:
            improved = False
            n = len(paths)
            for i in range(n):
                if time.time() > max_time:
                    break
                for j in range(i + 1, n):
                    if time.time() > max_time:
                        break
                    pi = paths[i]
                    pj = paths[j]
                    
                    # Try all 4 combinations of reversing
                    for rev_i in [False, True]:
                        for rev_j in [False, True]:
                            a = list(reversed(pi)) if rev_i else list(pi)
                            b = list(reversed(pj)) if rev_j else list(pj)
                            
                            # Try connecting end of a to start of b
                            if a[-1] == b[0]:
                                merged = a + b[1:]
                                if _is_valid_path(merged, edges):
                                    paths[i] = merged
                                    del paths[j]
                                    improved = True
                                    break
                            # Try connecting end of b to start of a
                            if b[-1] == a[0]:
                                merged = b + a[1:]
                                if _is_valid_path(merged, edges):
                                    paths[i] = merged
                                    del paths[j]
                                    improved = True
                                    break
                    if improved:
                        break
                if improved:
                    break
        
        return paths

    def _is_valid_path(path, edges):
        """Check if path is simple and uses only edges from the graph."""
        if len(set(path)) != len(path):
            return False
        for i in range(len(path) - 1):
            e = (min(path[i], path[i+1]), max(path[i], path[i+1]))
            if e not in edges:
                return False
        return True

    # Process each graph
    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    for spec in specs:
        edges = instance(*spec)
        
        if time.time() < deadline - 5:
            paths = _path_decomposition(edges, max_time=deadline - 5)
            if time.time() < deadline - 3:
                paths = _improve_by_merging(paths, edges, max_time=deadline - 3)
        else:
            paths = [list(e) for e in edges]
        
        results.append(paths)
    
    return results
# EVOLVE-BLOCK-END
