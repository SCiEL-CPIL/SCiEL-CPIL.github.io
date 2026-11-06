import numpy as np
from random import Random
import scipy.optimize as opt

def construct():
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = instance(*spec)
        n = spec[0]
        
        # Build adjacency list
        adj = {i: set() for i in range(n)}
        for (u, v) in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Enumerate simple cycles with vertex paths
        # Cycle represented as tuple of vertices (v0, v1, ..., vk-1)
        # where v0 is the smallest vertex.
        cycles = []
        
        def find_paths(start, end, min_vertex, forbidden_edge=None):
            paths = []
            stack = [(start, [start])]
            while stack:
                curr, path = stack.pop()
                if curr == end:
                    if len(path) > 1:
                        paths.append(path)
                    continue
                for neighbor in adj[curr]:
                    if neighbor < min_vertex:
                        continue
                    if neighbor in path:
                        continue
                    if forbidden_edge and tuple(sorted((curr, neighbor))) == forbidden_edge:
                        continue
                    stack.append((neighbor, path + [neighbor]))
            return paths

        for u in range(n):
            for v in adj[u]:
                if v <= u:
                    continue
                edge_to_remove = (u, v)
                paths = find_paths(u, v, u, forbidden_edge=edge_to_remove)
                for path in paths:
                    # path is [u, ..., v]
                    # Cycle vertices: u, ..., v
                    # We store the path as the cycle representation
                    # The edges are (path[i], path[i+1]) and (path[-1], path[0])
                    cycles.append(tuple(path))
        
        # Remove duplicates (though unlikely with min vertex constraint)
        cycles = list(set(cycles))
        
        if not cycles:
            results.append([list(e) for e in edges])
            continue
            
        num_edges = len(edges)
        num_cycles = len(cycles)
        
        # Map edge tuple to index
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        
        # Build incidence matrix A (Edges x Cycles)
        A = np.zeros((num_edges, num_cycles), dtype=np.int8)
        for c_idx, cycle in enumerate(cycles):
            # Add edges between consecutive vertices
            for i in range(len(cycle) - 1):
                e = tuple(sorted((cycle[i], cycle[i+1])))
                A[edge_to_idx[e], c_idx] = 1
            # Add closing edge
            e = tuple(sorted((cycle[-1], cycle[0])))
            A[edge_to_idx[e], c_idx] = 1
            
        # ILP: Maximize sum(x) subject to A @ x <= 1, x in {0, 1}
        c = np.ones(num_cycles)
        constraints = (A, np.ones(num_edges))
        integrality = np.ones(num_cycles)
        bounds = [(0, 1)] * num_cycles
        
        try:
            res = opt.milp(
                c=-c, 
                constraints=constraints, 
                integrality=integrality, 
                bounds=bounds
            )
            
            if res.status == 0: # Optimal
                x = res.x
                selected_cycles = [cycles[i] for i in range(num_cycles) if x[i] > 0.5]
                
                covered_edges = set()
                for cycle in selected_cycles:
                    for i in range(len(cycle) - 1):
                        covered_edges.add(tuple(sorted((cycle[i], cycle[i+1]))))
                    covered_edges.add(tuple(sorted((cycle[-1], cycle[0]))))
                    
                pieces = []
                # Add cycles
                for cycle in selected_cycles:
                    pieces.append(list(cycle))
                
                # Add remaining edges
                for e in edges:
                    if e not in covered_edges:
                        pieces.append(list(e))
                        
                results.append(pieces)
            else:
                # Fallback
                results.append([list(e) for e in edges])
        except Exception:
            results.append([list(e) for e in edges])
            
    return results
