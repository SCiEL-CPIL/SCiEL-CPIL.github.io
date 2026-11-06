import sys
import time
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    timeout = 140.0  # Safety margin

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_graph(edges):
        if not edges:
            return []
        
        # Build adjacency list and degree map
        adj = defaultdict(list)
        degree = defaultdict(int)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            degree[u] += 1
            degree[v] += 1
        
        # Identify odd-degree vertices
        odd_vertices = [v for v in degree if degree[v] % 2 != 0]
        
        # Pair odd vertices
        # We add dummy edges to make all degrees even
        # The number of dummy edges is len(odd_vertices) // 2
        dummy_edges = []
        if len(odd_vertices) > 0:
            # Pair them up
            for i in range(0, len(odd_vertices), 2):
                u = odd_vertices[i]
                v = odd_vertices[i+1]
                dummy_edges.append((u, v))
        
        # Create augmented graph
        # We need to distinguish real edges from dummy edges for splitting
        # We'll keep a list of all edges (real + dummy)
        all_edges = list(edges) + dummy_edges
        
        adj_aug = defaultdict(list)
        for u, v in all_edges:
            adj_aug[u].append(v)
            adj_aug[v].append(u)
        
        # Find Eulerian Circuit using Hierholzer's Algorithm
        # We need to track which edges are used
        # Since vertices can have multiple edges, we track (u, v) pairs
        # But (u,v) might appear multiple times? No, simple graph.
        # So (u,v) is unique.
        
        used_edges = set()
        circuit = []
        
        # Stack for Hierholzer
        # Start from any vertex
        start_node = all_edges[0][0] if all_edges else None
        if start_node is None:
            return []
            
        stack = [start_node]
        while stack:
            v = stack[-1]
            # Find an unused edge from v
            next_node = None
            for u in adj_aug[v]:
                e = tuple(sorted((v, u)))
                if e not in used_edges:
                    next_node = u
                    break
            
            if next_node is not None:
                # Use edge (v, next_node)
                e = tuple(sorted((v, next_node)))
                used_edges.add(e)
                stack.append(next_node)
            else:
                # Backtrack
                circuit.append(stack.pop())
        
        circuit.reverse()
        
        # Split the circuit at dummy edges
        # The circuit is a list of vertices: v0, v1, ..., vk, v0
        # We need to identify where dummy edges are.
        # A dummy edge is (u, v) where (u, v) is in dummy_edges.
        
        dummy_set = set(tuple(sorted(e)) for e in dummy_edges)
        
        paths = []
        current_path_vertices = []
        
        if not circuit:
            return []
            
        # The circuit is closed.
        # We iterate through the edges of the circuit.
        # If an edge is a dummy edge, we split here.
        
        # Note: The circuit might contain dummy edges.
        # We traverse the circuit.
        
        # We need to be careful: the circuit is a list of vertices.
        # Edges are (circuit[i], circuit[i+1]).
        
        # We will extract walks between dummy edges.
        
        walk_vertices = []
        for i in range(len(circuit)):
            u = circuit[i]
            v = circuit[(i + 1) % len(circuit)]
            
            # Add current vertex to walk
            if not walk_vertices:
                walk_vertices.append(u)
            
            # Check if edge (u, v) is a dummy edge
            e = tuple(sorted((u, v)))
            if e in dummy_set:
                # This edge is a dummy edge.
                # We should NOT include this edge in the path.
                # The path ends at u.
                # The next path will start at v.
                
                # Decompose walk_vertices into simple paths
                if walk_vertices:
                    paths.extend(decompose_walk_to_simple_paths(walk_vertices))
                
                # Start new walk at v
                walk_vertices = [v]
            else:
                # Real edge.
                # Include v in the walk
                walk_vertices.append(v)
        
        # Handle the last walk
        # Since the circuit is closed, the last walk connects back to the start?
        # Actually, if we split at dummy edges, the last segment ends at the vertex before the first dummy edge?
        # Let's trace:
        # Circuit: A - B (dummy) - C - D - A
        # i=0: A->B (dummy). walk=[A]. Split. paths from [A] (empty). walk=[B].
        # i=1: B->C (real). walk=[B, C].
        # i=2: C->D (real). walk=[B, C, D].
        # i=3: D->A (real). walk=[B, C, D, A].
        # End loop.
        # Last walk: [B, C, D, A].
        # But wait, the edge D->A is real.
        # Is A the start of the first walk?
        # The first walk was [A] (from A to B dummy).
        # The last walk ends at A.
        # So we have a path from B to A.
        # And the first part was A (single vertex, no edges).
        # This seems correct.
        
        if walk_vertices:
            paths.extend(decompose_walk_to_simple_paths(walk_vertices))
        
        return paths

    def decompose_walk_to_simple_paths(walk):
        """
        Decompose a walk (list of vertices) into simple paths.
        The walk represents a sequence of edges.
        We greedily split whenever a vertex is repeated.
        """
        if len(walk) < 2:
            # No edges
            return []
        
        simple_paths = []
        current_path = [walk[0]]
        visited = {walk[0]}
        
        for i in range(1, len(walk)):
            v = walk[i]
            if v in visited:
                # We have a cycle or repeat.
                # The current path is simple (up to previous vertex).
                # We output it.
                if len(current_path) > 1:
                    simple_paths.append(current_path)
                # Start new path at v
                current_path = [v]
                visited = {v}
            else:
                current_path.append(v)
                visited.add(v)
        
        # Output the last path
        if len(current_path) > 1:
            simple_paths.append(current_path)
            
        return simple_paths

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    for spec in specs:
        if time.time() - start_time > timeout:
            # Fallback: return single edges if we run out of time
            edges = instance(*spec)
            results.append([[u, v] for u, v in edges])
            continue
            
        edges = instance(*spec)
        paths = decompose_graph(edges)
        
        # Verify and fallback if necessary
        # Check if paths cover all edges and are edge-disjoint
        edge_count = len(edges)
        used_edges = set()
        valid = True
        for p in paths:
            if len(p) < 2:
                valid = False
                break
            for i in range(len(p) - 1):
                e = tuple(sorted((p[i], p[i+1])))
                if e in used_edges:
                    valid = False
                    break
                if e not in set(map(tuple, map(sorted, edges))):
                    valid = False
                    break
                used_edges.add(e)
            if not valid:
                break
        
        if not valid or len(used_edges) != edge_count:
            # Fallback to single edges
            paths = [[u, v] for u, v in edges]
            
        results.append(paths)
        
    return results

if __name__ == "__main__":
    # Test run
    res = construct()
    total_paths = sum(len(r) for r in res)
    print(f"Total paths: {total_paths}")
