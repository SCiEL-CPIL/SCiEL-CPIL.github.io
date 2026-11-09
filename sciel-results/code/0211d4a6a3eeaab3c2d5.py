"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 140

    from random import Random
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_eulerian_circuit(adj, n):
        """Hierholzer's algorithm. adj: dict v -> list of (neighbor, edge_id)"""
        used = set()
        circuit = []
        stack = []
        
        # Find start vertex
        start = 0
        for v in range(n):
            if adj[v]:
                start = v
                break
        
        stack.append(start)
        while stack:
            u = stack[-1]
            found = False
            for v, eid in adj[u]:
                if eid not in used:
                    used.add(eid)
                    stack.append(v)
                    found = True
                    break
            if not found:
                circuit.append(stack.pop())
        
        circuit.reverse()
        return circuit, used

    def decompose_graph(edges):
        n = max(max(u, v) for u, v in edges) + 1 if edges else 0
        if not edges:
            return []
        
        # Build adjacency with edge IDs
        adj = defaultdict(list)
        degree = [0] * n
        for eid, (u, v) in enumerate(edges):
            adj[u].append((v, eid))
            adj[v].append((u, eid))
            degree[u] += 1
            degree[v] += 1
        
        odd_verts = [v for v in range(n) if degree[v] % 2 == 1]
        k = len(odd_verts) // 2
        
        if k == 0:
            # Eulerian circuit
            circuit, _ = find_eulerian_circuit(dict(adj), n)
            if circuit and len(circuit) == len(edges) + 1:
                return [circuit]
            return [[list(e)] for e in edges]
        
        # Pair up odd vertices
        # Simple pairing: just pair consecutive odd vertices
        pairs = [(odd_verts[i], odd_verts[i + 1]) for i in range(0, len(odd_verts), 2)]
        
        # Add dummy edges
        dummy_edges = []
        for i, (u, v) in enumerate(pairs):
            dummy_id = len(edges) + i
            dummy_edges.append((u, v, dummy_id))
            adj[u].append((v, dummy_id))
            adj[v].append((u, dummy_id))
        
        # Find Eulerian circuit
        circuit, _ = find_eulerian_circuit(dict(adj), n)
        if not circuit or len(circuit) != len(edges) + k + 1:
            return [[list(e)] for e in edges]
        
        # Now split the circuit at dummy edges
        # The circuit is a vertex sequence. Between consecutive vertices, there's an edge.
        # We need to identify which edges are dummy.
        
        # Reconstruct edge sequence from vertex circuit
        # For each consecutive pair (circuit[i], circuit[i+1]), find the edge ID
        # There might be multiple edges between the same pair, so we need to be careful.
        
        # Better: redo Hierholzer to track edge IDs in the circuit
        used = set()
        stack = [0]
        for v in range(n):
            if adj[v]:
                stack = [v]
                break
        
        edge_circuit = []
        vertex_circuit = []
        
        while stack:
            u = stack[-1]
            found = False
            for v, eid in adj[u]:
                if eid not in used:
                    used.add(eid)
                    edge_circuit.append(eid)
                    stack.append(v)
                    found = True
                    break
            if not found:
                vertex_circuit.append(stack.pop())
        
        vertex_circuit.reverse()
        edge_circuit.reverse()
        
        # Now split at dummy edge IDs
        dummy_ids = set(eid for _, _, eid in dummy_edges)
        
        paths = []
        current_path = [vertex_circuit[0]]
        
        for i, eid in enumerate(edge_circuit):
            # The edge eid connects vertex_circuit[i] and vertex_circuit[i+1]
            u = vertex_circuit[i]
            v = vertex_circuit[i + 1]
            current_path.append(v)
            
            if eid in dummy_ids:
                # End of path
                if len(current_path) > 1:
                    paths.append(current_path)
                current_path = [v]
        
        if len(current_path) > 1:
            paths.append(current_path)
        
        # Verify
        if not paths:
            return [[list(e)] for e in edges]
        
        covered = set()
        valid = True
        for p in paths:
            # Check simple
            if len(set(p)) != len(p):
                valid = False
                break
            for i in range(len(p) - 1):
                a, b = p[i], p[i + 1]
                edge = tuple(sorted((a, b)))
                if edge in covered:
                    valid = False
                    break
                covered.add(edge)
            if not valid:
                break
        if not valid or covered != set(tuple(sorted(e)) for e in edges):
            return [[list(e)] for e in edges]
        
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        edges = instance(*spec)
        paths = decompose_graph(edges)
        results.append(paths)
    
    return results
# EVOLVE-BLOCK-END
