"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    from collections import defaultdict
    
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def find_eulerian_circuit(n, edge_list):
        """Find Eulerian circuit in a multigraph. 
        edge_list: list of (u,v) tuples.
        Returns list of vertices in circuit order, or None if no Eulerian circuit.
        """
        # Build adjacency with edge indices
        adj = defaultdict(list)  # adj[v] = list of (neighbor, edge_index)
        for idx, (u, v) in enumerate(edge_list):
            adj[u].append((v, idx))
            adj[v].append((u, idx))
        
        # Check all degrees even
        for v in range(n):
            if len(adj[v]) % 2 != 0:
                return None
        
        # Find start
        start = None
        for v in range(n):
            if adj[v]:
                start = v
                break
        if start is None:
            return []
        
        # Hierholzer's
        used = [False] * len(edge_list)
        stack = [start]
        circuit = []
        
        while stack:
            v = stack[-1]
            # Find an unused edge from v
            found = False
            for i, (nb, eidx) in enumerate(adj[v]):
                if not used[eidx]:
                    used[eidx] = True
                    stack.append(nb)
                    found = True
                    break
            if not found:
                circuit.append(stack.pop())
        
        circuit.reverse()
        
        # Verify
        if len(circuit) != len(edge_list) + 1:
            return None
        
        # Verify all edges used exactly once
        edge_count = defaultdict(int)
        for i in range(len(circuit) - 1):
            u, v = circuit[i], circuit[i + 1]
            edge_count[(min(u,v), max(u,v))] += 1
        for u, v in edge_list:
            if edge_count[(min(u,v), max(u,v))] != 1:
                return None
        
        return circuit
    
    def decompose_into_paths(n, edges):
        """Decompose graph into minimum number of edge-disjoint paths."""
        degree = defaultdict(int)
        adj = defaultdict(list)
        for u, v in edges:
            degree[u] += 1
            degree[v] += 1
            adj[u].append(v)
            adj[v].append(u)
        
        odd_vertices = [v for v in range(n) if degree[v] % 2 == 1]
        k = len(odd_vertices) // 2
        
        if k == 0:
            circuit = find_eulerian_circuit(n, edges)
            if circuit:
                return [circuit]
            else:
                return [[u, v] for u, v in edges]
        
        # Pair odd vertices
        dummy_pairs = []
        for i in range(0, k * 2, 2):
            a, b = odd_vertices[i], odd_vertices[i + 1]
            dummy_pairs.append((a, b))
        
        # Build augmented edge list
        all_edges = list(edges)
        for a, b in dummy_pairs:
            all_edges.append((a, b))
        
        circuit = find_eulerian_circuit(n, all_edges)
        if not circuit:
            return [[u, v] for u, v in edges]
        
        # Split at dummy edges
        dummy_set = set()
        for a, b in dummy_pairs:
            dummy_set.add((min(a, b), max(a, b)))
        
        paths = []
        current_path = [circuit[0]]
        for i in range(len(circuit) - 1):
            u, v = circuit[i], circuit[i + 1]
            key = (min(u, v), max(u, v))
            current_path.append(v)
            if key in dummy_set:
                paths.append(current_path)
                current_path = [v]
        
        if current_path:
            paths.append(current_path)
        
        paths = [p for p in paths if len(p) >= 2]
        
        return paths
    
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        n = spec[0]
        paths = decompose_into_paths(n, edges)
        results.append(paths)
    
    return results
# EVOLVE-BLOCK-END

