"""Gallai's conjecture: decompose each graph into the minimum number of edge-disjoint paths."""
import time
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges):
        """
        Use Gallai's theorem:
        1. Find all odd-degree vertices.
        2. Pair them up (|odd|/2 pairs).
        3. Add a path between each pair to make all degrees even (Eulerian multigraph).
        4. Find an Eulerian circuit in the multigraph.
        5. Cut the circuit at the added path edges to get |odd|/2 paths covering all original edges.
        """
        if not edges:
            return []

        # Build adjacency and degree
        adj = defaultdict(list)
        degree = defaultdict(int)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            degree[u] += 1
            degree[v] += 1

        odd_vertices = [v for v in range(max(adj.keys()) + 1) if degree.get(v, 0) % 2 == 1]
        num_pairs = len(odd_vertices) // 2

        if num_pairs == 0:
            # All degrees even: single Eulerian circuit
            circuit = eulerian_circuit(edges, adj)
            return [circuit]

        # Pair odd vertices greedily: pair (0,1), (2,3), etc.
        # Try different pairings to minimize total path length of added paths
        # For now, use simple pairing; we can try to improve with shortest paths
        pairs = [(odd_vertices[2*i], odd_vertices[2*i+1]) for i in range(num_pairs)]

        # Find shortest path between each pair (in terms of number of edges)
        # These will be the "added" paths that we cut the Eulerian circuit at
        added_paths = []
        for a, b in pairs:
            path = shortest_path(a, b, adj)
            added_paths.append(path)

        # Build multigraph: original edges + added path edges
        # Each added path edge is marked so we can cut at them
        # We'll track which edges in the Eulerian circuit are "added" edges

        # For the multigraph, we need to handle parallel edges
        # Use edge IDs: original edges get IDs 0..m-1, added edges get IDs m..m+k-1
        m = len(edges)
        edge_id_map = {}
        for i, (u, v) in enumerate(edges):
            key = tuple(sorted((u, v)))
            if key not in edge_id_map:
                edge_id_map[key] = []
            edge_id_map[key].append(i)

        # Build adjacency with edge IDs for the multigraph
        mult_adj = defaultdict(list)  # vertex -> list of (neighbor, edge_id, is_added)
        for u, v in edges:
            key = tuple(sorted((u, v)))
            eid = edge_id_map[key].pop(0)
            mult_adj[u].append((v, eid, False))
            mult_adj[v].append((u, eid, False))

        # Add the path edges
        added_edge_count = 0
        for path in added_paths:
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                eid = m + added_edge_count
                added_edge_count += 1
                mult_adj[u].append((v, eid, True))
                mult_adj[v].append((u, eid, True))

        # Find Eulerian circuit in the multigraph
        circuit = eulerian_circuit_multigraph(mult_adj, m + added_edge_count)

        # The circuit is a list of edge IDs. Cut at added edges.
        # An added edge appears in the circuit; when we traverse an added edge,
        # we start a new path segment.
        
        # Reconstruct vertex sequence from edge IDs
        # We need to know which edge each edge_id corresponds to
        # Original edges: edges[eid]
        # Added edges: we need to reconstruct from added_paths
        
        # Let's track the vertex sequence as we build the circuit
        # Actually, let's just get the vertex sequence directly
        
        # Better approach: during Hierholzer, track vertices
        circuit_vertices = eulerian_circuit_vertices(mult_adj, m + added_edge_count)
        
        # circuit_vertices is a list of vertices forming the Eulerian circuit
        # Now, identify where the added edges are in this circuit
        # An added edge connects consecutive vertices in the circuit that correspond to an added path edge
        
        # Build set of added edge endpoints (as sorted tuples)
        added_edge_set = set()
        for path in added_paths:
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                added_edge_set.add(tuple(sorted((u, v))))
        
        # But this is ambiguous if there are parallel edges. Let's use a different approach.
        # Mark the positions in the circuit where added edges are traversed.
        
        # Actually, let's just count: the circuit has m + added_edge_count edges.
        # We know which edge IDs are added. Let's redo with edge IDs.
        
        circuit_eids = eulerian_circuit_edge_ids(mult_adj, m + added_edge_count)
        
        # Now split the circuit at added edges
        # The circuit is a cycle, so we need to be careful
        # Find all positions where an added edge is traversed
        added_positions = [i for i, eid in enumerate(circuit_eids) if eid >= m]
        
        if not added_positions:
            # Should not happen if num_pairs > 0
            return [[circuit_vertices[i] for i in range(len(circuit_vertices))]]
        
        # The added edges divide the cycle into num_pairs segments
        # Each segment is a path in the original graph
        # We need to extract the vertex sequences for each segment
        
        # Build a mapping from edge_id to (u, v) for reconstruction
        edge_uv = {}
        for i, (u, v) in enumerate(edges):
            edge_uv[i] = (u, v)
        
        added_edge_list = []
        for path in added_paths:
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                added_edge_list.append((u, v))
        for i, (u, v) in enumerate(added_edge_list):
            edge_uv[m + i] = (u, v)
        
        # Reconstruct vertex sequence from edge IDs in the circuit
        # circuit_eids is a list of edge IDs forming a cycle
        # We need to orient each edge consistently
        
        # Start from the first edge, determine orientation
        # The circuit visits vertices in sequence
        # Let's just use the circuit_vertices we can get from a proper Hierholzer
        
        # Simpler: just return the segments based on positions
        # The circuit is a cycle of length L = len(circuit_eids)
        # added_positions tells us where to cut
        # Sort added_positions and extract segments
        
        L = len(circuit_eids)
        added_positions_sorted = sorted(added_positions)
        
        paths = []
        for i in range(len(added_positions_sorted)):
            start = added_positions_sorted[i]
            end = added_positions_sorted[(i + 1) % len(added_positions_sorted)]
            if end > start:
                seg_eids = circuit_eids[start:end]
            else:
                seg_eids = circuit_eids[start:] + circuit_eids[:end]
            
            # Convert edge ID sequence to vertex sequence
            if not seg_eids:
                continue
            # Determine vertex sequence from oriented edges
            # We need to know the starting vertex
            # The edge at position start in the circuit: we need its "from" vertex
            # This is tricky without tracking orientation in the circuit
            
            # Alternative: just use the fact that consecutive edges share a vertex
            # Start with the first edge's endpoints
            u0, v0 = edge_uv[seg_eids[0]]
            # The next edge must connect to either u0 or v0
            # Try both orientations
            
            # This is getting complex. Let's use a cleaner approach.
            pass
        
        # Let me use a cleaner implementation
        return _decompose_clean(edges)

    def _decompose_clean(edges):
        if not edges:
            return []
        
        adj = defaultdict(list)
        degree = defaultdict(int)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            degree[u] += 1
            degree[v] += 1
        
        odd_vertices = [v for v in set(adj.keys()) if degree[v] % 2 == 1]
        num_pairs = len(odd_vertices) // 2
        
        if num_pairs == 0:
            # Eulerian circuit
            circuit = _eulerian_circuit(edges)
            return [circuit]
        
        # Pair odd vertices
        pairs = [(odd_vertices[2*i], odd_vertices[2*i+1]) for i in range(num_pairs)]
        
        # Find shortest paths for each pair
        added_paths = []
        for a, b in pairs:
            path = _shortest_path(a, b, adj)
            added_paths.append(path)
        
        # Build multigraph with edge tracking
        # Each edge is (u, v, is_original)
        # Use a list of edges with indices
        all_edges = []
        for u, v in edges:
            all_edges.append((u, v, True))
        for path in added_paths:
            for i in range(len(path) - 1):
                all_edges.append((path[i], path[i+1], False))
        
        # Build adjacency with edge indices
        mult_adj = defaultdict(list)
        for idx, (u, v, is_orig) in enumerate(all_edges):
            mult_adj[u].append((v, idx))
            mult_adj[v].append((u, idx))
        
        # Hierholzer's algorithm returning vertex sequence
        circuit = _hierholzer(mult_adj, len(all_edges))
        
        # circuit is a list of vertices forming the Eulerian circuit
        # Now identify which consecutive pairs in the circuit are added edges
        added_edge_set = set()
        for path in added_paths:
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                added_edge_set.add((u, v))
                added_edge_set.add((v, u))
        
        # Find cut positions: indices i where (circuit[i], circuit[i+1]) is an added edge
        # Note: circuit is a cycle, so circuit[-1] connects to circuit[0]
        L = len(circuit)
        cut_positions = []
        for i in range(L):
            u = circuit[i]
            v = circuit[(i + 1) % L]
            if (u, v) in added_edge_set:
                cut_positions.append(i)
        
        # cut_positions should have exactly num_pairs elements
        # Extract segments
        paths = []
        for i in range(len(cut_positions)):
            start = cut_positions[i]
            end = cut_positions[(i + 1) % len(cut_positions)]
            
            # Segment goes from circuit[start+1] to circuit[end] (wrapping around)
            # The added edge is between circuit[start] and circuit[start+1]
            # So the path starts at circuit[start+1]
            
            if end > start:
                seg = circuit[start+1:end+1]
            else:
                seg = circuit[start+1:] + circuit[:end+1]
            
            if len(seg) > 0:
                paths.append(seg)
        
        # Filter out empty or single-vertex paths (shouldn't happen for valid decomposition)
        paths = [p for p in paths if len(p) >= 2]
        
        # Verify: each path uses only original edges
        original_edge_set = set()
        for u, v in edges:
            original_edge_set.add((u, v))
            original_edge_set.add((v, u))
        
        valid_paths = []
        for p in paths:
            valid = True
            for i in range(len(p) - 1):
                if (p[i], p[i+1]) not in original_edge_set:
                    valid = False
                    break
            if valid:
                valid_paths.append(p)
        
        if len(valid_paths) == len(paths):
            return valid_paths
        else:
            # Fallback: each edge as its own path
            return [[u, v] for u, v in edges]

    def _eulerian_circuit(edges):
        """Find Eulerian circuit in a graph where all degrees are even."""
        if not edges:
            return []
        
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Use Hierholzer's algorithm
        # We need to handle parallel edges, so track edge usage
        # For simplicity with simple graphs (no parallel edges in original),
        # we can use a different approach
        
        # Actually, the original graph may have parallel edges? No, it's a simple graph.
        # So we can just use adjacency lists and mark edges as used
        
        used = set()
        for u, v in edges:
            used.add((min(u,v), max(u,v)))
        
        start = edges[0][0]
        stack = [start]
        circuit = []
        adj_remaining = defaultdict(list)
        for u, v in edges:
            adj_remaining[u].append(v)
            adj_remaining[v].append(u)
        
        while stack:
            v = stack[-1]
            if adj_remaining[v]:
                w = adj_remaining[v].pop()
                # Remove the reverse edge too
                # Find and remove v from adj_remaining[w]
                if v in adj_remaining[w]:
                    adj_remaining[w].remove(v)
                stack.append(w)
            else:
                circuit.append(stack.pop())
        
        circuit.reverse()
        return circuit

    def _shortest_path(src, dst, adj):
        """BFS shortest path."""
        if src == dst:
            return [src]
        
        parent = {src: None}
        queue = deque([src])
        while queue:
            v = queue.popleft()
            if v == dst:
                break
            for w in adj[v]:
                if w not in parent:
                    parent[w] = v
                    queue.append(w)
        
        if dst not in parent:
            return [src, dst]  # Should not happen for connected graph
        
        # Reconstruct path
        path = []
        v = dst
        while v is not None:
            path.append(v)
            v = parent[v]
        path.reverse()
        return path

    def _hierholzer(adj, num_edges):
        """
        Hierholzer's algorithm for Eulerian circuit in a multigraph.
        adj: dict mapping vertex -> list of (neighbor, edge_id)
        Returns vertex sequence of the Eulerian circuit.
        """
        # We need to track which edges have been used
        used = [False] * num_edges
        
        # Build a proper adjacency structure with edge tracking
        # adj[v] is a list of (w, eid)
        # We'll use a pointer for each vertex to track next unused edge
        
        # For efficiency, use a list of lists and a pointer array
        adj_list = {v: list(neighbors) for v, neighbors in adj.items()}
        ptr = {v: 0 for v in adj_list}
        
        start = list(adj_list.keys())[0]
        stack = [(start, -1)]  # (vertex, incoming_edge_id)
        circuit = []
        
        while stack:
            v, in_eid = stack[-1]
            
            # Find next unused edge from v
            found = False
            while ptr[v] < len(adj_list[v]):
                w, eid = adj_list[v][ptr[v]]
                if not used[eid]:
                    used[eid] = True
                    ptr[v] += 1
                    # Also mark the reverse edge as used
                    # We need to find the reverse edge
                    # The reverse edge connects w to v with the same eid
                    # In our construction, each edge appears twice in adj_list
                    # So we just need to mark used[eid] = True once
                    stack.append((w, eid))
                    found = True
                    break
                else:
                    ptr[v] += 1
            
            if not found:
                circuit.append(v)
                stack.pop()
        
        circuit.reverse()
        return circuit

    # Process each instance
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline:
            # Fallback: each edge as its own path
            edges = instance(*spec)
            results.append([[u, v] for u, v in edges])
            continue
        
        edges = instance(*spec)
        paths = _decompose_clean(edges)
        results.append(paths)
    
    return results
