"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140  # 10s safety margin before 150s

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_eulerian_circuit(adj, start):
        """Find Eulerian circuit using Hierholzer's algorithm.
        adj: dict mapping vertex to list of neighbors (with edge indices for tracking)
        Returns list of vertices in the circuit.
        """
        # Use a stack-based approach
        # adj should be: vertex -> list of (neighbor, edge_id) pairs
        # We need to track which edges have been used
        
        # Convert to adjacency with edge tracking
        # We'll use a different approach: maintain adjacency as list of lists,
        # and mark used edges by removing them
        
        # For Hierholzer's, we need multigraph adjacency
        # Let's use a cleaner approach with edge IDs
        
        # Build adjacency as: vertex -> list of (neighbor, edge_id)
        # We'll use a dictionary for used edges
        used_edges = set()
        
        # Stack for DFS
        stack = [start]
        circuit = []
        
        while stack:
            v = stack[-1]
            # Find an unused edge from v
            found = False
            for i, (nb, eid) in enumerate(adj[v]):
                if eid not in used_edges:
                    used_edges.add(eid)
                    stack.append(nb)
                    found = True
                    break
            if not found:
                circuit.append(stack.pop())
        
        circuit.reverse()
        return circuit

    def min_path_cover_eulerian(edges, n):
        """Find minimum edge-disjoint path cover using Eulerian circuit technique.
        
        Minimum number of paths = max(1, num_odd_vertices // 2)
        """
        if not edges:
            return []
        
        # Compute degrees
        degree = {i: 0 for i in range(n)}
        for u, v in edges:
            degree[u] += 1
            degree[v] += 1
        
        odd_verts = [v for v in range(n) if degree[v] % 2 == 1]
        num_paths = max(1, len(odd_verts) // 2)
        
        if num_paths == 1 and len(odd_verts) == 0:
            # All even degrees: single Eulerian circuit
            # Build adjacency with edge IDs
            adj = {i: [] for i in range(n)}
            for eid, (u, v) in enumerate(edges):
                adj[u].append((v, eid))
                adj[v].append((u, eid))
            
            # Find Eulerian circuit starting from any vertex with edges
            start = next(iter(adj.keys()))
            while not adj[start]:
                start = next(iter(adj.keys()))
            
            circuit = find_eulerian_circuit(adj, start)
            # Convert circuit to path (remove last vertex which is same as first)
            if len(circuit) >= 2 and circuit[0] == circuit[-1]:
                return [circuit[:-1]]
            else:
                return [circuit]
        
        # Pair up odd-degree vertices
        # Greedy pairing
        pairs = []
        for i in range(0, len(odd_verts), 2):
            if i + 1 < len(odd_verts):
                pairs.append((odd_verts[i], odd_verts[i+1]))
        
        # Create augmented edge list: original edges + dummy edges
        # Edge IDs: 0 to len(edges)-1 for original, len(edges) to len(edges)+len(pairs)-1 for dummy
        all_edges = list(edges)
        dummy_edges = []
        for a, b in pairs:
            dummy_edges.append((a, b))
        
        total_edges = len(all_edges) + len(dummy_edges)
        
        # Build adjacency with edge IDs
        adj = {i: [] for i in range(n)}
        for eid in range(len(all_edges)):
            u, v = all_edges[eid]
            adj[u].append((v, eid))
            adj[v].append((u, eid))
        for eid in range(len(all_edges), total_edges):
            u, v = dummy_edges[eid - len(all_edges)]
            adj[u].append((v, eid))
            adj[v].append((u, eid))
        
        # Find Eulerian circuit in augmented graph
        # Start from a vertex with edges
        start = 0
        while not adj[start]:
            start += 1
        
        circuit = find_eulerian_circuit(adj, start)
        
        # Now break the circuit at dummy edge positions
        # The circuit is a list of vertices. Consecutive vertices in the circuit
        # correspond to edges. We need to identify which transitions are dummy edges.
        
        # For each consecutive pair in the circuit, determine the edge ID
        # Then split at positions where the edge is a dummy edge
        
        # Build a mapping from (u,v) pair to edge ID
        # Note: an edge (u,v) can appear as (u,v) or (v,u) in the circuit
        edge_map = {}
        for eid, (u, v) in enumerate(all_edges):
            edge_map[(min(u,v), max(u,v))] = eid
        for i, (a, b) in enumerate(pairs):
            eid = len(all_edges) + i
            edge_map[(min(a,b), max(a,b))] = eid
        
        # Find positions in the circuit where a dummy edge is used
        # Circuit: v0, v1, v2, ..., vk (where k+1 is the length)
        # Edge between vi and vi+1 has ID edge_map[(min(vi,vi+1), max(vi,vi+1))]
        
        split_positions = []  # Indices i such that edge between circuit[i] and circuit[i+1] is dummy
        for i in range(len(circuit) - 1):
            u, v = circuit[i], circuit[i+1]
            key = (min(u,v), max(u,v))
            if key in edge_map:
                eid = edge_map[key]
                if eid >= len(all_edges):  # It's a dummy edge
                    split_positions.append(i)
        
        # Also check the "wrap-around" edge from last to first
        # In an Eulerian circuit, the last vertex connects back to the first
        # But in our circuit representation, the circuit is v0, v1, ..., vk where vk = v0
        # So the edge from vk-1 to vk is the last edge, and there's no explicit wrap-around
        # Wait, let me reconsider. The circuit from Hierholzer's gives us the sequence of vertices.
        # The edges are between consecutive vertices. If the circuit is a cycle, then
        # the last vertex equals the first, and the edges are (v0,v1), (v1,v2), ..., (vk-1, vk=v0)
        
        # So we have len(circuit)-1 edges in the circuit
        # We need to split at dummy edge positions
        
        # Create the paths by splitting
        paths = []
        if not split_positions:
            # No dummy edges to break at (shouldn't happen if we have odd vertices)
            # Just return the whole circuit as one path
            path = circuit[:-1] if circuit[0] == circuit[-1] else circuit
            if len(path) >= 2:
                paths.append(path)
        else:
            # Split at each dummy edge position
            # The dummy edge at position i means we break between circuit[i] and circuit[i+1]
            # So one path ends at circuit[i] and the next starts at circuit[i+1]
            
            # We need to extract segments
            # Positions: 0, 1, ..., len(circuit)-1
            # Break after position i (i.e., between i and i+1)
            
            prev = 0
            for pos in split_positions:
                # Path from circuit[prev] to circuit[pos]
                segment = circuit[prev:pos+1]
                if len(segment) >= 2:
                    paths.append(segment)
                prev = pos + 1
            
            # Last segment: from circuit[prev] to circuit[-1]
            # But wait, the circuit is a cycle. If circuit[0] == circuit[-1], we need to handle that.
            # Actually, in our representation, the circuit from Hierholzer's is a closed walk,
            # so circuit[0] should equal circuit[-1].
            
            # The last segment goes from circuit[prev] to circuit[-1] = circuit[0]
            # But we also need to include the part from circuit[0] to... hmm, this is tricky.
            
            # Let me think again. The circuit is v0, v1, ..., vk where vk = v0.
            # We split at positions where a dummy edge occurs.
            # If we split at position i, we break the cycle into a path.
            
            # Actually, let's just collect all segments between consecutive splits
            # including the wrap-around
            
            all_splits = sorted(set(split_positions))
            if not all_splits:
                path = circuit[:-1] if circuit[0] == circuit[-1] else circuit
                if len(path) >= 2:
                    paths.append(path)
            else:
                # Insert 0 as a split point for the wrap-around
                # The segments are:
                # [circuit[0] to circuit[split0]], [circuit[split0+1] to circuit[split1]], ..., [circuit[last_split+1] to circuit[-1]=circuit[0]]
                
                # But we need to be careful: the last segment wraps around
                # Let's just collect all segments
                seg_starts = [0] + [s + 1 for s in all_splits]
                seg_ends = all_splits + [len(circuit) - 1]
                
                for s, e in zip(seg_starts, seg_ends):
                    segment = circuit[s:e+1]
                    if len(segment) >= 2:
                        paths.append(segment)
        
        # Filter out any paths that don't have at least 2 vertices
        paths = [p for p in paths if len(p) >= 2]
        
        # Ensure we have exactly num_paths paths
        # If we have fewer, something went wrong
        if len(paths) == 0:
            # Fallback: return each edge as a path
            return [list(e) for e in edges]
        
        return paths

    def greedy_path_cover(edges, n):
        """Fallback greedy path cover."""
        if not edges:
            return []
        
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        remaining_edges = set(edges)
        paths = []
        
        while remaining_edges:
            sub_adj = {i: set() for i in range(n)}
            for u, v in remaining_edges:
                sub_adj[u].add(v)
                sub_adj[v].add(u)
            
            active_verts = [v for v in range(n) if sub_adj[v]]
            if not active_verts:
                break
            
            best_path = None
            best_len = 0
            
            start_candidates = active_verts if len(active_verts) <= 10 else \
                              sorted(active_verts, key=lambda v: len(sub_adj[v]))[:10]
            
            for start in start_candidates:
                path = [start]
                visited = {start}
                current = start
                
                while True:
                    candidates = [nb for nb in sub_adj[current] if nb not in visited]
                    if not candidates:
                        break
                    best_next = min(candidates, key=lambda nb: len(sub_adj[nb]))
                    path.append(best_next)
                    visited.add(best_next)
                    current = best_next
                
                if len(path) > best_len:
                    best_len = len(path)
                    best_path = path
            
            if best_path is None or len(best_path) < 2:
                e = next(iter(remaining_edges))
                best_path = list(e)
            
            for i in range(len(best_path) - 1):
                e1 = tuple(sorted((best_path[i], best_path[i+1])))
                remaining_edges.discard(e1)
            
            paths.append(best_path)
        
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > DEADLINE:
            while len(results) < 3:
                edges = instance(*spec)
                results.append([list(e) for e in edges])
            break
        
        edges = instance(*spec)
        n = spec[0]
        
        # Try Eulerian-based minimum path cover
        paths = min_path_cover_eulerian(edges, n)
        
        # Validate: check that paths are valid
        valid = True
        used_edge_set = set()
        for path in paths:
            if len(path) < 2:
                valid = False
                break
            # Check simplicity
            if len(set(path)) != len(path):
                valid = False
                break
            for i in range(len(path) - 1):
                e = tuple(sorted((path[i], path[i+1])))
                if e not in set(edges):
                    valid = False
                    break
                if e in used_edge_set:
                    valid = False
                    break
                used_edge_set.add(e)
            if not valid:
                break
        
        if valid and used_edge_set == set(edges):
            results.append(paths)
        else:
            # Fallback to greedy
            paths = greedy_path_cover(edges, n)
            results.append(paths)
    
    return results
# EVOLVE-BLOCK-END
