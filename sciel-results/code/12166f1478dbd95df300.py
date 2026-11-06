import time
import sys
from collections import defaultdict
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_matching(edges, seed_offset=0):
        """
        Path decomposition via maximum matching on the "line graph" of half-edges.
        
        Each edge e=(u,v) has two ends: (u,e) and (v,e).
        We want to pair these ends so that:
        1. Two ends are paired only if they belong to different edges
        2. If end (u,e1) is paired with (u,e2), they share vertex u
        3. The resulting structure forms simple paths (no vertex repeated)
        
        Number of paths = |E| - |matching|
        """
        if not edges:
            return []

        edge_list = list(edges)
        E = len(edge_list)
        
        # Create half-edge nodes: for each edge (u,v), create two nodes: (u, edge_idx) and (v, edge_idx)
        # We'll use integer IDs for half-edges
        half_edges = []  # list of (vertex, edge_idx)
        vertex_to_halves = defaultdict(list)  # vertex -> list of half-edge IDs
        
        for idx, (u, v) in enumerate(edge_list):
            hid_u = len(half_edges)
            half_edges.append((u, idx))
            vertex_to_halves[u].append(hid_u)
            
            hid_v = len(half_edges)
            half_edges.append((v, idx))
            vertex_to_halves[v].append(hid_v)
        
        H = len(half_edges)  # = 2*E
        
        # Build compatibility graph: two half-edges can be matched if:
        # - They belong to different edges
        # - They share the same vertex
        # This is a perfect matching problem on a general graph (blossom needed)
        # But we can use a greedy approach with local search
        
        # For now, use greedy matching with randomization
        best_paths = None
        best_count = float('inf')
        
        rng = random.Random(42 + seed_offset)
        
        for attempt in range(50):
            # Greedy matching: randomly order half-edges, greedily match compatible pairs
            # that don't create cycles
            
            # Build adjacency: for each half-edge, which other half-edges are compatible
            # Two half-edges (v1, e1) and (v2, e2) are compatible if v1==v2 and e1!=e2
            
            # Use a simpler approach: for each vertex, we have a set of half-edges.
            # We need to pair them up (each half-edge can be in at most one pair).
            # This is a matching within each vertex's half-edge set, but pairs must
            # connect different edges.
            
            # Actually, let's think differently. We have 2E half-edges. We want to
            # find a matching M such that if (v,e1) and (v,e2) are matched, they form
            # a "continuation" at vertex v from edge e1 to edge e2.
            
            # The constraint is that the resulting directed graph (where each edge is
            # split into two directed half-edges, and matchings create connections)
            # should form simple paths.
            
            # Simpler: use the fact that any valid pairing of half-edges (respecting
            # same-vertex, different-edge) gives a path decomposition. The only issue
            # is simplicity (no vertex repeated in a path). For sparse graphs, this is
            # rare.
            
            # Greedy: shuffle half-edges, for each unpaired half-edge, find an unpaired
            # compatible half-edge and pair them.
            
            paired = [False] * H
            pairs = []  # list of (hid1, hid2)
            
            # Shuffle order of processing
            order = list(range(H))
            rng.shuffle(order)
            
            for hid in order:
                if paired[hid]:
                    continue
                v1, e1 = half_edges[hid]
                
                # Find unpaired compatible half-edges
                candidates = []
                for hid2 in vertex_to_halves[v1]:
                    if hid2 == hid or paired[hid2]:
                        continue
                    v2, e2 = half_edges[hid2]
                    if v2 == v1 and e2 != e1:
                        candidates.append(hid2)
                
                if not candidates:
                    continue
                
                # Pick a random candidate (or use heuristic)
                hid2 = rng.choice(candidates)
                paired[hid] = True
                paired[hid2] = True
                pairs.append((hid, hid2))
            
            # Reconstruct paths from the matching
            # Each edge has two half-edges. If both are matched, they connect to other edges.
            # Build a directed graph where each edge is a node, and matchings create
            # directed connections.
            
            # For each edge idx, get its two half-edge IDs
            edge_halves = {}
            for hid, (v, idx) in enumerate(half_edges):
                if idx not in edge_halves:
                    edge_halves[idx] = []
                edge_halves[idx].append(hid)
            
            # For each edge, determine if its half-edges are matched and to what
            # edge_next[idx] = (next_edge_idx, vertex) if the "out" half-edge is matched
            # Actually, let's build the path structure directly.
            
            # Each matched pair (hid1, hid2) means: at vertex v, edge e1 connects to edge e2.
            # This creates a directed relationship.
            
            # Build: for each edge, which edges it connects to at each endpoint
            # edge_connections[idx] = set of (other_idx, vertex)
            
            edge_connections = defaultdict(set)
            for hid1, hid2 in pairs:
                v1, e1 = half_edges[hid1]
                v2, e2 = half_edges[hid2]
                if e1 != e2:  # should always be true
                    edge_connections[e1].add((e2, v1))
                    edge_connections[e2].add((e1, v2))
            
            # Now trace paths. A path starts at an edge where one half-edge is unpaired.
            # Follow connections until reaching an edge with another unpaired half-edge.
            
            # Determine start and end of each edge in terms of half-edges
            # For edge idx with halves [h1, h2]:
            #   If h1 is unpaired, it's a "start" half-edge
            #   If h2 is unpaired, it's an "end" half-edge
            
            path_starts = []  # list of (edge_idx, start_half_hid)
            path_ends = []    # list of (edge_idx, end_half_hid)
            
            for idx in range(E):
                for hid in edge_halves[idx]:
                    if not paired[hid]:
                        # This half-edge is unpaired, so it's an endpoint of a path
                        v, e = half_edges[hid]
                        # Determine if this is a start or end based on convention
                        # For now, just mark it as an endpoint
                        path_starts.append((idx, hid))
            
            # Trace paths from each start
            paths = []
            used_edges = set()
            
            for start_idx, start_hid in path_starts:
                if start_idx in used_edges:
                    continue
                
                # Trace the path
                path_edges = [start_idx]
                current_idx = start_idx
                current_hid = start_hid
                v, _ = half_edges[current_hid]
                
                # Find where this edge connects from the other half-edge
                other_hid = edge_halves[current_idx][1] if edge_halves[current_idx][0] == current_hid else edge_halves[current_idx][0]
                
                if paired[other_hid]:
                    # Find the other half-edge in the pair
                    for hid1, hid2 in pairs:
                        if hid1 == other_hid:
                            next_hid = hid2
                            break
                        elif hid2 == other_hid:
                            next_hid = hid1
                            break
                    else:
                        next_hid = None
                    
                    if next_hid is not None:
                        next_v, next_idx = half_edges[next_hid]
                        if next_idx not in used_edges:
                            path_edges.append(next_idx)
                            current_idx = next_idx
                            current_hid = next_hid
                            v = next_v
                            
                            # Continue tracing
                            while True:
                                used_edges.add(current_idx)
                                other_hid = edge_halves[current_idx][1] if edge_halves[current_idx][0] == current_hid else edge_halves[current_idx][0]
                                
                                if not paired[other_hid]:
                                    break
                                
                                found_next = False
                                for hid1, hid2 in pairs:
                                    if hid1 == other_hid:
                                        next_hid = hid2
                                        found_next = True
                                        break
                                    elif hid2 == other_hid:
                                        next_hid = hid1
                                        found_next = True
                                        break
                                
                                if not found_next:
                                    break
                                
                                next_v, next_idx = half_edges[next_hid]
                                if next_idx in used_edges:
                                    break
                                
                                path_edges.append(next_idx)
                                current_idx = next_idx
                                current_hid = next_hid
                                v = next_v
                            used_edges.add(current_idx)
                
                # Convert edge indices to vertex lists
                # We need to determine the order of vertices in the path
                # This requires tracking the direction through each edge
                # For simplicity, we'll reconstruct by following the connections
                
                # Actually, let's just collect the edges and then order them
                # This is getting complex. Let's use a simpler reconstruction.
                
                if len(path_edges) > 0:
                    # Build vertex sequence from edge sequence
                    # We need to know the orientation of each edge in the path
                    # For now, just store the edge set; reconstruction can be done later
                    paths.append(path_edges)
            
            # This approach is getting too complex for the reconstruction.
            # Let's use a simpler greedy approach that directly builds paths.
            break  # Skip this attempt, use the simpler method below
        
        # Fallback: use greedy longest path with multiple restarts
        best_paths = None
        best_count = float('inf')
        
        for attempt in range(20):
            adj = defaultdict(set)
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            
            remaining_edges = set(tuple(sorted(e)) for e in edges)
            paths = []
            rng_local = random.Random(100 + attempt)
            
            while remaining_edges:
                active_nodes = set()
                for u, v in remaining_edges:
                    active_nodes.add(u)
                    active_nodes.add(v)
                
                if not active_nodes:
                    break

                best_path = []
                
                nodes_list = list(active_nodes)
                rng_local.shuffle(nodes_list)
                
                for start_node in nodes_list:
                    path = [start_node]
                    visited = {start_node}
                    current = start_node
                    
                    while True:
                        candidates = []
                        for nb in adj[current]:
                            edge = tuple(sorted((current, nb)))
                            if edge in remaining_edges and nb not in visited:
                                deg_nb = sum(1 for u, v in remaining_edges if u == nb or v == nb)
                                noise = rng_local.random() * 0.5
                                candidates.append((deg_nb + noise, nb))
                        
                        if not candidates:
                            break
                        
                        candidates.sort(reverse=True)
                        next_node = candidates[0][1]
                        
                        edge = tuple(sorted((current, next_node)))
                        if edge in remaining_edges:
                            path.append(next_node)
                            visited.add(next_node)
                            current = next_node
                        else:
                            break
                    
                    if len(path) > len(best_path):
                        best_path = path
                
                if not best_path:
                    e = next(iter(remaining_edges))
                    best_path = [e[0], e[1]]

                if len(best_path) > 1:
                    paths.append(best_path)
                    for i in range(len(best_path) - 1):
                        u, v = best_path[i], best_path[i+1]
                        edge = tuple(sorted((u, v)))
                        remaining_edges.discard(edge)
                else:
                    break
            
            if len(paths) < best_count:
                best_count = len(paths)
                best_paths = paths
            
            if time.time() > deadline - 15:
                break
        
        return best_paths if best_paths else []

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline:
            break
        edges = instance(*spec)
        
        time_left = deadline - time.time()
        if time_left > 50:
            paths = decompose_matching(edges, seed_offset=0)
        elif time_left > 20:
            paths = decompose_matching(edges, seed_offset=1)
        else:
            paths = decompose_matching(edges, seed_offset=2)
        
        if not paths:
            paths = [[u, v] for u, v in edges]
        
        results.append(paths)
    
    while len(results) < 3:
        spec = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))[len(results)]
        edges = instance(*spec)
        results.append([[u, v] for u, v in edges])
    
    return results
