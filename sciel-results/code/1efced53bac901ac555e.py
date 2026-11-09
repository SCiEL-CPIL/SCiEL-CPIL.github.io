import time
import random
from collections import defaultdict

def construct():
    t0 = time.time()
    DEADLINE = 145.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges, deadline):
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Map edges to indices for easier handling in matching
        edge_list = list(edges)
        m = len(edge_list)
        edge_to_idx = {e: i for i, e in enumerate(edge_list)}
        
        # For each vertex, list of incident edge indices
        vertex_edges = defaultdict(list)
        for i, (u, v) in enumerate(edge_list):
            vertex_edges[u].append(i)
            vertex_edges[v].append(i)

        def get_trails_from_matching(matching):
            # matching is a dict: (edge_idx, endpoint_vertex) -> (other_edge_idx, other_endpoint_vertex)
            # Or simpler: for each edge, we know which other edge it's connected to at each end.
            # Let's represent the connection as:
            # conn[edge_idx][vertex] = other_edge_idx
            # If conn[edge_idx][vertex] is None, it's an endpoint of a trail.
            
            conn = [None] * m
            # Initialize conn as a list of dicts or tuples
            # conn[i] = (neighbor_at_u, neighbor_at_v) where u,v are endpoints of edge i
            # If neighbor is None, it's an end.
            
            # We need to reconstruct trails.
            # A trail is a sequence of edges.
            # Start from an edge where one end is None.
            # Follow the connections.
            
            visited_edges = [False] * m
            trails = []
            
            for i in range(m):
                if visited_edges[i]:
                    continue
                
                # Start a trail from edge i
                trail = [i]
                visited_edges[i] = True
                
                # We are at one end of edge i.
                # Let's say we start at vertex u of edge i.
                u, v = edge_list[i]
                current_edge = i
                current_vertex = u # We are "at" u, coming from edge i? No, we start here.
                
                # Actually, let's just follow the links.
                # The matching pairs (edge, vertex) with (other_edge, other_vertex).
                # So if we are at edge `current_edge` and vertex `current_vertex`,
                # we look up the partner.
                
                # Let's use a dictionary for the matching for O(1) lookup
                # key: (edge_idx, vertex)
                # val: (other_edge_idx, other_vertex)
                
                # Reconstructing from the provided matching structure is tricky if not stored properly.
                # Let's store the matching as a list of pairs for each vertex.
                
                pass # This logic needs to be tied to the specific matching generation

        def generate_decomposition(seed_offset):
            rng = random.Random(seed_offset)
            
            # For each vertex, create a matching of its incident edges
            # matching[v] is a list of pairs of edge indices
            # e.g., matching[v] = [(e1, e2), (e3, e4), ...]
            # If odd number of edges, one is left unpaired.
            
            # We need to track for each edge end, which other edge end it is connected to.
            # conn[edge_idx][vertex] = other_edge_idx
            # If other_edge_idx is None, it's a trail endpoint.
            
            conn = [{} for _ in range(m)]
            
            for v in vertex_edges:
                e_list = vertex_edges[v]
                # Shuffle to randomize matching
                e_list_copy = e_list[:]
                rng.shuffle(e_list_copy)
                
                # Pair them up
                for k in range(0, len(e_list_copy) - 1, 2):
                    e1 = e_list_copy[k]
                    e2 = e_list_copy[k+1]
                    conn[e1][v] = e2
                    conn[e2][v] = e1
            
            # Now reconstruct trails
            visited = [False] * m
            trails = []
            
            for i in range(m):
                if visited[i]:
                    continue
                
                trail = [i]
                visited[i] = True
                
                # Determine the next edge
                u, v = edge_list[i]
                # We can go to u or v.
                # Let's pick u first.
                next_e = conn[i].get(u, None)
                # If next_e is None, the trail ends at u.
                # If next_e is not None, we go to next_e, and the next vertex to consider is the other end of next_e.
                
                # We need to keep track of the "current" vertex we are at.
                # Start at edge i, "entering" from nowhere?
                # Let's say we are at vertex u of edge i.
                current_vertex = u
                current_edge = i
                
                while True:
                    next_e = conn[current_edge].get(current_vertex, None)
                    if next_e is None:
                        break
                    
                    # Move to next_e
                    trail.append(next_e)
                    if visited[next_e]:
                        # This should not happen if the matching is valid
                        break
                    visited[next_e] = True
                    
                    # Update current_vertex
                    # The other endpoint of next_e
                    u2, v2 = edge_list[next_e]
                    if u2 == current_vertex:
                        current_vertex = v2
                    else:
                        current_vertex = u2
                    current_edge = next_e
                
                trails.append(trail)
            
            # Convert trails to paths
            paths = []
            for trail in trails:
                # trail is a list of edge indices
                # Convert to vertex list
                verts = []
                for idx in trail:
                    u, v = edge_list[idx]
                    if not verts:
                        verts.append(u)
                        verts.append(v)
                    else:
                        # Append the new vertex
                        # Check which vertex is shared
                        last_v = verts[-1]
                        if u == last_v:
                            verts.append(v)
                        else:
                            verts.append(u)
                
                # Check for simplicity
                # If not simple, split
                # Simple check: all vertices unique
                if len(verts) != len(set(verts)):
                    # Find the first repeated vertex
                    seen = set()
                    split_idx = -1
                    for k, vert in enumerate(verts):
                        if vert in seen:
                            split_idx = k
                            break
                        seen.add(vert)
                    
                    if split_idx > 0:
                        # Split into two paths: verts[0:split_idx+1] and verts[split_idx:]
                        # This removes the cycle
                        p1 = verts[:split_idx+1]
                        p2 = verts[split_idx:]
                        # Recursively split?
                        # For now, just split once. Might need multiple splits.
                        # Let's use a helper to fully decompose a trail into simple paths
                        def decompose_trail_to_paths(v_list):
                            if len(v_list) <= 1:
                                return []
                            if len(v_list) == len(set(v_list)):
                                return [v_list]
                            
                            seen = set()
                            split_idx = -1
                            for k, vert in enumerate(v_list):
                                if vert in seen:
                                    split_idx = k
                                    break
                                seen.add(vert)
                            
                            if split_idx == -1:
                                return [v_list]
                            
                            p1 = v_list[:split_idx+1]
                            p2 = v_list[split_idx:]
                            res1 = decompose_trail_to_paths(p1)
                            res2 = decompose_trail_to_paths(p2)
                            return res1 + res2
                            
                        paths.extend(decompose_trail_to_paths(verts))
                    else:
                        paths.append(verts)
                else:
                    paths.append(verts)
            
            return paths

        def try_merge(paths):
            merged = False
            new_paths = []
            for i in range(len(paths)):
                for j in range(i+1, len(paths)):
                    p1 = paths[i]
                    p2 = paths[j]
                    if len(p1) < 2 or len(p2) < 2:
                        continue
                    set1 = set(p1)
                    set2 = set(p2)
                    common = set1 & set2
                    if len(common) != 1:
                        continue
                    shared = list(common)[0]
                    is_ep1 = (p1[0] == shared or p1[-1] == shared)
                    is_ep2 = (p2[0] == shared or p2[-1] == shared)
                    if is_ep1 and is_ep2:
                        # Check if merge is simple
                        if p1[-1] == shared and p2[0] == shared:
                            new_p = p1 + p2[1:]
                        elif p1[0] == shared and p2[-1] == shared:
                            new_p = p2 + p1[1:]
                        elif p1[0] == shared and p2[0] == shared:
                            new_p = p1[::-1] + p2[1:]
                        elif p1[-1] == shared and p2[-1] == shared:
                            new_p = p2 + p1[::-1][1:]
                        else:
                            continue
                        if len(new_p) == len(set(new_p)):
                            # Merge
                            paths[i] = new_p
                            paths[j] = None
                            merged = True
                            break
                if merged:
                    break
            if merged:
                return [p for p in paths if p is not None]
            return None

        def local_search(paths, deadline):
            changed = True
            while changed and time.time() < deadline:
                changed = False
                new_paths = try_merge(paths)
                if new_paths is not None:
                    paths = new_paths
                    changed = True
            return paths

        best_paths = None
        best_count = float('inf')
        
        # Run multiple attempts with different random matchings
        attempts = 0
        while time.time() < deadline - 2:
            attempts += 1
            paths = generate_decomposition(attempts * 13 + 1)
            paths = local_search(paths, deadline - 1)
            count = len(paths)
            if count < best_count:
                best_count = count
                best_paths = paths
                # Check lower bound
                deg = defaultdict(int)
                for u, v in edges:
                    deg[u] += 1
                    deg[v] += 1
                lb = sum(1 for d in deg.values() if d % 2 == 1) // 2
                # Theoretical min for paths is at least lb
                # But Gallai bound is ceil((n-1)/2)
                n = max(max(u for u,v in edges), max(v for u,v in edges)) + 1
                gallai_lb = (n - 1 + 1) // 2 # ceil((n-1)/2)
                if best_count <= gallai_lb:
                    break

        if best_paths is None:
            best_paths = generate_decomposition(0)
        return best_paths

    result = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = decompose(edges, DEADLINE)
        result.append(paths)
    return result

