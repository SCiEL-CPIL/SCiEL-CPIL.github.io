"""Path decomposition of graphs using matching-based approach with greedy fallback."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    best_results = []
    
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline:
            # Fallback: each edge as its own path
            edges = _instance(*spec)
            best_results.append([list(e) for e in edges])
            continue
        edges = _instance(*spec)
        result = _decompose(edges, deadline)
        if result is not None:
            best_results.append(result)
        else:
            # Fallback: each edge as its own path
            best_results.append([list(e) for e in edges])
    
    return best_results


def _instance(n, p, seed):
    """Generate the instance graph exactly as in the verifier."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _decompose(edges, deadline):
    """Decompose edges into minimum number of edge-disjoint simple paths."""
    if not edges:
        return []
    
    best_paths = None
    best_count = len(edges)  # worst case: each edge is its own path
    
    # Try matching-based approach first
    remaining_time = deadline - time.time()
    
    # Multiple attempts with different randomizations
    num_attempts = max(10, int(remaining_time * 8))
    num_attempts = min(num_attempts, 80)
    
    for attempt in range(num_attempts):
        if time.time() > deadline - 2:
            break
        
        rng = random.Random(attempt * 7919 + 13)
        paths = _matching_decompose(edges, rng, deadline)
        
        if paths is not None:
            count = len(paths)
            if count < best_count:
                best_count = count
                best_paths = paths
                lower_bound = _lower_bound(edges)
                if count <= lower_bound:
                    break
    
    # Also try greedy approach as fallback/comparison
    if time.time() < deadline - 5:
        for attempt in range(5):
            if time.time() > deadline - 3:
                break
            rng = random.Random(100000 + attempt * 7919 + 13)
            paths = _greedy_decompose(edges, rng, deadline)
            if paths is not None:
                paths = _merge_paths(paths, edges)
                count = len(paths)
                if count < best_count:
                    best_count = count
                    best_paths = paths
    
    if best_paths is None:
        return [list(e) for e in edges]
    
    return best_paths


def _lower_bound(edges):
    """Compute theoretical lower bound on number of paths."""
    if not edges:
        return 0
    degree = defaultdict(int)
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
    odd_count = sum(1 for d in degree.values() if d % 2 == 1)
    return max(1, (odd_count + 1) // 2)


def _matching_decompose(edges, rng, deadline):
    """
    Matching-based path decomposition:
    - Create 2 nodes per edge (one for each endpoint)
    - Connect nodes that can be joined (same vertex, different edge)
    - Find maximum matching
    - Reconstruct paths
    """
    m = len(edges)
    if m == 0:
        return []
    
    # Create node IDs: for edge i with endpoints (u,v), 
    # node 2*i represents "entering edge i at u, leaving at v"
    # node 2*i+1 represents "entering edge i at v, leaving at u"
    
    # Build transition graph
    # For each vertex, group all edge-endpoints at that vertex
    vertex_edges = defaultdict(list)  # vertex -> list of (edge_idx, endpoint)
    for i, (u, v) in enumerate(edges):
        vertex_edges[u].append((i, 0))  # endpoint 0 is u
        vertex_edges[v].append((i, 1))  # endpoint 1 is v
    
    # Build adjacency for matching
    # Node 2*i = "use edge i from u to v" (i.e., arrive at u, leave at v)
    # Node 2*i+1 = "use edge i from v to u" (i.e., arrive at v, leave at u)
    # 
    # A match between node a and node b means: after traversing edge a, 
    # we continue to edge b. This is valid if the exit vertex of a 
    # equals the entry vertex of b.
    #
    # For node 2*i (edge i: u->v): exit vertex is v
    # For node 2*i+1 (edge i: v->u): exit vertex is u
    #
    # For node 2*j (edge j: u_j->v_j): entry vertex is u_j
    # For node 2*j+1 (edge j: v_j->u_j): entry vertex is v_j
    
    # Build transition graph: node a can connect to node b if exit(a) == entry(b)
    # and a != b (different edges)
    
    # We need to be careful about simplicity: the resulting path must not visit
    # the same vertex twice. We'll handle this by checking after reconstruction.
    
    # Build adjacency list for the transition graph
    n_nodes = 2 * m
    adj = defaultdict(set)
    
    # For each vertex, connect all "exit at v" nodes to all "entry at v" nodes
    # Exit at v: 
    #   - node 2*i if edges[i] = (u,v) and v is the second endpoint (exit)
    #   - node 2*i+1 if edges[i] = (u,v) and u is the second endpoint... wait
    #
    # Let me redefine more carefully:
    # Edge i has endpoints (u_i, v_i)
    # Direction 0: traverse from u_i to v_i. Entry: u_i, Exit: v_i. Node: 2*i
    # Direction 1: traverse from v_i to u_i. Entry: v_i, Exit: u_i. Node: 2*i+1
    #
    # So for vertex w:
    #   Entries at w: all nodes 2*i where u_i == w, and all nodes 2*i+1 where v_i == w
    #   Exits at w: all nodes 2*i where v_i == w, and all nodes 2*i+1 where u_i == w
    #
    # A valid transition: exit_node -> entry_node if exit_vertex == entry_vertex
    # and they use different edges
    
    # Build entry and exit lists per vertex
    entries_at = defaultdict(list)  # vertex -> list of node_ids that enter at this vertex
    exits_at = defaultdict(list)    # vertex -> list of node_ids that exit at this vertex
    
    for i, (u, v) in enumerate(edges):
        # Direction 0: u -> v, entry at u, exit at v
        entries_at[u].append(2 * i)
        exits_at[v].append(2 * i)
        # Direction 1: v -> u, entry at v, exit at u
        entries_at[v].append(2 * i + 1)
        exits_at[u].append(2 * i + 1)
    
    # Build transition graph
    for v in entries_at:
        for exit_node in exits_at[v]:
            for entry_node in entries_at[v]:
                # Check they use different edges
                exit_edge = exit_node // 2
                entry_edge = entry_node // 2
                if exit_edge != entry_node // 2:
                    adj[exit_node].add(entry_node)
                    adj[entry_node].add(exit_node)
    
    # Find maximum matching using greedy + augmenting paths
    matching = _find_max_matching(n_nodes, adj, rng, deadline)
    
    # Reconstruct paths from matching
    # Each matched pair (a, b) means: after node a, go to node b
    # Unmatched nodes are path endpoints
    
    # Build next pointer
    next_node = {}
    for a, b in matching.items():
        next_node[a] = b
    
    # Find all path starts (nodes with no predecessor)
    has_pred = set(next_node.values())
    starts = [n for n in range(n_nodes) if n not in has_pred]
    
    # Reconstruct paths
    paths = []
    for start in starts:
        path_nodes = []
        cur = start
        visited = set()
        while cur is not None:
            if cur in visited:
                break  # cycle detected, shouldn't happen
            visited.add(cur)
            path_nodes.append(cur)
            cur = next_node.get(cur)
        
        # Convert node sequence to vertex sequence
        vertices = []
        for node in path_nodes:
            edge_idx = node // 2
            direction = node % 2
            u, v = edges[edge_idx]
            if direction == 0:
                # u -> v
                if not vertices or vertices[-1] != u:
                    vertices.append(u)
                vertices.append(v)
            else:
                # v -> u
                if not vertices or vertices[-1] != v:
                    vertices.append(v)
                vertices.append(u)
        
        if len(vertices) >= 2:
            # Check simplicity
            if len(set(vertices)) == len(vertices):
                paths.append(vertices)
    
    # Check if all edges are covered
    if len(paths) < m:  # Should have at most m paths
        # Verify all edges are used exactly once
        used_edges = set()
        for path in paths:
            for i in range(len(path) - 1):
                e = tuple(sorted((path[i], path[i+1])))
                if e in used_edges:
                    return None  # Invalid
                used_edges.add(e)
        if used_edges != set(edges):
            return None  # Not all edges covered
    
    return paths if paths else [list(e) for e in edges]


def _find_max_matching(n_nodes, adj, rng, deadline):
    """
    Find a maximum cardinality matching using greedy initialization + augmenting paths.
    """
    # Greedy initialization
    matching = {}  # node -> matched node
    matched = set()
    
    # Randomize order
    nodes = list(range(n_nodes))
    rng.shuffle(nodes)
    
    for u in nodes:
        if u in matched:
            continue
        for v in adj[u]:
            if v not in matched:
                matching[u] = v
                matching[v] = u
                matched.add(u)
                matched.add(v)
                break
    
    # Try to improve with augmenting paths (BFS-based)
    improved = True
    while improved:
        if time.time() > deadline - 1:
            break
        improved = False
        
        # Find all free nodes
        free_nodes = [n for n in range(n_nodes) if n not in matched]
        rng.shuffle(free_nodes)
        
        for start in free_nodes:
            if time.time() > deadline - 1:
                break
            
            # BFS to find augmenting path from start
            # State: (node, is_unmatched_side)
            # We alternate: from unmatched side, go via unmatched edges; 
            # from matched side, go via matched edge
            parent = {}  # node -> (prev_node, edge_type)
            visited = {start}
            queue = [(start, True)]  # (node, is_unmatched_side)
            found = False
            end_node = None
            
            while queue and not found:
                if time.time() > deadline - 1:
                    break
                node, is_unmatched_side = queue.pop(0)
                
                if is_unmatched_side:
                    # Explore all edges from node that are not in matching
                    for nb in adj[node]:
                        if nb not in visited:
                            visited.add(nb)
                            parent[nb] = (node, 'unmatched')
                            if nb not in matched:
                                # Found augmenting path
                                found = True
                                end_node = nb
                                break
                            else:
                                # nb is matched, follow its matched edge
                                queue.append((nb, False))
                else:
                    # Follow the matched edge
                    if node in matching:
                        matched_nb = matching[node]
                        if matched_nb not in visited:
                            visited.add(matched_nb)
                            parent[matched_nb] = (node, 'matched')
                            queue.append((matched_nb, True))
            
            if found:
                # Augment along the path
                # Reconstruct path from end_node back to start
                path_nodes = []
                cur = end_node
                while cur is not None:
                    path_nodes.append(cur)
                    if cur == start:
                        break
                    if cur in parent:
                        prev, _ = parent[cur]
                        cur = prev
                    else:
                        break
                
                # path_nodes is from end to start
                path_nodes.reverse()  # now from start to end
                
                # Augment: flip matching along the path
                for i in range(0, len(path_nodes) - 1, 2):
                    a, b = path_nodes[i], path_nodes[i + 1]
                    # Remove old matches
                    if a in matching:
                        old_match = matching[a]
                        del matching[a]
                        if old_match in matching:
                            del matching[old_match]
                        matched.discard(old_match)
                    if b in matching:
                        old_match = matching[b]
                        del matching[b]
                        if old_match in matching:
                            del matching[old_match]
                        matched.discard(old_match)
                    # Add new match
                    matching[a] = b
                    matching[b] = a
                    matched.add(a)
                    matched.add(b)
                
                improved = True
    
    return matching


def _greedy_decompose(edges, rng, deadline):
    """
    Greedy path decomposition:
    1. Build adjacency from remaining edges
    2. Repeatedly find a long simple path using remaining edges
    3. Remove those edges and repeat
    Prioritize starting from odd-degree vertices.
    """
    remaining = set(edges)
    if not remaining:
        return []
    
    # Build adjacency
    adj = defaultdict(set)
    for u, v in remaining:
        adj[u].add(v)
        adj[v].add(u)
    
    degree = defaultdict(int)
    for u, v in remaining:
        degree[u] += 1
        degree[v] += 1
    
    paths = []
    
    while remaining:
        if time.time() > deadline:
            return None
        
        # Find a good starting vertex
        start = _pick_start(adj, degree, rng)
        if start is None:
            # Fallback: pick any vertex with edges
            candidates = [v for v in adj if adj[v]]
            if not candidates:
                break
            start = rng.choice(candidates)
        
        # Find a long simple path starting from start
        path = _find_long_path(start, adj, degree, rng, remaining, deadline)
        
        if path is None or len(path) < 2:
            # Can't find a path; take a single edge
            # Find any edge incident to start or any edge
            if adj[start]:
                next_v = rng.choice(list(adj[start]))
                path = [start, next_v]
            else:
                # Pick any remaining edge
                e = next(iter(remaining))
                path = [e[0], e[1]]
        
        # Extract edges from path and remove them
        path_edges = set()
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            edge = tuple(sorted((u, v)))
            if edge in remaining:
                path_edges.add(edge)
        
        if not path_edges:
            # Shouldn't happen, but safety
            break
        
        for e in path_edges:
            remaining.discard(e)
        
        # Update adjacency and degrees
        for u, v in path_edges:
            adj[u].discard(v)
            adj[v].discard(u)
            degree[u] -= 1
            degree[v] -= 1
        
        # Clean up empty adj entries
        for v in list(adj.keys()):
            if not adj[v]:
                del adj[v]
        
        paths.append(path)
    
    return paths


def _pick_start(adj, degree, rng):
    """Pick a starting vertex, preferring odd-degree vertices."""
    odd_vertices = [v for v in adj if degree.get(v, 0) % 2 == 1 and adj[v]]
    if odd_vertices:
        return rng.choice(odd_vertices)
    
    # Prefer high-degree vertices
    candidates = [v for v in adj if adj[v]]
    if not candidates:
        return None
    
    # Weight by degree
    weights = [max(1, degree.get(v, 0)) for v in candidates]
    total = sum(weights)
    r = rng.random() * total
    cum = 0
    for v, w in zip(candidates, weights):
        cum += w
        if r <= cum:
            return v
    return candidates[-1]


def _find_long_path(start, adj, degree, rng, remaining, deadline):
    """
    Find a long simple path starting from start using DFS with backtracking.
    Uses randomized exploration to find long paths.
    """
    best_path = [start]
    
    # Use a greedy approach with limited backtracking
    _greedy_extend(start, adj, best_path, rng, deadline)
    
    return best_path if len(best_path) >= 2 else None


def _greedy_extend(start, adj, best_path, rng, deadline):
    """
    Greedy path extension with occasional backtracking.
    """
    path = [start]
    visited = {start}
    
    max_depth = 80  # Allow longer paths
    for step in range(max_depth):
        if time.time() > deadline:
            break
        current = path[-1]
        neighbors = [nb for nb in adj[current] if nb not in visited]
        
        if not neighbors:
            break
        
        # Choose next vertex: combine Warnsdorff's rule with degree consideration
        def score(nb):
            unvisited_count = sum(1 for x in adj[nb] if x not in visited)
            # Prefer vertices with fewer unvisited neighbors (Warnsdorff)
            # but also consider overall degree to avoid isolating high-degree vertices
            deg_bonus = 1.0 / (1.0 + adj[nb].__len__() * 0.1)
            return unvisited_count * 1.0 - deg_bonus * 0.5
        
        # Sometimes pick greedily, sometimes randomly
        if rng.random() < 0.80:
            # Pick the neighbor with best score (Warnsdorff-like)
            neighbors.sort(key=score)
            # Pick from top 2-3 with some randomness
            top_k = min(3, len(neighbors))
            choice = rng.choice(neighbors[:top_k])
        else:
            choice = rng.choice(neighbors)
        
        path.append(choice)
        visited.add(choice)
        
        if len(path) > len(best_path):
            best_path[:] = path[:]
    
    return path


def _merge_paths(paths, edges):
    """
    Post-processing: merge consecutive paths that share an endpoint.
    If path A ends at vertex v and path B starts at vertex v (or vice versa),
    we can merge them into a single path, reducing the count by 1.
    """
    if len(paths) <= 1:
        return paths
    
    # Build a map from vertex to list of (path_index, position) where position is 0 for start, 1 for end
    vertex_map = defaultdict(list)
    for i, path in enumerate(paths):
        if len(path) >= 2:
            vertex_map[path[0]].append((i, 0))
            vertex_map[path[-1]].append((i, 1))
    
    merged = True
    while merged:
        merged = False
        for v in list(vertex_map.keys()):
            if len(vertex_map[v]) < 2:
                continue
            
            # Find two paths that can be merged at vertex v
            entries = vertex_map[v]
            
            # Find pairs that can be merged
            end_paths = [e for e in entries if e[1] == 1]
            start_paths = [e for e in entries if e[1] == 0]
            
            # Try to merge an end path with a start path
            if end_paths and start_paths:
                # Don't merge a path with itself
                for ep in end_paths:
                    for sp in start_paths:
                        if ep[0] != sp[0]:
                            # Merge paths[ep[0]] (ends at v) with paths[sp[0]] (starts at v)
                            path_a = paths[ep[0]]
                            path_b = paths[sp[0]]
                            # path_a ends at v, path_b starts at v
                            # New path: path_a + path_b[1:]
                            new_path = path_a + path_b[1:]
                            # Check if new path is simple (no repeated vertices)
                            if len(set(new_path)) == len(new_path):
                                # Replace both paths with the new one
                                new_paths = []
                                for idx, p in enumerate(paths):
                                    if idx != ep[0] and idx != sp[0]:
                                        new_paths.append(p)
                                new_paths.append(new_path)
                                paths = new_paths
                                # Rebuild vertex_map
                                vertex_map = defaultdict(list)
                                for i, path in enumerate(paths):
                                    if len(path) >= 2:
                                        vertex_map[path[0]].append((i, 0))
                                        vertex_map[path[-1]].append((i, 1))
                                merged = True
                                break
                    if merged:
                        break
            
            # If no end-start merge, try end-end or start-start merges
            if not merged and len(end_paths) >= 2:
                # Merge two paths that both end at v
                i1, i2 = end_paths[0][0], end_paths[1][0]
                path_a = paths[i1]
                path_b = paths[i2]
                # Both end at v. Reverse path_b, then merge path_a + reversed_path_b[1:]
                reversed_b = path_b[::-1]
                new_path = path_a + reversed_b[1:]
                if len(set(new_path)) == len(new_path):
                    new_paths = []
                    for idx, p in enumerate(paths):
                        if idx != i1 and idx != i2:
                            new_paths.append(p)
                    new_paths.append(new_path)
                    paths = new_paths
                    vertex_map = defaultdict(list)
                    for i, path in enumerate(paths):
                        if len(path) >= 2:
                            vertex_map[path[0]].append((i, 0))
                            vertex_map[path[-1]].append((i, 1))
                    merged = True
                    break
            
            if not merged and len(start_paths) >= 2:
                # Merge two paths that both start at v
                i1, i2 = start_paths[0][0], start_paths[1][0]
                path_a = paths[i1]
                path_b = paths[i2]
                # Both start at v. Reverse path_a, then merge reversed_path_a[1:] + path_b
                reversed_a = path_a[::-1]
                new_path = reversed_a[1:] + path_b
                if len(set(new_path)) == len(new_path):
                    new_paths = []
                    for idx, p in enumerate(paths):
                        if idx != i1 and idx != i2:
                            new_paths.append(p)
                    new_paths.append(new_path)
                    paths = new_paths
                    vertex_map = defaultdict(list)
                    for i, path in enumerate(paths):
                        if len(path) >= 2:
                            vertex_map[path[0]].append((i, 0))
                            vertex_map[path[-1]].append((i, 1))
                    merged = True
                    break
    
    return paths

# EVOLVE-BLOCK-END
