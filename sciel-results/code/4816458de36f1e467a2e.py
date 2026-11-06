"""Path decomposition of graphs using greedy + randomized restarts with improved heuristics."""
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
    
    # Number of restarts based on available time
    num_restarts = max(50, int((deadline - time.time()) * 15))
    num_restarts = min(num_restarts, 300)
    
    for attempt in range(num_restarts):
        if time.time() > deadline - 3:
            break
        
        rng = random.Random(attempt * 7919 + 13)
        paths = _greedy_decompose(edges, rng, deadline)
        
        if paths is not None:
            # Post-process: merge consecutive paths that share an endpoint
            paths = _merge_paths(paths, edges)
            count = len(paths)
            if count < best_count:
                best_count = count
                best_paths = paths
                # Early stop if we hit theoretical lower bound
                lower_bound = _lower_bound(edges)
                if count <= lower_bound:
                    break
    
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
    Post-process: try to merge consecutive paths that share an endpoint.
    If path A ends at vertex v and path B starts at vertex v (or vice versa),
    we can concatenate them into one path, reducing the total count by 1.
    Only merge if the resulting path is simple (no repeated vertices).
    """
    if not paths:
        return paths
    
    # Build edge set for quick lookup
    edge_set = set(tuple(sorted(e)) for e in edges)
    
    changed = True
    while changed:
        changed = False
        new_paths = list(paths)
        
        for i in range(len(new_paths)):
            if time.time() > 145:  # safety
                break
            for j in range(len(new_paths)):
                if i == j:
                    continue
                # Try merging path i and path j
                merged = _try_merge(new_paths[i], new_paths[j], edge_set)
                if merged is not None:
                    # Replace both with the merged path
                    new_paths.pop(j)
                    new_paths[i] = merged
                    changed = True
                    break
            if changed:
                break
        
        paths = new_paths
    
    return paths


def _try_merge(path_a, path_b, edge_set):
    """
    Try to merge two paths into one. Returns the merged path or None.
    Path A: [a0, a1, ..., ak]
    Path B: [b0, b1, ..., bm]
    
    Possible merges:
    1. ak == b0: [a0, ..., ak, b1, ..., bm]
    2. ak == bm: [a0, ..., ak, b_{m-1}, ..., b0]
    3. a0 == b0: [a1, ..., ak, b1, ..., bm] (if we reverse A)
    4. a0 == bm: [a1, ..., ak, b_{m-1}, ..., b0] (if we reverse A)
    Also consider reversing B similarly.
    """
    if not path_a or not path_b:
        return None
    
    # Try all 4 combinations of orientations
    for a in (path_a, path_a[::-1]):
        for b in (path_b, path_b[::-1]):
            # Check if a[-1] == b[0]
            if a[-1] == b[0]:
                merged = a + b[1:]
                if _is_simple_path(merged, edge_set):
                    return merged
            # Check if a[0] == b[-1] (reverse both and check)
            if a[0] == b[-1]:
                merged = a[::-1] + b[:-1][::-1]
                if _is_simple_path(merged, edge_set):
                    return merged
    
    return None


def _is_simple_path(path, edge_set):
    """Check if path is simple (no repeated vertices) and all edges exist."""
    if len(path) < 2:
        return True
    
    # Check for repeated vertices
    if len(set(path)) != len(path):
        return False
    
    # Check all edges exist
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        edge = tuple(sorted((u, v)))
        if edge not in edge_set:
            return False
    
    return True

# EVOLVE-BLOCK-END
