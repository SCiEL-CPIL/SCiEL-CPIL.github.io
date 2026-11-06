"""Path decomposition of graphs using greedy + randomized restarts with improved heuristics and local search."""
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
            
            # Local search: try to improve by swapping edges between paths
            paths = _local_search(paths, edges, rng, deadline)
            
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
    Merge consecutive paths that share an endpoint, reducing total path count.
    Two paths can be merged if they share an endpoint and the resulting path is simple.
    """
    if len(paths) <= 1:
        return paths
    
    edge_set = set(edges)
    merged = True
    while merged:
        merged = False
        new_paths = list(paths)
        for i in range(len(new_paths)):
            for j in range(i + 1, len(new_paths)):
                if _try_merge(new_paths[i], new_paths[j], edge_set):
                    # Merge paths i and j
                    merged_path = _do_merge(new_paths[i], new_paths[j], edge_set)
                    if merged_path is not None:
                        new_paths[i] = merged_path
                        new_paths.pop(j)
                        merged = True
                        break
            if merged:
                break
        paths = new_paths
    
    return paths


def _try_merge(path1, path2, edge_set):
    """Check if two paths can be merged into a single simple path."""
    # Try all four combinations of endpoints
    for p1_start, p1_end in [(path1[0], path1[-1]), (path1[-1], path1[0])]:
        for p2_start, p2_end in [(path2[0], path2[-1]), (path2[-1], path2[0])]:
            # Check if they share an endpoint
            if p1_end == p2_start:
                # path1 -> path2
                if _is_simple_path(path1 + path2[1:], edge_set):
                    return True
            if p1_start == p2_end:
                # path2 -> path1
                if _is_simple_path(path2 + path1[1:], edge_set):
                    return True
    return False


def _do_merge(path1, path2, edge_set):
    """Merge two paths into a single simple path."""
    # Try all four combinations of endpoints
    for p1_start, p1_end in [(path1[0], path1[-1]), (path1[-1], path1[0])]:
        for p2_start, p2_end in [(path2[0], path2[-1]), (path2[-1], path2[0])]:
            if p1_end == p2_start:
                # path1 -> path2
                merged = path1 + path2[1:]
                if _is_simple_path(merged, edge_set):
                    return merged
            if p1_start == p2_end:
                # path2 -> path1
                merged = path2 + path1[1:]
                if _is_simple_path(merged, edge_set):
                    return merged
    return None


def _is_simple_path(path, edge_set):
    """Check if a path is simple (no repeated vertices) and uses only edges from the graph."""
    # Check for repeated vertices
    if len(set(path)) != len(path):
        return False
    
    # Check that all consecutive pairs are edges in the graph
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        edge = tuple(sorted((u, v)))
        if edge not in edge_set:
            return False
    
    return True


def _local_search(paths, edges, rng, deadline):
    """
    Local search: try to improve the decomposition by swapping edges between paths.
    This can reduce the total number of paths.
    """
    if len(paths) <= 1:
        return paths
    
    edge_set = set(edges)
    improved = True
    max_iterations = 100  # Limit local search iterations
    
    for _ in range(max_iterations):
        if time.time() > deadline:
            break
        if not improved:
            break
        
        improved = False
        # Try to merge any pair of paths
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                if _try_merge(paths[i], paths[j], edge_set):
                    merged_path = _do_merge(paths[i], paths[j], edge_set)
                    if merged_path is not None:
                        paths[i] = merged_path
                        paths.pop(j)
                        improved = True
                        break
            if improved:
                break
    
    return paths

# EVOLVE-BLOCK-END
