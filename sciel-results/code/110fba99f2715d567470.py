import time
import random
from collections import defaultdict

def construct():
    """
    Decomposes the edges of three fixed instance graphs into the minimum number of
    edge-disjoint simple paths.
    
    Returns:
        A list of three lists of paths. Each path is a list of vertices.
    """
    # Deadline guard: 150s limit, leave 10s safety margin
    start_time = time.time()
    deadline = start_time + 140.0
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def eulerian_trails(edges, rng, num_trails_hint=None):
        """
        Generate edge-disjoint trails using a randomized Eulerian-like approach.
        We build trails by greedily following unused edges, preferring to continue
        from the current end. When stuck, we start a new trail.
        """
        if not edges:
            return []
        
        # Build adjacency with edge tracking
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        remaining = set(edges)
        trails = []
        
        def get_unused_neighbors(u):
            """Get neighbors of u connected by an unused edge."""
            result = []
            for v in adj[u]:
                e = tuple(sorted((u, v)))
                if e in remaining:
                    result.append(v)
            return result
        
        def remove_edge(u, v):
            e = tuple(sorted((u, v)))
            if e in remaining:
                remaining.remove(e)
        
        # Start trails from vertices with highest degree first
        # to encourage long trails
        vertices = list(adj.keys())
        
        while remaining:
            # Pick starting vertex: prefer high degree
            degrees = defaultdict(int)
            for u, v in remaining:
                degrees[u] += 1
                degrees[v] += 1
            
            if degrees:
                max_deg = max(degrees.values())
                candidates = [v for v, d in degrees.items() if d == max_deg]
                start_v = rng.choice(candidates)
            else:
                break
            
            # Build trail from start_v
            trail = [start_v]
            current = start_v
            
            while True:
                neighbors = get_unused_neighbors(current)
                if not neighbors:
                    break
                
                # Choose next vertex: prefer vertices with more unused edges
                # (this helps create longer trails)
                if len(neighbors) > 1:
                    # Score by degree in remaining graph
                    def score(v):
                        deg = 0
                        for e in remaining:
                            if e[0] == v or e[1] == v:
                                deg += 1
                        return deg
                    max_score = max(score(v) for v in neighbors)
                    best = [v for v in neighbors if score(v) == max_score]
                    next_v = rng.choice(best)
                else:
                    next_v = neighbors[0]
                
                # Check if next_v is already in trail (would create a cycle/revisit)
                # For now, allow it (trail can revisit vertices)
                trail.append(next_v)
                remove_edge(current, next_v)
                current = next_v
            
            if len(trail) > 1:
                trails.append(trail)
        
        return trails
    
    def split_trail_to_paths(trail):
        """
        Split a trail into simple paths by cutting at repeated vertices.
        A trail may revisit vertices; we split it into maximal simple sub-paths.
        """
        if len(trail) <= 1:
            return [trail] if trail else []
        
        paths = []
        current_path = [trail[0]]
        visited_in_path = {trail[0]}
        
        for i in range(1, len(trail)):
            v = trail[i]
            if v in visited_in_path:
                # This vertex was already visited in current path
                # End current path before this vertex
                if len(current_path) > 1:
                    paths.append(current_path)
                # Start new path from this vertex
                current_path = [v]
                visited_in_path = {v}
            else:
                current_path.append(v)
                visited_in_path.add(v)
        
        if len(current_path) > 1:
            paths.append(current_path)
        
        return paths
    
    def merge_adjacent_paths(paths, edges_set):
        """
        Try to merge adjacent paths if they can be connected.
        Two paths can be merged if the end of one is adjacent to the start of another
        and the edge exists and is not used by either path.
        """
        if len(paths) <= 1:
            return paths
        
        # Build set of all edges used
        used_edges = set()
        for path in paths:
            for i in range(len(path) - 1):
                e = tuple(sorted((path[i], path[i+1])))
                used_edges.add(e)
        
        # Build adjacency from edges_set
        adj = defaultdict(set)
        for u, v in edges_set:
            adj[u].add(v)
            adj[v].add(u)
        
        merged = True
        while merged:
            merged = False
            new_paths = []
            i = 0
            while i < len(paths):
                if i + 1 < len(paths):
                    p1 = paths[i]
                    p2 = paths[i + 1]
                    # Try to merge p1 and p2
                    # Option 1: end of p1 connects to start of p2
                    end1 = p1[-1]
                    start2 = p2[0]
                    can_merge = False
                    if (end1, start2) in [(tuple(sorted(e)) for e in edges_set)] and \
                       tuple(sorted((end1, start2))) not in used_edges:
                        # Check simplicity
                        merged_path = p1 + p2[1:]
                        if len(set(merged_path)) == len(merged_path):
                            new_paths.append(merged_path)
                            # Update used edges
                            e = tuple(sorted((end1, start2)))
                            used_edges.add(e)
                            i += 2
                            merged = True
                            continue
                    # Option 2: end of p1 connects to end of p2 (reverse p2)
                    end2 = p2[-1]
                    if (end1, end2) in [(tuple(sorted(e)) for e in edges_set)] and \
                       tuple(sorted((end1, end2))) not in used_edges:
                        reversed_p2 = list(reversed(p2))
                        merged_path = p1 + reversed_p2[1:]
                        if len(set(merged_path)) == len(merged_path):
                            new_paths.append(merged_path)
                            e = tuple(sorted((end1, end2)))
                            used_edges.add(e)
                            i += 2
                            merged = True
                            continue
                    # Option 3: start of p1 connects to start of p2 (reverse p1)
                    start1 = p1[0]
                    if (start1, start2) in [(tuple(sorted(e)) for e in edges_set)] and \
                       tuple(sorted((start1, start2))) not in used_edges:
                        reversed_p1 = list(reversed(p1))
                        merged_path = reversed_p1 + p2[1:]
                        if len(set(merged_path)) == len(merged_path):
                            new_paths.append(merged_path)
                            e = tuple(sorted((start1, start2)))
                            used_edges.add(e)
                            i += 2
                            merged = True
                            continue
                    # Option 4: start of p1 connects to end of p2 (reverse both)
                    if (start1, end2) in [(tuple(sorted(e)) for e in edges_set)] and \
                       tuple(sorted((start1, end2))) not in used_edges:
                        reversed_p1 = list(reversed(p1))
                        reversed_p2 = list(reversed(p2))
                        merged_path = reversed_p2 + reversed_p1[1:]
                        if len(set(merged_path)) == len(merged_path):
                            new_paths.append(merged_path)
                            e = tuple(sorted((start1, end2)))
                            used_edges.add(e)
                            i += 2
                            merged = True
                            continue
                
                new_paths.append(paths[i])
                i += 1
            
            paths = new_paths
        
        return paths
    
    def local_search_improve(paths, edges_set, rng, max_iters=100):
        """
        Local search: try to reduce number of paths by moving endpoints
        or reconnecting paths.
        """
        if len(paths) <= 1:
            return paths
        
        # Build used edge set
        def get_used_edges(path_list):
            used = set()
            for path in path_list:
                for i in range(len(path) - 1):
                    e = tuple(sorted((path[i], path[i+1])))
                    used.add(e)
            return used
        
        current_paths = [list(p) for p in paths]
        best_paths = [list(p) for p in paths]
        best_count = len(best_paths)
        
        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges_set:
            adj[u].add(v)
            adj[v].add(u)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try to merge any two paths
            for i in range(len(current_paths)):
                for j in range(i + 1, len(current_paths)):
                    p1 = current_paths[i]
                    p2 = current_paths[j]
                    
                    # Try all 4 combinations of endpoints
                    endpoints_p1 = [(p1[0], 'start'), (p1[-1], 'end')]
                    endpoints_p2 = [(p2[0], 'start'), (p2[-1], 'end')]
                    
                    for (v1, pos1) in endpoints_p1:
                        for (v2, pos2) in endpoints_p2:
                            if v1 == v2:
                                continue
                            e = tuple(sorted((v1, v2)))
                            if e not in edges_set:
                                continue
                            
                            # Check if edge is unused
                            used = get_used_edges(current_paths)
                            if e in used:
                                continue
                            
                            # Try merging
                            # Determine how to connect
                            # If pos1 is 'end' and pos2 is 'start': p1 + p2[1:]
                            # If pos1 is 'end' and pos2 is 'end': p1 + reversed(p2)[1:]
                            # If pos1 is 'start' and pos2 is 'start': reversed(p1) + p2[1:]
                            # If pos1 is 'start' and pos2 is 'end': reversed(p2) + reversed(p1)[1:]
                            
                            if pos1 == 'end' and pos2 == 'start':
                                merged = p1 + p2[1:]
                            elif pos1 == 'end' and pos2 == 'end':
                                merged = p1 + list(reversed(p2))[1:]
                            elif pos1 == 'start' and pos2 == 'start':
                                merged = list(reversed(p1)) + p2[1:]
                            else:  # pos1 == 'start' and pos2 == 'end'
                                merged = list(reversed(p2)) + list(reversed(p1))[1:]
                            
                            if len(set(merged)) != len(merged):
                                continue
                            
                            # Valid merge!
                            new_paths = [p for k, p in enumerate(current_paths) 
                                         if k != i and k != j] + [merged]
                            if len(new_paths) < best_count:
                                best_paths = new_paths
                                best_count = len(new_paths)
                                current_paths = new_paths
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                # Try endpoint swapping
                for i in range(len(current_paths)):
                    for j in range(i + 1, len(current_paths)):
                        p1 = current_paths[i]
                        p2 = current_paths[j]
                        
                        # Try swapping endpoints
                        # Option: reverse p1
                        new_p1 = list(reversed(p1))
                        # Check if this creates a valid configuration
                        # (it always does since it's just a reorientation)
                        # But we need to check if it enables further merges
                        # For now, just keep it if it doesn't break anything
                        current_paths[i] = new_p1
                        improved = True
                        break
                    if improved:
                        break
            
            if not improved:
                break
        
        return best_paths
    
    def decompose_greedy(edges, rng, strategy='random'):
        """
        Runs a greedy path decomposition algorithm.
        Tries to build long paths by extending from both ends.
        Uses randomization in tie-breaking to explore different solutions.
        """
        if not edges:
            return []
        
        # Copy edges to a mutable set for removal
        remaining = set(edges)
        
        paths = []
        
        # Helper to get neighbors of a node in the remaining graph
        def get_neighbors(u):
            neighbors = []
            for e in remaining:
                if e[0] == u:
                    neighbors.append(e[1])
                elif e[1] == u:
                    neighbors.append(e[0])
            return neighbors

        # Helper to remove edge
        def remove_edge(u, v):
            if (u, v) in remaining:
                remaining.remove((u, v))
            else:
                remaining.remove((v, u))

        # Helper to extend path
        def extend_path(path):
            # path is a list of vertices
            # Try to extend from both ends until no more extension is possible
            changed = True
            while changed:
                changed = False
                # Try extending the back end
                if len(path) > 0:
                    back = path[-1]
                    neighbors = get_neighbors(back)
                    # Filter neighbors that are not already in the path
                    valid_neighbors = [n for n in neighbors if n not in path]
                    if valid_neighbors:
                        if strategy == 'random':
                            next_node = rng.choice(valid_neighbors)
                        elif strategy == 'low_deg':
                            # Prefer neighbors with lower remaining degree
                            # Calculate degree in remaining graph
                            degs = {}
                            for n in valid_neighbors:
                                degs[n] = len(get_neighbors(n))
                            min_deg = min(degs.values())
                            candidates = [n for n in valid_neighbors if degs[n] == min_deg]
                            next_node = rng.choice(candidates)
                        else:
                            next_node = rng.choice(valid_neighbors)
                        path.append(next_node)
                        remove_edge(back, next_node)
                        changed = True
                
                # Try extending the front end
                if len(path) > 0:
                    front = path[0]
                    neighbors = get_neighbors(front)
                    valid_neighbors = [n for n in neighbors if n not in path]
                    if valid_neighbors:
                        if strategy == 'random':
                            next_node = rng.choice(valid_neighbors)
                        elif strategy == 'low_deg':
                            degs = {}
                            for n in valid_neighbors:
                                degs[n] = len(get_neighbors(n))
                            min_deg = min(degs.values())
                            candidates = [n for n in valid_neighbors if degs[n] == min_deg]
                            next_node = rng.choice(candidates)
                        else:
                            next_node = rng.choice(valid_neighbors)
                        path.insert(0, next_node)
                        remove_edge(front, next_node)
                        changed = True
                        
        # Main loop: construct paths
        while remaining:
            # Pick a starting edge
            # Heuristic: Pick an edge incident to a vertex with the highest degree
            # Calculate degrees
            degrees = defaultdict(int)
            for u, v in remaining:
                degrees[u] += 1
                degrees[v] += 1
            
            # Find vertex with max degree
            max_deg = max(degrees.values())
            candidates = [v for v, d in degrees.items() if d == max_deg]
            start_u = rng.choice(candidates)
            
            # Pick a neighbor for the starting edge
            neighbors = get_neighbors(start_u)
            if not neighbors:
                continue 
            
            start_v = rng.choice(neighbors)
            
            # Initialize path
            path = [start_u, start_v]
            remove_edge(start_u, start_v)
            
            # Extend path
            extend_path(path)
            
            paths.append(path)
            
            # Check deadline
            if time.time() > deadline:
                break

        return paths

    def decompose_greedy_deterministic(edges, start_vertex=None):
        """
        Deterministic greedy decomposition.
        Starts from a specific vertex if provided, otherwise picks highest degree.
        Always picks the first valid neighbor in sorted order.
        """
        if not edges:
            return []
        
        remaining = set(edges)
        paths = []
        
        def get_neighbors(u):
            neighbors = []
            for e in remaining:
                if e[0] == u:
                    neighbors.append(e[1])
                elif e[1] == u:
                    neighbors.append(e[0])
            return neighbors

        def remove_edge(u, v):
            if (u, v) in remaining:
                remaining.remove((u, v))
            else:
                remaining.remove((v, u))

        def extend_path(path):
            changed = True
            while changed:
                changed = False
                if len(path) > 0:
                    back = path[-1]
                    neighbors = sorted(get_neighbors(back))
                    valid_neighbors = [n for n in neighbors if n not in path]
                    if valid_neighbors:
                        next_node = valid_neighbors[0]
                        path.append(next_node)
                        remove_edge(back, next_node)
                        changed = True
                
                if len(path) > 0:
                    front = path[0]
                    neighbors = sorted(get_neighbors(front))
                    valid_neighbors = [n for n in neighbors if n not in path]
                    if valid_neighbors:
                        next_node = valid_neighbors[0]
                        path.insert(0, next_node)
                        remove_edge(front, next_node)
                        changed = True
                        
        while remaining:
            if start_vertex is not None and start_vertex in [u for u, v in remaining] + [v for u, v in remaining]:
                start_u = start_vertex
                start_vertex = None  # Only use once
            else:
                degrees = defaultdict(int)
                for u, v in remaining:
                    degrees[u] += 1
                    degrees[v] += 1
                max_deg = max(degrees.values())
                candidates = sorted([v for v, d in degrees.items() if d == max_deg])
                start_u = candidates[0]
            
            neighbors = sorted(get_neighbors(start_u))
            if not neighbors:
                continue 
            
            start_v = neighbors[0]
            
            path = [start_u, start_v]
            remove_edge(start_u, start_v)
            
            extend_path(path)
            
            paths.append(path)
            
            if time.time() > deadline:
                break

        return paths

    def solve_instance(n, p, seed):
        edges = instance(n, p, seed)
        if not edges:
            return []
        
        edges_set = set(tuple(sorted(e)) for e in edges)
        best_paths = None
        best_count = float('inf')
        
        # Strategy 1: Eulerian trail decomposition with splitting
        # Generate multiple random Eulerian trails and split them
        num_eulerian_trials = 500
        for i in range(num_eulerian_trials):
            if time.time() > deadline - 2:
                break
            
            rng = random.Random(seed * 100000 + i)
            trails = eulerian_trails(edges, rng)
            
            # Split each trail into simple paths
            all_paths = []
            for trail in trails:
                paths = split_trail_to_paths(trail)
                all_paths.extend(paths)
            
            if validate_paths(all_paths, edges):
                if len(all_paths) < best_count:
                    best_count = len(all_paths)
                    best_paths = [list(p) for p in all_paths]
                
                # Try local search improvement
                if time.time() < deadline - 3:
                    improved = local_search_improve(all_paths, edges_set, rng, max_iters=20)
                    if validate_paths(improved, edges):
                        if len(improved) < best_count:
                            best_count = len(improved)
                            best_paths = [list(p) for p in improved]
                
                # Early termination if we hit a very good score
                if best_count <= 5:
                    break
        
        # Strategy 2: Greedy decomposition (original approach)
        # Try deterministic
        if time.time() < deadline - 1:
            paths = decompose_greedy_deterministic(edges)
            if validate_paths(paths, edges):
                if len(paths) < best_count:
                    best_count = len(paths)
                    best_paths = [list(p) for p in paths]
        
        # Try randomized greedy
        greedy_iterations = 1000
        for i in range(greedy_iterations):
            if time.time() > deadline - 1:
                break
            
            if i % 3 == 0:
                strategy = 'random'
            elif i % 3 == 1:
                strategy = 'low_deg'
            else:
                strategy = 'random'
            
            rng = random.Random(seed * 10000 + i)
            paths = decompose_greedy(edges, rng, strategy)
            
            if validate_paths(paths, edges):
                if len(paths) < best_count:
                    best_count = len(paths)
                    best_paths = [list(p) for p in paths]
                
                if best_count <= 5:
                    break

        if best_paths is None:
            # Fallback: return single edges if greedy failed completely (shouldn't happen)
            return [[list(e)] for e in edges]
            
        return best_paths

    def validate_paths(paths, edges):
        """Validate that paths are simple, edge-disjoint, and cover all edges."""
        if not paths:
            return len(edges) == 0
        
        edge_set = set(edges)
        used_edges = set()
        
        for path in paths:
            # Check simple path
            if len(set(path)) != len(path):
                return False
            
            # Check edges exist and are not reused
            for j in range(len(path) - 1):
                u, v = path[j], path[j+1]
                e = tuple(sorted((u, v)))
                if e not in edge_set:
                    return False
                if e in used_edges:
                    return False
                used_edges.add(e)
        
        # Check all edges are covered
        if used_edges != edge_set:
            return False
        
        return True

    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    results = []
    
    for spec in specs:
        n, p, seed = spec
        paths = solve_instance(n, p, seed)
        results.append(paths)
        
        if time.time() > deadline:
            break
            
    # If we didn't finish all specs, ensure we return a valid structure
    while len(results) < 3:
        # Generate a trivial valid solution: each edge as its own path
        n, p, seed = specs[len(results)]
        edges = instance(n, p, seed)
        results.append([[u, v] for u, v in edges])

    return results

if __name__ == "__main__":
    # Quick test
    res = construct()
    total = sum(len(r) for r in res)
    print(f"Total paths: {total}")
    for i, r in enumerate(res):
        print(f"Instance {i}: {len(r)} paths")
