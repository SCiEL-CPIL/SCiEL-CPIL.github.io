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

    def validate_paths(paths, edges):
        """
        Validate that the paths are simple, use graph edges, are edge-disjoint and cover all edges.
        """
        if not paths:
            return len(edges) == 0
        
        used_edges = set()
        edge_set = set(edges)
        
        for path in paths:
            # Check simple path
            if len(path) < 1:
                return False
            if len(set(path)) != len(path):
                return False
            
            for j in range(len(path) - 1):
                u, v = path[j], path[j+1]
                e = tuple(sorted((u, v)))
                if e not in edge_set:
                    return False
                if e in used_edges:
                    return False
                used_edges.add(e)
        
        return len(used_edges) == len(edge_set)

    def greedy_decompose(edges, rng, strategy='random'):
        """
        Greedy path decomposition. Returns a list of paths.
        """
        if not edges:
            return []
        
        remaining = set(edges)
        paths = []
        
        # Pre-compute adjacency for fast lookup, but we need to update it as edges are removed.
        # Since n is small (<=30), O(E) neighbor lookup is fine.
        
        def get_neighbors(u):
            neighbors = []
            for e in remaining:
                if e[0] == u:
                    neighbors.append(e[1])
                elif e[1] == u:
                    neighbors.append(e[0])
            return neighbors

        def remove_edge(u, v):
            e = tuple(sorted((u, v)))
            remaining.remove(e)

        def extend_path(path):
            changed = True
            while changed:
                changed = False
                # Try extending the back end
                if len(path) > 0:
                    back = path[-1]
                    neighbors = get_neighbors(back)
                    valid_neighbors = [n for n in neighbors if n not in path]
                    if valid_neighbors:
                        if strategy == 'random':
                            next_node = rng.choice(valid_neighbors)
                        elif strategy == 'low_deg':
                            degs = {n: len(get_neighbors(n)) for n in valid_neighbors}
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
                            degs = {n: len(get_neighbors(n)) for n in valid_neighbors}
                            min_deg = min(degs.values())
                            candidates = [n for n in valid_neighbors if degs[n] == min_deg]
                            next_node = rng.choice(candidates)
                        else:
                            next_node = rng.choice(valid_neighbors)
                        path.insert(0, next_node)
                        remove_edge(front, next_node)
                        changed = True
                        
        while remaining:
            if time.time() > deadline:
                break
                
            # Pick a starting edge
            # Heuristic: Pick an edge incident to a vertex with the highest degree
            degrees = defaultdict(int)
            for u, v in remaining:
                degrees[u] += 1
                degrees[v] += 1
            
            max_deg = max(degrees.values())
            candidates = [v for v, d in degrees.items() if d == max_deg]
            start_u = rng.choice(candidates)
            
            neighbors = get_neighbors(start_u)
            if not neighbors:
                continue 
            
            start_v = rng.choice(neighbors)
            
            path = [start_u, start_v]
            remove_edge(start_u, start_v)
            
            extend_path(path)
            
            paths.append(path)
            
        return paths

    def merge_paths(paths, edges, rng, max_merges=100):
        """
        Attempt to merge paths by finding valid connections between endpoints.
        This reduces the number of paths.
        """
        # Work on a copy of the paths
        current_paths = [list(p) for p in paths]
        
        if len(current_paths) <= 1:
            return current_paths
            
        # We need to know which edges are used to ensure we don't reuse them?
        # No, the paths already cover all edges. We are just reordering vertices.
        # The set of edges used is fixed.
        # We just need to ensure that when we connect end of A to start of B,
        # the edge (end_A, start_B) exists in the original graph.
        # And that the resulting path is simple (no repeated vertices).
        
        edge_set = set(edges)
        
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break
                
            # Try all pairs of paths (i, j) with i != j
            # We can try to connect end of i to start of j, or start of i to end of j, etc.
            # There are 4 possible ways to join two paths:
            # 1. end(i) -> start(j)  => new_path = i + j[1:]
            # 2. end(i) -> end(j)    => new_path = i + reverse(j)[1:]
            # 3. start(i) -> start(j) => new_path = reverse(i) + j[1:]
            # 4. start(i) -> end(j)   => new_path = reverse(i) + reverse(j)[1:]
            
            n_paths = len(current_paths)
            found_merge = False
            
            for i in range(n_paths):
                if found_merge:
                    break
                for j in range(n_paths):
                    if i == j:
                        continue
                    if found_merge:
                        break
                    
                    p_i = current_paths[i]
                    p_j = current_paths[j]
                    
                    # Check 4 connections
                    # 1. p_i[-1] connected to p_j[0]
                    if (p_i[-1], p_j[0]) in edge_set or (p_j[0], p_i[-1]) in edge_set:
                        new_path = p_i + p_j[1:]
                        if len(set(new_path)) == len(new_path):
                            # Valid merge
                            current_paths[i] = new_path
                            del current_paths[j]
                            improved = True
                            found_merge = True
                            break
                    
                    # 2. p_i[-1] connected to p_j[-1]
                    if (p_i[-1], p_j[-1]) in edge_set or (p_j[-1], p_i[-1]) in edge_set:
                        new_path = p_i + p_j[::-1][1:]
                        if len(set(new_path)) == len(new_path):
                            current_paths[i] = new_path
                            del current_paths[j]
                            improved = True
                            found_merge = True
                            break
                    
                    # 3. p_i[0] connected to p_j[0]
                    if (p_i[0], p_j[0]) in edge_set or (p_j[0], p_i[0]) in edge_set:
                        new_path = p_i[::-1] + p_j[1:]
                        if len(set(new_path)) == len(new_path):
                            current_paths[i] = new_path
                            del current_paths[j]
                            improved = True
                            found_merge = True
                            break
                    
                    # 4. p_i[0] connected to p_j[-1]
                    if (p_i[0], p_j[-1]) in edge_set or (p_j[-1], p_i[0]) in edge_set:
                        new_path = p_i[::-1] + p_j[::-1][1:]
                        if len(set(new_path)) == len(new_path):
                            current_paths[i] = new_path
                            del current_paths[j]
                            improved = True
                            found_merge = True
                            break
                            
        return current_paths

    def solve_instance(n, p, seed):
        edges = instance(n, p, seed)
        if not edges:
            return []
        
        best_paths = None
        best_count = float('inf')
        
        # Run multiple iterations with different random seeds for the greedy choices
        # Graphs are small. One iteration is very fast.
        
        iterations = 5000
        for i in range(iterations):
            if time.time() > deadline:
                break
            rng = random.Random(seed * 1000 + i)
            
            # Alternate strategies
            if i % 2 == 0:
                strategy = 'random'
            else:
                strategy = 'low_deg'
                
            paths = greedy_decompose(edges, rng, strategy)
            
            # Verify validity before merging
            if not validate_paths(paths, edges):
                continue
                
            # Try to merge
            merged_paths = merge_paths(paths, edges, rng)
            
            # Verify validity of merged paths
            if validate_paths(merged_paths, edges):
                if len(merged_paths) < best_count:
                    best_count = len(merged_paths)
                    best_paths = [list(p) for p in merged_paths]
            
            # If we reached the theoretical lower bound, we can stop early
            # Lower bound for paths covering all edges:
            # Let d(v) be degree. Number of paths >= max(1, (sum(d(v)-1) for odd d(v)) / 2) ?
            # Actually, for a graph, the minimum number of edge-disjoint paths covering all edges is at least:
            # 1 if Eulerian.
            # Otherwise, at least (number of odd degree vertices) / 2.
            # Wait, an Eulerian path exists if 0 or 2 vertices have odd degree.
            # If there are 2k odd degree vertices, we need at least k paths?
            # Yes, each path has 2 endpoints. Odd degree vertices must be endpoints of paths.
            # So min paths >= odd_count / 2.
            
            odd_count = 0
            degs = defaultdict(int)
            for u, v in edges:
                degs[u] += 1
                degs[v] += 1
            for v, d in degs.items():
                if d % 2 != 0:
                    odd_count += 1
                    
            lower_bound = max(1, odd_count // 2)
            
            if best_count == lower_bound:
                break

        if best_paths is None:
            # Fallback: return single edges if greedy failed completely (shouldn't happen)
            return [[list(e)] for e in edges]
            
        return best_paths

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
        # This should not happen if time management is correct
        # But if it does, we need to return something valid.
        # We can't easily generate a valid solution quickly if we ran out of time.
        # However, the greedy step is very fast, so we should have finished.
        # If we truly ran out of time, we might return an empty list or a trivial one.
        # The task says "return the best result found".
        # If we didn't find a valid result for an instance, what to do?
        # The prompt implies we should return valid paths.
        # If we fail, score is 0.
        # Let's try to generate a trivial valid solution: single edges.
        # This requires knowing the edges.
        edges = instance(n, p, seed)
        results.append([[list(e)] for e in edges])

    return results

if __name__ == "__main__":
    # Quick test
    res = construct()
    total = sum(len(r) for r in res)
    print(f"Total paths: {total}")
    for i, r in enumerate(res):
        print(f"Instance {i}: {len(r)} paths")
