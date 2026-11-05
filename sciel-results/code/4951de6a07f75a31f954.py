import time
import random
from collections import defaultdict, deque


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 150s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def validate_paths(paths, edges):
        """Validate that paths are simple, edge-disjoint, and cover all edges."""
        if not paths:
            return len(edges) == 0

        edge_set = set(edges)
        used_edges = set()

        for path in paths:
            if len(path) < 2:
                continue
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

        return used_edges == edge_set

    def greedy_decompose_init(edges, adj, vertices, rng):
        """
        Greedy path decomposition to create an initial set of paths.
        This is used as a starting point for the merging phase.
        """
        if not edges:
            return []
        
        remaining = set(edges)
        paths = []
        
        def get_neighbors(u):
            neighbors = []
            for nb in adj[u]:
                if (u, nb) in remaining or (nb, u) in remaining:
                    neighbors.append(nb)
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
                # Try extending the back end
                if len(path) > 0:
                    back = path[-1]
                    neighbors = get_neighbors(back)
                    valid_neighbors = [n for n in neighbors if n not in path]
                    if valid_neighbors:
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
                        next_node = rng.choice(valid_neighbors)
                        path.insert(0, next_node)
                        remove_edge(front, next_node)
                        changed = True
                        
        # Main loop: construct paths
        while remaining:
            # Pick a starting edge
            degrees = defaultdict(int)
            for u, v in remaining:
                degrees[u] += 1
                degrees[v] += 1
            
            if not degrees:
                break
                
            max_deg = max(degrees.values())
            candidates = [v for v, d in degrees.items() if d == max_deg]
            start_u = rng.choice(candidates)
            
            neighbors = get_neighbors(start_u)
            if not neighbors:
                e = next(iter(remaining))
                start_u, start_v = e
                remove_edge(start_u, start_v)
                paths.append([start_u, start_v])
                continue
            
            start_v = rng.choice(neighbors)
            
            path = [start_u, start_v]
            remove_edge(start_u, start_v)
            
            extend_path(path)
            
            paths.append(path)
            
            if time.time() > deadline - 10:
                break

        return paths

    def merge_paths(paths, max_iters=1000):
        """
        Iteratively merge paths that share endpoints.
        A merge is valid if the resulting path is simple (no repeated vertices).
        Returns the improved list of paths.
        """
        current_paths = [list(p) for p in paths]
        
        # Map vertex to list of (path_index, is_end)
        # is_end=True if vertex is the last element, False if first
        endpoint_map = defaultdict(list)
        
        def rebuild_endpoint_map():
            endpoint_map.clear()
            for i, p in enumerate(current_paths):
                if not p:
                    continue
                endpoint_map[p[0]].append((i, False))
                endpoint_map[p[-1]].append((i, True))
        
        rebuild_endpoint_map()
        
        improved = True
        iter_count = 0
        
        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1
            
            # Try to find a mergeable pair
            # We look for vertices that are endpoints of at least 2 paths
            merge_candidates = []
            for v, entries in endpoint_map.items():
                if len(entries) >= 2:
                    merge_candidates.append(v)
            
            if not merge_candidates:
                break
            
            # Randomize candidate order
            random.shuffle(merge_candidates)
            
            for v in merge_candidates:
                entries = endpoint_map[v]
                if len(entries) < 2:
                    continue
                
                # Get unique path indices
                path_indices = list(set([idx for idx, _ in entries]))
                
                if len(path_indices) < 2:
                    continue
                
                # Try to merge path_indices[0] and path_indices[1]
                idx1, idx2 = path_indices[0], path_indices[1]
                
                p1 = current_paths[idx1]
                p2 = current_paths[idx2]
                
                if not p1 or not p2:
                    continue
                
                # Determine how to merge
                # Case 1: p1 ends at v, p2 starts at v -> p1 + p2
                # Case 2: p1 ends at v, p2 ends at v -> p1 + reversed(p2)
                # Case 3: p1 starts at v, p2 starts at v -> reversed(p1) + p2
                # Case 4: p1 starts at v, p2 ends at v -> p2 + p1
                
                p1_end_v = (p1[-1] == v)
                p2_end_v = (p2[-1] == v)
                
                if p1_end_v and not p2_end_v:
                    new_path = p1 + p2
                elif p1_end_v and p2_end_v:
                    new_path = p1 + list(reversed(p2))
                elif not p1_end_v and not p2_end_v:
                    new_path = list(reversed(p1)) + p2
                else: # p1 starts at v, p2 ends at v
                    new_path = p2 + p1
                
                # Check simplicity
                if len(set(new_path)) == len(new_path):
                    # Valid merge
                    # Remove old paths and add new one
                    # We need to be careful with indices if we remove them
                    # Easier: mark as empty and filter later, or just rebuild list
                    current_paths[idx1] = new_path
                    current_paths[idx2] = []
                    
                    improved = True
                    # Rebuild endpoint map for next iteration
                    # Actually, we can just break and rebuild to be safe
                    break
            
            if improved:
                # Clean up empty paths
                current_paths = [p for p in current_paths if p]
                rebuild_endpoint_map()
                if time.time() > deadline - 15:
                    break
        
        return current_paths

    def solve_instance(n, p, seed):
        edges = instance(n, p, seed)
        if not edges:
            return []

        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        vertices = sorted(adj.keys())
        
        # Compute lower bound
        deg = {v: len(neigh) for v, neigh in adj.items()}
        odd_count = sum(1 for d in deg.values() if d % 2 == 1)
        lower_bound = max(1, (odd_count + 1) // 2)
        
        # Allocate time per instance
        time_per_instance = 35.0
        inst_start = time.time()
        inst_deadline = inst_start + time_per_instance
        
        best_paths = None
        best_count = float('inf')
        
        # Strategy: Multiple restarts of (Greedy Init + Merging)
        for restart in range(2000):
            if time.time() > inst_deadline - 2:
                break
            
            rng = random.Random(seed * 100000 + restart)
            
            # 1. Initial Greedy Decomposition
            init_paths = greedy_decompose_init(edges, adj, vertices, rng)
            
            if not init_paths:
                continue
            
            # 2. Merging Phase
            merged_paths = merge_paths(init_paths, max_iters=100)
            
            # 3. Validate
            if validate_paths(merged_paths, edges):
                if len(merged_paths) < best_count:
                    best_count = len(merged_paths)
                    best_paths = merged_paths
                    
                    # If we hit the lower bound, we can't do better
                    if best_count == lower_bound:
                        break
        
        if best_paths is None:
            # Fallback: Return single edges if all else fails (should not happen)
            return [[u, v] for u, v in edges]
            
        return best_paths

    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    results = []

    for idx, spec in enumerate(specs):
        n, p, seed = spec
        if time.time() > deadline - 10:
            edges = instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue
        paths = solve_instance(n, p, seed)
        results.append(paths)

    while len(results) < 3:
        n, p, seed = specs[len(results)]
        edges = instance(n, p, seed)
        results.append([[u, v] for u, v in edges])

    return results


if __name__ == "__main__":
    res = construct()
    total = sum(len(r) for r in res)
    print(f"Total paths: {total}")
    for i, r in enumerate(res):
        print(f"Instance {i}: {len(r)} paths")
