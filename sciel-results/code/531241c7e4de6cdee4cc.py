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

    def decompose_greedy(edges, rng, max_iter=1000):
        """
        Runs a greedy path decomposition algorithm.
        Tries to build long paths by extending from both ends.
        Uses randomization in tie-breaking to explore different solutions.
        """
        if not edges:
            return []
        
        # Copy edges to a mutable set for removal
        remaining = set(edges)
        # Build adjacency list for the remaining graph
        # We need to update this as edges are removed, or just check membership in 'remaining'
        
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
                        # Choose a neighbor. Heuristic: Random selection to allow exploration.
                        # Or highest degree? Let's stick to random for now, 
                        # but maybe bias towards higher degree is better for long paths.
                        # Let's try random first, as the outer loop will run many times.
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
                # Should not happen if remaining is not empty and connected? 
                # But graph might be disconnected.
                # If isolated edge, it's just [u, v]
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

    def solve_instance(n, p, seed):
        edges = instance(n, p, seed)
        if not edges:
            return []
        
        best_paths = None
        best_count = float('inf')
        
        # Run multiple iterations with different random seeds for the greedy choices
        # The number of iterations depends on the time budget
        # We have ~140s. Let's allocate time proportionally to graph size?
        # Or just run a fixed number of fast iterations.
        # Graphs are small (N=20-30, E~100-200). One iteration is very fast (<1ms).
        # We can run thousands of iterations.
        
        iterations = 2000
        for i in range(iterations):
            if time.time() > deadline:
                break
            rng = random.Random(seed * 1000 + i)
            paths = decompose_greedy(edges, rng)
            
            # Verify validity (simple, edge-disjoint, cover all)
            # This is important to ensure we don't return an invalid solution
            # 1. Edge-disjoint and cover all:
            used_edges = set()
            valid = True
            for path in paths:
                for j in range(len(path) - 1):
                    u, v = path[j], path[j+1]
                    e = tuple(sorted((u, v)))
                    if e in used_edges:
                        valid = False
                        break
                    used_edges.add(e)
                    # Check if edge exists in original graph
                    if e not in set(edges):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid and len(used_edges) == len(edges):
                # 2. Simple paths:
                for path in paths:
                    if len(set(path)) != len(path):
                        valid = False
                        break
            
            if valid:
                if len(paths) < best_count:
                    best_count = len(paths)
                    best_paths = paths
            
            # If we reached the theoretical lower bound, we can stop early
            # Lower bound: max(1, ceil(sum(degrees(v)-1)/2) ) ? No, that's for trails.
            # For paths, a lower bound is related to the number of odd degree vertices?
            # Actually, just checking if we hit a very good score like 1 (Eulerian path) 
            # or close to it is enough.
            if best_count == 1:
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
            # If we ran out of time, fill remaining with trivial solution?
            # But we should have finished.
            break
            
    # If we didn't finish all specs, ensure we return a valid structure
    while len(results) < 3:
         # This should not happen if time management is correct
         results.append([])

    return results

if __name__ == "__main__":
    # Quick test
    res = construct()
    total = sum(len(r) for r in res)
    print(f"Total paths: {total}")
    for i, r in enumerate(res):
        print(f"Instance {i}: {len(r)} paths")
