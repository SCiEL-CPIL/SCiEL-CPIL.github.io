import time
import random
import itertools

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    deadline = time.time() + 55
    rng = random.Random(42)
    
    # Build adjacency structures
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    all_edges_set = set((min(u, v), max(u, v)) for u, v in edges)
    
    def get_cycle_edges(cycle):
        es = set()
        for i in range(len(cycle)):
            u, v = cycle[i], cycle[(i+1) % len(cycle)]
            es.add((min(u, v), max(u, v)))
        return es
    
    def find_ham_cycle(adj_set, forbidden_edges, time_limit=5.0):
        """
        Find a Hamiltonian cycle in the graph defined by adj_set,
        avoiding edges in forbidden_edges.
        Uses backtracking with heuristic ordering.
        """
        start_time = time.time()
        
        # Precompute valid neighbors for each node given forbidden edges
        # valid_adj[v] = list of neighbors u such that (v,u) is in adj_set and not forbidden
        valid_adj = {i: [] for i in range(n)}
        for u in range(n):
            for v in adj_set[u]:
                ekey = (min(u, v), max(u, v))
                if ekey not in forbidden_edges:
                    valid_adj[u].append(v)
        
        # Sort neighbors by degree (Warnsdorff's rule) to find solutions faster
        # Lower degree neighbors first
        for u in range(n):
            valid_adj[u].sort(key=lambda x: len(valid_adj[x]))
            
        # Try starting from each vertex? Or just 0?
        # Trying 0 is usually enough if the graph is symmetric or well-connected.
        # But to be safe, we can try a few starts if the first fails quickly.
        
        def backtrack(path, visited):
            if time.time() - start_time > time_limit:
                return None
            
            if len(path) == n:
                last = path[-1]
                start = path[0]
                if start in valid_adj[last]:
                    # Check if closing edge is valid (it is, by construction of valid_adj)
                    return list(path)
                return None
            
            last = path[-1]
            
            # Heuristic: if we are near the end, prioritize neighbors that can connect back to start
            if len(path) >= n - 2:
                # Create a sorted list where 'start' is prioritized if it's a valid neighbor
                # Actually, valid_adj[last] already contains only valid neighbors.
                # We just need to check if 'start' is in valid_adj[last] if we are at n-1.
                # For n-2, we want neighbors that have 'start' in their valid_adj.
                
                # Simple approach: just iterate valid_adj[last]
                # But let's sort dynamically if we are close to the end
                pass
            
            for neighbor in valid_adj[last]:
                if neighbor not in visited:
                    # Additional pruning: if this is the last step, neighbor must connect to start
                    if len(path) == n - 1:
                        if path[0] not in valid_adj[neighbor]:
                            continue
                    
                    path.append(neighbor)
                    visited.add(neighbor)
                    result = backtrack(path, visited)
                    if result:
                        return result
                    path.pop()
                    visited.remove(neighbor)
            
            return None
        
        # Try starting from 0
        path = [0]
        visited = {0}
        result = backtrack(path, visited)
        if result:
            return result
            
        # If 0 didn't work, try a few random starts
        for start in range(1, n):
            if time.time() - start_time > time_limit:
                break
            path = [start]
            visited = {start}
            result = backtrack(path, visited)
            if result:
                return result
                
        return None

    def solve():
        best_result = []
        
        # Step 1: Find C1
        c1 = find_ham_cycle(adj, set(), time_limit=10.0)
        if not c1:
            return []
        
        used_1 = get_cycle_edges(c1)
        
        # Step 2: Find C2
        c2 = find_ham_cycle(adj, used_1, time_limit=10.0)
        if not c2:
            return [c1]
        
        used_2 = get_cycle_edges(c2)
        all_used = used_1 | used_2
        
        # Step 3: Find C3
        c3 = find_ham_cycle(adj, all_used, time_limit=15.0)
        if c3:
            return [c1, c2, c3]
        else:
            return [c1, c2]

    result = solve()
    
    # Validation and Type Conversion
    if not result:
        return []
        
    final_result = []
    seen_edges = set()
    vset = set(range(n))
    
    for c in result:
        # Ensure it's a list of ints
        c = [int(v) for v in c]
        # Check vertex set
        if set(c) != vset or len(c) != n:
            continue
            
        edges_in_cycle = get_cycle_edges(c)
        # Check edge disjointness
        if edges_in_cycle & seen_edges:
            continue
            
        # Check if edges are in host graph
        if not edges_in_cycle.issubset(all_edges_set):
            continue
            
        seen_edges.update(edges_in_cycle)
        final_result.append(c)
        
    return final_result
