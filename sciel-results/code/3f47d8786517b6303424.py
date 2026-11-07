"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list for the full graph
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # We want to find 3 edge-disjoint Hamiltonian cycles.
    # State: set of available edges.
    # Since n=21 is small, we can represent the graph state by a set of frozensets or just a set of (min(u,v), max(u,v)).
    
    def find_hamiltonian_cycle(adj, used_edges, deadline):
        # adj is a dict of lists, but we must check 'used_edges' dynamically.
        # To speed up, we can precompute neighbors.
        # We use DFS with backtracking.
        # Start node 0 is fixed to reduce symmetry.
        
        path = [0]
        visited = {0}
        
        # Heuristic: Order neighbors? Maybe sort by degree in remaining graph?
        # For simplicity and speed on small graph, we can just iterate.
        # However, to find a solution quickly, we might want to try "greedy" paths or use Warnsdorff's rule?
        # Let's stick to simple DFS with random restarts if it gets stuck, but here we just need to find *one*.
        
        def dfs(u):
            if time.time() > deadline:
                return None
            
            if len(path) == n:
                # Check if edge (u, 0) exists and is available
                e = tuple(sorted((u, 0)))
                if e not in used_edges:
                    return list(path)
                return None
            
            # Get available neighbors
            neighbors = adj[u]
            # Shuffle to break symmetry and allow random restarts effectively if we call this function again?
            # Actually, inside a single DFS run, deterministic order is fine if we restart the whole function.
            # But we can shuffle locally to explore different branches.
            random.shuffle(neighbors)
            
            for v in neighbors:
                if v in visited:
                    continue
                e = tuple(sorted((u, v)))
                if e in used_edges:
                    continue
                
                # Additional pruning: 
                # If any unvisited vertex w (w != v) has no available edges to any other unvisited vertex or to u (if it were the last step, but we are not at last step yet), 
                # it might be a dead end. This is expensive to check fully.
                # Simple check: if v is the only unvisited neighbor of some unvisited node?
                
                # Let's just proceed.
                path.append(v)
                visited.add(v)
                
                res = dfs(v)
                if res is not None:
                    return res
                
                path.pop()
                visited.remove(v)
                
            return None
        
        return dfs(0)

    # Main loop
    deadline = time.time() + 50  # 50s limit to be safe
    
    used_edges = set()
    cycles = []
    
    # We need to find 3 cycles.
    # To increase probability of finding them, we can try multiple times if one fails?
    # But if we find one, we commit to it and remove its edges.
    # If we get stuck later, we might want to backtrack the choice of the first cycle?
    # That's complex. 
    # Given the graph is 6-regular and likely symmetric, any Hamiltonian cycle should leave a 5-regular graph (or close to it, if we remove 21 edges from 63, remaining is 42 edges, avg degree 4).
    # Actually, removing a Hamiltonian cycle from a 6-regular graph leaves a 5-regular graph? No.
    # Each vertex has degree 6. In a Hamiltonian cycle, each vertex has degree 2.
    # So remaining degree is 4. The remaining graph is 4-regular.
    # A 4-regular graph on 21 vertices has 42 edges.
    # We need 2 more Hamiltonian cycles (21 edges each).
    # 42 edges is exactly enough for 2 cycles.
    # So we need to partition the remaining 4-regular graph into 2 Hamiltonian cycles.
    # This implies the remaining graph must be 2-edge-colorable? Or just contain 2 disjoint HC.
    # Actually, if the original graph is 3-edge-colorable, the 3 cycles are the color classes.
    
    # Strategy:
    # 1. Find HC 1.
    # 2. Remove edges.
    # 3. Find HC 2 in remaining graph.
    # 4. Remove edges.
    # 5. Find HC 3 in remaining graph.
    
    # To handle cases where a bad first choice leads to a dead end for the second,
    # we can implement a simple backtracking on the set of cycles found, 
    # or simply restart the whole process if we fail to find 3.
    # Given the symmetry, it's highly probable that the first few random HCs found will work.
    
    best_cycles = []
    
    for attempt in range(1000): # Cap attempts
        if time.time() > deadline - 5:
            break
            
        current_used = set()
        current_cycles = []
        
        success = True
        for k in range(3):
            # Build adj for current available edges
            adj_cur = defaultdict(list)
            for u, v in edges:
                e = tuple(sorted((u, v)))
                if e not in current_used:
                    adj_cur[u].append(v)
                    adj_cur[v].append(u)
            
            # Check if any vertex has degree < 2, impossible for HC
            possible = True
            for i in range(n):
                if len(adj_cur[i]) < 2:
                    possible = False
                    break
            
            if not possible:
                success = False
                break
                
            # Find HC
            # We need to pass the dynamic adjacency and used edges.
            # My find_hamiltonian_cycle takes adj and used_edges.
            # But adj is built from *available* edges, so used_edges is effectively empty relative to adj_cur?
            # No, find_hamiltonian_cycle checks `if e in used_edges`.
            # If I pass adj_cur (which only has available edges), I should pass an empty set for used_edges?
            # Yes, because adj_cur already filters them out.
            # BUT, find_hamiltonian_cycle uses the global `adj` if I don't modify it.
            # I modified the function signature in my head, but the code above uses `adj` from outer scope?
            # Let's refactor find_hamiltonian_cycle to take the adjacency list as an argument.
            
            # Let's rewrite the inner search to be self-contained and fast.
            
            def find_hc_in_graph(adj_list):
                # adj_list: dict int -> list of int
                # Returns list of vertices or None
                path = [0]
                visited = {0}
                
                def dfs(u):
                    if time.time() > deadline - 5:
                        return None
                    if len(path) == n:
                        e = tuple(sorted((u, 0)))
                        # Check if e is in adj_list (implicitly, if 0 is in adj_list[u] and u is in adj_list[0])
                        # Since we built adj_list from available edges, we just check existence.
                        if 0 in adj_list[u]:
                            return list(path)
                        return None
                    
                    neighbors = adj_list[u]
                    # Heuristic: prioritize neighbors with fewer available options (Warnsdorff)
                    # This often helps in Hamiltonian path/cycle problems.
                    scored_neighbors = []
                    for v in neighbors:
                        if v in visited:
                            continue
                        # Count available edges from v to unvisited nodes (excluding u which is visited)
                        # Actually, just count degree in remaining graph?
                        # Simpler: just count len(adj_list[v])
                        scored_neighbors.append((len(adj_list[v]), v))
                    
                    scored_neighbors.sort()
                    
                    for deg, v in scored_neighbors:
                        path.append(v)
                        visited.add(v)
                        
                        res = dfs(v)
                        if res is not None:
                            return res
                        
                        path.pop()
                        visited.remove(v)
                    return None
                
                return dfs(0)

            hc = find_hc_in_graph(adj_cur)
            
            if hc is None:
                success = False
                break
            
            # Record cycle
            current_cycles.append(hc)
            
            # Remove edges from current_used
            for i in range(len(hc)):
                u = hc[i]
                v = hc[(i + 1) % len(hc)]
                e = tuple(sorted((u, v)))
                current_used.add(e)
        
        if success and len(current_cycles) == 3:
            # Verify edge disjointness just in case
            all_edges = set()
            for c in current_cycles:
                for i in range(len(c)):
                    u = c[i]
                    v = c[(i + 1) % len(c)]
                    e = tuple(sorted((u, v)))
                    if e in all_edges:
                        success = False
                        break
                    all_edges.add(e)
            if success:
                return [list(c) for c in current_cycles]
                
    # If we didn't find 3, return whatever we found? 
    # The task says "Maximize number of cycles". 
    # But "Any broken rule makes the whole result invalid".
    # Rules: "nonempty list", "edge-disjoint", "simple cycles", "same vertex set".
    # It does NOT say we MUST return 3. It says maximize.
    # So if we find 2, we should return 2.
    # However, the loop above only updates `best_cycles` if it fails? 
    # I should track the best partial result.
    
    # Let's restart the logic to track best.
    
    best_found = []
    
    for attempt in range(200):
        if time.time() > deadline - 5:
            break
            
        current_used = set()
        current_cycles = []
        
        for k in range(3):
            adj_cur = defaultdict(list)
            for u, v in edges:
                e = tuple(sorted((u, v)))
                if e not in current_used:
                    adj_cur[u].append(v)
                    adj_cur[v].append(u)
            
            possible = True
            for i in range(n):
                if len(adj_cur[i]) < 2:
                    possible = False
                    break
            
            if not possible:
                break
                
            def find_hc_in_graph(adj_list):
                path = [0]
                visited = {0}
                
                def dfs(u):
                    if time.time() > deadline - 5:
                        return None
                    if len(path) == n:
                        if 0 in adj_list[u]:
                            return list(path)
                        return None
                    
                    neighbors = adj_list[u]
                    scored_neighbors = []
                    for v in neighbors:
                        if v in visited:
                            continue
                        scored_neighbors.append((len(adj_list[v]), v))
                    
                    random.shuffle(scored_neighbors) # Add randomness to break ties and explore
                    scored_neighbors.sort(key=lambda x: x[0])
                    
                    for deg, v in scored_neighbors:
                        path.append(v)
                        visited.add(v)
                        
                        res = dfs(v)
                        if res is not None:
                            return res
                        
                        path.pop()
                        visited.remove(v)
                    return None
                
                return dfs(0)

            hc = find_hc_in_graph(adj_cur)
            
            if hc is None:
                break
            
            current_cycles.append(hc)
            
            for i in range(len(hc)):
                u = hc[i]
                v = hc[(i + 1) % len(hc)]
                e = tuple(sorted((u, v)))
                current_used.add(e)
                
        if len(current_cycles) > len(best_found):
            best_found = current_cycles
            
        if len(best_found) == 3:
            break

    return [list(c) for c in best_found]

# EVOLVE-BLOCK-END
