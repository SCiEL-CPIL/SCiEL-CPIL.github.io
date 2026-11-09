import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    # Helper to count simple paths of length 5 (5 edges) between u and v
    # in the graph defined by adj (list of sets/lists of neighbors)
    def count_paths_len5(adj, u, v, n):
        # We want paths u -> n1 -> n2 -> n3 -> n4 -> v
        # Length 5 edges.
        # u, n1, n2, n3, n4, v must be distinct.
        # Note: v is the target.
        # We start at u, depth 0.
        # We want to reach v at depth 5.
        
        count = 0
        # Use a stack for DFS to avoid recursion overhead if possible, 
        # but recursion depth 5 is very small.
        # Let's use recursion for clarity and speed (function call overhead is small for depth 5).
        
        # Optimization: Prune if remaining depth > distance to v?
        # We don't have distance.
        
        # visited set
        visited = set([u])
        
        def dfs(node, depth):
            nonlocal count
            if depth == 5:
                if node == v:
                    count += 1
                return
            
            for nei in adj[node]:
                if nei not in visited:
                    # If nei is v, we can only take it if depth == 4?
                    # No, path length is 5 edges.
                    # u (0) -> n1 (1) -> n2 (2) -> n3 (3) -> n4 (4) -> v (5).
                    # So at depth 4, we are at n4. We look at neighbors of n4.
                    # If neighbor is v, we go to v at depth 5.
                    # If neighbor is not v, we go to it at depth 5, but then we don't check if it's v.
                    # Wait.
                    # Depth 0: u
                    # Depth 1: n1
                    # Depth 2: n2
                    # Depth 3: n3
                    # Depth 4: n4
                    # Depth 5: v
                    # So we only count if we arrive at v at depth 5.
                    
                    if nei == v:
                        if depth == 4:
                             # This would mean path length 4?
                             # u -> n1 -> n2 -> n3 -> v. That's 4 edges.
                             # We want 5 edges.
                             # So we should NOT count this as a complete path unless depth becomes 5.
                             # But if we move to v, depth becomes 5.
                             # So we allow moving to v.
                             pass
                            
                    visited.add(nei)
                    dfs(nei, depth + 1)
                    visited.remove(nei)

        dfs(u, 0)
        return count

    def total_cost(adj, n, edges):
        cost = 0
        for c in range(4):
            for u, v in edges:
                # Check if edge (u,v) is in color c
                # To check, we can see if v is in adj[c][u]
                if v in adj[c][u]:
                    # Remove edge
                    adj[c][u].remove(v)
                    adj[c][v].remove(u)
                    cost += count_paths_len5(adj[c], u, v, n)
                    # Restore edge
                    adj[c][u].add(v)
                    adj[c][v].add(u)
        return cost

    def find_coloring(n, time_limit):
        m = n * (n - 1) // 2
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
        
        best_result = None
        max_restarts = 50 # Adjust based on time
        
        # Pre-allocate sets for adjacency
        # adj[c][u] is a set of neighbors
        
        while time.time() < start_time + time_limit:
            # Init random
            colors = [random.randint(0, 3) for _ in range(m)]
            
            adj = [[set() for _ in range(n)] for _ in range(4)]
            for idx, (u, v) in enumerate(edges):
                c = colors[idx]
                adj[c][u].add(v)
                adj[c][v].add(u)
            
            cost = total_cost(adj, n, edges)
            
            if cost == 0:
                return colors
            
            # SA Loop
            T = 1.0
            min_T = 0.01
            cooling = 0.999
            stuck = 0
            
            while time.time() < start_time + time_limit:
                if cost == 0:
                    return colors
                
                # Pick random edge
                idx = random.randint(0, m - 1)
                u, v = edges[idx]
                old_c = colors[idx]
                new_c = random.choice([0, 1, 2, 3])
                if new_c == old_c:
                    continue
                
                # Calculate Delta
                # 1. Remove from old
                adj[old_c][u].remove(v)
                adj[old_c][v].remove(u)
                paths_lost = count_paths_len5(adj[old_c], u, v, n)
                
                # 2. Count paths in new (current state, edge NOT present)
                paths_gained = count_paths_len5(adj[new_c], u, v, n)
                
                delta = paths_gained - paths_lost
                
                # Accept?
                if delta < 0:
                    accept = True
                elif delta == 0:
                    accept = random.random() < 0.5
                else:
                    accept = random.random() < pow(2.71828, -delta / T)
                
                if accept:
                    # Apply
                    adj[old_c][u].add(v) # Wait, we removed it. We need to re-add if we reject? No, we removed it for calc.
                    # If accept, we leave it removed from old_c.
                    # We need to add it to new_c.
                    adj[new_c][u].add(v)
                    adj[new_c][v].add(u)
                    colors[idx] = new_c
                    cost += delta
                    if cost < 0: cost = 0 # Safety
                    stuck = 0
                else:
                    # Reject: Restore old state
                    adj[old_c][u].add(v)
                    adj[old_c][v].add(u)
                    stuck += 1
                    if stuck > 1000:
                        break # Restart SA for this instance
                
                T *= cooling
                if T < min_T:
                    T = min_T
        
        return None

    # Main Loop
    # Try n=17 first (fast), then 18 (hard), then 19 (very hard)
    
    best_n = 9
    # Fallback: A trivially valid small graph
    best_colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]
    
    # 1. Find a valid 17 quickly
    # We allocate 5 seconds for this. It should be very fast.
    res_17 = find_coloring(17, 5)
    if res_17:
        best_n = 17
        best_colors = res_17
    
    # 2. Try to find a valid 18
    # Allocate remaining time minus 5s for safety
    time_for_18 = 100
    res_18 = find_coloring(18, time_for_18)
    if res_18:
        best_n = 18
        best_colors = res_18
        
        # 3. If 18 found, try 19 with remaining time
        elapsed = time.time() - start_time
        remaining = deadline - elapsed - 2
        if remaining > 10:
            res_19 = find_coloring(19, remaining)
            if res_19:
                best_n = 19
                best_colors = res_19
    
    return {'n': best_n, 'colors': best_colors}
