import time
import itertools

def construct():
    start_time = time.time()
    deadline = 90.0
    
    # 1. Generate odds
    odds = [i for i in range(1, 257) if i % 2 != 0]
    
    # 2. Generate candidate evens (those that don't conflict with any odd)
    evens = [i for i in range(1, 257) if i % 2 == 0]
    
    def conflicts(a, b):
        return (a * b) % (a + b) == 0
    
    cand_evens = []
    for e in evens:
        valid = True
        for o in odds:
            if conflicts(e, o):
                valid = False
                break
        if valid:
            cand_evens.append(e)
            
    # 3. Find Max Independent Set in cand_evens
    # Graph nodes: indices of cand_evens
    n = len(cand_evens)
    # Build adjacency list or matrix
    # adj[i][j] = True if cand_evens[i] conflicts with cand_evens[j]
    adj = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if conflicts(cand_evens[i], cand_evens[j]):
                adj[i][j] = True
                adj[j][i] = True
                
    # Solve Max Independent Set
    # Since n is small, we can use a simple branch and bound or even brute force if n < 25
    # Let's use a simple greedy + local search or DFS
    
    # Greedy initialization
    selected = []
    for i in range(n):
        # Check if i conflicts with any already selected
        can_add = True
        for j in selected:
            if adj[i][j]:
                can_add = False
                break
        if can_add:
            selected.append(i)
            
    # Local search to improve
    improved = True
    while improved:
        improved = False
        # Try to add any unselected node
        for i in range(n):
            if i not in selected:
                can_add = True
                for j in selected:
                    if adj[i][j]:
                        can_add = False
                        break
                if can_add:
                    selected.append(i)
                    improved = True
        
        # Try to swap: remove one, add two? 
        # Or just simple hill climbing: remove one, see if we can add more
        # This is complex. Given small n, let's try all subsets if n is small enough
        # Or use a better heuristic.
        
    # Actually, let's just use a simple recursive Max Independent Set solver
    # which is fast for n < 30
    
    def max_independent_set(vertices, adj):
        if not vertices:
            return []
        
        # Pick a vertex with max degree in the induced subgraph?
        # Or just pick the first one.
        v = vertices[0]
        rest = vertices[1:]
        
        # Option 1: Include v
        # Must exclude neighbors of v
        neighbors = {u for u in rest if adj[v][u]}
        non_neighbors = [u for u in rest if u not in neighbors]
        
        # Recurse on non-neighbors
        sol_with_v = [v] + max_independent_set(non_neighbors, adj)
        
        # Option 2: Exclude v
        sol_without_v = max_independent_set(rest, adj)
        
        if len(sol_with_v) >= len(sol_without_v):
            return sol_with_v
        else:
            return sol_without_v
            
    # The recursive above might be slow if not careful, but n is small.
    # Let's use a more efficient backtracking with pruning or just the recursive one
    # since n is likely < 20.
    
    # Re-verify n
    if n > 25:
        # Fallback to greedy if too many
        selected = []
        for i in range(n):
            can_add = True
            for j in selected:
                if adj[i][j]:
                    can_add = False
                    break
            if can_add:
                selected.append(i)
    else:
        selected = max_independent_set(list(range(n)), adj)
        
    final_evens = [cand_evens[i] for i in selected]
    
    result = odds + final_evens
    
    return result
