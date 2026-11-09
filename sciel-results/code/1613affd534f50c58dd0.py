import random
import time

def construct():
    n = 20
    best_edges = []
    best_count = 0
    
    # Pre-calculate all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    start_time = time.time()
    time_limit = 115.0
    
    # Helper to check if adding edge (u,v) to current graph is valid
    # We maintain adj and C (common neighbor counts)
    
    def try_greedy_construction():
        random.shuffle(all_edges)
        adj = [[0]*n for _ in range(n)]
        C = [[0]*n for _ in range(n)]
        edges = []
        
        for u, v in all_edges:
            valid = True
            if C[u][v] > 2:
                valid = False
            else:
                for w in range(n):
                    if w == u or w == v:
                        continue
                    if C[u][w] + adj[v][w] > 2:
                        valid = False
                        break
                    if C[v][w] + adj[u][w] > 2:
                        valid = False
                        break
            
            if valid:
                edges.append((u, v))
                adj[u][v] = 1
                adj[v][u] = 1
                for w in range(n):
                    if w == u or w == v:
                        continue
                    if adj[u][w]:
                        C[v][w] += 1
                        C[w][v] += 1
                    if adj[v][w]:
                        C[u][w] += 1
                        C[w][u] += 1
        return edges, adj, C

    # Helper to check validity of a specific edge addition given adj and C
    def is_edge_valid(u, v, adj, C):
        if C[u][v] > 2:
            return False
        for w in range(n):
            if w == u or w == v:
                continue
            if C[u][w] + adj[v][w] > 2:
                return False
            if C[v][w] + adj[u][w] > 2:
                return False
        return True

    # Helper to add edge to state
    def add_edge(u, v, adj, C, edges_set):
        edges_set.add((u, v))
        adj[u][v] = 1
        adj[v][u] = 1
        for w in range(n):
            if w == u or w == v:
                continue
            if adj[u][w]:
                C[v][w] += 1
                C[w][v] += 1
            if adj[v][w]:
                C[u][w] += 1
                C[w][u] += 1

    # Helper to remove edge from state
    def remove_edge(u, v, adj, C, edges_set):
        edges_set.remove((u, v))
        adj[u][v] = 0
        adj[v][u] = 0
        for w in range(n):
            if w == u or w == v:
                continue
            if adj[u][w]:
                C[v][w] -= 1
                C[w][v] -= 1
            if adj[v][w]:
                C[u][w] -= 1
                C[w][u] -= 1

    while time.time() - start_time < time_limit:
        # 1. Greedy Initialization
        edges, adj, C = try_greedy_construction()
        edges_set = set(edges)
        current_count = len(edges)
        
        if current_count > best_count:
            best_count = current_count
            best_edges = [list(e) for e in edges]
            
        # 2. Local Search: Try to improve by removing one edge and adding more
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            # Shuffle edges to randomize order
            current_edges_list = list(edges_set)
            random.shuffle(current_edges_list)
            
            for e_out in current_edges_list:
                u_out, v_out = e_out
                
                # Save state
                # We can't easily deepcopy adj/C quickly, so we do manual undo
                # Or we can just try the swap and revert if not better
                
                # Remove e_out
                remove_edge(u_out, v_out, adj, C, edges_set)
                
                added_count = 0
                added_edges = []
                
                # Try to add any non-edge
                # To be fast, we can just iterate through all non-edges
                # But we only care if we can add MORE than 1 (net gain)
                # If we add 1, it's a swap (no gain in count, but might enable future gains)
                # For now, let's look for net gain > 0.
                
                # Optimization: Only try to add edges that were invalid before?
                # Simpler: Try all non-edges.
                non_edges = [e for e in all_edges if e not in edges_set]
                random.shuffle(non_edges)
                
                for e_in in non_edges:
                    u_in, v_in = e_in
                    if is_edge_valid(u_in, v_in, adj, C):
                        add_edge(u_in, v_in, adj, C, edges_set)
                        added_count += 1
                        added_edges.append(e_in)
                        
                        # If we added 2, we have a net gain of 1.
                        # If we added 3, net gain 2.
                        # We can stop early if we know we can't beat best?
                        # But we want to maximize.
                        # Let's keep adding if possible.
                
                if added_count > 1:
                    # Net gain is added_count - 1
                    # This is an improvement
                    improved = True
                    # State is already updated (e_out removed, e_in added)
                    # Update best if better
                    if len(edges_set) > best_count:
                        best_count = len(edges_set)
                        best_edges = [list(e) for e in edges_set]
                    # Break out of e_out loop to restart local search with new state?
                    # Or continue?
                    # Usually, after a big change, restart the local search phase.
                    break 
                
                elif added_count == 1:
                    # Swap: remove 1, add 1. Count unchanged.
                    # This might help escape local optima.
                    # We accept this move to change the structure.
                    improved = True # Mark as changed to continue loop
                    # State is updated
                    # We don't update best_count as it's same
                    # But we break to restart the "for e_out" loop with new state?
                    # Yes, because the set of edges changed.
                    break
                    
                else:
                    # added_count == 0
                    # Revert removal of e_out
                    add_edge(u_out, v_out, adj, C, edges_set)
                    
        # End local search for this restart
        
    return best_edges
