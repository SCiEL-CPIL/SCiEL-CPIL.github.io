import random
import time
import math

def construct():
    n = 16
    target_edges = 65
    
    # Helper to calculate score
    def get_score(adj, degs):
        max_sum = 0
        # Iterate over all potential triangles
        for u in range(n):
            for v in range(u + 1, n):
                if not adj[u][v]:
                    continue
                for w in range(v + 1, n):
                    if adj[u][w] and adj[v][w]:
                        s = degs[u] + degs[v] + degs[w]
                        if s > max_sum:
                            max_sum = s
        return max_sum

    def random_graph():
        adj = [[False] * n for _ in range(n)]
        edges = set()
        # Generate a random graph with exactly target_edges
        # To ensure connectivity or just random, we just pick random pairs
        # But we need exactly 65.
        # Total possible edges = 16*15/2 = 120.
        # We pick 65 randomly.
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_edges)
        for u, v in all_edges[:target_edges]:
            adj[u][v] = True
            adj[v][u] = True
        return adj

    def calculate_degrees(adj):
        degs = [0] * n
        for u in range(n):
            for v in range(n):
                if adj[u][v]:
                    degs[u] += 1
        return degs

    best_adj = None
    best_score = float('inf')
    
    start_time = time.time()
    time_limit = 110 # Leave 10s for safety
    
    # Simulated Annealing
    # We will run SA for a while, then polish.
    
    # Initial state: A structured graph might be better than random.
    # K_8,8 + 1 edge is a good start (Score 26).
    # Let's start with that and also try random restarts.
    
    def init_k88_plus():
        adj = [[False] * n for _ in range(n)]
        for u in range(8):
            for v in range(8, 16):
                adj[u][v] = True
                adj[v][u] = True
        # Add one edge within A
        adj[0][1] = True
        adj[1][0] = True
        return adj

    current_adj = init_k88_plus()
    current_degs = calculate_degrees(current_adj)
    current_score = get_score(current_adj, current_degs)
    
    best_adj = [row[:] for row in current_adj]
    best_score = current_score
    
    T = 10.0
    T_min = 0.01
    cooling = 0.995
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            break
            
        # Generate a neighbor: 2-opt swap
        # Remove a random edge
        # Find all edges
        edge_list = [(u, v) for u in range(n) for v in range(u + 1, n) if current_adj[u][v]]
        if not edge_list:
            break
        u1, v1 = random.choice(edge_list)
        
        # Add a random non-edge
        # Find all non-edges
        non_edge_list = [(u, v) for u in range(n) for v in range(u + 1, n) if not current_adj[u][v]]
        if not non_edge_list:
            break
        u2, v2 = random.choice(non_edge_list)
        
        # Apply move
        current_adj[u1][v1] = False
        current_adj[v1][u1] = False
        current_adj[u2][v2] = True
        current_adj[v2][u2] = True
        
        # Update degrees
        current_degs[u1] -= 1
        current_degs[v1] -= 1
        current_degs[u2] += 1
        current_degs[v2] += 1
        
        new_score = get_score(current_adj, current_degs)
        delta = new_score - current_score
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = [row[:] for row in current_adj]
                if best_score < 24:
                    # Found a very good solution, maybe try to polish or exit
                    # Let's continue to see if we can go lower
                    pass
        else:
            # Revert
            current_adj[u1][v1] = True
            current_adj[v1][u1] = True
            current_adj[u2][v2] = False
            current_adj[v2][u2] = False
            current_degs[u1] += 1
            current_degs[v1] += 1
            current_degs[u2] -= 1
            current_degs[v2] -= 1
            
        T *= cooling

    # Final Polish: Hill Climbing on best_adj
    # Try to improve best_adj with strict local search
    adj = [row[:] for row in best_adj]
    degs = calculate_degrees(adj)
    score = get_score(adj, degs)
    
    improved = True
    while improved and time.time() - start_time < time_limit - 2:
        improved = False
        # Try all possible 2-opt swaps? 
        # Number of swaps is E * (Non-E) ~ 65 * 55 ~ 3575.
        # Checking each is O(N^3) ~ 560.
        # 3575 * 560 ~ 2 million ops per pass.
        # This is fast enough for a few passes.
        
        edge_list = [(u, v) for u in range(n) for v in range(u + 1, n) if adj[u][v]]
        non_edge_list = [(u, v) for u in range(n) for v in range(u + 1, n) if not adj[u][v]]
        
        for u1, v1 in edge_list:
            for u2, v2 in non_edge_list:
                # Apply
                adj[u1][v1] = False
                adj[v1][u1] = False
                adj[u2][v2] = True
                adj[v2][u2] = True
                
                degs[u1] -= 1
                degs[v1] -= 1
                degs[u2] += 1
                degs[v2] += 1
                
                new_score = get_score(adj, degs)
                
                if new_score < score:
                    score = new_score
                    if score < best_score:
                        best_score = score
                        best_adj = [row[:] for row in adj]
                    improved = True
                else:
                    # Revert
                    adj[u1][v1] = True
                    adj[v1][u1] = True
                    adj[u2][v2] = False
                    adj[v2][u2] = False
                    degs[u1] += 1
                    degs[v1] += 1
                    degs[u2] -= 1
                    degs[v2] -= 1

    # Construct final edges
    final_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u][v]:
                final_edges.append([u, v])
                
    return {'n': n, 'edges': final_edges}
