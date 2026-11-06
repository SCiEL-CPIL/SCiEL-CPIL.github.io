import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # We want to find a C4-free graph G on n vertices with min degree >= n - 39.
    # This is equivalent to finding a C4-free graph H (complement of G) on n vertices
    # with max degree <= 38.
    # We search for H, then return edges of G.
    
    best_n = 39
    best_edges = []
    
    # Try n from 47 down to 39
    for n in range(47, 38, -1):
        if time.time() > deadline:
            break
            
        max_deg_limit = 38
        
        # Simulated Annealing to find H
        # State: list of sets of neighbors
        # Cost: num_C4s + 100 * max(0, max_degree - 38)
        
        def count_c4s(adj):
            # Count number of C4s in the graph
            # A C4 is a set of 4 vertices {u,v,w,x} such that edges (u,v), (v,w), (w,x), (x,u) exist.
            # Or simply count paths of length 3 and divide by 4?
            # Number of C4s = (1/4) * sum_{u,v} (common_neighbors(u,v)^2 - common_neighbors(u,v)) ? No.
            # Standard formula: Number of C4s = (1/4) * (sum_{u,v} C(common_neighbors(u,v), 2) - sum_{u} C(deg(u), 2) ? No.
            # Let's just iterate over pairs of edges (u,v) and (w,x) and check if they form a C4?
            # For n=47, m is small.
            # Better: For each pair of vertices (u, v), let c = |N(u) \cap N(v)|.
            # The number of C4s containing the "diagonal" or sides?
            # Each C4 has 2 diagonals.
            # For a pair (u, v), the number of C4s where u and v are opposite is C(c, 2).
            # So total C4s = (1/2) * sum_{u < v} C(|N(u) \cap N(v)|, 2).
            
            count = 0
            for u in range(n):
                for v in range(u + 1, n):
                    # Count common neighbors
                    # adj[u] and adj[v] are sets
                    common = len(adj[u].intersection(adj[v]))
                    if common >= 2:
                        count += common * (common - 1) // 2
            return count // 2
        
        def get_cost(adj):
            c4s = count_c4s(adj)
            max_deg = max(len(adj[i]) for i in range(n))
            deg_penalty = max(0, max_deg - max_deg_limit) * 100
            return c4s + deg_penalty

        found_valid = False
        best_adj = None
        best_cost = float('inf')
        
        # Initial state: random sparse graph
        # Start with a random graph with average degree ~ 20
        for trial in range(50):
            if time.time() > deadline:
                break
            
            adj = [set() for _ in range(n)]
            # Add random edges
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.3:
                        adj[i].add(j)
                        adj[j].add(i)
            
            current_cost = get_cost(adj)
            best_adj = [set(s) for s in adj]
            best_cost = current_cost
            
            T = 100.0
            T_min = 0.01
            cooling = 0.995
            steps = 0
            max_steps = 20000
            
            while T > T_min and steps < max_steps:
                if time.time() > deadline:
                    break
                
                # Generate neighbor state
                # Choose a random move: add edge, remove edge, or swap
                move_type = random.choice(['add', 'remove', 'swap'])
                new_adj = [set(s) for s in adj]
                
                if move_type == 'add':
                    u = random.randint(0, n - 1)
                    v = random.randint(0, n - 1)
                    if u != v and v not in adj[u]:
                        new_adj[u].add(v)
                        new_adj[v].add(u)
                elif move_type == 'remove':
                    # Pick a random edge to remove
                    # To avoid empty list, pick a vertex with degree > 0
                    candidates = [i for i in range(n) if len(adj[i]) > 0]
                    if not candidates:
                        continue
                    u = random.choice(candidates)
                    v = random.choice(list(adj[u]))
                    new_adj[u].remove(v)
                    new_adj[v].remove(u)
                else: # swap
                    # Remove one edge, add another
                    candidates = [i for i in range(n) if len(adj[i]) > 0]
                    if not candidates:
                        continue
                    u1 = random.choice(candidates)
                    v1 = random.choice(list(adj[u1]))
                    new_adj[u1].remove(v1)
                    new_adj[v1].remove(u1)
                    
                    u2 = random.randint(0, n - 1)
                    v2 = random.randint(0, n - 1)
                    if u2 != v2 and v2 not in new_adj[u2]:
                        new_adj[u2].add(v2)
                        new_adj[v2].add(u2)
                    else:
                        # If can't add, just undo removal? Or skip move.
                        new_adj[u1].add(v1)
                        new_adj[v1].add(u1)
                        continue
                
                new_cost = get_cost(new_adj)
                delta_cost = new_cost - current_cost
                
                if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                    adj = new_adj
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_adj = [set(s) for s in adj]
                        
                    if current_cost == 0:
                        found_valid = True
                        break
                else:
                    # Revert
                    pass
                
                T *= cooling
                steps += 1
                
            if found_valid:
                break
        
        if found_valid:
            # best_adj is H. We need G = complement of H.
            edges_G = []
            for u in range(n):
                for v in range(u + 1, n):
                    if v not in best_adj[u]:
                        edges_G.append([u, v])
            
            # Verify
            # Check C4 in G
            adj_G = [set() for _ in range(n)]
            for u, v in edges_G:
                adj_G[u].add(v)
                adj_G[v].add(u)
                
            # Check min degree
            min_deg = min(len(adj_G[i]) for i in range(n))
            if min_deg < n - 39:
                # Should not happen if best_cost == 0, but just in case
                continue
                
            # Check C4 in G
            has_c4 = False
            for u in range(n):
                for v in range(u + 1, n):
                    if v in adj_G[u]:
                        continue
                    common = len(adj_G[u].intersection(adj_G[v]))
                    if common >= 2:
                        has_c4 = True
                        break
                if has_c4:
                    break
            
            if not has_c4:
                best_n = n
                best_edges = edges_G
                return {'n': n, 'edges': best_edges}
        
    # If no valid graph found for n > 39, return a trivial one or best found
    # A graph with n=39, min degree 0 is always valid (empty graph)
    # But we want to maximize n.
    # If we failed to find one, we return the best we have or a fallback.
    # Let's try to construct a valid one for n=39 just to be safe?
    # For n=39, min deg >= 0. Empty graph is C4-free.
    return {'n': best_n, 'edges': best_edges}
