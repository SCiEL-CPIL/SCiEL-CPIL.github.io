import random
import time
import math

def construct():
    n = 20
    
    # Precompute edge indices
    edge_list = []
    edge_indices = {}
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_indices[(u, v)] = idx
            edge_list.append((u, v))
            idx += 1
    m = idx  # total possible edges = 190

    # Build conflict graph: two edges conflict if they form a K_{2,3} subgraph
    # Edge e1=(a,b) and e2=(c,d) conflict if {a,b} and {c,d} share exactly 2 vertices
    # and there exist 3 common neighbors, OR they are disjoint and form a K_{2,3}
    # Actually, K_{2,3} means two vertices u,v with 3 common neighbors w1,w2,w3
    # This happens when edges (u,w1), (u,w2), (u,w3), (v,w1), (v,w2), (v,w3) all exist
    # So edges (u,w_i) and (v,w_j) for i!=j can be part of K_{2,3}
    # More precisely: two edges (a,b) and (c,d) are in conflict if adding both to a graph
    # could create a K_{2,3}. The key insight: edges (u,x) and (v,y) where u!=v, x!=y
    # conflict if there exists a third vertex z such that (u,z) and (v,z) are also present.
    # But for the conflict graph approach, we need: edges e1 and e2 cannot both be in
    # a K_{2,3}-free graph if they are part of some K_{2,3}.
    
    # Simpler characterization: Two edges (a,b) and (c,d) conflict if:
    # 1. They share a vertex, say a=c, and there exist 3 vertices x,y,z such that
    #    (a,x),(a,y),(a,z),(b,x),(b,y),(b,z) could form K_{2,3} - but this requires 6 edges
    # 2. Actually, the standard approach: edges (u,v) and (u,w) conflict if there are
    #    at least 2 other common neighbors of v and w... this is getting complex.
    
    # Let me use a different approach: directly check if adding an edge to a current
    # K_{2,3}-free graph keeps it K_{2,3}-free, and use branch-and-bound with greedy upper bounds.
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Greedy initialization: build a good starting graph
    def greedy_build():
        adj = [0] * n
        # Start with 5 disjoint K4s
        for k in range(5):
            base = 4 * k
            for a in range(4):
                for b in range(a + 1, 4):
                    u = base + a
                    v = base + b
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
        return adj
    
    def edge_count(adj):
        count = 0
        for u in range(n):
            count += bin(adj[u]).count('1')
        return count // 2
    
    def is_valid_addition(adj, u, v):
        # Check if adding (u, v) keeps graph K2,3-free
        # After adding (u,v), we need no two vertices to have 3+ common neighbors
        # The only new common neighbor relationships involve u or v
        # For any w != u,v: N(u) ∩ N(w) and N(v) ∩ N(w) must have size <= 2
        # But wait, we need <= 1 for K_{2,3} to be absent (no 3 common neighbors)
        
        # Check pairs involving u: for each w in N(v) \ {u}, check |N(u) ∩ N(w)| <= 1
        # Note: u is not in N(v) yet, so N(v) doesn't include u
        neighbors_v = adj[v]
        temp_v = neighbors_v
        while temp_v:
            lsb = temp_v & -temp_v
            w = lsb.bit_length() - 1
            temp_v ^= lsb
            if w == u:
                continue
            common = adj[u] & adj[w]
            if bin(common).count('1') > 1:
                return False
        
        # Check pairs involving v: for each w in N(u) \ {v}, check |N(v) ∩ N(w)| <= 1
        neighbors_u = adj[u]
        temp_u = neighbors_u
        while temp_u:
            lsb = temp_u & -temp_u
            w = lsb.bit_length() - 1
            temp_u ^= lsb
            if w == v:
                continue
            common = adj[v] & adj[w]
            if bin(common).count('1') > 1:
                return False
        
        return True
    
    def add_edge(adj, u, v):
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    
    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    
    # Get a good initial solution using greedy + local search
    current_adj = greedy_build()
    current_edges = edge_count(current_adj)
    
    best_adj = [x for x in current_adj]
    best_edges = current_edges
    
    # Local search to improve initial solution
    all_edges = list(edge_list)
    improved = True
    while improved:
        improved = False
        for u, v in all_edges:
            if not ((current_adj[u] >> v) & 1):
                if is_valid_addition(current_adj, u, v):
                    add_edge(current_adj, u, v)
                    current_edges += 1
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = [x for x in current_adj]
                    improved = True
    
    # Now use branch-and-bound with the current best as lower bound
    # State: (adj, edges_count, next_edge_index)
    # We try to add edges from next_edge_index onwards
    
    best_solution = [x for x in best_adj]
    best_solution_edges = best_edges
    
    # Precompute for speed: for each edge, which other edges it conflicts with
    # Two edges (a,b) and (c,d) conflict if they cannot both be in a K_{2,3}-free graph
    # This is complex, so we'll use a simpler bound: the number of remaining edges
    # that can potentially be added
    
    def branch_and_bound(adj, edges_count, start_idx):
        nonlocal best_solution, best_solution_edges, deadline, start_time
        
        if time.time() > deadline:
            return
        
        # Upper bound: edges_count + (m - start_idx) [optimistic: all remaining edges can be added]
        upper_bound = edges_count + (m - start_idx)
        if upper_bound <= best_solution_edges:
            return
        
        if start_idx >= m:
            if edges_count > best_solution_edges:
                best_solution_edges = edges_count
                best_solution = [x for x in adj]
            return
        
        # Try to add edge at start_idx
        u, v = edge_list[start_idx]
        if not ((adj[u] >> v) & 1):
            if is_valid_addition(adj, u, v):
                # Branch: include this edge
                new_adj = [x for x in adj]
                add_edge(new_adj, u, v)
                branch_and_bound(new_adj, edges_count + 1, start_idx + 1)
        
        # Branch: exclude this edge
        branch_and_bound(adj, edges_count, start_idx + 1)
    
    # The full B&B is too slow, so use a hybrid: restart local search with different seeds
    # and keep the best result
    
    # Multiple restarts with simulated annealing
    for restart in range(100):
        if time.time() > deadline:
            break
        
        # Random restart: start from a random subset of edges
        random_adj = [0] * n
        for u, v in all_edges:
            if random.random() < 0.3:  # 30% chance to include
                if is_valid_addition(random_adj, u, v):
                    add_edge(random_adj, u, v)
        
        random_edges = edge_count(random_adj)
        
        # Simulated annealing from this start
        T = 1.0
        cooling_rate = 0.9999
        current = [x for x in random_adj]
        curr_edges = random_edges
        
        for step in range(5000):
            if time.time() > deadline:
                break
            
            move_type = random.random()
            
            if move_type < 0.7:
                # Try to add a random edge
                u, v = random.choice(all_edges)
                if not ((current[u] >> v) & 1):
                    if is_valid_addition(current, u, v):
                        add_edge(current, u, v)
                        curr_edges += 1
                        if curr_edges > best_solution_edges:
                            best_solution_edges = curr_edges
                            best_solution = [x for x in current]
                    else:
                        T *= 0.99
                else:
                    T *= 0.999
            else:
                # Try a swap
                u, v = random.choice(all_edges)
                if ((current[u] >> v) & 1):
                    x, y = random.choice(all_edges)
                    if ((current[x] >> y) & 1) or (x == u and y == v) or (x == v and y == u):
                        T *= 0.999
                        continue
                    
                    new_adj = [a for a in current]
                    remove_edge(new_adj, u, v)
                    
                    if is_valid_addition(new_adj, x, y):
                        add_edge(new_adj, x, y)
                        # Same edge count, accept to explore
                        current = new_adj
                        T *= 0.999
                    else:
                        T *= 0.99
                else:
                    T *= 0.999
            
            T *= cooling_rate
            if T < 0.01:
                T = 0.01
    
    # Convert best solution to list of edges
    result = []
    for (u, v), idx in edge_indices.items():
        if (best_solution[u] >> v) & 1:
            result.append([u, v])
    return result
