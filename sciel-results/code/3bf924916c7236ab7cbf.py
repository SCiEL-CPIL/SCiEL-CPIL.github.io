import random
import time
import math

def construct():
    n = 16
    target_edges = 65
    
    def build_k88_plus_one():
        """Build K_{8,8} + one edge within A"""
        adj = [[False]*n for _ in range(n)]
        for u in range(8):
            for v in range(8, 16):
                adj[u][v] = adj[v][u] = True
        adj[0][1] = adj[1][0] = True
        return adj
    
    def build_k88_plus_one_b():
        """Build K_{8,8} + one edge within B"""
        adj = [[False]*n for _ in range(n)]
        for u in range(8):
            for v in range(8, 16):
                adj[u][v] = adj[v][u] = True
        adj[8][9] = adj[9][8] = True
        return adj
    
    def build_k88_plus_cross():
        """Build K_{8,8} with one cross edge removed and two within-part edges added"""
        adj = [[False]*n for _ in range(n)]
        for u in range(8):
            for v in range(8, 16):
                adj[u][v] = adj[v][u] = True
        # Remove one cross edge
        adj[0][8] = adj[8][0] = False
        # Add two within-part edges
        adj[0][1] = adj[1][0] = True
        adj[2][3] = adj[3][2] = True
        return adj
    
    def get_edges(adj):
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    edges.append((u, v))
        return edges
    
    def get_degrees(adj):
        degs = [0]*n
        for u in range(n):
            for v in range(n):
                if adj[u][v]:
                    degs[u] += 1
        return degs
    
    def calc_score(adj, degs):
        max_sum = 0
        for u in range(n):
            for v in range(u+1, n):
                if not adj[u][v]:
                    continue
                for w in range(v+1, n):
                    if adj[u][w] and adj[v][w]:
                        s = degs[u] + degs[v] + degs[w]
                        if s > max_sum:
                            max_sum = s
        return max_sum
    
    def full_score(adj):
        degs = get_degrees(adj)
        return calc_score(adj, degs)
    
    def try_all_swaps(adj, degs, score):
        """Try all possible single edge swaps, return best improvement"""
        edges = get_edges(adj)
        non_edges = []
        for u in range(n):
            for v in range(u+1, n):
                if not adj[u][v]:
                    non_edges.append((u, v))
        
        best_score = score
        best_swap = None
        
        for rem_u, rem_v in edges:
            for add_u, add_v in non_edges:
                # Skip if same edge
                if (rem_u, rem_v) == (add_u, add_v) or (rem_u, rem_v) == (add_v, add_u):
                    continue
                
                # Apply swap
                adj[rem_u][rem_v] = adj[rem_v][rem_u] = False
                adj[add_u][add_v] = adj[add_v][add_u] = True
                
                # Update degrees
                new_degs = degs[:]
                new_degs[rem_u] -= 1
                new_degs[rem_v] -= 1
                new_degs[add_u] += 1
                new_degs[add_v] += 1
                
                new_score = calc_score(adj, new_degs)
                
                # Revert
                adj[rem_u][rem_v] = adj[rem_v][rem_u] = True
                adj[add_u][add_v] = adj[add_v][add_u] = False
                
                if new_score < best_score:
                    best_score = new_score
                    best_swap = (rem_u, rem_v, add_u, add_v)
        
        return best_swap, best_score
    
    def local_search(adj, max_time=5.0):
        """Run local search until no improvement or timeout"""
        start = time.time()
        degs = get_degrees(adj)
        score = calc_score(adj, degs)
        
        while time.time() - start < max_time:
            best_swap, new_score = try_all_swaps(adj, degs, score)
            
            if best_swap is None:
                break
            
            rem_u, rem_v, add_u, add_v = best_swap
            adj[rem_u][rem_v] = adj[rem_v][rem_u] = False
            adj[add_u][add_v] = adj[add_v][add_u] = True
            
            degs[rem_u] -= 1
            degs[rem_v] -= 1
            degs[add_u] += 1
            degs[add_v] += 1
            
            score = new_score
            
            if score < 24:
                break
        
        return adj, score
    
    # Try multiple initial constructions
    initial_graphs = [
        build_k88_plus_one(),
        build_k88_plus_one_b(),
        build_k88_plus_cross(),
    ]
    
    # Also try some random perturbations of K_{8,8}+1
    base = build_k88_plus_one()
    for _ in range(3):
        perturbed = [row[:] for row in base]
        # Remove 2 random cross edges
        cross_edges = [(u, v) for u in range(8) for v in range(8, 16) if perturbed[u][v]]
        if len(cross_edges) >= 2:
            e1, e2 = random.sample(cross_edges, 2)
            perturbed[e1[0]][e1[1]] = perturbed[e1[1]][e1[0]] = False
            perturbed[e2[0]][e2[1]] = perturbed[e2[1]][e2[0]] = False
            # Add 2 random within-part edges
            within_a = [(u, v) for u in range(8) for v in range(u+1, 8) if not perturbed[u][v]]
            within_b = [(u, v) for u in range(8, 16) for v in range(u+1, 16) if not perturbed[u][v]]
            all_within = within_a + within_b
            if len(all_within) >= 2:
                ne1, ne2 = random.sample(all_within, 2)
                perturbed[ne1[0]][ne1[1]] = perturbed[ne1[1]][ne1[0]] = True
                perturbed[ne2[0]][ne2[1]] = perturbed[ne2[1]][ne2[0]] = True
            initial_graphs.append(perturbed)
    
    best_adj = None
    best_final_score = float('inf')
    
    total_start = time.time()
    time_per_graph = 8.0
    
    for i, init_adj in enumerate(initial_graphs):
        if time.time() - total_start > 100:
            break
        
        # Verify edge count
        edge_count = sum(sum(row) for row in init_adj) // 2
        if edge_count != target_edges:
            continue
        
        adj_copy = [row[:] for row in init_adj]
        adj_copy, score = local_search(adj_copy, max_time=time_per_graph)
        
        if score < best_final_score:
            best_final_score = score
            best_adj = adj_copy
        
        if best_final_score < 24:
            break
    
    # If no valid graph found, use the first initial graph
    if best_adj is None:
        best_adj = initial_graphs[0]
    
    # Convert to edge list
    final_edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u][v]:
                final_edges.append([u, v])
    
    return {'n': n, 'edges': final_edges}
