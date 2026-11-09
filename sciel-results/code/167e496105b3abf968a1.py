import time
import random
import math

def construct():
    start_time = time.time()
    deadline = 148
    
    n = 16
    adj = [[False]*n for _ in range(n)]
    edge_set = set()
    deg = [0]*n
    
    def add_edge(u, v):
        u, v = min(u,v), max(u,v)
        if u != v and (u,v) not in edge_set:
            edge_set.add((u,v))
            adj[u][v] = adj[v][u] = True
            deg[u] += 1
            deg[v] += 1
    
    def remove_edge(u, v):
        u, v = min(u,v), max(u,v)
        if (u,v) in edge_set:
            edge_set.remove((u,v))
            adj[u][v] = adj[v][u] = False
            deg[u] -= 1
            deg[v] -= 1
    
    def compute_score():
        max_sum = 0
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    for k in range(j+1, n):
                        if adj[i][k] and adj[j][k]:
                            s = deg[i] + deg[j] + deg[k]
                            if s > max_sum:
                                max_sum = s
        return max_sum
    
    def fast_score_after_swap(e_remove, e_add):
        """Compute score after removing e_remove and adding e_add.
        Only need to check triangles involving vertices in e_remove or e_add."""
        u1, v1 = e_remove
        u2, v2 = e_add
        # Vertices whose degrees change
        changed_vertices = {u1, v1, u2, v2}
        
        # Temporarily apply the swap
        remove_edge(u1, v1)
        add_edge(u2, v2)
        
        # Find all triangles and compute max degree sum
        max_sum = 0
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    for k in range(j+1, n):
                        if adj[i][k] and adj[j][k]:
                            s = deg[i] + deg[j] + deg[k]
                            if s > max_sum:
                                max_sum = s
        
        # Undo the swap
        remove_edge(u2, v2)
        add_edge(u1, v1)
        
        return max_sum
    
    all_edges = [(i,j) for i in range(n) for j in range(i+1, n)]
    
    best_score = float('inf')
    best_edge_set = None
    
    def try_local_search(init_edge_set, max_iterations=50000):
        nonlocal best_score, best_edge_set
        edge_set.clear()
        for i in range(n):
            adj[i] = [False]*n
            deg[i] = 0
        for e in init_edge_set:
            add_edge(*e)
        
        current_score = compute_score()
        
        if current_score < best_score:
            best_score = current_score
            best_edge_set = set(edge_set)
        
        no_improve = 0
        for it in range(max_iterations):
            if time.time() > deadline - 3:
                break
            
            improved = False
            # Try all possible swaps
            for e_remove in list(edge_set):
                if time.time() > deadline - 3:
                    break
                for e_add in all_edges:
                    if e_add in edge_set or e_add == e_remove:
                        continue
                    
                    new_score = fast_score_after_swap(e_remove, e_add)
                    
                    if new_score < current_score:
                        # Apply the swap permanently
                        remove_edge(*e_remove)
                        add_edge(*e_add)
                        current_score = new_score
                        improved = True
                        if current_score < best_score:
                            best_score = current_score
                            best_edge_set = set(edge_set)
                        break
                if improved:
                    break
            
            if not improved:
                no_improve += 1
                if no_improve > 100:
                    break
    
    # Initial construction: K_{8,8} + one edge
    base_edges = set()
    for u in range(8):
        for v in range(8, 16):
            base_edges.add((u, v))
    
    # Try different initial intra-part edges
    for part in [0, 1]:
        for u in range(8):
            for v in range(u+1, 8):
                if part == 0:
                    edge = (u, v)
                else:
                    edge = (u+8, v+8)
                
                init_edges = base_edges | {edge}
                try_local_search(init_edges, max_iterations=20000)
                
                if time.time() > deadline - 10:
                    break
            if time.time() > deadline - 10:
                break
        if time.time() > deadline - 10:
            break
    
    if best_edge_set is None:
        edge_set.clear()
        for i in range(n):
            adj[i] = [False]*n
            deg[i] = 0
        for u in range(8):
            for v in range(8, 16):
                add_edge(u, v)
        add_edge(0, 1)
        best_edge_set = set(edge_set)
    
    edge_list = [[u, v] for (u, v) in best_edge_set]
    return {'n': n, 'edges': edge_list}
