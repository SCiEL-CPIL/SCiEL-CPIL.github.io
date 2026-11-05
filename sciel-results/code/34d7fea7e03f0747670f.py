import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin
    
    n = 16
    target_edges = 65
    
    A = list(range(8))
    B = list(range(8, 16))
    
    def build_adj(edges):
        adj = [set() for _ in range(n)]
        deg = [0] * n
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
        return adj, deg
    
    def max_tri_sum(adj, deg):
        best = 0
        for u in range(n):
            for v in adj[u]:
                if v <= u:
                    continue
                common = adj[u] & adj[v]
                for w in common:
                    if w <= v:
                        continue
                    ds = deg[u] + deg[v] + deg[w]
                    if ds > best:
                        best = ds
        return best
    
    def score_edges(edges):
        adj, deg = build_adj(edges)
        return max_tri_sum(adj, deg)
    
    # Strategy: K_8,8 + 1 internal edge gives score 26
    # Try to do better with local search
    
    random.seed(123)
    
    def get_k88_plus_internal(int_edge):
        edges = [[u, v] for u in A for v in B]
        edges.append(list(int_edge))
        return edges
    
    # Find best single internal edge
    all_int_A = [list(e) for e in combinations(range(8), 2)]
    all_int_B = [list(e) for e in combinations(range(8, 16), 2)]
    all_int = all_int_A + all_int_B
    
    best_int_edge = all_int[0]
    best_int_score = float('inf')
    for e in all_int:
        edges = get_k88_plus_internal(e)
        s = score_edges(edges)
        if s < best_int_score:
            best_int_score = s
            best_int_edge = e
    
    best_edges = get_k88_plus_internal(best_int_edge)
    best_score = best_int_score
    
    # Local search: try to improve by edge swaps
    def local_search(edges, max_iters=1000):
        current = edges[:]
        adj, deg = build_adj(current)
        current_score = max_tri_sum(adj, deg)
        
        edge_set = set(tuple(sorted(e)) for e in current)
        
        for it in range(max_iters):
            if time.time() - start_time > deadline - 8:
                break
            
            improved = False
            
            # Find a high-degree-sum triangle
            tri_list = []
            for u in range(n):
                for v in adj[u]:
                    if v <= u:
                        continue
                    common = adj[u] & adj[v]
                    for w in common:
                        if w <= v:
                            continue
                        ds = deg[u] + deg[v] + deg[w]
                        tri_list.append((ds, u, v, w))
            
            if not tri_list:
                break
            
            tri_list.sort(reverse=True)
            ds_max, u, v, w = tri_list[0]
            
            if ds_max < current_score:
                current_score = ds_max
            
            # Try to break the worst triangle
            tri_edges = [(u,v), (v,w), (u,w)]
            random.shuffle(tri_edges)
            
            for rem_u, rem_v in tri_edges:
                rem_e = tuple(sorted((rem_u, rem_v)))
                if rem_e not in edge_set:
                    continue
                
                # Try adding a random edge
                for _ in range(20):
                    new_u, new_v = random.sample(range(n), 2)
                    add_e = tuple(sorted((new_u, new_v)))
                    if add_e in edge_set or add_e == rem_e:
                        continue
                    
                    # Check if this creates a new triangle with high sum
                    # Quick check: new edge connects new_u and new_v
                    # New triangles would be (new_u, new_v, x) where x is common neighbor
                    # This is complex, so just compute full score
                    
                    new_edges = [e for e in current if tuple(sorted(e)) != rem_e]
                    new_edges.append(list(add_e))
                    
                    new_adj, new_deg = build_adj(new_edges)
                    new_score = max_tri_sum(new_adj, new_deg)
                    
                    if new_score < current_score:
                        current = new_edges
                        edge_set = set(tuple(sorted(e)) for e in current)
                        adj, deg = new_adj, new_deg
                        current_score = new_score
                        improved = True
                        break
                    elif new_score == current_score and random.random() < 0.2:
                        current = new_edges
                        edge_set = set(tuple(sorted(e)) for e in current)
                        adj, deg = new_adj, new_deg
                        improved = True
                        break
                
                if improved:
                    break
            
            if not improved and random.random() < 0.1:
                # Random perturbation: swap a random edge
                rem_e = random.choice(list(edge_set))
                while True:
                    new_u, new_v = random.sample(range(n), 2)
                    add_e = tuple(sorted((new_u, new_v)))
                    if add_e not in edge_set and add_e != rem_e:
                        break
                
                new_edges = [e for e in current if tuple(sorted(e)) != rem_e]
                new_edges.append(list(add_e))
                current = new_edges
                edge_set = set(tuple(sorted(e)) for e in current)
                adj, deg = build_adj(current)
                current_score = max_tri_sum(adj, deg)
        
        return current, current_score
    
    # Run local search from best K_8,8+internal start
    if time.time() - start_time < deadline - 20:
        edges_ls, score_ls = local_search(best_edges, max_iters=500)
        if score_ls < best_score:
            best_score = score_ls
            best_edges = edges_ls
    
    # Try a few random restarts
    for r in range(10):
        if time.time() - start_time > deadline - 15:
            break
        
        # Random graph
        all_possible = []
        for i in range(n):
            for j in range(i+1, n):
                all_possible.append((i,j))
        random.shuffle(all_possible)
        rand_edges = [list(e) for e in all_possible[:65]]
        
        edges_ls, score_ls = local_search(rand_edges, max_iters=300)
        if score_ls < best_score:
            best_score = score_ls
            best_edges = edges_ls
    
    # Validate
    assert len(best_edges) == 65
    assert len(set(tuple(sorted(e)) for e in best_edges)) == 65
    
    return {'n': n, 'edges': best_edges}
