import random
import time
import numpy as np

def construct():
    n = 16
    target_edges = 65
    start_time = time.time()
    max_time = 115.0  # Safety margin for 150s limit

    # Initialize with a random graph
    # Better: Start with a structure closer to regular
    # Let's try to generate a random graph with 65 edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Initialize adjacency matrix
    adj = np.zeros((n, n), dtype=bool)
    deg = np.zeros(n, dtype=int)
    edge_set = set()
    
    def random_init():
        nonlocal adj, deg, edge_set
        adj = np.zeros((n, n), dtype=bool)
        deg = np.zeros(n, dtype=int)
        edge_set = set()
        random.shuffle(all_edges)
        for u, v in all_edges[:target_edges]:
            adj[u, v] = True
            adj[v, u] = True
            deg[u] += 1
            deg[v] += 1
            edge_set.add((u, v))

    random.seed(42) # Reproducibility for debugging, but we want to search
    random_init()

    def get_score():
        max_sum = 0
        # Iterate over all triangles
        for u in range(n):
            for v in range(u+1, n):
                if adj[u, v]:
                    for w in range(v+1, n):
                        if adj[u, w] and adj[v, w]:
                            s = deg[u] + deg[v] + deg[w]
                            if s > max_sum:
                                max_sum = s
        return max_sum

    best_score = get_score()
    best_adj = adj.copy()
    best_deg = deg.copy()
    best_edge_set = set(edge_set)
    
    current_score = best_score
    current_adj = adj.copy()
    current_deg = deg.copy()
    current_edge_set = set(edge_set)
    
    # Precompute list of non-edges for faster random selection?
    # It changes, so we might just sample.
    # Sampling a random non-edge:
    # Pick u, v randomly. If edge exists, reject.
    # Or maintain a list. With 120 possible edges and 65 present, 55 missing.
    # List maintenance is O(1) swap.
    
    non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (i, j) not in edge_set]
    
    def add_edge(u, v):
        if u > v: u, v = v, u
        current_adj[u, v] = True
        current_adj[v, u] = True
        current_deg[u] += 1
        current_deg[v] += 1
        current_edge_set.add((u, v))
        # Remove from non_edges
        idx = non_edges.index((u, v))
        last = non_edges[-1]
        non_edges[idx] = last
        non_edges.pop()

    def remove_edge(u, v):
        if u > v: u, v = v, u
        current_adj[u, v] = False
        current_adj[v, u] = False
        current_deg[u] -= 1
        current_deg[v] -= 1
        current_edge_set.remove((u, v))
        non_edges.append((u, v))

    def swap_edges(rem_u, rem_v, add_u, add_v):
        remove_edge(rem_u, rem_v)
        add_edge(add_u, add_v)
        
    def undo_swap(rem_u, rem_v, add_u, add_v):
        remove_edge(add_u, add_v)
        add_edge(rem_u, rem_v)

    T = 5.0
    cool = 0.999
    
    while time.time() - start_time < max_time:
        if T < 0.01:
            T = 5.0 # Restart if cooled down too much? Or just continue
            # Actually, let's do multiple restarts if stuck
            pass

        # Pick random edge to remove
        rem_edge = random.choice(list(current_edge_set))
        rem_u, rem_v = rem_edge
        
        # Pick random non-edge to add
        if not non_edges:
            break
        add_edge_idx = random.randint(0, len(non_edges)-1)
        add_u, add_v = non_edges[add_edge_idx]
        
        if rem_edge == (add_u, add_v) or rem_edge == (add_v, add_u):
            continue
            
        # Apply swap
        swap_edges(rem_u, rem_v, add_u, add_v)
        new_score = get_score()
        
        delta = new_score - current_score
        if delta < 0 or (T > 0 and random.random() < np.exp(-delta / T)):
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_adj = current_adj.copy()
                best_deg = current_deg.copy()
                best_edge_set = set(current_edge_set)
        else:
            undo_swap(rem_u, rem_v, add_u, add_v)
            
        T *= cool
        if T < 0.05:
            # Reset temperature occasionally to escape local minima
            T = 2.0

    # Final polish: Greedy local search on best solution
    # Restore best state
    current_adj = best_adj
    current_deg = best_deg
    current_edge_set = best_edge_set
    current_score = best_score
    non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (i, j) not in current_edge_set]
    
    # Greedy hill climbing
    improved = True
    while improved and time.time() - start_time < max_time - 5:
        improved = False
        # Find worst triangle
        max_s = 0
        worst_tri = None
        for u in range(n):
            for v in range(u+1, n):
                if current_adj[u, v]:
                    for w in range(v+1, n):
                        if current_adj[u, w] and current_adj[v, w]:
                            s = current_deg[u] + current_deg[v] + current_deg[w]
                            if s > max_s:
                                max_s = s
                                worst_tri = (u, v, w)
        
        if not worst_tri:
            break
            
        u, v, w = worst_tri
        # Try to improve by swapping edges incident to u, v, w
        candidates = []
        for vert in (u, v, w):
            for i in range(n):
                if current_adj[vert, i]:
                    candidates.append((vert, i))
                    
        random.shuffle(candidates)
        
        for rem_u, rem_v in candidates:
            # Try a few random add edges
            for _ in range(10):
                if not non_edges: break
                add_idx = random.randint(0, len(non_edges)-1)
                add_u, add_v = non_edges[add_idx]
                if (rem_u, rem_v) == (add_u, add_v) or (rem_v, rem_u) == (add_u, add_v):
                    continue
                swap_edges(rem_u, rem_v, add_u, add_v)
                new_score = get_score()
                if new_score < current_score:
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_adj = current_adj.copy()
                        best_deg = current_deg.copy()
                        best_edge_set = set(current_edge_set)
                    improved = True
                    break
                else:
                    undo_swap(rem_u, rem_v, add_u, add_v)
            if improved:
                break
                
    return {'n': n, 'edges': [list(e) for e in best_edge_set]}
