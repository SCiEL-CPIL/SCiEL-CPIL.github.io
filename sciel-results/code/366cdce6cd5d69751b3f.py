import time
import random
import math

def construct():
    start_time = time.time()
    deadline = 140
    
    n = 16
    adj = [[False]*n for _ in range(n)]
    edge_set = set()
    
    def add_edge(u, v):
        u, v = min(u,v), max(u,v)
        if u != v and (u,v) not in edge_set:
            edge_set.add((u,v))
            adj[u][v] = adj[v][u] = True
    
    def remove_edge(u, v):
        u, v = min(u,v), max(u,v)
        if (u,v) in edge_set:
            edge_set.remove((u,v))
            adj[u][v] = adj[v][u] = False
    
    def compute_score():
        deg = [sum(adj[i]) for i in range(n)]
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
    
    all_edges = [(i,j) for i in range(n) for j in range(i+1, n)]
    
    best_score = float('inf')
    best_edge_set = None
    
    # Try multiple random restarts
    for restart in range(50):
        if time.time() > deadline - 5:
            break
        
        # Random initialization: start with K_{8,8} + random edge, then perturb
        edge_set.clear()
        for i in range(n):
            adj[i] = [False]*n
        
        # Create K_{8,8}
        for u in range(8):
            for v in range(8, 16):
                add_edge(u, v)
        
        # Add one random edge within a part
        if random.random() < 0.5:
            u = random.randint(0, 7)
            v = random.randint(0, 7)
            if u != v:
                add_edge(u, v)
        else:
            u = random.randint(8, 15)
            v = random.randint(8, 15)
            if u != v:
                add_edge(u, v)
        
        # Perturb: do 10 random edge swaps
        for _ in range(10):
            e_remove = random.choice(list(edge_set))
            e_add = random.choice(all_edges)
            if e_add not in edge_set and e_add != e_remove:
                remove_edge(*e_remove)
                add_edge(*e_add)
        
        current_score = compute_score()
        
        # Local search with simulated annealing
        temp = 5.0
        iterations_this_restart = 0
        while time.time() < deadline - 5:
            iterations_this_restart += 1
            if iterations_this_restart % 10000 == 0:
                temp *= 0.95
            
            e_remove = random.choice(list(edge_set))
            e_add = random.choice(all_edges)
            if e_add in edge_set or e_add == e_remove:
                continue
            
            remove_edge(*e_remove)
            add_edge(*e_add)
            
            new_score = compute_score()
            
            if new_score < current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_edge_set = set(edge_set)
            else:
                if random.random() < math.exp(-(new_score - current_score) / max(temp, 0.1)):
                    current_score = new_score
                else:
                    remove_edge(*e_add)
                    add_edge(*e_remove)
        
        if current_score < best_score:
            best_score = current_score
            best_edge_set = set(edge_set)
    
    if best_edge_set is None:
        # Fallback to original
        edge_set.clear()
        for i in range(n):
            adj[i] = [False]*n
        for u in range(8):
            for v in range(8, 16):
                add_edge(u, v)
        add_edge(0, 1)
        best_edge_set = set(edge_set)
    
    edge_list = [[u, v] for (u, v) in best_edge_set]
    return {'n': n, 'edges': edge_list}
