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
    
    def local_search(edge_set_copy, initial_score, time_limit):
        """Run local search from a given state, return (best_score, best_set, final_score)"""
        nonlocal best_score, best_edge_set
        adj_local = [[adj[i][j] for j in range(n)] for i in range(n)]
        es_local = set(edge_set_copy)
        current_score = initial_score
        
        def local_add(u, v):
            u, v = min(u,v), max(u,v)
            if u != v and (u,v) not in es_local:
                es_local.add((u,v))
                adj_local[u][v] = adj_local[v][u] = True
        
        def local_remove(u, v):
            u, v = min(u,v), max(u,v)
            if (u,v) in es_local:
                es_local.remove((u,v))
                adj_local[u][v] = adj_local[v][u] = False
        
        def local_score():
            deg = [sum(adj_local[i]) for i in range(n)]
            max_s = 0
            for i in range(n):
                for j in range(i+1, n):
                    if adj_local[i][j]:
                        for k in range(j+1, n):
                            if adj_local[i][k] and adj_local[j][k]:
                                s = deg[i] + deg[j] + deg[k]
                                if s > max_s:
                                    max_s = s
            return max_s
        
        # Simulated annealing
        temp = 3.0
        while time.time() < time_limit:
            # Try a random edge swap
            e_remove = random.choice(list(es_local))
            e_add = random.choice(all_edges)
            if e_add in es_local or e_add == e_remove:
                continue
            
            local_remove(*e_remove)
            local_add(*e_add)
            
            new_score = local_score()
            
            if new_score < current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_edge_set = set(es_local)
            else:
                if random.random() < math.exp(-(new_score - current_score) / max(temp, 0.1)):
                    current_score = new_score
                else:
                    local_remove(*e_add)
                    local_add(*e_remove)
            
            # Cool down
            temp *= 0.999
            
            # Occasionally try a 2-edge swap (remove one, add two, or similar)
            if random.random() < 0.1:
                # Try removing one edge and adding two
                e_rem2 = random.choice(list(es_local))
                e_add1 = random.choice(all_edges)
                e_add2 = random.choice(all_edges)
                if e_add1 not in es_local and e_add2 not in es_local and e_add1 != e_add2:
                    old_score = current_score
                    local_remove(*e_rem2)
                    local_add(*e_add1)
                    local_add(*e_add2)
                    # Now we have 66 edges, need to remove one more
                    e_rem3 = random.choice(list(es_local))
                    local_remove(*e_rem3)
                    new_score = local_score()
                    if new_score < old_score:
                        current_score = new_score
                        if current_score < best_score:
                            best_score = current_score
                            best_edge_set = set(es_local)
                    else:
                        # Revert
                        local_add(*e_rem3)
                        local_remove(*e_add1)
                        local_remove(*e_add2)
                        local_add(*e_rem2)
        
        return current_score, set(es_local)
    
    # Try multiple random restarts
    for restart in range(30):
        if time.time() > deadline - 10:
            break
        
        # Random initialization: start with K_{8,8} + random edge
        es_init = set()
        for u in range(8):
            for v in range(8, 16):
                es_init.add((u, v))
        
        # Add one random edge within a part
        if random.random() < 0.5:
            u = random.randint(0, 7)
            v = random.randint(0, 7)
            if u != v:
                es_init.add((u, v))
        else:
            u = random.randint(8, 15)
            v = random.randint(8, 15)
            if u != v:
                es_init.add((u, v))
        
        # Perturb: do 5 random edge swaps
        for _ in range(5):
            e_remove = random.choice(list(es_init))
            e_add = random.choice(all_edges)
            if e_add not in es_init and e_add != e_remove:
                es_init.remove(e_remove)
                es_init.add(e_add)
        
        # Compute initial score
        # Set up adj for scoring
        for i in range(n):
            for j in range(n):
                adj[i][j] = (i,j) in es_init or (j,i) in es_init
        
        init_score = compute_score()
        
        # Run local search with time limit
        time_left = deadline - 10 - time.time()
        if time_left < 1:
            break
        
        # Use a fraction of remaining time for this restart
        search_time = time.time() + min(time_left * 0.7, 10)
        final_score, final_set = local_search(es_init, init_score, search_time)
        
        # Update global best
        if final_score < best_score:
            best_score = final_score
            best_edge_set = final_set
    
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
