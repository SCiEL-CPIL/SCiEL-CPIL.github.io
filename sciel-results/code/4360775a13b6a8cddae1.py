import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n = 40
    target_edges = 401
    
    def init_graph_k2020():
        adj = [set() for _ in range(n)]
        edges = set()
        for u in range(20):
            for v in range(20, 40):
                adj[u].add(v)
                adj[v].add(u)
                edges.add((u, v))
        u, v = random.sample(range(20), 2)
        adj[u].add(v)
        adj[v].add(u)
        edges.add((u, v))
        return adj, edges

    def init_graph_random():
        # Random graph with 401 edges
        adj = [set() for _ in range(n)]
        edges = set()
        possible = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(possible)
        for u, v in possible[:target_edges]:
            adj[u].add(v)
            adj[v].add(u)
            edges.add((u, v))
        return adj, edges

    def get_max_book(adj, edges):
        max_b = 0
        for u, v in edges:
            b = len(adj[u] & adj[v])
            if b > max_b:
                max_b = b
        return max_b

    def swap_edges(adj, edges, e1, e2):
        u, v = e1
        x, y = e2
        if e1 == e2:
            return False
        if (x, y) in edges or (y, x) in edges:
            return False
        if x == y:
            return False
        
        adj[u].remove(v)
        adj[v].remove(u)
        edges.remove(e1)
        
        adj[x].add(y)
        adj[y].add(x)
        edges.add(e2)
        
        return True

    def unswap_edges(adj, edges, e1, e2):
        u, v = e1
        x, y = e2
        
        adj[x].remove(y)
        adj[y].remove(x)
        edges.remove(e2)
        
        adj[u].add(v)
        adj[v].add(u)
        edges.add(e1)

    best_adj, best_edges = init_graph_k2020()
    best_score = get_max_book(best_adj, best_edges)
    
    for restart in range(100): # Max 100 restarts
        if time.time() > deadline - 5:
            break
            
        if restart == 0:
            current_adj, current_edges = [set(s) for s in best_adj], set(best_edges)
        else:
            # Mix of initializations
            if random.random() < 0.5:
                current_adj, current_edges = init_graph_k2020()
            else:
                current_adj, current_edges = init_graph_random()
                
        current_score = get_max_book(current_adj, current_edges)
        
        if current_score < best_score:
            best_score = current_score
            best_adj = [set(s) for s in current_adj]
            best_edges = set(current_edges)
            
        T = 5.0
        cooling = 0.9995
        no_improve_steps = 0
        max_no_improve = 1000
        
        while time.time() < deadline - 5 and no_improve_steps < max_no_improve:
            # Pick edge to remove
            if random.random() < 0.7:
                # Pick edge with high book
                # Find max book edge quickly?
                # Just pick random for now, or precompute
                # Let's pick random to avoid bias
                e1 = random.choice(list(current_edges))
            else:
                # Try to find a bad edge
                max_b = 0
                bad_e = None
                for u, v in current_edges:
                    b = len(current_adj[u] & current_adj[v])
                    if b > max_b:
                        max_b = b
                        bad_e = (u, v)
                e1 = bad_e
            
            # Pick edge to add
            attempts = 0
            e2 = None
            while attempts < 50:
                x, y = random.sample(range(n), 2)
                if (x, y) not in current_edges and (y, x) not in current_edges:
                    e2 = (x, y)
                    break
                attempts += 1
            
            if e2 is None:
                continue
                
            if not swap_edges(current_adj, current_edges, e1, e2):
                continue
                
            new_score = get_max_book(current_adj, current_edges)
            
            if new_score < current_score:
                current_score = new_score
                no_improve_steps = 0
                if current_score < best_score:
                    best_score = current_score
                    best_adj = [set(s) for s in current_adj]
                    best_edges = set(current_edges)
            else:
                delta = new_score - current_score
                if T > 0.01:
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        current_score = new_score
                        no_improve_steps += 1
                    else:
                        unswap_edges(current_adj, current_edges, e1, e2)
                        no_improve_steps += 1
                else:
                    unswap_edges(current_adj, current_edges, e1, e2)
                    no_improve_steps += 1
            
            T *= cooling
            
        if best_score <= 8:
            break

    result = [[u, v] for u, v in best_edges]
    return result
