import time
import random

def construct():
    n = 40
    start_time = time.time()
    deadline = 145  # Safety margin
    
    def time_left():
        return deadline - (time.time() - start_time)
    
    def can_add_in(adj, u, v):
        Nv = adj[v]
        Nu = adj[u]
        for x in Nv:
            if x == u: continue
            for w in Nu:
                if w == v: continue
                if x == w: continue
                for y in adj[x]:
                    if y == u or y == v or y == x or y == w: continue
                    for z in adj[w]:
                        if z == u or z == v or z == x or z == w or z == y: continue
                        if z in adj[y]:
                            return False
        return True

    # Phase 1: Greedy
    best_adj = None
    best_edges = 0
    random.seed(42)
    
    for trial in range(10):
        if time_left() < 20:
            break
        adj_try = [set() for _ in range(n)]
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        ec = 0
        for u, v in all_edges:
            if time_left() < 10:
                break
            if can_add_in(adj_try, u, v):
                adj_try[u].add(v)
                adj_try[v].add(u)
                ec += 1
        if ec > best_edges:
            best_edges = ec
            best_adj = [set(s) for s in adj_try]

    if best_adj is None:
        best_adj = [set() for _ in range(n)]
        for k in range(8):
            for a in range(5):
                for b in range(a+1, 5):
                    best_adj[5*k+a].add(5*k+b)
                    best_adj[5*k+b].add(5*k+a)
        best_edges = 80

    # Phase 2: Local Search
    adj = best_adj
    current_edges = best_edges
    all_edges_list = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Track best found during local search
    global_best_adj = [set(s) for s in adj]
    global_best_edges = current_edges
    
    stagnation = 0
    
    while time_left() > 2:
        improved = False
        
        # Try to add
        for _ in range(50):
            u, v = random.choice(all_edges_list)
            if v in adj[u]:
                continue
            if can_add_in(adj, u, v):
                adj[u].add(v)
                adj[v].add(u)
                current_edges += 1
                improved = True
                break
        
        if improved:
            stagnation = 0
            if current_edges > global_best_edges:
                global_best_edges = current_edges
                global_best_adj = [set(s) for s in adj]
        else:
            stagnation += 1
            # If stuck for a while, remove an edge
            if stagnation > 10 or random.random() < 0.1:
                if current_edges > 0:
                    # Remove random edge
                    u = random.randint(0, n-1)
                    if adj[u]:
                        v = random.choice(list(adj[u]))
                        adj[u].remove(v)
                        adj[v].remove(u)
                        current_edges -= 1
                    else:
                        # Find any edge
                        for i in range(n):
                            if adj[i]:
                                u = i
                                v = random.choice(list(adj[i]))
                                adj[u].remove(v)
                                adj[v].remove(u)
                                current_edges -= 1
                                break
                else:
                    break

    # Return the best graph found
    adj = global_best_adj
    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append([i, j])
    return edges
