import random
import time

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 110.0
    
    adj = [[0]*n for _ in range(n)]
    neighbors = [set() for _ in range(n)]
    
    def add_edge(u, v):
        adj[u][v] = 1
        adj[v][u] = 1
        neighbors[u].add(v)
        neighbors[v].add(u)
        
    def remove_edge(u, v):
        adj[u][v] = 0
        adj[v][u] = 0
        neighbors[u].remove(v)
        neighbors[v].remove(v) # Bug: should be remove(v)
        
    # Corrected remove_edge
    def remove_edge(u, v):
        adj[u][v] = 0
        adj[v][u] = 0
        neighbors[u].discard(v)
        neighbors[v].discard(u)
        
    def edge_count():
        return sum(len(neighbors[u]) for u in range(n)) // 2
        
    def find_k33_conflict(u, v):
        A = neighbors[v] - {u}
        B = neighbors[u] - {v}
        
        if len(A) < 2 or len(B) < 2:
            return None
            
        A_list = list(A)
        B_list = list(B)
        
        for i in range(len(A_list)):
            a1 = A_list[i]
            common1 = neighbors[a1] & set(B_list)
            if len(common1) < 2:
                continue
            for j in range(1, len(A_list)):
                a2 = A_list[j]
                common2 = neighbors[a2] & set(B_list)
                common_inter = common1 & common2
                if len(common_inter) >= 2:
                    b_list = list(common_inter)
                    b1 = b_list[0]
                    b2 = b_list[1]
                    return [
                        [u, b1], [u, b2],
                        [v, a1], [v, a2],
                        [a1, b1], [a1, b2],
                        [a2, b1], [a2, b2]
                    ]
        return None

    # Initial random graph
    for u in range(n):
        for v in range(u+1, n):
            if random.random() < 0.1:
                add_edge(u, v)
                
    best_edges = edge_count()
    best_adj = [row[:] for row in adj]
    best_neighbors = [set(s) for s in neighbors]
    
    # Greedy phase
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    random.shuffle(all_edges)
    for u, v in all_edges:
        if not adj[u][v]:
            if not find_k33_conflict(u, v):
                add_edge(u, v)
                
    current_edges = edge_count()
    if current_edges > best_edges:
        best_edges = current_edges
        best_adj = [row[:] for row in adj]
        best_neighbors = [set(s) for s in neighbors]
        
    while time.time() < deadline:
        if random.random() < 0.8:
            while True:
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v: continue
                if adj[u][v]: continue
                break
                
            conflict = find_k33_conflict(u, v)
            if conflict is None:
                add_edge(u, v)
                current_edges = edge_count()
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = [row[:] for row in adj]
                    best_neighbors = [set(s) for s in neighbors]
            else:
                # Try to resolve conflict
                # Save state
                u_rem, v_rem = random.choice(conflict)
                
                # Perform move
                remove_edge(u_rem, v_rem)
                add_edge(u, v)
                
                # Check if new edge (u,v) creates a conflict
                new_conflict = find_k33_conflict(u, v)
                if new_conflict is not None:
                    # Revert
                    remove_edge(u, v)
                    add_edge(u_rem, v_rem)
                else:
                    # Accept move
                    # Edge count is same, so no update to best
                    pass
        else:
            if edge_count() == 0: continue
            while True:
                u = random.randint(0, n-1)
                if not neighbors[u]: continue
                v = random.choice(list(neighbors[u]))
                break
            remove_edge(u, v)
            current_edges = edge_count()
            
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u][v]:
                edges.append([u, v])
                
    return edges
