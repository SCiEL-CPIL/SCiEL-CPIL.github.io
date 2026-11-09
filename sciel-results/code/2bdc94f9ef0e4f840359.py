import random
import time
import itertools

N = 16
SUBSETS_8 = list(itertools.combinations(range(N), 8))

def is_q3_induced(subset, adj):
    """
    Check if induced subgraph on subset is Q3.
    adj is a list of lists of bools.
    """
    # 1. Count edges and degrees
    edge_count = 0
    degrees = [0] * 8
    
    # Unroll for speed? Maybe not necessary.
    for i in range(8):
        u = subset[i]
        for j in range(i+1, 8):
            v = subset[j]
            if adj[u][v]:
                edge_count += 1
                degrees[i] += 1
                degrees[j] += 1
                
    if edge_count != 12:
        return False
    if degrees != [3]*8:
        return False
        
    # 2. Check bipartite
    # Build local adj list
    local_adj = [[] for _ in range(8)]
    for i in range(8):
        u = subset[i]
        for j in range(8):
            if i != j and adj[u][subset[j]]:
                local_adj[i].append(j)
                
    color = [-1] * 8
    for start in range(8):
        if color[start] != -1:
            continue
        color[start] = 0
        stack = [start]
        while stack:
            u = stack.pop()
            for v in local_adj[u]:
                if color[v] == -1:
                    color[v] = 1 - color[u]
                    stack.append(v)
                elif color[v] == color[u]:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    best_edges = 0
    best_graph = None
    
    # Seed: K_{3,13}
    adj_seed = [[False]*N for _ in range(N)]
    for a in range(3):
        for b in range(3, 16):
            adj_seed[a][b] = True
            adj_seed[b][a] = True
    best_graph = adj_seed
    best_edges = 39
    
    # Randomized Search
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        if restarts > 200: # Hard cap on restarts to avoid infinite loop if time check fails
            break
            
        # Random dense graph
        # Try different densities
        p = random.choice([0.6, 0.7, 0.8, 0.9])
        adj = [[False]*N for _ in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                if random.random() < p:
                    adj[i][j] = True
                    adj[j][i] = True
                    
        edge_count = sum(sum(row) for row in adj) // 2
        
        # Iteratively remove edges to destroy Q3s
        # To avoid getting stuck in a local minimum or slow loop, 
        # we can limit the number of removals per restart
        max_removals = 100
        removed = 0
        
        while removed < max_removals and time.time() < deadline:
            q3_found = False
            # Shuffle subsets to randomize which Q3 is found first?
            # This might help escape local minima.
            # But shuffling 12870 items is fast.
            subsets_copy = SUBSETS_8[:]
            random.shuffle(subsets_copy)
            
            for subset in subsets_copy:
                if is_q3_induced(subset, adj):
                    q3_found = True
                    # Find edges in this Q3
                    q3_edges = []
                    for i in range(8):
                        u = subset[i]
                        for j in range(i+1, 8):
                            v = subset[j]
                            if adj[u][v]:
                                q3_edges.append((u, v))
                    
                    if q3_edges:
                        u, v = random.choice(q3_edges)
                        adj[u][v] = False
                        adj[v][u] = False
                        edge_count -= 1
                        removed += 1
                    break
                    
            if not q3_found:
                break
                
        # Check if this graph is Q3-free (it should be, but verify)
        # Actually, the loop breaks when no Q3 is found in a full pass.
        # So it is Q3-free.
        
        if edge_count > best_edges:
            best_edges = edge_count
            best_graph = [row[:] for row in adj]
            
    # Convert best_graph to list of edges
    edges = []
    for i in range(N):
        for j in range(i+1, N):
            if best_graph[i][j]:
                edges.append([i, j])
                
    return edges
