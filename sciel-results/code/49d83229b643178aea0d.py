import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin for safety

    n = 16
    
    # Core: Grötzsch graph on vertices 0..10
    # Vertices 0..4 form C5
    # Vertices 5..9 are copies of 0..4
    # Vertex 10 is connected to 5..9
    
    def make_grotzsch():
        edges = []
        # C5
        for i in range(5):
            edges.append((i, (i+1)%5))
        # Mycielski edges
        for u, v in [(i, (i+1)%5) for i in range(5)]:
            edges.append((u, 5+v))
            edges.append((v, 5+u))
        for u in range(5):
            edges.append((10, 5+u))
        return edges

    core_edges = make_grotzsch()
    
    best_edges = set(core_edges)
    best_adj = [0] * n
    for u, v in core_edges:
        best_adj[u] |= 1 << v
        best_adj[v] |= 1 << u
        
    def is_triangle_free(adj):
        for i in range(n):
            for j in range(i+1, n):
                if adj[i] & (1 << j):
                    if adj[i] & adj[j]:
                        return False
        return True

    def is_3_colorable(adj):
        # Backtracking 3-coloring
        color = [-1] * n
        order = sorted(range(n), key=lambda x: bin(adj[x]).count('1'), reverse=True)
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            avail = 7 # 0b111
            for u in order[:idx]:
                if color[u] != -1 and (adj[v] & (1 << u)):
                    avail &= ~(1 << color[u])
            if avail == 0:
                return False
            while avail:
                c = avail & (-avail)
                avail &= avail - 1
                color[v] = c.bit_length() - 1
                if backtrack(idx + 1):
                    return True
            color[v] = -1
            return False
            
        return backtrack(0)

    # Initial check
    if not is_triangle_free(best_adj) or is_3_colorable(best_adj):
        return [[u, v] for u, v in best_edges]

    best_edge_count = len(best_edges)
    
    all_possible_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Local search with random restarts
    while time.time() < deadline:
        candidate_edges = list(all_possible_edges)
        random.shuffle(candidate_edges)
        
        current_edges = set(core_edges)
        current_adj = [0] * n
        for u, v in current_edges:
            current_adj[u] |= 1 << v
            current_adj[v] |= 1 << u
            
        for u, v in candidate_edges:
            if u == v: continue
            if (1 << v) & current_adj[u]:
                continue
                
            if current_adj[u] & current_adj[v]:
                continue
                
            current_adj[u] |= 1 << v
            current_adj[v] |= 1 << u
            
            if not is_3_colorable(current_adj):
                current_edges.add((u, v))
            else:
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
                
        if len(current_edges) > best_edge_count:
            best_edge_count = len(current_edges)
            best_edges = current_edges
            best_adj = current_adj

    return [[u, v] for u, v in best_edges]
