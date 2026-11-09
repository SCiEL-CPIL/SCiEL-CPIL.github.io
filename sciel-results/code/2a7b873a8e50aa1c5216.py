import time
import random
import sys

def construct():
    N = 16
    target_edges = 65
    deadline = time.time() + 115.0
    
    # Initialize with seed graph to have a valid baseline
    # But we will overwrite with random restarts
    
    def generate_random_graph():
        all_edges = [(u, v) for u in range(N) for v in range(u+1, N)]
        random.shuffle(all_edges)
        return set(all_edges[:target_edges])
    
    def get_score(adj):
        deg = [0]*N
        for i in range(N):
            deg[i] = bin(adj[i]).count('1')
        
        max_sum = 0
        for u in range(N):
            for v in range(u+1, N):
                if adj[u] & (1 << v):
                    common = adj[u] & adj[v]
                    c = common
                    while c:
                        lsb = c & -c
                        w = lsb.bit_length() - 1
                        if w > v:
                            s = deg[u] + deg[v] + deg[w]
                            if s > max_sum:
                                max_sum = s
                        c -= lsb
        
        if max_sum == 0:
            return 1000 # Penalty for no triangles
        return max_sum

    def build_adj(edge_set):
        adj = [0]*N
        for u, v in edge_set:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    best_score = 1000
    best_edges = None
    
    # Run multiple restarts if time permits
    # Or one long SA run
    
    # Let's do a long SA run starting from a random graph
    # And keep the global best
    
    current_edges = generate_random_graph()
    current_adj = build_adj(current_edges)
    current_score = get_score(current_adj)
    
    if current_score < best_score:
        best_score = current_score
        best_edges = current_edges.copy()
        
    T = 2.0
    cooling = 0.9995
    min_T = 0.05
    
    # List of edges for fast random selection
    # We need to maintain the set of edges
    
    while time.time() < deadline:
        # Random swap
        # Pick random edge
        e1 = random.choice(list(current_edges))
        
        # Pick random non-edge
        while True:
            u = random.randint(0, N-1)
            v = random.randint(0, N-1)
            if u == v: continue
            if u > v: u, v = v, u
            e2 = (u, v)
            if e2 not in current_edges:
                break
        
        # Apply
        new_edges = current_edges.copy()
        new_edges.remove(e1)
        new_edges.add(e2)
        
        new_adj = build_adj(new_edges)
        new_score = get_score(new_adj)
        
        # Accept
        if new_score < current_score:
            current_edges = new_edges
            current_adj = new_adj
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_edges = current_edges.copy()
        elif new_score == current_score:
            # Accept to move around
            current_edges = new_edges
            current_adj = new_adj
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_edges = current_edges.copy()
        else:
            delta = new_score - current_score
            # Avoid overflow in exp
            if delta / T < 50:
                if random.random() < pow(2.718281828, -delta / T):
                    current_edges = new_edges
                    current_adj = new_adj
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_edges = current_edges.copy()
        
        T *= cooling
        if T < min_T:
            T = min_T
            
    # If best_edges is None (should not happen), fallback to seed
    if best_edges is None:
        best_edges = set([(u, v) for u in range(8) for v in range(8, 16)] + [(0, 1)])
        
    edge_list = [list(e) for e in best_edges]
    return {'n': N, 'edges': edge_list}
