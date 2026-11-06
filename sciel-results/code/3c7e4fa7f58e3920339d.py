import random
import time

def construct():
    n = 16
    # Initial graph: K_{8,8} + (0,1)
    edges = set()
    for u in range(8):
        for v in range(8, 16):
            edges.add((u, v))
    edges.add((0, 1))
    
    # Precompute all triples
    triples = [(i, j, k) for i in range(n) for j in range(i+1, n) for k in range(j+1, n)]
    
    def get_degrees(edges):
        degs = [0]*n
        for u, v in edges:
            degs[u] += 1
            degs[v] += 1
        return degs
    
    def is_triangle(triple, edges):
        u, v, w = triple
        return (u,v) in edges or (v,u) in edges and \
               (v,w) in edges or (w,v) in edges and \
               (u,w) in edges or (w,u) in edges
               
    # Note: edges stored as frozenset or sorted tuple for fast lookup?
    # Using set of frozensets is slow. Set of tuples (min, max) is better.
    
    def calc_max_sum(edges):
        degs = get_degrees(edges)
        max_sum = 0
        for u, v, w in triples:
            if ((u,v) in edges or (v,u) in edges) and \
               ((v,w) in edges or (w,v) in edges) and \
               ((u,w) in edges or (w,u) in edges):
                s = degs[u] + degs[v] + degs[w]
                if s > max_sum:
                    max_sum = s
        return max_sum

    current_edges = edges
    current_score = calc_max_sum(current_edges)
    best_score = current_score
    best_edges = set(current_edges)
    
    start_time = time.time()
    timeout = 110 # Safety margin
    
    # Hill Climbing
    while time.time() - start_time < timeout:
        # Pick random edge to remove
        if not current_edges: break
        u, v = random.choice(list(current_edges))
        # Pick random non-edge to add
        # Generate random pair
        while True:
            x = random.randint(0, n-1)
            y = random.randint(0, n-1)
            if x == y: continue
            if x > y: x, y = y, x
            if (x, y) not in current_edges:
                break
        
        # Apply move
        current_edges.remove((u, v))
        current_edges.add((x, y))
        
        new_score = calc_max_sum(current_edges)
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_edges = set(current_edges)
        else:
            # Revert
            current_edges.remove((x, y))
            current_edges.add((u, v))
            
    # Convert best_edges to list of lists
    final_edges = [[u, v] for u, v in best_edges]
    return {'n': n, 'edges': final_edges}
