import time
import random

def construct():
    n = 40
    start_time = time.time()
    deadline = 145.0  # Safety margin

    def is_safe(u, v, neighbors):
        # Check if there is a simple path of length 5 from u to v
        visited = 1 << u
        current = 1 << u
        
        for step in range(5):
            next_set = 0
            c = current
            while c:
                lsb = c & -c
                idx = lsb.bit_length() - 1
                c ^= lsb
                next_set |= neighbors[idx]
            
            next_set &= ~visited
            visited |= next_set
            
            if step == 4:
                if next_set & (1 << v):
                    return False
                return True
                
            current = next_set
            if current == 0:
                return True
                
        return True

    def greedy_build(seed):
        random.seed(seed)
        neighbors = [0] * n
        edges = []
        
        all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if is_safe(u, v, neighbors):
                neighbors[u] |= (1 << v)
                neighbors[v] |= (1 << u)
                edges.append([u, v])
                
        return edges

    best_edges = []
    best_count = -1
    
    seed = 0
    while time.time() < deadline:
        edges = greedy_build(seed)
        count = len(edges)
        if count > best_count:
            best_count = count
            best_edges = edges
        seed += 1
        
    return best_edges
