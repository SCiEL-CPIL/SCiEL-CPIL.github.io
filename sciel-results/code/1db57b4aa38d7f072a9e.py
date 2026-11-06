import random
import time

def construct():
    N = 40
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    best_edges = []
    best_count = 0
    
    def creates_c6(adj_bits, u, v):
        visited_mask = 1 << u
        stack = [(u, 0)]
        while stack:
            node, depth = stack.pop()
            if depth == 5:
                if node == v:
                    return True
                continue
            next_depth = depth + 1
            neighbors = adj_bits[node]
            while neighbors:
                lsb = neighbors & -neighbors
                neighbor = lsb.bit_length() - 1
                neighbors ^= lsb
                
                if neighbor == v and next_depth == 5:
                    return True
                
                if not (visited_mask & (1 << neighbor)):
                    visited_mask |= (1 << neighbor)
                    stack.append((neighbor, next_depth))
        return False

    def add_edge(adj_bits, u, v):
        adj_bits[u] |= (1 << v)
        adj_bits[v] |= (1 << u)

    def remove_edge(adj_bits, u, v):
        adj_bits[u] &= ~(1 << v)
        adj_bits[v] &= ~(1 << u)

    # Initial best: 8 K5s
    for k in range(8):
        for a in range(5):
            for b in range(a+1, 5):
                best_edges.append([5*k+a, 5*k+b])
    best_count = len(best_edges)
    
    # Search
    # We will run multiple restarts
    # Each restart: 
    # 1. Start from empty or random
    # 2. Greedy fill
    # 3. Local improve
    
    for restart in range(100):
        if time.time() > deadline:
            break
        
        # Initialize
        adj_bits = [0] * N
        edges = []
        
        # Random start? Or empty?
        # Empty is safer to avoid initial C6s, but slower to fill.
        # Let's try empty.
        
        all_edges = [(i, j) for i in range(N) for j in range(i+1, N)]
        random.shuffle(all_edges)
        
        # Greedy Phase
        for u, v in all_edges:
            if time.time() > deadline:
                break
            if not creates_c6(adj_bits, u, v):
                add_edge(adj_bits, u, v)
                edges.append((u, v))
        
        # Local Search Phase
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            # Try to add missing edges
            for u in range(N):
                for v in range(u+1, N):
                    if not (adj_bits[u] & (1 << v)):
                        if not creates_c6(adj_bits, u, v):
                            add_edge(adj_bits, u, v)
                            edges.append((u, v))
                            improved = True
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = [[u, v] for u, v in edges]
            
    return best_edges
