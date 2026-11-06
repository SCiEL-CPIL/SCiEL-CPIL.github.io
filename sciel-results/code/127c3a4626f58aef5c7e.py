import time
import random

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    def is_valid(adj):
        # Check if graph is K3,3 free
        # No 3 vertices have 3 common neighbors
        for i in range(n):
            ai = adj[i]
            for j in range(i + 1, n):
                aj = adj[j]
                common_ij = ai & aj
                # Optimization: if common neighbors of i,j < 2, they can't share 3 with any k
                if common_ij.bit_count() < 2:
                    continue
                for k in range(j + 1, n):
                    if (common_ij & adj[k]).bit_count() >= 3:
                        return False
        return True

    def count_edges(adj):
        return sum(a.bit_count() for a in adj) // 2

    # Initial random graph
    def random_graph():
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < 0.3:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    best_adj = [0] * n
    best_count = 0
    current_adj = random_graph()
    
    if is_valid(current_adj):
        best_adj = current_adj[:]
        best_count = count_edges(current_adj)

    # Local search
    while time.time() < deadline:
        # Try to add an edge
        added = False
        # Randomly shuffle pairs to avoid bias
        pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(pairs)
        
        for u, v in pairs:
            if time.time() > deadline:
                break
            if current_adj[u] & (1 << v):
                continue
            
            # Try adding edge
            current_adj[u] |= (1 << v)
            current_adj[v] |= (1 << u)
            
            if is_valid(current_adj):
                added = True
                new_count = count_edges(current_adj)
                if new_count > best_count:
                    best_count = new_count
                    best_adj = current_adj[:]
            else:
                # Rollback
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
        
        if not added:
            # Stuck, remove a random edge
            edges_to_remove = []
            for u in range(n):
                for v in range(u + 1, n):
                    if current_adj[u] & (1 << v):
                        edges_to_remove.append((u, v))
            
            if edges_to_remove:
                u, v = random.choice(edges_to_remove)
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
            else:
                # Empty graph, restart
                current_adj = random_graph()
                if is_valid(current_adj):
                    current_count = count_edges(current_adj)
                    if current_count > best_count:
                        best_count = current_count
                        best_adj = current_adj[:]
    
    # Convert best_adj to list of edges
    result = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u] & (1 << v):
                result.append([u, v])
    return result
