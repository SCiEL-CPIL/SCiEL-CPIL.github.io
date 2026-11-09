import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Grötzsch graph
    cycle = [(i, (i + 1) % 5) for i in range(5)]
    base_edges = [list(e) for e in cycle]
    for u, w in cycle:
        base_edges += [[u, 5 + w], [w, 5 + u]]
    base_edges += [[5 + u, 10] for u in range(5)]
    
    # Bitset representation
    # adj[i] is an integer where bit j is set if (i,j) is an edge
    adj = [0] * 16
    for u, v in base_edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    best_edges = base_edges[:]
    best_count = len(base_edges)
    best_adj = adj[:]
    
    def can_add(u, v, adj):
        # Check if (u,v) creates a triangle
        # Common neighbors of u and v
        common = adj[u] & adj[v]
        return common == 0
        
    def greedy_search(adj, deadline):
        # Candidates: all edges not in adj
        candidates = []
        for i in range(16):
            for j in range(i + 1, 16):
                if not (adj[i] & (1 << j)):
                    candidates.append((i, j))
        
        random.shuffle(candidates)
        
        current_adj = adj[:]
        current_edges = []
        # Reconstruct edges from adj? Or just track count.
        # We need to return the list of edges.
        # Let's track the list.
        
        # Initialize edges list from adj
        edges = []
        for i in range(16):
            for j in range(i + 1, 16):
                if current_adj[i] & (1 << j):
                    edges.append([i, j])
                    
        for u, v in candidates:
            if time.time() > deadline:
                break
            if can_add(u, v, current_adj):
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                edges.append([u, v])
                
        return edges, current_adj

    # Run multiple trials
    for _ in range(2000):
        if time.time() > deadline:
            break
        edges, adj = greedy_search(best_adj, deadline)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
            best_adj = adj

    # Local search: remove an edge, then greedily add
    for _ in range(1000):
        if time.time() > deadline:
            break
        current_adj = best_adj[:]
        current_edges = best_edges[:]
        
        # Remove a random edge
        if not current_edges:
            continue
        idx = random.randint(0, len(current_edges) - 1)
        u, v = current_edges[idx]
        current_adj[u] &= ~(1 << v)
        current_adj[v] &= ~(1 << u)
        current_edges.pop(idx)
        
        # Greedy add
        candidates = []
        for i in range(16):
            for j in range(i + 1, 16):
                if not (current_adj[i] & (1 << j)):
                    candidates.append((i, j))
        random.shuffle(candidates)
        
        for u2, v2 in candidates:
            if time.time() > deadline:
                break
            if can_add(u2, v2, current_adj):
                current_adj[u2] |= (1 << v2)
                current_adj[v2] |= (1 << u2)
                current_edges.append([u2, v2])
                
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges
            best_adj = current_adj

    return best_edges
