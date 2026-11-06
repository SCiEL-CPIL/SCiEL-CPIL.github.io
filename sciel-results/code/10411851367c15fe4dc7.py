import time
import random

def construct():
    N = 40
    all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
    num_edges = len(all_edges)
    
    # Deadline: 110 seconds to be safe
    start_time = time.time()
    deadline = start_time + 110
    
    best_edge_count = 0
    best_adj = None
    best_edge_list = []
    
    # Helper to check if adding (u, v) creates a C6
    # Path: u - a - b - c - d - v
    # All distinct.
    def creates_c6(adj, u, v):
        # adj is a list of lists of neighbors
        # We need to find a, b, c, d such that:
        # a in adj[u]
        # b in adj[a]
        # c in adj[b]
        # d in adj[c]
        # v in adj[d]
        # and {u, a, b, c, d, v} are all distinct.
        
        # Optimization: 
        # a must be in adj[u] and a != v
        # b must be in adj[a] and b not in {u, v}
        # c must be in adj[b] and c not in {u, v, a}
        # d must be in adj[c] and d not in {u, v, a, b}
        # and v in adj[d]
        
        for a in adj[u]:
            if a == v:
                continue
            for b in adj[a]:
                if b == u or b == v or b == a:
                    continue
                for c in adj[b]:
                    if c == u or c == v or c == a or c == b:
                        continue
                    for d in adj[c]:
                        if d == u or d == v or d == a or d == b or d == c:
                            continue
                        if v in adj[d]:
                            return True
        return False

    # Initial seed: 8 disjoint K5s
    # We will run randomized greedy from scratch.
    
    while time.time() < deadline:
        random.shuffle(all_edges)
        adj = [[] for _ in range(N)]
        current_edges = 0
        
        for u, v in all_edges:
            # Check if adding (u, v) creates a C6
            if not creates_c6(adj, u, v):
                adj[u].append(v)
                adj[v].append(u)
                current_edges += 1
        
        if current_edges > best_edge_count:
            best_edge_count = current_edges
            best_adj = [list(lst) for lst in adj]
            # Reconstruct edge list for return
            best_edge_list = []
            for u in range(N):
                for v in best_adj[u]:
                    if u < v:
                        best_edge_list.append([u, v])
            
            # If we reached the theoretical max or known best, we might stop early?
            # Known best is 105. If we hit 105, we can keep searching for 106?
            # The task says "Maximize".
            
    # If no better graph found, return seed
    if best_edge_count <= 100:
         # Return the seed graph
         edges = []
         for k in range(8):
             edges += [[5 * k + a, 5 * k + b] for a in range(5) for b in range(a + 1, 5)]
         return edges
         
    return best_edge_list
