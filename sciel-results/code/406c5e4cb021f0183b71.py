import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    V = 40
    adj = [set() for _ in range(V)]
    
    # Initial structure: 8 disjoint K5s
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                u = 5 * k + a
                v = 5 * k + b
                adj[u].add(v)
                adj[v].add(u)
                
    def edge_creates_c6(u, v):
        # Check if adding edge (u, v) creates a C6
        # Path: u - a - b - c - d - v
        # a in N(u), d in N(v)
        # b in N(a) \ {u, v, d}
        # c in N(d) \ {u, v, a}
        # c in N(b)
        
        Nu = adj[u]
        Nv = adj[v]
        
        # Optimization: If u and v have many common neighbors, it's harder to be C6 free?
        # Actually, common neighbors create short cycles, not necessarily C6.
        
        for a in Nu:
            for d in Nv:
                if a == d:
                    continue
                
                # b candidates
                # c candidates
                # We iterate b in N(a) and check if N(b) intersects N(d) appropriately
                
                # To speed up, we can check intersection of N(a) and N(d) first?
                # No, we need b-c edge.
                
                # Iterate b
                for b in adj[a]:
                    if b == u or b == v or b == d:
                        continue
                    # Check if any neighbor of b is a valid c
                    # c must be in N(d) \ {u, v, a}
                    # and c != b
                    # and c in N(b)
                    
                    # We can check if (N(b) intersect N(d)) contains a valid c
                    # Valid c: in N(d), not u, not v, not a, not b
                    # Let's just iterate neighbors of b
                    for c in adj[b]:
                        if c == u or c == v or c == a or c == b:
                            continue
                        if c in adj[d]:
                            return True
        return False

    def count_edges():
        return sum(len(s) for s in adj) // 2

    def greedy_fill():
        # Try to add all possible non-edges
        # To avoid O(V^2) scan every time, we can just shuffle and try
        # But for 40 vertices, V^2 is small.
        non_edges = []
        for i in range(V):
            for j in range(i + 1, V):
                if j not in adj[i]:
                    non_edges.append((i, j))
        
        random.shuffle(non_edges)
        added = 0
        for u, v in non_edges:
            if not edge_creates_c6(u, v):
                adj[u].add(v)
                adj[v].add(u)
                added += 1
        return added

    # Initial greedy fill
    greedy_fill()
    
    best_edges = count_edges()
    best_adj = [set(s) for s in adj]
    
    while time.time() < deadline:
        # Try to add random edges
        improved = False
        for _ in range(50):
            u = random.randint(0, V - 1)
            v = random.randint(0, V - 1)
            if u == v:
                continue
            if v in adj[u]:
                continue
            
            if not edge_creates_c6(u, v):
                adj[u].add(v)
                adj[v].add(u)
                improved = True
                curr_edges = count_edges()
                if curr_edges > best_edges:
                    best_edges = curr_edges
                    best_adj = [set(s) for s in adj]
        
        if improved:
            continue
        
        # Stuck. Perturb: Remove 1 random edge
        all_edges = []
        for i in range(V):
            for j in adj[i]:
                if i < j:
                    all_edges.append((i, j))
        
        if not all_edges:
            break
        
        u, v = random.choice(all_edges)
        adj[u].remove(v)
        adj[v].remove(u)
        
        # Greedy fill again
        greedy_fill()
        
        curr_edges = count_edges()
        if curr_edges > best_edges:
            best_edges = curr_edges
            best_adj = [set(s) for s in adj]
        else:
            # Revert
            adj = [set(s) for s in best_adj]
            
    # Construct result
    edges = []
    for i in range(V):
        for j in best_adj[i]:
            if i < j:
                edges.append([i, j])
    return edges
