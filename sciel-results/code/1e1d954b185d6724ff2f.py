import random
import time

def construct():
    n = 40
    best_edges = []
    best_count = 0
    start_time = time.time()
    time_limit = 115  # Leave 5s buffer
    
    # Precompute bit masks for vertices
    # bit(i) = 1 << i
    
    def creates_c6(u, v, adj):
        # Check if adding (u, v) creates a C6.
        # This is true if there is a path of length 4 from u to v in the current graph.
        # Path: u - a - b - c - d - v
        # a in N(u), d in N(v)
        # a-b-c-d is a path of length 3.
        
        # Neighbors of u
        mask_u = adj[u]
        # Neighbors of v
        mask_v = adj[v]
        
        # Iterate a in N(u)
        # We can iterate over set bits
        mu = mask_u
        while mu:
            # Get lowest set bit
            lb = mu & -mu
            a = lb.bit_length() - 1
            mu ^= lb
            
            # a cannot be v (but v is not in N(u) yet, so this is safe)
            # Iterate b in N(a) \ {u}
            mask_a = adj[a]
            mask_b = mask_a & ~(1 << u)
            
            mb = mask_b
            while mb:
                lb_b = mb & -mb
                b = lb_b.bit_length() - 1
                mb ^= lb_b
                
                # Iterate c in N(b) \ {a}
                mask_c = adj[b]
                mask_c = mask_c & ~(1 << a)
                
                # We need d in N(c) AND d in N(v)
                # d must not be a, b, u, v
                # mask_v does not contain u (since uv not added)
                # mask_c does not contain b (simple graph)
                # We need to mask out a and b from the intersection?
                # Intersection = N(c) & N(v)
                # If d=a, then a in N(v). Path u-a-b-c-a-v. Not simple.
                # If d=b, then b in N(v). Path u-a-b-c-b-v. Not simple.
                
                intersection = mask_c & mask_v
                intersection &= ~(1 << a)
                intersection &= ~(1 << b)
                
                if intersection:
                    return True
                    
        return False

    def run_greedy(seed):
        random.seed(seed)
        adj = [0] * n
        edges = []
        
        # Generate all possible edges
        candidates = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(candidates)
        
        for u, v in candidates:
            if not (adj[u] >> v) & 1:
                if not creates_c6(u, v, adj):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append((u, v))
        
        return edges

    # Run multiple greedy passes with different random seeds
    seed = 0
    while time.time() - start_time < time_limit:
        edges = run_greedy(seed)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
        seed += 1
        
    return [list(e) for e in best_edges]
