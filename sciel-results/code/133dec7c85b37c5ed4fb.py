import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 135
    
    n_base = 32
    adj = [0] * n_base
    edges_base = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                u, v = base + i, base + j
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges_base.append([u, v])
    
    def has_c5(n, adj):
        # For each vertex, check if there's a C5 through it
        for v in range(n):
            neighbors = []
            v_bits = adj[v]
            while v_bits:
                bit = v_bits & -v_bits
                u = bit.bit_length() - 1
                neighbors.append(u)
                v_bits ^= bit
            # Check all pairs of neighbors
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    a, b = neighbors[i], neighbors[j]
                    # Check if a and b have a common neighbor other than v
                    common = adj[a] & adj[b] & ~(1 << v)
                    if common:
                        # Find c in common
                        c_bits = common
                        while c_bits:
                            bit = c_bits & -c_bits
                            c = bit.bit_length() - 1
                            c_bits ^= bit
                            # Check if c has a neighbor d that is also a neighbor of v (other than a,b,c)
                            d_candidates = adj[c] & adj[v] & ~(1 << v) & ~(1 << a) & ~(1 << b) & ~(1 << c)
                            if d_candidates:
                                return True
        return False
    
    def alpha_at_least(n, adj, target):
        # Check if alpha >= target using greedy + backtracking
        remaining = list(range(n))
        def backtrack(remaining, size):
            if size >= target:
                return True
            if not remaining:
                return False
            if size + len(remaining) < target:
                return False
            # Pick first vertex
            v = remaining[0]
            rest = remaining[1:]
            # Include v
            if backtrack([x for x in rest if not (adj[v] >> x) & 1], size + 1):
                return True
            # Exclude v
            if backtrack(rest, size):
                return True
            return False
        return backtrack(remaining, 0)
    
    best_n = n_base
    best_edges = edges_base[:]
    
    # Try random modifications
    for attempt in range(100):
        if time.time() > deadline:
            break
        
        # Try adding a 33rd vertex with random edges
        n = 33
        adj2 = adj[:] + [0]
        
        # Randomly connect vertex 32
        k = random.randint(4, 12)
        neighbors = random.sample(range(32), k)
        for v in neighbors:
            adj2[32] |= (1 << v)
            adj2[v] |= (1 << 32)
        
        if has_c5(n, adj2):
            continue
        if alpha_at_least(n, adj2, 9):
            continue
        
        # Found valid 33-vertex graph
        best_n = n
        best_edges = []
        for i in range(n):
            bits = adj2[i]
            while bits:
                bit = bits & -bits
                j = bit.bit_length() - 1
                bits ^= bit
                if i < j:
                    best_edges.append([i, j])
        break
    
    return {'n': best_n, 'edges': best_edges}
