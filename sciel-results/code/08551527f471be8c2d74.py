import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 112.0
    n = 18
    
    # Helper: Check if a set of vertices (mask) is an independent set in a graph defined by adj
    def is_independent(mask, adj):
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            if adj[v] & (mask ^ lsb):
                return False
            m ^= lsb
        return True

    # Helper: Check if adding edge (u,v) keeps graph K4-free
    def can_add_edge(adj, u, v):
        if (adj[u] >> v) & 1:
            return True
        common = adj[u] & adj[v]
        return is_independent(common, adj)

    # Helper: Calculate largest triangle-free induced subgraph size
    # Greedy heuristic with random restarts or simple greedy
    def calc_score(adj):
        # Greedy: pick vertices that don't complete a triangle
        # A triangle is formed if we have u, v, w in S and all 3 edges exist.
        # When adding v to S, check if there are u, w in S such that (u,v), (w,v), (u,w) exist.
        # This is equivalent to: v has two neighbors in S that are connected.
        
        best = 0
        # Try multiple random orders to get a better bound
        for _ in range(10):
            order = list(range(n))
            random.shuffle(order)
            S = 0
            for v in order:
                # Check if v can be added to S
                # v forms a triangle if there exist u, w in S such that 
                # adj[v] has u, adj[v] has w, and adj[u] has w.
                # Let N_v_in_S = adj[v] & S
                # We need to check if N_v_in_S contains an edge.
                # If N_v_in_S is an independent set, v can be added.
                if is_independent(adj[v] & S, adj):
                    S |= (1 << v)
            sz = bin(S).count('1')
            if sz > best:
                best = sz
        return best

    def get_edge_list(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    # Initialize empty graph
    adj = [0] * n
    
    # Generate a random maximal K4-free graph
    random.seed(42)
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_edges)
    
    for u, v in all_edges:
        if can_add_edge(adj, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
    current_score = calc_score(adj)
    best_score = current_score
    best_adj = adj[:]
    
    # Local Search
    T = 1.0
    T_min = 0.01
    alpha = 0.99
    
    while time.time() < deadline - 5.0 and T > T_min:
        T *= alpha
        
        # Random move: Add or Remove an edge
        if random.random() < 0.5:
            # Try adding an edge
            non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not ((adj[i] >> j) & 1)]
            if not non_edges:
                continue
            u, v = random.choice(non_edges)
            if can_add_edge(adj, u, v):
                # Add
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                new_score = calc_score(adj)
                delta = new_score - current_score
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_adj = adj[:]
                else:
                    # Revert
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
            else:
                continue
        else:
            # Try removing an edge
            edges = [(i, j) for i in range(n) for j in range(i+1, n) if ((adj[i] >> j) & 1)]
            if not edges:
                continue
            u, v = random.choice(edges)
            # Remove
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            new_score = calc_score(adj)
            delta = new_score - current_score
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_adj = adj[:]
            else:
                # Revert
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)

    return get_edge_list(best_adj)
