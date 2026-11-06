import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    time_limit = 115  # Leave some buffer
    
    n = 16
    
    def find_q3_edges(adj):
        """
        Finds a Q3 subgraph in the graph represented by adj (list of bitmasks).
        Returns a list of 12 edges (tuples (u,v)) if found, else None.
        """
        for v0 in range(n):
            neigh_v0 = adj[v0]
            # Get list of neighbors of v0
            neigh_list = []
            mask = neigh_v0
            while mask:
                v = (mask & -mask).bit_length() - 1
                mask &= mask - 1
                if v != v0:
                    neigh_list.append(v)
            
            # We need to choose 3 neighbors for v1, v2, v4 (neighbors of 0 in Q3)
            # Q3 neighbors of 0 are 1, 2, 4.
            for v1, v2, v4 in combinations(neigh_list, 3):
                # v3 must be neighbor of v1 and v2
                common_12 = adj[v1] & adj[v2]
                # v5 must be neighbor of v1 and v4
                common_14 = adj[v1] & adj[v4]
                # v6 must be neighbor of v2 and v4
                common_24 = adj[v2] & adj[v4]
                
                # Iterate v3
                mask3 = common_12
                while mask3:
                    v3 = (mask3 & -mask3).bit_length() - 1
                    mask3 &= mask3 - 1
                    if v3 in (v0, v1, v2, v4): continue
                    
                    # Iterate v5
                    mask5 = common_14
                    while mask5:
                        v5 = (mask5 & -mask5).bit_length() - 1
                        mask5 &= mask5 - 1
                        if v5 in (v0, v1, v2, v4, v3): continue
                        
                        # Iterate v6
                        mask6 = common_24
                        while mask6:
                            v6 = (mask6 & -mask6).bit_length() - 1
                            mask6 &= mask6 - 1
                            if v6 in (v0, v1, v2, v4, v3, v5): continue
                            
                            # v7 must be neighbor of v3, v5, v6
                            common_356 = adj[v3] & adj[v5] & adj[v6]
                            mask7 = common_356
                            while mask7:
                                v7 = (mask7 & -mask7).bit_length() - 1
                                mask7 &= mask7 - 1
                                if v7 in (v0, v1, v2, v3, v4, v5, v6): continue
                                
                                # Found Q3
                                edges = [
                                    (min(v0,v1), max(v0,v1)),
                                    (min(v0,v2), max(v0,v2)),
                                    (min(v0,v4), max(v0,v4)),
                                    (min(v1,v3), max(v1,v3)),
                                    (min(v1,v5), max(v1,v5)),
                                    (min(v2,v3), max(v2,v3)),
                                    (min(v2,v6), max(v2,v6)),
                                    (min(v3,v7), max(v3,v7)),
                                    (min(v4,v5), max(v4,v5)),
                                    (min(v4,v6), max(v4,v6)),
                                    (min(v5,v7), max(v5,v7)),
                                    (min(v6,v7), max(v6,v7))
                                ]
                                return edges
        return None

    best_edges = 0
    best_adj = None
    
    # Run multiple random restarts
    num_restarts = 1000 # High number, will break on time
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        random.seed(42 + restart)
        
        # Start with K16
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
        # Greedy removal
        while True:
            if time.time() - start_time > time_limit:
                break
                
            q3_edges = find_q3_edges(adj)
            if q3_edges is None:
                # No Q3 found, graph is valid
                edge_count = sum(bin(adj[i]).count('1') for i in range(n)) // 2
                if edge_count > best_edges:
                    best_edges = edge_count
                    best_adj = adj[:]
                break
            else:
                # Pick a random edge from the Q3 to remove
                u, v = random.choice(q3_edges)
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                
    # Convert best_adj to list of edges
    if best_adj is None:
        # Fallback to seed if something went wrong
        return [[a, b] for a in range(3) for b in range(3, 16)]
        
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
                
    return edges
