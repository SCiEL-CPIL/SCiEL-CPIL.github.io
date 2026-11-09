"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations

def construct():
    n = 20
    max_colors = 20
    
    # Precompute all 4-cycles in K_n
    # A 4-cycle is defined by 4 distinct vertices and a cyclic ordering
    # For vertices a,b,c,d, the 4-cycles are:
    # a-b-c-d-a, a-b-d-c-a, a-c-b-d-a
    # Each 4-cycle is represented by its 4 edges
    
    all_4cycles = []
    for a, b, c, d in combinations(range(n), 4):
        # 3 distinct 4-cycles on 4 vertices
        # Cycle 1: a-b-c-d-a
        all_4cycles.append(((a,b), (b,c), (c,d), (d,a)))
        # Cycle 2: a-b-d-c-a
        all_4cycles.append(((a,b), (b,d), (d,c), (c,a)))
        # Cycle 3: a-c-b-d-a
        all_4cycles.append(((a,c), (c,b), (b,d), (d,a)))
    
    # Map each edge to a tuple (u,v) with u<v for consistent lookup
    def edge_key(u, v):
        return (min(u,v), max(u,v))
    
    # Build edge-to-4cycles incidence: for each edge, which 4-cycles contain it
    edge_to_cycles = {edge_key(e[0], e[1]): [] for e in [(i,j) for i in range(n) for j in range(i+1,n)]}
    for cycle_idx, cycle in enumerate(all_4cycles):
        for edge in cycle:
            ek = edge_key(edge[0], edge[1])
            edge_to_cycles[ek].append(cycle_idx)
    
    # Store cycles as arrays for fast access
    # Each cycle: 4 edges, each edge has 2 endpoints
    # We'll store: cycle_edges[i] = [(u0,v0), (u1,v1), (u2,v2), (u3,v3)]
    cycle_edges = []
    for cycle in all_4cycles:
        cycle_edges.append([edge_key(e[0], e[1]) for e in cycle])
    
    cycle_edges = np.array([list(c) for c in cycle_edges])  # shape: (num_cycles, 4)
    # Actually, we need to keep them as tuples for dictionary lookup, so let's keep both
    cycle_edges_tuples = [tuple(c) for c in cycle_edges]
    
    # For fast checking, we'll maintain:
    # - adj: 2D array of colors (-1 if no edge)
    # - When we add an edge, we check all 4-cycles containing that edge
    
    import random
    import time
    
    def run_greedy(seed):
        random.seed(seed)
        adj = np.full((n, n), -1, dtype=np.int8)
        
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        
        def is_rainbow(c1, c2, c3, c4):
            return len({c1, c2, c3, c4}) == 4
        
        def can_add_edge(u, v, c):
            ek = edge_key(u, v)
            for cycle_idx in edge_to_cycles[ek]:
                # Get the 4 edges of this cycle
                c_edges = cycle_edges_tuples[cycle_idx]
                # Get colors of all 4 edges
                colors = []
                for ce in c_edges:
                    cu, cv = ce
                    if ce == ek:
                        colors.append(c)
                    else:
                        cc = adj[cu][cv]
                        if cc == -1:
                            # Edge doesn't exist yet, so this cycle is incomplete
                            # We can't check rainbow validity until all edges exist
                            # But we need to ensure that when all edges exist, it will be rainbow
                            # For now, if any edge is missing, we skip this check
                            # Actually, we should only check cycles where all other 3 edges exist
                            break
                        colors.append(cc)
                else:
                    # All 4 edges exist (or are being added)
                    if not is_rainbow(colors[0], colors[1], colors[2], colors[3]):
                        return False
            return True
        
        for (u, v) in all_edges:
            colors_to_try = list(range(max_colors))
            random.shuffle(colors_to_try)
            for c in colors_to_try:
                if can_add_edge(u, v, c):
                    adj[u][v] = c
                    adj[v][u] = c
                    break
        
        # Count edges
        edge_count = np.sum(adj > 0) // 2
        return edge_count, adj
    
    best_edges = 0
    best_adj = None
    
    start_time = time.time()
    trial = 0
    while time.time() - start_time < 110:
        edges, adj = run_greedy(trial)
        if edges > best_edges:
            best_edges = edges
            best_adj = adj
        trial += 1
    
    result = []
    if best_adj is not None:
        for i in range(n):
            for j in range(i+1, n):
                if best_adj[i][j] != -1:
                    result.append([i, j, int(best_adj[i][j])])
    
    return result
# EVOLVE-BLOCK-END
