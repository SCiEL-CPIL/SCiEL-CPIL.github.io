import numpy as np
import time
import sys

sys.setrecursionlimit(10000)

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    k = 2
    
    # 1. Precompute forbidden differences (non-zero squares)
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, k, m))
    forbidden.discard(0)
    
    # 2. Build adjacency as bitmasks
    # adj[u] is a bitmask where bit v is 1 if (u,v) is an edge (forbidden pair)
    adj = [0] * m
    for u in range(m):
        mask = 0
        for v in range(m):
            if u != v:
                d = (u - v) % m
                if d in forbidden:
                    mask |= (1 << v)
        adj[u] = mask
    
    # Precompute degrees for heuristic ordering
    degrees = [adj[i].bit_count() for i in range(m)]
    
    # We want to find a Maximum Independent Set (MIS) in G.
    # This is equivalent to finding a Maximum Clique in the complement graph G'.
    # In G', an edge exists between u and v if they are NOT forbidden in G.
    # Let's work directly on G for MIS using Branch and Bound.
    
    # Greedy coloring of the subgraph induced by P.
    # We want an upper bound on the size of an independent set in G[P].
    # This is equivalent to an upper bound on the clique number of G'[P].
    # A valid upper bound is the chromatic number of G[P] (since an independent set in G is a clique in G', and a clique in G' is an independent set in G... wait).
    
    # Let's stick to the standard MIS B&B:
    # We are searching for an Independent Set in G.
    # The coloring bound: If we color the graph G[P] with k colors, then any Independent Set in G[P] can have at most 1 vertex from each color class?
    # NO. An independent set in G is a set of vertices with NO edges between them in G.
    # A color class in G is an independent set in G.
    # So, if G[P] has chromatic number k, it can be partitioned into k independent sets.
    # This doesn't directly bound the size of a SINGLE independent set.
    
    # CORRECTION:
    # The standard bound for MIS using coloring is:
    # If we color the COMPLEMENT graph G'[P] with k colors, then any Clique in G'[P] (which is an Independent Set in G[P]) has size at most k.
    # So we need to compute an upper bound on the chromatic number of G'[P].
    # Greedy coloring gives a valid upper bound on the chromatic number.
    
    # Let's define adj_comp[u] as the bitmask of neighbors in the complement graph (excluding self).
    # adj_comp[u] = ~adj[u] & ~(1<<u) & ((1<<m)-1)
    
    all_mask = (1 << m) - 1
    adj_comp = [0] * m
    for u in range(m):
        # Neighbors in complement are those v != u such that (u,v) is NOT an edge in G.
        # adj[u] has bits for neighbors in G.
        # So ~adj[u] has bits for non-neighbors (including u).
        # We exclude u.
        adj_comp[u] = (all_mask ^ adj[u]) & ~(1 << u)
    
    # Greedy coloring of G'[P]
    # Returns number of colors used.
    def greedy_color_comp(P):
        colors = 0
        U = P
        while U:
            colors += 1
            # Find a maximal independent set in G'[U], which is a maximal clique in G[U].
            # Wait, greedy coloring picks a maximal independent set in the graph being colored.
            # We are coloring G'[P].
            # So we pick a maximal independent set in G'[U].
            # An independent set in G' is a clique in G.
            # So we are picking a maximal clique in G[U] to assign the current color.
            
            # Let's just implement standard greedy coloring on G'[P].
            # We process vertices in some order.
            # To be fast, we can pick the vertex with the smallest index or use a heuristic.
            # Let's pick the vertex with the smallest index for simplicity, or maybe the one with highest degree in G'[U]?
            # For speed, simple order is often fine.
            
            # Actually, a faster way to do greedy coloring is:
            # While U is not empty:
            #   colors++
            #   I = empty
            #   While U is not empty:
            #      pick v from U
            #      add v to I
            #      remove from U all vertices w in U such that w is adjacent to v in G' (i.e., w is NOT adjacent to v in G)
            #      i.e., remove w if (adj_comp[v] & (1<<w)) != 0
            #      i.e., keep w if (adj_comp[v] & (1<<w)) == 0
            
            I = 0
            # Pick vertices greedily.
            # To make the bound tighter, we should pick vertices that are "hard" to color, i.e., high degree in G'?
            # Or just lowest index.
            
            # Let's use a heuristic: pick the vertex in U with the highest degree in G'[U].
            # But computing degree dynamically is expensive.
            # Let's stick to lowest index for now, it's O(V^2) per coloring which is fast enough for 205.
            
            while U:
                # Pick lowest set bit
                v_bit = U & (-U)
                v = v_bit.bit_length() - 1
                I |= v_bit
                # Remove v and its neighbors in G' from U
                # Neighbors in G' are adj_comp[v]
                # So we remove U & adj_comp[v]
                U &= ~adj_comp[v]
                U &= ~v_bit
        
        return colors

    best_size = 0
    best_set_mask = 0
    start_time = time.time()
    deadline = start_time + 590  # Leave 10s margin
    
    # Recursive search
    # P: bitmask of available vertices in G
    # C: current size of independent set
    def search(P, C, current_set):
        nonlocal best_size, best_set_mask
        
        if time.time() > deadline:
            return
        
        if not P:
            if C > best_size:
                best_size = C
                best_set_mask = current_set
            return
        
        # Bound 1: Cardinality
        if C + P.bit_count() <= best_size:
            return
            
        # Bound 2: Coloring
        # Calculate upper bound on MIS size in G[P]
        # This is the chromatic number of G'[P]
        colors = greedy_color_comp(P)
        if C + colors <= best_size:
            return
            
        # Branching
        # Choose a vertex to branch on.
        # Heuristic: Choose a vertex with high degree in G[P]?
        # High degree in G means it forbids many others, so including it prunes a lot.
        # Let's pick the vertex in P with the highest degree in G[P].
        
        # To find the vertex with max degree in G[P]:
        # deg(v) in G[P] = popcount(adj[v] & P)
        
        max_deg = -1
        v_bit = 0
        v = 0
        temp_P = P
        while temp_P:
            b = temp_P & (-temp_P)
            i = b.bit_length() - 1
            d = (adj[i] & P).bit_count()
            if d > max_deg:
                max_deg = d
                v_bit = b
                v = i
            temp_P ^= b
        
        # Branch 1: Include v
        # New available vertices: P minus v and its neighbors in G
        P_include = P & ~(adj[v] | v_bit)
        search(P_include, C + 1, current_set | v_bit)
        
        # Branch 2: Exclude v
        P_exclude = P & ~v_bit
        search(P_exclude, C, current_set)

    # Initial P is all vertices
    initial_P = (1 << m) - 1
    
    # Run search
    search(initial_P, 0, 0)
    
    # Extract result
    result = []
    for i in range(m):
        if (best_set_mask >> i) & 1:
            result.append(i)
            
    return np.array(result, dtype=np.int64)
