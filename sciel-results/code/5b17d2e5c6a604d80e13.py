"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations, permutations

def construct():
    start = time.time()
    deadline = start + 140
    n = 16
    
    # Q3 structure: 8 vertices (0-7 as 3-bit), 12 edges
    q3_edges = []
    for u in range(8):
        for v in range(u+1, 8):
            d = u ^ v
            if d & (d - 1) == 0:  # power of 2
                q3_edges.append((u, v))
    assert len(q3_edges) == 12
    
    # All edges in K16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    m = len(all_edges)  # 120
    edge_idx = {e: i for i, e in enumerate(all_edges)}
    
    # Precompute all Q3 embeddings as frozensets of K16 edges
    # An embedding is a bijection from Q3 vertices {0..7} to 8 distinct vertices in 0..15
    # such that all 12 Q3 edges map to existing K16 edges
    q3_embeddings = []  # list of frozensets of edge indices
    
    # For each 8-subset of vertices, check if it contains a Q3
    # But that's C(16,8) = 12870 subsets, each needing isomorphism check
    # Alternative: enumerate all injective maps from Q3 vertices to 16 vertices
    # That's P(16,8) = 16!/(16-8)! = 518,918,400 - too many
    
    # Better: for each pair of edges that could be "first" in a Q3, extend
    # Or: use the fact that Q3 is determined by its structure
    
    # Actually, let's precompute differently:
    # For each potential Q3 embedding, we need 8 distinct vertices.
    # The Q3 has a specific edge structure. Let's enumerate systematically.
    
    # Key insight: Q3 is vertex-transitive and edge-transitive.
    # Any two edges in Q3 that share a vertex can be the "starting" pair.
    # Let's enumerate by choosing which K16 edges form the Q3.
    
    # More practical approach: for each 8-subset S of {0..15}, check if G[S] contains Q3
    # But we want to precompute which edge sets form Q3s regardless of the graph.
    
    # Let's think differently. A Q3 in K16 is determined by:
    # - 8 distinct vertices
    # - A labeling of these vertices as 0..7 (the Q3 vertices)
    # - All 12 Q3 edges must be present in K16 (they always are, since K16 is complete)
    # So ANY 8 distinct vertices form a Q3 in K16? No! Only if the induced subgraph 
    # is isomorphic to Q3. In K16, the induced subgraph on any 8 vertices is K8, not Q3.
    
    # I was confused. Q3 is a specific graph on 8 vertices with 12 edges.
    # In K16, a Q3 subgraph exists if there are 8 vertices where exactly the 12 Q3 edges
    # are present (and possibly other edges, but Q3 as a subgraph only requires the 12 edges).
    
    # Wait, "subgraph isomorphic to Q3" means there exists an injection f: {0..7} -> {0..15}
    # such that for every edge (a,b) in Q3, (f(a), f(b)) is an edge in G.
    # It does NOT require that non-edges of Q3 are non-edges in G.
    
    # So in K16, ANY injection f: {0..7} -> {0..15} gives a Q3 subgraph!
    # Because all edges exist in K16.
    
    # That means K16 itself contains Q3 as a subgraph. So we can't use K16.
    
    # The problem is: find a Q3-FREE graph, i.e., one with NO subgraph isomorphic to Q3.
    # So we need a graph where no 8 vertices have all 12 Q3 edges present.
    
    # Precomputation: for each set of 12 edges that forms a Q3 pattern, 
    # we want to know if all 12 are present in our graph.
    
    # Let's precompute all "Q3 edge patterns": for each 8-tuple of distinct vertices,
    # and each bijection to Q3 vertices, the 12 edges form a Q3.
    # But that's too many. Instead, let's use the structure.
    
    # Q3 has vertices 0-7. Edges are between vertices differing in one bit.
    # Let's precompute for each choice of 8 vertices (v0,...,v7) and each permutation
    # assigning them to Q3 positions, whether the 12 required edges exist.
    
    # This is still large. Let's use a different strategy:
    # Precompute a "conflict" structure: for each edge e, which other edges 
    # "conflict" with it (i.e., together with some other edges, they form a Q3).
    
    # Actually, the most practical approach for n=16:
    # Use the bitmask-based check from the current program, but optimize it.
    # The current program's has_q3_fast is already quite good.
    
    # Let me refine the approach:
    # 1. Start with a good initial graph (K_{3,13} or better)
    # 2. Use efficient incremental Q3 checking
    # 3. Multiple greedy restarts with different orderings
    # 4. Kick + greedy local search
    
    # Bitmask adjacency
    def has_q3_fast(adj, u, v):
        """Check if adding edge (u,v) creates a Q3. adj is list of bitmasks."""
        mask_u = adj[u]
        mask_v = adj[v]
        others = ((1 << n) - 1) & ~(1 << u) & ~(1 << v)
        
        cands_24 = mask_u & others
        cands_35 = mask_v & others
        
        m24 = cands_24
        while m24:
            lsb2 = m24 & -m24
            x2 = lsb2.bit_length() - 1
            m24 ^= lsb2
            
            m4 = cands_24 & ~(1 << x2)
            while m4:
                lsb4 = m4 & -m4
                x4 = lsb4.bit_length() - 1
                m4 ^= lsb4
                
                cands_3 = cands_35 & adj[x2] & ~(1 << u) & ~(1 << v) & ~(1 << x2) & ~(1 << x4)
                cands_5 = cands_35 & adj[x4] & ~(1 << u) & ~(1 << v) & ~(1 << x2) & ~(1 << x4)
                
                m3 = cands_3
                while m3:
                    lsb3 = m3 & -m3
                    x3 = lsb3.bit_length() - 1
                    m3 ^= lsb3
                    
                    m5 = cands_5 & ~(1 << x3)
                    while m5:
                        lsb5 = m5 & -m5
                        x5 = lsb5.bit_length() - 1
                        m5 ^= lsb5
                        
                        used = (1 << u) | (1 << v) | (1 << x2) | (1 << x3) | (1 << x4) | (1 << x5)
                        cands_6 = adj[x2] & adj[x4] & ~used
                        cands_7 = adj[x3] & adj[x5] & ~used
                        
                        m6 = cands_6
                        while m6:
                            lsb6 = m6 & -m6
                            x6 = lsb6.bit_length() - 1
                            m6 ^= lsb6
                            
                            m7 = cands_7 & ~(1 << x6)
                            while m7:
                                lsb7 = m7 & -m7
                                x7 = lsb7.bit_length() - 1
                                m7 ^= lsb7
                                
                                if adj[x6] & (1 << x7):
                                    return True
        return False
    
    def run_greedy(adj, edges, seed, deadline):
        random.seed(seed)
        missing = [e for e in all_edges if e not in edges]
        random.shuffle(missing)
        
        for e in missing:
            if time.time() > deadline - 15:
                break
            u, v = e
            if has_q3_fast(adj, u, v):
                continue
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.add(e)
        return adj, edges
    
    # Start with K_{3,13}
    best_adj = [0] * n
    best_edges = set()
    for a in range(3):
        for b in range(3, 16):
            best_adj[a] |= (1 << b)
            best_adj[b] |= (1 << a)
            best_edges.add((a, b))
    best_count = len(best_edges)
    
    # Multiple greedy restarts
    for i in range(80):
        if time.time() > deadline - 25:
            break
        
        adj, edges = run_greedy(best_adj[:], set(best_edges), i, deadline)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_adj = adj
            best_edges = edges
            
            # Kick + greedy on new best
            if time.time() < deadline - 20:
                for k in range(5):
                    if time.time() > deadline - 15:
                        break
                    kick_adj = best_adj[:]
                    kick_edges = set(best_edges)
                    kick_list = list(kick_edges)
                    random.seed(i * 100 + k)
                    random.shuffle(kick_list)
                    for e in kick_list[:3]:
                        u, v = e
                        kick_adj[u] &= ~(1 << v)
                        kick_adj[v] &= ~(1 << u)
                        kick_edges.remove(e)
                    
                    k_adj, k_edges = run_greedy(kick_adj, kick_edges, i * 100 + k + 1, deadline)
                    
                    if len(k_edges) > best_count:
                        best_count = len(k_edges)
                        best_adj = k_adj
                        best_edges = k_edges
    
    # Final polish: try adding any remaining edges
    if time.time() < deadline - 5:
        for e in all_edges:
            if time.time() > deadline - 3:
                break
            if e in best_edges:
                continue
            u, v = e
            if not has_q3_fast(best_adj, u, v):
                best_adj[u] |= (1 << v)
                best_adj[v] |= (1 << u)
                best_edges.add(e)
                best_count += 1
    
    return [[u, v] for u, v in best_edges]
# EVOLVE-BLOCK-END
