"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start = time.time()
    deadline = start + 140
    n = 16

    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]

    # Optimized Q3 check using bitmasks
    def has_q3_fast(adj, u, v):
        """
        Check if adding edge (u, v) to graph adj creates a Q3 subgraph.
        adj is a list of bitmasks.
        """
        # We need to find 6 other vertices x2, x4, x3, x5, x6, x7 such that
        # they form Q3 with u, v.
        # Q3 structure:
        # u (0) is connected to v (1), x2 (2), x4 (4)
        # v (1) is connected to u (0), x3 (3), x5 (5)
        # x2 (2) is connected to u (0), x3 (3), x6 (6)
        # x4 (4) is connected to u (0), x5 (5), x6 (6)
        # x3 (3) is connected to v (1), x2 (2), x7 (7)
        # x5 (5) is connected to v (1), x4 (4), x7 (7)
        # x6 (6) is connected to x2 (2), x4 (4), x7 (7)
        # x7 (7) is connected to x3 (3), x5 (5), x6 (6)
        
        # Try both orientations: u->0, v->1 and u->1, v->0
        # Since the graph is undirected, we can just fix u as 0 and v as 1,
        # but we must ensure we don't miss cases where u plays role 1 and v plays role 0?
        # Actually, Q3 is vertex-transitive? No, but edge-transitive.
        # Any edge in Q3 can be mapped to (0,1).
        # So checking (u,v) as (0,1) is sufficient for the existence of *an* embedding
        # containing edge (u,v).
        
        mask_u = adj[u]
        mask_v = adj[v]
        
        # Candidates for x2, x4 must be neighbors of u (not v, not u)
        # Candidates for x3, x5 must be neighbors of v (not u, not v)
        
        # Bitmask of all vertices except u and v
        others_mask = ((1 << n) - 1) & ~(1 << u) & ~(1 << v)
        
        # x2, x4 in N(u) \ {v}
        cands_24_mask = mask_u & others_mask
        # x3, x5 in N(v) \ {u}
        cands_35_mask = mask_v & others_mask
        
        # Iterate over pairs (x2, x4) from cands_24_mask
        # We need x2 != x4.
        m24 = cands_24_mask
        while m24:
            lsb2 = m24 & -m24
            x2 = lsb2.bit_length() - 1
            m24 ^= lsb2
            
            # x4 must be in cands_24_mask, not x2
            m4 = cands_24_mask & ~(1 << x2)
            while m4:
                lsb4 = m4 & -m4
                x4 = lsb4.bit_length() - 1
                m4 ^= lsb4
                
                # Now we have x2, x4.
                # x3 must be neighbor of x2 and x3.
                # x5 must be neighbor of x4 and x5.
                # x3, x5 in cands_35_mask.
                # Also x3 != x5.
                
                # Constraints on x3:
                # 1. x3 in N(v) (cands_35_mask)
                # 2. x3 in N(x2) (adj[x2])
                # 3. x3 not in {u, v, x2, x4}
                
                cands_3_mask = cands_35_mask & adj[x2] & ~(1 << u) & ~(1 << v) & ~(1 << x2) & ~(1 << x4)
                
                # Constraints on x5:
                # 1. x5 in N(v) (cands_35_mask)
                # 2. x5 in N(x4) (adj[x4])
                # 3. x5 not in {u, v, x2, x4}
                
                cands_5_mask = cands_35_mask & adj[x4] & ~(1 << u) & ~(1 << v) & ~(1 << x2) & ~(1 << x4)
                
                m3 = cands_3_mask
                while m3:
                    lsb3 = m3 & -m3
                    x3 = lsb3.bit_length() - 1
                    m3 ^= lsb3
                    
                    # x5 must be in cands_5_mask and not x3
                    m5 = cands_5_mask & ~(1 << x3)
                    while m5:
                        lsb5 = m5 & -m5
                        x5 = lsb5.bit_length() - 1
                        m5 ^= lsb5
                        
                        # Now we have x2, x4, x3, x5.
                        # Check remaining edges:
                        # (x2, x3) - ensured by cands_3_mask
                        # (x4, x5) - ensured by cands_5_mask
                        # (x2, x4) - NO, x2 and x4 are NOT adjacent in Q3.
                        # Wait, in Q3, 2^4 = 6, not a power of 2. So (2,4) is NOT an edge.
                        # So we must ensure (x2, x4) is NOT an edge in our graph?
                        # No, we are looking for a subgraph isomorphic to Q3.
                        # The edges of Q3 are a subset of the edges of our graph.
                        # Non-edges of Q3 can be edges in our graph.
                        # So we don't care if (x2, x4) is an edge in G.
                        
                        # We need to find x6, x7.
                        # x6 must be neighbor of x2 and x4.
                        # x7 must be neighbor of x3 and x5.
                        # Also x6 connected to x7.
                        # x6, x7 distinct from {u, v, x2, x3, x4, x5}.
                        
                        used_mask = (1 << u) | (1 << v) | (1 << x2) | (1 << x3) | (1 << x4) | (1 << x5)
                        
                        # Candidates for x6: N(x2) & N(x4) & ~used_mask
                        cands_6_mask = adj[x2] & adj[x4] & ~used_mask
                        # Candidates for x7: N(x3) & N(x5) & ~used_mask
                        cands_7_mask = adj[x3] & adj[x5] & ~used_mask
                        
                        m6 = cands_6_mask
                        while m6:
                            lsb6 = m6 & -m6
                            x6 = lsb6.bit_length() - 1
                            m6 ^= lsb6
                            
                            # x7 must be in cands_7_mask, not x6
                            m7 = cands_7_mask & ~(1 << x6)
                            while m7:
                                lsb7 = m7 & -m7
                                x7 = lsb7.bit_length() - 1
                                m7 ^= lsb7
                                
                                # Check final edge (x6, x7)
                                if adj[x6] & (1 << x7):
                                    return True
        return False

    def run_greedy(start_adj, start_edges, seed):
        adj = start_adj[:]
        edges = set(start_edges)
        random.seed(seed)
        
        missing = [e for e in all_edges if e not in edges]
        random.shuffle(missing)
        
        for e in missing:
            if time.time() > deadline - 10:
                break
            u, v = e
            if has_q3_fast(adj, u, v):
                continue
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.add(e)
        return adj, edges

    def local_search(adj, edges, seed):
        random.seed(seed)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            adj_copy = adj[:]
            edges_copy = set(edges)
            
            # Try to swap one edge for another
            # We remove one edge and try to add a different one.
            # To be efficient, we can just try adding random missing edges
            # and if it fails, we don't change anything.
            # But we want to increase density.
            # Strategy: Pick a random existing edge to remove? No, that decreases count.
            # Strategy: Pick a random missing edge to add. If it fails, try to remove
            # an edge that is "blocking" it? That's complex.
            
            # Simpler local search:
            # 1. Try to add random missing edges.
            # 2. If stuck, try removing a random edge and see if we can add 2 edges?
            #    Or just remove 1 and add 1?
            
            # Let's do 1-swap: Remove edge e1, add edge e2.
            # We iterate through edges and try to swap them with missing edges.
            
            edge_list = list(edges_copy)
            random.shuffle(edge_list)
            
            for e1 in edge_list:
                if time.time() > deadline - 5:
                    break
                u1, v1 = e1
                # Remove e1
                adj_copy[u1] &= ~(1 << v1)
                adj_copy[v1] &= ~(1 << u1)
                edges_copy.remove(e1)
                
                # Try to add a random missing edge
                missing = [e for e in all_edges if e not in edges_copy]
                random.shuffle(missing)
                
                added = False
                for e2 in missing:
                    u2, v2 = e2
                    if has_q3_fast(adj_copy, u2, v2):
                        continue
                    
                    # Add e2
                    adj_copy[u2] |= (1 << v2)
                    adj_copy[v2] |= (1 << u2)
                    edges_copy.add(e2)
                    added = True
                    break
                
                if not added:
                    # Restore e1
                    adj_copy[u1] |= (1 << v1)
                    adj_copy[v1] |= (1 << u1)
                    edges_copy.add(e1)
                else:
                    # We swapped e1 for e2. Count is same.
                    # This doesn't improve count.
                    # We need a strategy that increases count.
                    # Greedy already maximizes count for a given order.
                    # Local search should try to find a better order or escape local optima.
                    
                    # Actually, if we remove e1 and add e2, count is constant.
                    # To increase count, we need to add an edge without removing one.
                    # But greedy did that.
                    # Maybe the greedy order was bad.
                    # Let's try "restart" local search:
                    # Remove a random edge, then run greedy again on the remaining missing edges?
                    
                    # Let's just keep the best graph found by greedy restarts.
                    # The local search above doesn't improve count.
                    # Let's implement a "kick" + greedy.
                    # Kick: Remove k random edges.
                    # Greedy: Add edges back greedily.
                    
                    # If we remove e1 and run greedy, we might add e1 back and more.
                    # So:
                    # 1. Remove random edge e1.
                    # 2. Run greedy on the rest.
                    # 3. If result is better, keep it.
                    
                    # This is expensive but we have time.
                    
                    # Let's restructure:
                    # The loop above is for "swaps" which don't help count.
                    # I will replace this with a "kick" strategy in the main loop.
                    
                    # For now, just restore and break if no improvement is possible via simple swap?
                    # Actually, let's just return the current best from greedy.
                    # The "kick" will be handled in the main loop.
                    
                    # Wait, if I restore e1, I wasted time.
                    # Let's just break and let the main loop handle kicks.
                    break 

        return adj, edges

    best_adj = None
    best_edges = None
    best_count = 0
    
    # Initial graph: Empty or small seed?
    # Empty graph is Q3-free.
    # Greedy from empty might be slow to fill.
    # Let's start with a known Q3-free graph?
    # K_{3,13} is Q3-free?
    # Q3 has min degree 3. K_{3,13} has part of size 3.
    # If Q3 is in K_{3,13}, it must use vertices from both parts.
    # Bipartite graph. Q3 is bipartite.
    # Q3 has 4 vertices in each part.
    # K_{3,13} has 3 vertices in one part.
    # So Q3 cannot be embedded in K_{3,13} because it needs 4 vertices in one part.
    # So K_{3,13} is Q3-free.
    
    # Start with K_{3,13}
    init_adj = [0] * n
    init_edges = set()
    for a in range(3):
        for b in range(3, 16):
            init_adj[a] |= (1 << b)
            init_adj[b] |= (1 << a)
            init_edges.add((a, b))
            
    best_adj = init_adj[:]
    best_edges = set(init_edges)
    best_count = len(init_edges)
    
    # Multiple greedy restarts with randomization
    for i in range(100):
        if time.time() > deadline - 20:
            break
        
        # Kick: Remove some random edges from current best?
        # Or just start from init and shuffle missing edges?
        # Starting from init is good.
        
        adj, edges = run_greedy(best_adj, best_edges, i)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_adj = adj
            best_edges = edges
            
            # Try local search (kick + greedy) on the new best
            if time.time() < deadline - 10:
                for _ in range(10):
                    if time.time() > deadline - 10:
                        break
                    # Kick: remove 3 random edges
                    kick_adj = best_adj[:]
                    kick_edges = set(best_edges)
                    kick_edges_list = list(kick_edges)
                    random.shuffle(kick_edges_list)
                    for e in kick_edges_list[:3]:
                        u, v = e
                        kick_adj[u] &= ~(1 << v)
                        kick_adj[v] &= ~(1 << u)
                        kick_edges.remove(e)
                        
                    # Greedy from kicked state
                    k_adj, k_edges = run_greedy(kick_adj, kick_edges, i * 100 + 1)
                    
                    if len(k_edges) > best_count:
                        best_count = len(k_edges)
                        best_adj = k_adj
                        best_edges = k_edges
    
    return [[u, v] for u, v in best_edges]
# EVOLVE-BLOCK-END
