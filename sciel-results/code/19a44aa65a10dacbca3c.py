"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start = time.time()
    deadline = start + 145
    n = 16

    # Q3 structure constants
    # Q3 vertices: 0..7
    # Edges: (u,v) where u^v is power of 2
    # Bipartition A={0,3,5,6}, B={1,2,4,7}
    
    # Precompute neighbor masks for Q3 vertices
    # q3_mask[v] is bitmask of Q3 neighbors of v
    q3_mask = [0] * 8
    for u in range(8):
        for v in range(8):
            if u != v and (u ^ v) & ((u ^ v) - 1) == 0:
                q3_mask[u] |= (1 << v)

    # All possible edges in K16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    m = len(all_edges)

    # Fast Q3 check for adding edge (u, v) to graph defined by adj_bitmasks
    # adj_bitmasks[i] is an integer bitmask of neighbors of i
    def has_q3_fast(adj, u, v):
        # We check if adding (u,v) creates a Q3.
        # The new edge must be part of the Q3.
        # In Q3, every edge is equivalent. WLOG map u->0, v->1.
        # Also try u->1, v->0? No, Q3 is vertex transitive? 
        # Q3 is vertex transitive. Edge transitive? Yes.
        # So mapping u->0, v->1 covers all cases where (u,v) is an edge in Q3.
        # Wait, we need to ensure we cover all embeddings.
        # If (u,v) is an edge in the Q3 subgraph, it corresponds to some edge (a,b) in Q3.
        # Due to edge transitivity, we can assume WLOG that (a,b) = (0,1).
        # So we only need to check the mapping u->0, v->1.
        
        # However, the automorphism group of Q3 acts transitively on edges.
        # So yes, one check is sufficient for the "new edge is in Q3" scenario.
        
        # But wait, the Q3 might exist *without* the new edge?
        # No, we are checking if *adding* the edge creates a Q3.
        # If a Q3 existed before, the graph was invalid. We assume the current graph is Q3-free.
        # So any new Q3 must use the new edge (u,v).
        
        # Mapping: 0 -> u, 1 -> v
        # Q3 neighbors of 0: {1, 2, 4}
        # Q3 neighbors of 1: {0, 3, 5}
        
        # Candidates for Q3 vertex 2: must be neighbor of u in K16.
        # Candidates for Q3 vertex 4: must be neighbor of u in K16.
        # Candidates for Q3 vertex 3: must be neighbor of v in K16.
        # Candidates for Q3 vertex 5: must be neighbor of v in K16.
        
        mask_u = adj[u]
        mask_v = adj[v]
        
        # Exclude u and v themselves
        mask_u &= ~(1 << u)
        mask_u &= ~(1 << v)
        mask_v &= ~(1 << u)
        mask_v &= ~(1 << v)
        
        # Get candidate lists
        # To speed up, we can iterate over set bits
        cands_u = []
        m = mask_u
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            cands_u.append(idx)
            m ^= lsb
            
        cands_v = []
        m = mask_v
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            cands_v.append(idx)
            m ^= lsb
            
        # We need to pick c2, c4 from cands_u and c3, c5 from cands_v
        # such that:
        # 1. c2, c4, c3, c5 are distinct.
        # 2. Edge (c2, c3) exists.
        # 3. Edge (c4, c5) exists.
        # 4. There exists c6 in N(c2) \cap N(c4) distinct from {u,v,c2,c4,c3,c5}.
        # 5. There exists c7 in N(c3) \cap N(c5) distinct from {u,v,c2,c4,c3,c5}.
        # 6. Edge (c6, c7) exists.
        
        # Optimization: 
        # For each pair (c2, c4) from cands_u (distinct):
        #   For each pair (c3, c5) from cands_v (distinct):
        #     Check distinctness.
        #     Check edges (c2,c3) and (c4,c5).
        #     Compute common neighbors for c6 and c7.
        
        # To reduce loops, we can precompute intersections?
        # No, adj changes.
        
        # Let's iterate.
        # cands_u and cands_v are typically small (degree < 16).
        
        # Further pruning: 
        # c2 and c4 must have a common neighbor c6.
        # c3 and c5 must have a common neighbor c7.
        
        for i in range(len(cands_u)):
            c2 = cands_u[i]
            for j in range(i+1, len(cands_u)):
                c4 = cands_u[j]
                
                # Pre-check: do c2 and c4 have a common neighbor?
                # If not, skip.
                common_24 = adj[c2] & adj[c4]
                # Exclude u, v, c2, c4
                common_24 &= ~(1 << u)
                common_24 &= ~(1 << v)
                common_24 &= ~(1 << c2)
                common_24 &= ~(1 << c4)
                
                if common_24 == 0:
                    continue
                
                for k in range(len(cands_v)):
                    c3 = cands_v[k]
                    if c3 == c2 or c3 == c4:
                        continue
                    # Check edge (c2, c3)
                    if not (adj[c2] & (1 << c3)):
                        continue
                    
                    for l in range(k+1, len(cands_v)):
                        c5 = cands_v[l]
                        if c5 == c2 or c5 == c4 or c5 == c3:
                            continue
                        # Check edge (c4, c5)
                        if not (adj[c4] & (1 << c5)):
                            continue
                        
                        # Check distinctness of all 6 vertices
                        # c2, c4, c3, c5 are distinct by loop structure and checks above?
                        # c2 != c4 (i!=j)
                        # c3 != c5 (k!=l)
                        # c3 != c2, c3 != c4 checked.
                        # c5 != c2, c5 != c4, c5 != c3 checked.
                        
                        # Now check c6 and c7
                        # c6 must be in common_24, and not c3, c5
                        # c7 must be in common_35, and not c2, c4
                        
                        # Compute common_35
                        common_35 = adj[c3] & adj[c5]
                        common_35 &= ~(1 << u)
                        common_35 &= ~(1 << v)
                        common_35 &= ~(1 << c3)
                        common_35 &= ~(1 << c5)
                        
                        if common_35 == 0:
                            continue
                        
                        # Filter common_24 to exclude c3, c5
                        c6_candidates = common_24
                        c6_candidates &= ~(1 << c3)
                        c6_candidates &= ~(1 << c5)
                        
                        if c6_candidates == 0:
                            continue
                        
                        # Filter common_35 to exclude c2, c4
                        c7_candidates = common_35
                        c7_candidates &= ~(1 << c2)
                        c7_candidates &= ~(1 << c4)
                        
                        if c7_candidates == 0:
                            continue
                        
                        # We need to find c6 in c6_candidates and c7 in c7_candidates
                        # such that c6 != c7 and edge (c6, c7) exists.
                        
                        # Iterate over smaller set?
                        # Usually small.
                        # Let's just check if intersection of "neighbors of c6_candidates" and "c7_candidates" is non-empty.
                        # i.e., is there c6 in c6_candidates such that (adj[c6] & c7_candidates) != 0?
                        
                        found = False
                        m6 = c6_candidates
                        while m6:
                            lsb = m6 & -m6
                            c6 = lsb.bit_length() - 1
                            # Check if any neighbor of c6 is in c7_candidates
                            if adj[c6] & c7_candidates:
                                found = True
                                break
                            m6 ^= lsb
                        
                        if found:
                            return True
        
        return False

    # Build initial graph: K_{3,13}
    def build_initial():
        adj = [0] * n
        edges = set()
        for a in range(3):
            for b in range(3, 16):
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
                edges.add((a, b))
        return adj, edges

    best_adj, best_edges = build_initial()
    best_count = len(best_edges)
    
    random.seed(42)
    
    # Greedy Add
    for trial in range(100):
        if time.time() > deadline - 30:
            break
            
        adj, edges = build_initial()
        
        missing = [e for e in all_edges if e not in edges]
        random.shuffle(missing)
        
        for e in missing:
            if time.time() > deadline - 25:
                break
            u, v = e
            if has_q3_fast(adj, u, v):
                continue
            # Add edge
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.add(e)
            
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = set(edges)
            best_adj = adj[:]

    # Local Search: Remove one, try to add one (or more)
    if time.time() < deadline - 20:
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            adj = best_adj[:]
            edges = set(best_edges)
            
            # Get list of edges to try removing
            # Prioritize edges with low "degree of participation" in potential Q3s?
            # Just random for now.
            edge_list = list(edges)
            random.shuffle(edge_list)
            
            for e in edge_list:
                if time.time() > deadline - 10:
                    break
                
                u, v = e
                # Remove
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                edges.remove(e)
                
                # Try to add a missing edge
                missing = [ne for ne in all_edges if ne not in edges]
                random.shuffle(missing)
                
                added = False
                for ne in missing:
                    if time.time() > deadline - 10:
                        break
                    nu, nv = ne
                    if has_q3_fast(adj, nu, nv):
                        continue
                    # Add
                    adj[nu] |= (1 << nv)
                    adj[nv] |= (1 << nu)
                    edges.add(ne)
                    added = True
                    # Try to add one more?
                    # If we added one, we might be able to add another.
                    # Let's do a quick greedy pass on the remaining missing
                    # But keep it fast.
                    
                    if len(edges) > best_count:
                        best_count = len(edges)
                        best_edges = set(edges)
                        best_adj = adj[:]
                        improved = True
                    break # Only add one per removal step to keep it simple/fast
    
    return [[u, v] for u, v in best_edges]
# EVOLVE-BLOCK-END
