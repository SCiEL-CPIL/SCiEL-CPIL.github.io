import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    num_all_edges = len(all_edges)
    
    def creates_q3(adj, u, v):
        """
        Check if adding edge (u, v) to graph 'adj' creates a Q3.
        'adj' is a list of int bitmasks. adj[i] has bit j set if (i,j) is an edge.
        Assumes (u, v) is NOT in adj.
        """
        # Neighbors of u excluding v
        nu = []
        mask_u = adj[u]
        for x in range(n):
            if x != v and (mask_u >> x) & 1:
                nu.append(x)
        
        nv = []
        mask_v = adj[v]
        for x in range(n):
            if x != u and (mask_v >> x) & 1:
                nv.append(x)
                
        if len(nu) < 2 or len(nv) < 2:
            return False

        # We need to choose b1, b2 from nu and a1, a2 from nv.
        for b1 in range(len(nu)):
            u_b1 = nu[b1]
            for b2 in range(b1+1, len(nu)):
                u_b2 = nu[b2]
                
                for a1 in range(len(nv)):
                    v_a1 = nv[a1]
                    for a2 in range(a1+1, len(nv)):
                        v_a2 = nv[a2]
                        
                        # S = {u, v, u_b1, u_b2, v_a1, v_a2}
                        # Find a3 not in S, not connected to v.
                        # Find b3 not in S U {a3}, not connected to u.
                        
                        # Optimization: Pre-check a1, a2 connections to u_b1, u_b2?
                        # a1 must be connected to v.
                        # a1's match is in {u_b1, u_b2, b3}.
                        # If a1 is connected to BOTH u_b1 and u_b2, then its match MUST be b3.
                        # If a1 is connected to NEITHER, impossible (match is one of them, so it must be connected to the other).
                        # So a1 must be connected to exactly one of {u_b1, u_b2} OR both (if match is b3).
                        # Wait, if match is b3, it is connected to u_b1 and u_b2.
                        # If match is u_b1, it is NOT connected to u_b1, but IS connected to u_b2.
                        
                        # Let's check connections of a1, a2 to u_b1, u_b2.
                        c_a1_b1 = (adj[v_a1] >> u_b1) & 1
                        c_a1_b2 = (adj[v_a1] >> u_b2) & 1
                        c_a2_b1 = (adj[v_a2] >> u_b1) & 1
                        c_a2_b2 = (adj[v_a2] >> u_b2) & 1
                        
                        # Possibilities for matches of a1, a2 within {u_b1, u_b2}:
                        # 1. a1->u_b1 (so c_a1_b1=0), a2->u_b2 (so c_a2_b2=0).
                        #    Then c_a1_b2 must be 1, c_a2_b1 must be 1.
                        #    b3 is match of u.
                        # 2. a1->u_b2 (so c_a1_b2=0), a2->u_b1 (so c_a2_b1=0).
                        #    Then c_a1_b1 must be 1, c_a2_b2 must be 1.
                        # 3. a1->b3, a2->b3? No, matching.
                        #    If a1->b3, c_a1_b1=1, c_a1_b2=1.
                        #    If a2->b3, impossible (b3 already used).
                        #    So one of a1, a2 matches b3, the other matches u_b1 or u_b2.
                        
                        # Let's just iterate a3, b3 as before but with pruning.
                        
                        # Candidates for a3: not in S, not connected to v.
                        # We can precompute list of non-neighbors of v excluding S?
                        # S changes with a1, a2.
                        
                        # Fast path: Check if there is ANY a3, b3.
                        # If not, skip.
                        
                        # To speed up, we can precompute non_neighbors of v and u.
                        # But S excludes 6 vertices.
                        
                        # Let's try the direct loop but break early if possible.
                        
                        # Check Case 1: a1->u_b1, a2->u_b2
                        if not c_a1_b1 and c_a1_b2 and not c_a2_b2 and c_a2_b1:
                            # Matches: a1-u_b1, a2-u_b2.
                            # u's match is b3.
                            # a3's match is v.
                            # Need a3 not in S, not connected to v.
                            # Need b3 not in S U {a3}, not connected to u.
                            # Also a3 must be connected to u_b1, u_b2.
                            # b3 must be connected to v_a1, v_a2.
                            
                            # Find a3
                            found_a3 = False
                            for a3 in range(n):
                                if a3 in (u, v, u_b1, u_b2, v_a1, v_a2):
                                    continue
                                if (adj[v] >> a3) & 1: # connected to v?
                                    continue
                                # Check a3 connected to u_b1, u_b2
                                if not ((adj[a3] >> u_b1) & 1) or not ((adj[a3] >> u_b2) & 1):
                                    continue
                                
                                # Found candidate a3. Now find b3.
                                # b3 not in S U {a3}, not connected to u.
                                # b3 connected to v_a1, v_a2.
                                for b3 in range(n):
                                    if b3 in (u, v, u_b1, u_b2, v_a1, v_a2, a3):
                                        continue
                                    if (adj[u] >> b3) & 1: # connected to u?
                                        continue
                                    if not ((adj[b3] >> v_a1) & 1) or not ((adj[b3] >> v_a2) & 1):
                                        continue
                                    
                                    # Check all conditions?
                                    # We verified:
                                    # a1-u_b1 (no edge), a1-u_b2 (edge)
                                    # a2-u_b2 (no edge), a2-u_b1 (edge)
                                    # a3-v (no edge)
                                    # u-b3 (no edge)
                                    # a3-u_b1 (edge), a3-u_b2 (edge)
                                    # b3-v_a1 (edge), b3-v_a2 (edge)
                                    # u-v (edge, added)
                                    # u-u_b1, u-u_b2 (edges, by def)
                                    # v-v_a1, v-v_a2 (edges, by def)
                                    
                                    # Are there any other edges?
                                    # a1-v (edge), a2-v (edge)
                                    # a3-v (no edge)
                                    # u-b3 (no edge)
                                    
                                    # What about a1-b3?
                                    # a1's match is u_b1. So a1 is connected to b3.
                                    # We did NOT check a1-b3.
                                    # What about a2-b3?
                                    # a2's match is u_b2. So a2 is connected to b3.
                                    # We did NOT check a2-b3.
                                    
                                    # What about u-a3? No, same partition.
                                    # What about v-b3? No, same partition.
                                    
                                    # So we need to check:
                                    # (adj[v_a1] >> b3) & 1
                                    # (adj[v_a2] >> b3) & 1
                                    
                                    if not ((adj[v_a1] >> b3) & 1) or not ((adj[v_a2] >> b3) & 1):
                                        continue
                                        
                                    return True
            # Case 2: a1->u_b2, a2->u_b1
            if not c_a1_b2 and c_a1_b1 and not c_a2_b1 and c_a2_b2:
                if not c_a1_b2 and c_a1_b1 and not c_a2_b1 and c_a2_b2:
                    for a3 in range(n):
                        if a3 in (u, v, u_b1, u_b2, v_a1, v_a2):
                            continue
                        if (adj[v] >> a3) & 1:
                            continue
                        if not ((adj[a3] >> u_b1) & 1) or not ((adj[a3] >> u_b2) & 1):
                            continue
                        
                        for b3 in range(n):
                            if b3 in (u, v, u_b1, u_b2, v_a1, v_a2, a3):
                                continue
                            if (adj[u] >> b3) & 1:
                                continue
                            if not ((adj[b3] >> v_a1) & 1) or not ((adj[b3] >> v_a2) & 1):
                                continue
                            if not ((adj[v_a1] >> b3) & 1) or not ((adj[v_a2] >> b3) & 1):
                                continue
                            return True
                            
        return False

    best_edges = []
    best_count = 0
    
    for attempt in range(500):
        if time.time() > deadline - 5:
            break
            
        random.shuffle(all_edges)
        adj = [0] * n
        cur_edges = []
        
        for u, v in all_edges:
            if time.time() > deadline - 2:
                break
            if not creates_q3(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                cur_edges.append((u, v))
                
        if len(cur_edges) > best_count:
            best_count = len(cur_edges)
            best_edges = cur_edges[:]
            
    return [[u, v] for u, v in best_edges]
