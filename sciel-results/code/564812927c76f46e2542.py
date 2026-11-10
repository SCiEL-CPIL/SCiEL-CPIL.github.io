import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    best_edges = []
    best_count = 0
    
    # Precompute common neighbors for all pairs
    # common[u][v] is a bitmask of vertices adjacent to both u and v
    common = [[0]*n for _ in range(n)]
    for u in range(n):
        for v in range(u+1, n):
            # We can compute this on the fly or precompute for the empty graph?
            # No, common neighbors depend on the current graph.
            # So we cannot precompute common neighbors for a fixed graph.
            # We must compute intersections on the fly.
            # However, we can precompute the list of edges or just use bitmasks.
            pass
            
    def creates_q3(u, v, adj):
        # adj is list of 16 ints
        mask_u = adj[u] & ~(1 << v)
        mask_v = adj[v] & ~(1 << u)
        
        # Get lists for a, b
        # To speed up, we can just iterate bits
        
        # Iterate a in mask_u
        mask_a = mask_u
        while mask_a:
            lsb_a = mask_a & -mask_a
            a = lsb_a.bit_length() - 1
            mask_a ^= lsb_a
            
            # Iterate b in mask_u excluding a
            mask_b = mask_u & ~lsb_a
            while mask_b:
                lsb_b = mask_b & -mask_b
                b = lsb_b.bit_length() - 1
                mask_b ^= lsb_b
                
                # c must be in N(v) \ {u} AND N(a)
                # d must be in N(v) \ {u} AND N(b)
                
                mask_c = mask_v & adj[a]
                mask_d = mask_v & adj[b]
                
                # Iterate c in mask_c
                mask_c_iter = mask_c
                while mask_c_iter:
                    lsb_c = mask_c_iter & -mask_c_iter
                    c = lsb_c.bit_length() - 1
                    mask_c_iter ^= lsb_c
                    
                    # Check distinctness of a, b, c
                    if c == a or c == b:
                        continue
                        
                    # d must be in mask_d, excluding c
                    mask_d_iter = mask_d & ~lsb_c
                    while mask_d_iter:
                        lsb_d = mask_d_iter & -mask_d_iter
                        d = lsb_d.bit_length() - 1
                        mask_d_iter ^= lsb_d
                        
                        # Check distinctness of a, b, c, d
                        if d == a or d == b or d == c:
                            continue
                        
                        # Now we have u, v, a, b, c, d distinct.
                        # Check if edges (a,c) and (b,d) exist?
                        # c was chosen from mask_c = mask_v & adj[a], so (a,c) exists.
                        # d was chosen from mask_d = mask_v & adj[b], so (b,d) exists.
                        # Wait, the pairing in Q3 is specific.
                        # In Q3, a is connected to c. b is connected to d.
                        # Is it possible that a is connected to d and b to c?
                        # In the cube 000(u)-001(v), a=010, b=100.
                        # c=011, d=101.
                        # a(010) connects to c(011). b(100) connects to d(101).
                        # a(010) does NOT connect to d(101).
                        # b(100) does NOT connect to c(011).
                        # So the pairing is fixed by the geometry.
                        # However, we don't know which neighbor is "a" and which is "b" in the abstract.
                        # We just pick two neighbors a, b of u.
                        # And two neighbors c, d of v.
                        # The condition for forming a Q3 with edge (u,v) is:
                        # There exist a,b in N(u)\{v} and c,d in N(v)\{u} such that:
                        # 1. (a,c) is an edge
                        # 2. (b,d) is an edge
                        # 3. (a,e) and (b,e) for some e
                        # 4. (c,f) and (d,f) for some f
                        # 5. (e,f) is an edge
                        # AND all 8 vertices distinct.
                        
                        # My current loop ensures (a,c) and (b,d) are edges.
                        # This covers one of the two possible matchings.
                        # Do we need to check (a,d) and (b,c)?
                        # If (a,d) and (b,c) are edges, that would correspond to a different labeling?
                        # If we swap c and d, then (a,d) and (b,c) becomes (a,c') and (b,d') where c'=d, d'=c.
                        # Since we iterate all c, d, we will find the valid pairing if it exists.
                        # Specifically, if the valid pairing is a-d and b-c, then when we pick c=d_valid and d=c_valid, we check (a,c_valid) which is (a,d_valid) - wait.
                        # If valid edges are (a,d) and (b,c).
                        # My loop checks if (a,c) and (b,d) are edges.
                        # So if the graph has (a,d) and (b,c), my loop with c=d, d=c will check:
                        # Is (a,d) an edge? (Yes, because c=d).
                        # Is (b,c) an edge? (Yes, because d=c).
                        # So yes, iterating all pairs covers both matchings.
                        
                        # Find e in N(a) \cap N(b) \ {u,v,a,b,c,d}
                        mask_e = (adj[a] & adj[b]) & ~((1 << u) | (1 << v) | (1 << a) | (1 << b) | (1 << c) | (1 << d))
                        if mask_e == 0:
                            continue
                        
                        # Find f in N(c) \cap N(d) \ {u,v,a,b,c,d}
                        mask_f = (adj[c] & adj[d]) & ~((1 << u) | (1 << v) | (1 << a) | (1 << b) | (1 << c) | (1 << d))
                        if mask_f == 0:
                            continue
                        
                        # Check if e and f are distinct
                        # mask_e and mask_f are disjoint?
                        # e is not c,d. f is not a,b.
                        # e could be f?
                        # If e == f, then e is in N(a) \cap N(b) \cap N(c) \cap N(d).
                        # In Q3, e=110, f=111. They are distinct.
                        # So we must ensure e != f.
                        # Also (e,f) must be an edge.
                        
                        # Iterate e in mask_e
                        mask_e_iter = mask_e
                        found = False
                        while mask_e_iter:
                            lsb_e = mask_e_iter & -mask_e_iter
                            e = lsb_e.bit_length() - 1
                            mask_e_iter ^= lsb_e
                            
                            # f must be in mask_f, excluding e
                            # And (e,f) must be an edge.
                            # So f in mask_f & adj[e] & ~(1<<e)
                            mask_f_valid = mask_f & adj[e] & ~lsb_e
                            if mask_f_valid:
                                return True
                                
        return False

    # Main search loop
    while time.time() < deadline:
        random.shuffle(all_edges)
        adj = [0] * n
        current_edges = []
        
        # Greedy Phase
        for u, v in all_edges:
            if not creates_q3(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges.append((u, v))
                
        # Local Search Phase (Kicks)
        # Try to improve by removing an edge and re-adding
        for _ in range(5):
            if time.time() > deadline: break
            if not current_edges: break
            
            idx = random.randint(0, len(current_edges) - 1)
            u_rem, v_rem = current_edges[idx]
            
            adj[u_rem] &= ~(1 << v_rem)
            adj[v_rem] &= ~(1 << u_rem)
            current_edges.pop(idx)
            
            # Try to add any missing edge
            random.shuffle(all_edges)
            added_any = False
            for u, v in all_edges:
                if (adj[u] >> v) & 1:
                    continue
                if not creates_q3(u, v, adj):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    current_edges.append((u, v))
                    added_any = True
                    # Don't break, try to add more
            
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges.copy()
            
    return [[u, v] for u, v in best_edges]
