import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    n = 16
    vertices = range(n)
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = len(all_edges)
    
    # Precompute all Q3 embeddings in K16
    # A Q3 is a 3-regular bipartite graph on 8 vertices.
    # We enumerate all 8-subsets and all bipartitions (4+4) and check for a Q3.
    # Actually, it's faster to just enumerate all isomorphic copies of Q3 in K16.
    # Number of Q3s in K16 is C(16,8) * (number of Q3s in K8).
    # Number of Q3s in K8 is 8! / 48 = 840.
    # Total ~ 10.8 million. Too many to store explicitly.
    
    # Instead, we use a local check: 
    # When adding an edge (u,v), check if it completes a Q3.
    # A Q3 containing (u,v) must have u and v at distance 1.
    # In Q3, neighbors of u are 3 vertices. Let them be a, b, c.
    # Neighbors of v are 3 vertices. Let them be x, y, z.
    # Since (u,v) is an edge, in Q3, u and v are in different parts of the bipartition?
    # No, Q3 is bipartite. If u is in Part A, v is in Part B.
    # Neighbors of u (a,b,c) are in Part B.
    # Neighbors of v (x,y,z) are in Part A.
    # So {a,b,c} and {x,y,z} are disjoint sets of 3 vertices.
    # Also, in Q3, each vertex in Part A is connected to exactly 3 vertices in Part B.
    # u is connected to a,b,c.
    # x,y,z are connected to v.
    # Are a,b,c connected to x,y,z?
    # Let's trace the structure.
    # Vertices: 0..7. 000..111.
    # 000 neighbors: 001, 010, 100.
    # 001 neighbors: 000, 011, 101.
    # If edge is (000, 001).
    # N(000) = {001, 010, 100}.
    # N(001) = {000, 011, 101}.
    # The Q3 vertices are {000, 001, 010, 100, 011, 101, 110, 111}.
    # The "other" 4 vertices are 011, 101, 110, 111.
    # 010 is connected to 000, 011, 110.
    # 100 is connected to 000, 101, 110.
    # 011 is connected to 001, 010, 111.
    # 101 is connected to 001, 100, 111.
    # 110 is connected to 010, 100, 111.
    # 111 is connected to 011, 101, 110.
    
    # So, given edge (u,v), to check if it's in a Q3, we need to find 6 other vertices.
    # Let N(u) = {a,b,c} and N(v) = {x,y,z}.
    # The Q3 vertices are {u, v, a, b, c, x, y, z}? Not necessarily.
    # In the example, N(000)={001,010,100}, N(001)={000,011,101}.
    # The Q3 vertices are {000,001,010,100,011,101,110,111}.
    # Note that 110 and 111 are NOT in N(000) or N(001).
    # So the 8 vertices are not just u,v and their neighbors.
    
    # However, we can check:
    # Does there exist a set of 4 vertices S such that {u,v} U S forms a Q3?
    # This is still complex.
    
    # Let's use the "remove edge" strategy from the parent, which worked well (1.38 -> 1.46).
    # The parent used a local Q3 check.
    # I will implement a very fast Q3 check using bitmasks.
    
    # Bitmask representation for adjacency.
    # adj[v] is a 16-bit integer.
    
    def has_q3(adj, u, v):
        # Check if adding edge (u,v) creates a Q3.
        # We assume (u,v) is NOT yet in adj.
        # We need to check if there is a Q3 containing (u,v).
        
        # In Q3, u and v are adjacent.
        # Let's iterate over possible "neighbors" of u in the Q3 (excluding v).
        # Let a, b be two other neighbors of u in the Q3.
        # Let x, y be two other neighbors of v in the Q3.
        
        # Actually, a simpler check:
        # A Q3 is a 3-regular bipartite graph.
        # If we add (u,v), do we create a 3-regular bipartite subgraph on some 8 vertices?
        
        # Let's try to find the 8 vertices.
        # u and v are in the Q3.
        # Let the Q3 vertices be Q.
        # u, v in Q.
        # N_Q(u) = {v, a, b}. So a, b are neighbors of u in G.
        # N_Q(v) = {u, x, y}. So x, y are neighbors of v in G.
        # a, b, x, y are distinct from u, v.
        # Also, a, b are in the same part as v? No.
        # If u in A, v in B. Then N(u) in B. So a, b in B.
        # N(v) in A. So x, y in A.
        # So a, b, v are in B. x, y, u are in A.
        # The 8th vertex w?
        # In Q3, each vertex has degree 3.
        # a is connected to u, and two others in A.
        # b is connected to u, and two others in A.
        # v is connected to a, b, x, y? No, degree 3.
        # Wait, N_Q(v) = {u, x, y}. So v is connected to u, x, y.
        # Is v connected to a or b?
        # In the example: v=001. N(001)={000, 011, 101}.
        # a,b from N(000)={001, 010, 100}. So a=010, b=100.
        # Is 001 connected to 010? No.
        # Is 001 connected to 100? No.
        # So v is NOT connected to a or b.
        # Similarly, u is NOT connected to x or y.
        
        # So:
        # 1. a, b are neighbors of u.
        # 2. x, y are neighbors of v.
        # 3. a, b, x, y are distinct.
        # 4. a is not connected to v.
        # 5. b is not connected to v.
        # 6. x is not connected to u.
        # 7. y is not connected to u.
        
        # Now, who is w?
        # a is connected to u. It needs 2 more neighbors in A.
        # b is connected to u. It needs 2 more neighbors in A.
        # x is connected to v. It needs 2 more neighbors in B.
        # y is connected to v. It needs 2 more neighbors in B.
        
        # The vertices in A are {u, x, y, w}.
        # The vertices in B are {v, a, b, w'}? No, w is in A or B?
        # Total 8 vertices.
        # A has 4, B has 4.
        # A = {u, x, y, ?}.
        # B = {v, a, b, ?}.
        # Let the 4th vertex in A be w_a and in B be w_b?
        # No, Q3 is connected.
        # Let's look at the example again.
        # A = {000, 011, 101, 111}.
        # B = {001, 010, 100, 110}.
        # u=000, v=001.
        # a=010, b=100.
        # x=011, y=101.
        # The 4th vertex in A is 111.
        # The 4th vertex in B is 110.
        
        # So we need to find w_a (in A) and w_b (in B).
        # w_a must be connected to a, b?
        # 111 is connected to 011(x), 101(y), 110(w_b).
        # Wait, 111 is in A. It is connected to B vertices.
        # N(111) = {011, 101, 110}.
        # So w_a (111) is connected to x, y, w_b.
        # Similarly, w_b (110) is in B.
        # N(110) = {010(a), 100(b), 111(w_a)}.
        # So w_b is connected to a, b, w_a.
        
        # So the conditions are:
        # 1. a, b in N(u) \ {v}.
        # 2. x, y in N(v) \ {u}.
        # 3. a, b, x, y distinct.
        # 4. a not adj v, b not adj v.
        # 5. x not adj u, y not adj u.
        # 6. There exists w_a such that:
        #    - w_a is adj to x, y, w_b.
        #    - w_a is NOT adj to u, a, b, v? (w_a in A, u,a,b in A? No, a,b in B).
        #    - w_a in A. So not adj to u, x, y? No, w_a is adj to x,y.
        #    - w_a is not adj to a, b? (a,b in B).
        # 7. There exists w_b such that:
        #    - w_b is adj to a, b, w_a.
        #    - w_b in B.
        
        # This is getting complicated to check on the fly.
        # The parent's method was to find ANY Q3 in the graph and remove an edge.
        # This is O(V^k) but k is small.
        
        # I will use the parent's `find_q3` but optimize it with bitmasks.
        
        pass

    # Let's implement the bitmask find_q3.
    # adj is a list of 16 ints.
    
    def find_q3_bitmask(adj):
        # Try to find a Q3.
        # Iterate over all pairs (u,v) that are edges.
        # Check if they can be part of a Q3.
        
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] & (1 << v)):
                    continue
                
                # Check if (u,v) is in a Q3.
                # We need to find a, b, x, y, w_a, w_b.
                
                # Candidates for a, b: neighbors of u (excluding v).
                # Candidates for x, y: neighbors of v (excluding u).
                
                nu = adj[u] & ~(1 << v) & ~(1 << u)
                nv = adj[v] & ~(1 << u) & ~(1 << v)
                
                # Get lists of candidates
                a_candidates = [i for i in range(n) if (nu & (1 << i))]
                x_candidates = [i for i in range(n) if (nv & (1 << i))]
                
                # We need a, b from a_candidates, x, y from x_candidates.
                # a, b distinct, x, y distinct.
                # a, b not in x_candidates? No, a,b in B, x,y in A. Disjoint.
                # So a, b must not be in x_candidates?
                # a is in B. x is in A. So a != x.
                # So a_candidates and x_candidates should be disjoint?
                # If a is in both, it's adjacent to u and v.
                # In Q3, can a vertex be adjacent to both u and v?
                # u in A, v in B.
                # a in B. a adj u.
                # If a adj v, then a is adjacent to two vertices in B? No, v is in B.
                # a is in B. v is in B.
                # In bipartite graph, no edges within B.
                # So a cannot be adjacent to v.
                # So a_candidates and x_candidates must be disjoint?
                # a is in B. x is in A.
                # So a != x.
                # But a could be in x_candidates if a is adjacent to v?
                # But a is NOT adjacent to v in Q3.
                # So if a is in x_candidates (adj to v), it's not a valid 'a' for this Q3.
                # So we filter a_candidates to exclude those adjacent to v.
                # And x_candidates to exclude those adjacent to u.
                
                valid_a = [i for i in a_candidates if not (adj[i] & (1 << v))]
                valid_x = [i for i in x_candidates if not (adj[i] & (1 << u))]
                
                # Now pick a, b from valid_a, x, y from valid_x.
                for i in range(len(valid_a)):
                    for j in range(i + 1, len(valid_a)):
                        a, b = valid_a[i], valid_a[j]
                        
                        for k in range(len(valid_x)):
                            for l in range(k + 1, len(valid_x)):
                                x, y = valid_x[k], valid_x[l]
                                
                                # We have u, v, a, b, x, y.
                                # Need w_a, w_b.
                                # w_a must be adj to x, y.
                                # w_b must be adj to a, b.
                                # w_a must be adj to w_b.
                                # w_a not adj to u, a, b, v?
                                # w_a in A. u, x, y in A.
                                # So w_a not adj to u, x, y? No, w_a adj to x, y.
                                # w_a not adj to a, b (in B)? No, w_a can be adj to a, b?
                                # In Q3, w_a (111) is adj to x(011), y(101), w_b(110).
                                # Is 111 adj to a(010)? No.
                                # Is 111 adj to b(100)? No.
                                # So w_a is NOT adj to a, b.
                                # Similarly, w_b is NOT adj to x, y.
                                
                                # Find candidates for w_a:
                                # Must be adj to x, y.
                                # Must NOT be adj to u, a, b, v.
                                # Must be distinct from u, v, a, b, x, y.
                                
                                forbidden_a = (1 << u) | (1 << v) | (1 << a) | (1 << b) | (1 << x) | (1 << y)
                                
                                # w_a must be in N(x) & N(y)
                                # and not in forbidden_a
                                # and not adj to a, b? (Check later)
                                # and not adj to u, v? (Check later)
                                
                                cand_wa = (adj[x] & adj[y]) & ~forbidden_a
                                
                                # Filter cand_wa
                                wa_list = [i for i in range(n) if (cand_wa & (1 << i))]
                                
                                for w_a in wa_list:
                                    # Check w_a not adj to a, b, u, v
                                    if adj[w_a] & ((1 << a) | (1 << b) | (1 << u) | (1 << v)):
                                        continue
                                    
                                    # Find w_b
                                    # w_b must be adj to a, b, w_a.
                                    # w_b not adj to x, y, u, v.
                                    # w_b distinct from others.
                                    
                                    forbidden_b = (1 << u) | (1 << v) | (1 << a) | (1 << b) | (1 << x) | (1 << y) | (1 << w_a)
                                    
                                    cand_wb = (adj[a] & adj[b] & adj[w_a]) & ~forbidden_b
                                    
                                    wb_list = [i for i in range(n) if (cand_wb & (1 << i))]
                                    
                                    for w_b in wb_list:
                                        # Check w_b not adj to x, y, u, v
                                        if adj[w_b] & ((1 << x) | (1 << y) | (1 << u) | (1 << v)):
                                            continue
                                        
                                        # Found a Q3!
                                        # Vertices: u, v, a, b, x, y, w_a, w_b
                                        # Edges:
                                        # (u,v)
                                        # (u,a), (u,b)
                                        # (v,x), (v,y)
                                        # (a,w_b), (b,w_b)
                                        # (x,w_a), (y,w_a)
                                        # (w_a, w_b)
                                        
                                        return [(u, v), (u, a), (u, b), (v, x), (v, y), 
                                                (a, w_b), (b, w_b), (x, w_a), (y, w_a), (w_a, w_b)]
        
        return None

    # Main Search
    best_edges = []
    best_count = 0
    
    for restart in range(50):
        if time.time() > deadline - 5:
            break
        
        # Random init
        edges = set()
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    edges.add((i, j))
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Remove Q3s
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            q3 = find_q3_bitmask(adj)
            if q3:
                # Remove random edge from Q3
                e = random.choice(q3)
                u, v = e
                edges.remove((u, v))
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                changed = True
        
        # Add edges greedily
        missing = [(i, j) for i in range(n) for j in range(i + 1, n) if (i, j) not in edges]
        random.shuffle(missing)
        
        for e in missing:
            if time.time() > deadline - 5:
                break
            u, v = e
            # Check if adding (u,v) creates a Q3
            # Temporarily add
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            if find_q3_bitmask(adj) is None:
                edges.add((u, v))
            else:
                # Revert
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = list(edges)
            
        # Local Search: Try to swap edges?
        # Or just try to add more?
        # The greedy add should be maximal.
        
    # Final cleanup
    adj = [0] * n
    for u, v in best_edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    changed = True
    while changed and time.time() < deadline - 2:
        changed = False
        q3 = find_q3_bitmask(adj)
        if q3:
            e = random.choice(q3)
            u, v = e
            best_edges.remove((u, v))
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            changed = True
            
    return [[u, v] for u, v in best_edges]
