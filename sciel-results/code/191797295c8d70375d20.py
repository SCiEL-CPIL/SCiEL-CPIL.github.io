import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    m_total = len(all_edges)
    
    # Precompute Q3 edge masks for fast checking
    # A Q3 is a cube: 8 vertices, 12 edges
    # We enumerate all possible Q3 subgraphs on 16 vertices
    # For each Q3, we store the set of edges as a bitmask
    
    # Actually, let's use a different approach:
    # For a graph to contain a Q3, there must exist 8 distinct vertices
    # forming a 3-cube structure.
    # 
    # Key insight: In Q3, each vertex has degree 3 within the cube.
    # So if we pick a vertex u and 3 of its neighbors a, b, c,
    # then in Q3, a, b, c must be mutually non-adjacent (they are at distance 2 from each other via u).
    # Wait, no. In Q3, u=000, a=100, b=010, c=001. a and b are NOT adjacent.
    # So a, b, c are an independent set in the Q3.
    # Then there must be vertices a', b', c' such that:
    # a' adjacent to u? No. a' adjacent to a (the "other" edge from a).
    # Let's label properly.
    # u = 000
    # a = 100, b = 010, c = 001
    # a' = 110 (adjacent to a and b)
    # b' = 101 (adjacent to a and c)
    # c' = 011 (adjacent to b and c)
    # d = 111 (adjacent to a', b', c')
    #
    # Edges:
    # u-a, u-b, u-c
    # a-a', b-b', c-c'
    # a'-b', a'-c', b'-c'  -- wait, no.
    # a'=110 is adjacent to a=100 and b=010. Also to d=111.
    # b'=101 is adjacent to a=100 and c=001. Also to d=111.
    # c'=011 is adjacent to b=010 and c=001. Also to d=111.
    # d=111 is adjacent to a', b', c'.
    #
    # So edges are:
    # (u,a), (u,b), (u,c)
    # (a,a'), (b,b'), (c,c')
    # (a',b'), (a',c'), (b',c') -- NO! a'=110, b'=101. 110 and 101 differ in 2 bits. NOT adjacent.
    # 
    # Let me re-check. 110 and 101: 1^1=0, 1^0=1, 0^1=1. Hamming distance 2. Not adjacent.
    # So a' and b' are NOT adjacent.
    # 
    # The edges of Q3 are:
    # (000,100), (000,010), (000,001)
    # (100,110), (100,101)
    # (010,110), (010,011)
    # (001,101), (001,011)
    # (110,111), (101,111), (011,111)
    #
    # So the structure is:
    # u=000 connects to a=100, b=010, c=001
    # a=100 connects to a1=110, a2=101
    # b=010 connects to b1=110, b2=011
    # c=001 connects to c1=101, c2=011
    # Note: a1=b1=110, a2=c1=101, b2=c2=011
    # So there are only 3 "middle" vertices: 110, 101, 011
    # And 1 "far" vertex: 111
    #
    # Let's rename:
    # u = 000
    # a = 100, b = 010, c = 001
    # m1 = 110 (adjacent to a, b, and d)
    # m2 = 101 (adjacent to a, c, and d)
    # m3 = 011 (adjacent to b, c, and d)
    # d = 111 (adjacent to m1, m2, m3)
    #
    # Edges:
    # (u,a), (u,b), (u,c)
    # (a,m1), (a,m2)
    # (b,m1), (b,m3)
    # (c,m2), (c,m3)
    # (m1,d), (m2,d), (m3,d)
    #
    # Total 12 edges. Correct.
    
    def has_q3(adj, new_u, new_v):
        """Check if adding edge (new_u, new_v) creates a Q3 containing that edge."""
        # The new edge must be part of some Q3.
        # In Q3, each edge connects two vertices at Hamming distance 1.
        # So new_u and new_v play the role of two adjacent vertices in the cube.
        
        # Without loss of generality, let new_u be the "lower" vertex (like 000)
        # and new_v be the "higher" vertex (like 100).
        # But we don't know which bit flipped. So we consider both orientations.
        
        # Actually, let's just check all Q3s that contain the edge (new_u, new_v).
        # A Q3 containing (new_u, new_v) must have:
        # - new_u has 2 other neighbors in the Q3 (besides new_v)
        # - new_v has 2 other neighbors in the Q3 (besides new_u)
        
        nu = adj[new_u] & ~(1 << new_v)
        nv = adj[new_v] & ~(1 << new_u)
        
        # new_u must have at least 2 other neighbors, new_v at least 2 other neighbors
        if nu.bit_count() < 2 or nv.bit_count() < 2:
            return False
        
        # Get neighbor lists
        nu_list = [i for i in range(n) if (nu >> i) & 1]
        nv_list = [i for i in range(n) if (nv >> i) & 1]
        
        # Try all pairs of neighbors of new_u
        for i in range(len(nu_list)):
            a = nu_list[i]
            for j in range(i+1, len(nu_list)):
                b = nu_list[j]
                
                # In Q3, a and b are NOT adjacent (they are at distance 2 from each other)
                # Wait, in the structure above, a=100 and b=010 are NOT adjacent.
                # But in a general graph, they might be. The Q3 is a SUBGRAPH,
                # so extra edges are allowed. So we don't need to check non-adjacency.
                
                # Now we need to find m1, m2, m3, d such that:
                # m1 adjacent to a, b, d
                # m2 adjacent to a, c, d  (where c is the third neighbor of new_u)
                # m3 adjacent to b, c, d
                # 
                # Wait, I assumed new_u has neighbors a, b, c in the Q3.
                # But here I'm only picking 2 neighbors a, b. The third one c is determined
                # by the structure? No, c is another neighbor of new_u.
                # 
                # Let me reconsider. The Q3 containing edge (new_u, new_v) has:
                # new_u connected to new_v, and 2 other vertices (call them x, y)
                # new_v connected to new_u, and 2 other vertices (call them p, q)
                # 
                # The full structure:
                # new_u - new_v
                # new_u - x, new_u - y
                # new_v - p, new_v - q
                # x - p, x - q  (x is adjacent to both p and q? Let's check)
                # 
                # Using the bit model:
                # new_u = 000, new_v = 100
                # x = 010, y = 001
                # p = 110, q = 101
                # m3 = 011 (adjacent to x=010, y=001, and d=111)
                # d = 111
                # 
                # Edges:
                # (000,100), (000,010), (000,001)
                # (100,110), (100,101)
                # (010,110), (010,011)
                # (001,101), (001,011)
                # (110,111), (101,111), (011,111)
                # 
                # So:
                # x=010 is adjacent to p=110 and m3=011
                # y=001 is adjacent to q=101 and m3=011
                # p=110 is adjacent to new_v=100, x=010, d=111
                # q=101 is adjacent to new_v=100, y=001, d=111
                # m3=011 is adjacent to x=010, y=001, d=111
                # d=111 is adjacent to p, q, m3
                # 
                # So the structure is:
                # new_u (000) -- new_v (100)
                # new_u -- x (010), new_u -- y (001)
                # new_v -- p (110), new_v -- q (101)
                # x -- p, x -- m3 (011)
                # y -- q, y -- m3 (011)
                # p -- d (111), q -- d, m3 -- d
                # 
                # Total vertices: new_u, new_v, x, y, p, q, m3, d (8 vertices)
                # Edges: 12
                
                # So for a fixed pair (x, y) of neighbors of new_u:
                # We need to find:
                # p: neighbor of new_v and x
                # q: neighbor of new_v and y
                # m3: neighbor of x and y
                # d: neighbor of p, q, m3
                # 
                # And all 8 vertices must be distinct.
                
                # Find p: common neighbor of new_v and x
                p_candidates = adj[new_v] & adj[x]
                # Find q: common neighbor of new_v and y
                q_candidates = adj[new_v] & adj[y]
                # Find m3: common neighbor of x and y
                m3_candidates = adj[x] & adj[y]
                
                if p_candidates == 0 or q_candidates == 0 or m3_candidates == 0:
                    continue
                
                p_list = [i for i in range(n) if (p_candidates >> i) & 1]
                q_list = [i for i in range(n) if (q_candidates >> i) & 1]
                m3_list = [i for i in range(n) if (m3_candidates >> i) & 1]
                
                for p in p_list:
                    for q in q_list:
                        for m3 in m3_list:
                            # Distinctness check
                            # Vertices: new_u, new_v, x, y, p, q, m3, d
                            # new_u, new_v, x, y are distinct by construction
                            # p, q, m3 must be distinct from each other and from new_u, new_v, x, y
                            if p in (new_u, new_v, x, y, q, m3):
                                continue
                            if q in (new_u, new_v, x, y, p, m3):
                                continue
                            if m3 in (new_u, new_v, x, y, p, q):
                                continue
                            
                            # Find d: common neighbor of p, q, m3
                            d_candidates = adj[p] & adj[q] & adj[m3]
                            if d_candidates == 0:
                                continue
                            
                            d_list = [i for i in range(n) if (d_candidates >> i) & 1]
                            for d in d_list:
                                if d in (new_u, new_v, x, y, p, q, m3):
                                    continue
                                # All 8 vertices distinct, all edges present
                                # Check that d is adjacent to p, q, m3 (already ensured by d_candidates)
                                # Check that p is adjacent to new_v and x (ensured)
                                # Check that q is adjacent to new_v and y (ensured)
                                # Check that m3 is adjacent to x and y (ensured)
                                # Check that new_u is adjacent to x, y (ensured)
                                # Check that new_v is adjacent to p, q (ensured)
                                # Check that x is adjacent to p, m3 (ensured)
                                # Check that y is adjacent to q, m3 (ensured)
                                return True
        return False

    best_edges = []
    best_count = 0
    
    def run_search(seed):
        random.seed(seed)
        adj = [0] * n
        edge_set = set()
        
        # Greedy construction with random edge order
        edges_to_try = all_edges[:]
        random.shuffle(edges_to_try)
        
        for u, v in edges_to_try:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            if not has_q3(adj, u, v):
                edge_set.add((u, v))
            else:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        
        # Local search: try to add missing edges
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try to add any missing edge
            for u, v in all_edges:
                if (u, v) not in edge_set:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    if not has_q3(adj, u, v):
                        edge_set.add((u, v))
                        improved = True
                    else:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
            
            # If stuck, try swaps
            if not improved and len(edge_set) > 0:
                # Remove a random edge
                u, v = random.choice(list(edge_set))
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                edge_set.remove((u, v))
                
                # Try to add a different edge
                added = False
                for u2, v2 in all_edges:
                    if (u2, v2) not in edge_set and (u2, v2) != (u, v):
                        adj[u2] |= (1 << v2)
                        adj[v2] |= (1 << u2)
                        if not has_q3(adj, u2, v2):
                            edge_set.add((u2, v2))
                            added = True
                            break
                        else:
                            adj[u2] &= ~(1 << v2)
                            adj[v2] &= ~(1 << u2)
                
                if not added:
                    # Revert
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_set.add((u, v))
                else:
                    improved = True
        
        return len(edge_set), list(edge_set)
    
    # Run multiple restarts
    restart = 0
    while time.time() < deadline:
        count, edges_list = run_search(restart)
        if count > best_count:
            best_count = count
            best_edges = edges_list
        
        restart += 1
        
        if best_count >= 60:
            break
    
    # Fallback
    if best_count == 0:
        random.seed(0)
        adj = [0] * n
        edge_set = set()
        for u, v in all_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            if not has_q3(adj, u, v):
                edge_set.add((u, v))
            else:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        best_count = len(edge_set)
        best_edges = list(edge_set)
    
    return [[u, v] for u, v in best_edges]

if __name__ == "__main__":
    print(construct())
