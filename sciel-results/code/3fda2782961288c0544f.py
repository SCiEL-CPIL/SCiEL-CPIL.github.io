import time
import random

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a C5-free graph with alpha < 9, maximizing n.
    
    Strategy:
    1. Start with 8 disjoint K4's (n=32, alpha=8, C5-free).
    2. Try to add 1-3 more vertices (n=33,34,35) by connecting them to the base
       in a way that preserves C5-freeness and keeps alpha < 9.
    3. If no valid extension is found quickly, fall back to n=32.
    
    The key insight: to add a vertex v to a C5-free graph, v must not create a C5.
    Also, v must not be part of an independent set of size 9.
    
    For the base 8xK4 structure:
    - Each K4 is a clique, so any vertex connected to >=2 vertices in a K4 
      could potentially create a C5 with other vertices.
    - Actually, in a C5-free graph, if v is connected to two vertices u,w in a 
      clique, and there's a path of length 2 from u to w through v, that's fine.
      A C5 requires a cycle of length exactly 5.
    
    Let's think more carefully. The base graph is 8 disjoint K4's.
    Adding a new vertex v:
    - If v is isolated: alpha increases by 1 (from 8 to 9) -> invalid.
    - If v is connected to some vertices: need to check C5 and alpha.
    
    For alpha: the maximum independent set in the base is 8 (one from each K4).
    If v is connected to at least one vertex in each K4, then v cannot be in any 
    independent set of size 9 (since any such set would need one from each K4 plus v,
    but v is adjacent to at least one in each K4... wait, that's not quite right).
    
    Actually, an independent set of size 9 would consist of 9 vertices no two adjacent.
    In the base, max independent set is 8. To get 9, we'd need to include v and 
    8 others, but those 8 others must be independent and none adjacent to v.
    
    If v is connected to exactly one vertex in each K4, then from each K4 we can 
    still pick at most 1 vertex (since it's a clique), but we must avoid the one 
    adjacent to v. So from each K4, we can pick at most 1 from the remaining 3.
    So we could pick 8 vertices (one from each K4, avoiding the neighbor of v) 
    plus v? No, v is adjacent to those 8, so v can't be in the independent set 
    with them.
    
    So if v is connected to at least one vertex in each K4, then any independent set 
    containing v can have at most... well, v's neighbors are excluded. The remaining 
    graph has 8 K4's each with one vertex removed, so each becomes a K3 (clique of size 3).
    Max independent set in that is 8 (one from each K3). But v is adjacent to all 
    of those? No, v is adjacent to one specific vertex in each K4, not all.
    
    This is getting complex. Let me just do a quick search.
    """
    deadline = time.time() + 140  # 140 second deadline
    
    # Base: 8 disjoint K4's
    base_n = 32
    base_edges = []
    for b in range(8):
        base = b * 4
        for i in range(4):
            for j in range(i + 1, 4):
                base_edges.append([base + i, base + j])
    
    def is_c5_free(n, edges):
        """Check if graph has no C5 subgraph."""
        # Build adjacency list
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Check all 5-tuples for cycles
        # This is O(n^5) which is too slow for n=35
        # Use a smarter approach: check for cycles of length 5
        
        # For each edge (u,v), check if there's a path of length 4 from u to v
        # that doesn't use edge (u,v) and forms a simple cycle
        # Actually, let's use a different approach: enumerate all simple cycles of length 5
        
        # Better: use BFS/DFS to find cycles
        # For each pair of vertices (u,v) with u < v, check if there's a path 
        # of length 4 from u to v that doesn't include the direct edge
        
        # Even better: for each vertex, do a BFS to find all paths of length 4
        # back to itself
        
        # Let's use a combinatorial approach: for each 5-tuple (a,b,c,d,e) with 
        # a < b < c < d < e, check if they form a C5
        # This is C(n,5) * 12 (number of C5s on 5 labeled vertices) which is 
        # C(35,5) * 12 = 324632 * 12 = ~3.9M, which is feasible
        
        from itertools import combinations
        for comb in combinations(range(n), 5):
            a, b, c, d, e = comb
            # Check all 12 ways to arrange these 5 vertices as a C5
            # The 12 distinct C5s on 5 vertices (up to rotation and reflection)
            # We can check: is there a permutation (a,b,c,d,e) such that
            # a-b, b-c, c-d, d-e, e-a are all edges?
            # There are 12 such cycles for 5 labeled vertices
            cycles = [
                (a,b,c,d,e),
                (a,b,c,e,d),
                (a,b,d,c,e),
                (a,b,d,e,c),
                (a,b,e,c,d),
                (a,b,e,d,c),
                (a,c,b,d,e),
                (a,c,b,e,d),
                (a,c,d,b,e),
                (a,c,d,e,b),
                (a,c,e,b,d),
                (a,c,e,d,b),
            ]
            for cyc in cycles:
                if (cyc[0] in adj[cyc[1]] and cyc[1] in adj[cyc[2]] and 
                    cyc[2] in adj[cyc[3]] and cyc[3] in adj[cyc[4]] and 
                    cyc[4] in adj[cyc[0]]):
                    return False
        return True
    
    def alpha_less_than_9(n, edges):
        """Check if independence number < 9."""
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Find maximum independent set using a simple branch and bound
        # For n <= 35, this should be fast enough
        
        def max_independent_set(vertices, adj):
            if not vertices:
                return 0
            # Pick a vertex
            v = vertices[0]
            # Option 1: include v
            # Remove v and its neighbors
            remaining = [u for u in vertices[1:] if u not in adj[v]]
            opt1 = 1 + max_independent_set(remaining, adj)
            # Option 2: exclude v
            opt2 = max_independent_set(vertices[1:], adj)
            return max(opt1, opt2)
        
        # This exponential approach may be too slow for n=35
        # Use a greedy + local search or a better algorithm
        
        # Simpler: try all subsets of size 9 and check if any is independent
        # C(35,9) = 70,607,460 which is too many
        
        # Use a different approach: maximum independent set via complement
        # alpha(G) = n - omega(complement(G))
        # Find maximum clique in complement
        
        # Build complement
        comp_adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u+1, n):
                if v not in adj[u]:
                    comp_adj[u].add(v)
                    comp_adj[v].add(u)
        
        # Find maximum clique in complement
        # Use Bron-Kerbosch with pivot
        max_clique_size = 0
        
        def bron_kerbosch(R, P, X):
            nonlocal max_clique_size
            if not P and not X:
                max_clique_size = max(max_clique_size, len(R))
                return
            
            # Pivot selection
            if P:
                u = list(P)[0]
                candidates = P - comp_adj[u]
            else:
                candidates = X
            
            for v in list(candidates):
                bron_kerbosch(R | {v}, P & comp_adj[v], X & comp_adj[v])
                P.remove(v)
                X.add(v)
        
        bron_kerbosch(set(), set(range(n)), set())
        return n - max_clique_size < 9
    
    # Try to extend the base graph
    best_n = base_n
    best_edges = base_edges.copy()
    
    # Try adding 1 vertex
    for extra in range(1, 4):  # Try n=33, 34, 35
        if time.time() > deadline - 10:
            break
        
        n = base_n + extra
        # Try different connection patterns for the new vertices
        # Each new vertex can be connected to a subset of the 32 base vertices
        
        # Strategy: connect each new vertex to exactly one vertex in each K4
        # This ensures that no C5 is created (need to verify) and alpha stays < 9
        
        found = False
        attempts = 0
        max_attempts = 2000
        
        while not found and attempts < max_attempts and time.time() < deadline - 5:
            attempts += 1
            edges = base_edges.copy()
            
            # For each new vertex, choose one vertex from each K4 to connect to
            valid = True
            for k in range(extra):
                new_v = base_n + k
                for b in range(8):
                    base = b * 4
                    # Choose a random vertex in this K4
                    choice = random.randint(0, 3)
                    edges.append([new_v, base + choice])
            
            # Check if C5-free
            if is_c5_free(n, edges):
                # Check alpha < 9
                if alpha_less_than_9(n, edges):
                    best_n = n
                    best_edges = edges
                    found = True
                    break
            else:
                # C5 found, try different connections
                pass
        
        if found:
            break
    
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
