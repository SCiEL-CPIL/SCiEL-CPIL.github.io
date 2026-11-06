import time
import random
import sys

def _paley_coloring(n):
    """
    Generate the coloring for the Paley graph on n vertices.
    n must be a prime p = 1 mod 4.
    Edge (i, j) is color 0 if (j-i) is a quadratic residue mod n, else 1.
    """
    squares = set()
    for x in range(1, n):
        squares.add((x * x) % n)
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            d = (j - i) % n
            colors.append(0 if d in squares else 1)
    return colors

def _has_k222_fast(colors, n):
    """
    Check if the coloring contains K2,2,2 in either color.
    Uses bitset adjacency for efficiency.
    Returns True if K2,2,2 found, False otherwise.
    
    K2,2,2 (octahedron) is the complete 3-partite graph K_{2,2,2}.
    A graph G contains K2,2,2 if there exist 6 vertices partitioned into 3 pairs
    (a,b), (c,d), (e,f) such that:
    1. (a,b), (c,d), (e,f) are non-edges in G.
    2. All edges between different pairs are present in G.
    
    Strategy:
    For each color c:
      For each non-edge (i, j) in G_c:
        Let S = N(i) ∩ N(j) (common neighbors in G_c).
        We need to find two disjoint non-edges (k, l) and (m, n) in S
        such that all cross edges between {k,l} and {m,n} are present in G_c.
        This is equivalent to finding a K4 in the graph H induced by S,
        where H's edges are the non-edges of G_c? No.
        
        Let's re-verify.
        We need:
        - i, j non-adjacent in G_c
        - k, l non-adjacent in G_c
        - m, n non-adjacent in G_c
        - i-k, i-l, i-m, i-n present in G_c
        - j-k, j-l, j-m, j-n present in G_c
        - k-m, k-n, l-m, l-n present in G_c
        
        So k, l, m, n must be in S = N(i) ∩ N(j).
        And in the induced subgraph on {k,l,m,n} in G_c, we need:
        - k-m, k-n, l-m, l-n present.
        - k-l, m-n absent.
        
        This means {k,l,m,n} induces a K4 minus two disjoint edges (k,l) and (m,n).
        Or equivalently, in the complement graph \bar{G}_c induced by S,
        we need edges (k,l) and (m,n) to be present, and no other edges among these 4?
        No, if k-m is present in G_c, it is absent in \bar{G}_c.
        So in \bar{G}_c induced by S, we need:
        - (k,l) present
        - (m,n) present
        - (k,m), (k,n), (l,m), (l,n) absent
        
        So we need to find a matching of size 2 in \bar{G}_c[S] such that
        the 4 vertices form an independent set in \bar{G}_c[S] except for the two matching edges?
        Actually, if (k,l) and (m,n) are edges in \bar{G}_c, and all cross edges are absent in \bar{G}_c,
        then {k,l,m,n} induces a 2K2 (two disjoint edges) in \bar{G}_c.
        
        So the condition is:
        There exists a non-edge (i,j) in G_c such that the complement of G_c induced by
        N_Gc(i) ∩ N_Gc(j) contains a 2K2 (two disjoint edges).
        
        This is much faster to check!
        We just need to check if \bar{G}_c[S] contains 2K2.
        2K2 is a graph with 4 vertices and 2 disjoint edges.
        It is a subgraph of any graph with at least 4 vertices that has at least 2 disjoint edges.
        Actually, if \bar{G}_c[S] has a matching of size 2, we are done.
        
        Wait, is that sufficient?
        If \bar{G}_c[S] has a matching of size 2, say (k,l) and (m,n),
        then k-l and m-n are non-edges in G_c.
        And k-m, k-n, l-m, l-n are edges in G_c (because they are NOT in \bar{G}_c).
        This is exactly the condition.
        
        So the check is:
        For each color c:
          For each non-edge (i, j) in G_c:
            S = N_Gc(i) ∩ N_Gc(j)
            If \bar{G}_c[S] has a matching of size 2, return True.
            
        How to check if \bar{G}_c[S] has a matching of size 2?
        Let E_bar be the set of edges in \bar{G}_c[S].
        We need two disjoint edges (u,v) and (w,x) in E_bar.
        This is equivalent to:
        There exists an edge (u,v) in E_bar such that E_bar contains an edge
        disjoint from {u,v}.
        
        We can precompute for each vertex v in S, the set of neighbors in \bar{G}_c[S].
        Then for each edge (u,v) in \bar{G}_c[S], check if there is an edge in \bar{G}_c[S]
        not incident to u or v.
        
        Alternatively, just check if the maximum matching size in \bar{G}_c[S] is >= 2.
        For small S, we can just iterate.
        
        Let's implement this.
    """
    adj0 = [0] * n
    adj1 = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            idx += 1
            if c == 0:
                adj0[i] |= (1 << j)
                adj0[j] |= (1 << i)
            else:
                adj1[i] |= (1 << j)
                adj1[j] |= (1 << i)
    
    full_mask = (1 << n) - 1
    
    for c in [0, 1]:
        adj = adj0 if c == 0 else adj1
        # Build complement adjacency for this color
        # comp_adj[v] = full_mask ^ (1<<v) ^ adj[v]
        # But we only care about edges within S later.
        
        for i in range(n):
            for j in range(i + 1, n):
                # Check if (i, j) is a non-edge in G_c
                if adj[i] & (1 << j):
                    continue # It is an edge, skip
                
                # S = N(i) ∩ N(j) in G_c
                S = adj[i] & adj[j]
                S &= ~(1 << i) & ~(1 << j) # Exclude i, j (though they are non-neighbors)
                
                # Check if \bar{G}_c[S] has a matching of size 2
                # We need to find two disjoint non-edges in G_c within S.
                # i.e., two pairs (k,l) and (m,n) in S such that
                # (k,l) and (m,n) are non-edges in G_c, and {k,l} ∩ {m,n} = empty.
                
                # Let's extract the list of vertices in S
                verts = []
                m = S
                while m:
                    v = (m & -m).bit_length() - 1
                    verts.append(v)
                    m &= m - 1
                
                # If |S| < 4, no matching of size 2
                if len(verts) < 4:
                    continue
                
                # Build adjacency list for \bar{G}_c induced by verts
                # We need to check if there is a matching of size 2.
                # A matching of size 2 exists iff there are two disjoint edges in \bar{G}_c[S].
                
                # Let's just check all pairs of edges in \bar{G}_c[S].
                # Number of edges in \bar{G}_c[S] can be up to |S|*(|S|-1)/2.
                # For n=30, |S| can be up to 28. Edges ~ 378. Pairs of edges ~ 70k.
                # This is fast enough.
                
                edges_bar = []
                for a in range(len(verts)):
                    u = verts[a]
                    for b in range(a + 1, len(verts)):
                        v = verts[b]
                        # Check if (u, v) is a non-edge in G_c
                        if not (adj[u] & (1 << v)):
                            edges_bar.append((u, v))
                
                # Now check if there are two disjoint edges in edges_bar
                # We can do this by iterating over pairs of edges.
                # To speed up, we can use a bitmask for vertices.
                # For each edge e1, check if there is an edge e2 disjoint from e1.
                
                # Even simpler:
                # For each edge (u, v) in edges_bar,
                # check if there is an edge (w, x) in edges_bar such that
                # w, x not in {u, v}.
                
                # We can precompute a set of edges for fast lookup?
                # Or just double loop.
                
                L = len(edges_bar)
                for a in range(L):
                    u, v = edges_bar[a]
                    mask_uv = (1 << u) | (1 << v)
                    for b in range(a + 1, L):
                        w, x = edges_bar[b]
                        # Check if {w, x} is disjoint from {u, v}
                        if not (mask_uv & ((1 << w) | (1 << x))):
                            return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    # The known record is n=29. Paley(29) is a valid K2,2,2-free coloring.
    # R(K2,2,2) = 30 means that for n=30, every 2-coloring contains a monochromatic K2,2,2.
    # So we cannot find a valid coloring for n=30.
    # The maximum n is 29.
    
    n = 29
    colors = _paley_coloring(n)
    
    # Validate
    if _has_k222_fast(colors, n):
        # Fallback to smaller known-good Paley graph if validation fails
        n = 17
        colors = _paley_coloring(n)
    
    return {'n': n, 'colors': colors}
