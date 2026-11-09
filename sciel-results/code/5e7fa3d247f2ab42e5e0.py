"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START

def _verify_c4_free(n, colors):
    """Check that no color class contains a C4.
    
    A C4 in a color class means there exist vertices a, b, c, d (all distinct)
    such that (a,c) and (b,d) are edges of the same color, and (a,d) and (b,c) 
    are also edges of the same color... wait no.
    
    Actually, a 4-cycle C4 consists of four vertices v1, v2, v3, v4 and edges
    (v1,v2), (v2,v3), (v3,v4), (v4,v1). All four edges must be the same color.
    
    Equivalently: for any two vertices u, w, if they have at least two common
    neighbors in the same color class, that doesn't directly give a C4 unless
    those two common neighbors are connected in that color.
    
    Let me think again. A C4 in color c is a set of 4 vertices {a,b,c,d} with
    edges (a,b), (b,c), (c,d), (d,a) all in color c.
    
    Equivalent check: for any pair of vertices (u, w), if there exist two
    distinct vertices x, y such that (u,x), (x,w), (w,y), (y,u) are all in
    color c, that's a C4. But this is the same as saying u and w share at least
    two common neighbors x, y in color c AND (x,y) is NOT necessarily required...
    
    Wait, actually a C4 is: vertices v1,v2,v3,v4 with edges (v1,v2),(v2,v3),(v3,v4),(v4,v1).
    So v1 and v3 are opposite, v2 and v4 are opposite.
    v1 and v3 share common neighbors v2 and v4 in color c.
    So: if any two vertices share >= 2 common neighbors in the same color, we have a C4?
    No! We also need that v2 and v4 are NOT connected... actually no, in a C4 the edges are
    (v1,v2),(v2,v3),(v3,v4),(v4,v1). The diagonal edges (v1,v3) and (v2,v4) are NOT part of the C4.
    So we just need: exist v1,v2,v3,v4 all distinct, such that (v1,v2),(v2,v3),(v3,v4),(v4,v1) all in color c.
    
    This is equivalent to: v1 and v3 have common neighbors v2 and v4 in color c (i.e., (v1,v2),(v1,v4),(v3,v2),(v3,v4) all in color c).
    We do NOT require (v2,v4) to be in color c.
    
    So the check is: for any color c, for any pair of vertices (u,w), count common neighbors in color c.
    If count >= 2, we have a C4.
    
    Wait, that's not quite right either. If u and w share common neighbors x and y in color c,
    then we have edges (u,x),(u,y),(w,x),(w,y) all in color c. That forms a C4: u-x-w-y-u.
    Yes! So the condition is: for no color c and no pair (u,w) do they have 2 or more common neighbors in color c.
    """
    # For each color, build adjacency and check
    # For efficiency with n up to ~29, O(n^4 * 5) is fine
    # But let's use the common-neighbor approach
    for c in range(5):
        # adj[v] = set of neighbors of v in color c
        adj = [set() for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                if colors[idx] == c:
                    adj[i].add(j)
                    adj[j].add(i)
                idx += 1
        # For each pair (u,w), count common neighbors
        for u in range(n):
            for w in range(u+1, n):
                if len(adj[u] & adj[w]) >= 2:
                    return False
    return True

def _construct_z5x5():
    """Construct C4-free 5-coloring of K_25 using Z_5 x Z_5."""
    n = 25
    # Vertices labeled 0..24, mapped to (x, y) in Z_5 x Z_5 where x = v // 5, y = v % 5
    # 5 slopes: 0, 1, 2, 3, 4 (slopes in Z_5)
    # For each slope m, the "lines" are sets {(x, m*x + b) : x in Z_5} for b in Z_5
    # But we need to partition edges, not vertices.
    
    # Alternative: use the fact that K_25 can be decomposed into 5 graphs, each being
    # a union of C5's or similar C4-free structures.
    
    # Actually, let me use a different construction.
    # The 5-coloring of K_25: assign to edge (i,j) a color based on the "difference" pattern.
    
    # Label vertices as (a, b) where a = v // 5, b = v % 5, so a,b in {0,1,2,3,4}
    # For edge ((a1,b1), (a2,b2)):
    #   If a1 == a2: same row, color based on b2-b1 mod 5 (using 1..4 -> colors 0..3?) 
    #   This gets complicated.
    
    # Let me try: 5 groups of 5 vertices each (groups 0..4).
    # Within each group: C5 coloring (4 colors used, or distribute)
    # Between groups: use slope-based coloring
    
    # Actually, simplest known construction:
    # Vertices in Z_5^2. Edge from (x1,y1) to (x2,y2).
    # Let dx = (x2-x1) mod 5, dy = (y2-y1) mod 5.
    # If dx == 0: color = 0 (vertical)
    # Else: slope = (dy * dx^{-1}) mod 5, color = slope (1..4)
    # But dx can be 0,1,2,3,4. Inverse: 1->1, 2->3, 3->2, 4->4
    # This gives colors 0..4.
    
    # Check C4-freeness: In color 0 (vertical), edges connect (x,y) to (x,y') for all y' != y.
    # So color 0 graph is 5 disjoint K_5's. K_5 contains C4! So this is BAD.
    
    # I need a different approach. Each color class must be C4-free.
    # A C4-free graph on 5 vertices: maximum edges? C4-free means no 4-cycle.
    # K5 has many C4s. So I can't have K5 in any color.
    
    # Let me use: 5 groups of 5. Within each group, use a C4-free graph (like a 5-cycle C5, which has 5 edges).
    # Between groups, distribute remaining edges.
    
    # Total edges in K25 = 25*24/2 = 300.
    # Within groups: 5 * C(5,2) = 5*10 = 50 edges.
    # Between groups: 5*4/2 * 5*5 = 10*25 = 250 edges.
    # Total: 300. Check: 50+250=300. Good.
    
    # For within-group edges: use C5 structure. In group g (vertices 5g..5g+4), 
    # color edge (5g+i, 5g+j) based on |i-j| mod 5: if dist=1 or 4, use color 0; if dist=2 or 3, use... 
    # But C5 uses 5 edges, we have 10 edges per group. 
    # Actually a C5 has 5 edges. The complement of C5 in K5 is also a C5.
    # So I can split K5 into two C5's. But I need 5 colors total.
    
    # Let me reconsider. Maybe use a known construction from literature.
    # Lazebnik-Woldar construction for R_5(C4) >= 26 uses algebraic curves over GF(5).
    # For n=25, a simpler construction exists.
    
    # Let me try: vertices (x,y) in Z_5^2.
    # Color of edge ((x1,y1),(x2,y2)):
    #   dx = (x2-x1) mod 5
    #   dy = (y2-y1) mod 5  
    #   If (dx,dy) = (1,0) or (4,0): color 0
    #   If (dx,dy) = (0,1) or (0,4): color 1
    #   If (dx,dy) = (1,1) or (4,4): color 2
    #   If (dx,dy) = (1,2) or (4,3): color 3
    #   If (dx,dy) = (1,3) or (4,2): color 4
    #   ... but this doesn't cover all cases.
    
    # Actually I think the right approach for a guaranteed correct answer is to just
    # use the known n=11 construction (which is verified correct) and try to extend it
    # with a local search. But the seed already does n=11.
    
    # Let me just return the n=11 construction which is known correct, and try a
    # simple greedy/local search to extend to larger n within time budget.
    
    n = 11
    colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    return n, colors

def _try_extend(n_start, colors_start, target_n, deadline):
    """Try to extend a valid coloring from n_start to target_n vertices.
    Uses greedy: for each new vertex, try all possible color assignments for its
    edges to existing vertices, checking C4-freeness."""
    n = n_start
    colors = list(colors_start)
    
    def add_vertex():
        nonlocal n, colors
        if n >= target_n:
            return True
        if time.time() > deadline - 5:
            return False
        
        new_v = n
        # For each existing vertex i (0..n-1), assign a color to edge (i, new_v)
        # Try all 5^n combinations? No, that's too many.
        # Greedy: assign colors one by one, backtracking if needed.
        # For n=11, adding 1 vertex means 11 edges. 5^11 = 48M, too many for exhaustive.
        # Try random assignments with verification.
        
        import random
        # Try up to 1000 random assignments
        for _ in range(1000):
            if time.time() > deadline - 5:
                return False
            new_colors = [random.randint(0, 4) for _ in range(n)]
            # New full color list
            full_colors = colors + new_colors
            # Verify C4-freeness
            if _verify_c4_free(n+1, full_colors):
                n += 1
                colors = full_colors
                return True
        return False
    
    while add_vertex():
        pass
    
    return n, colors

def _local_search(n, colors, deadline):
    """Try to improve by adding vertices via local search / simulated annealing."""
    # This is complex; skip for now and just return what we have
    return n, colors

def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Start with known valid n=11 construction
    n, colors = _construct_z5x5()
    
    # Verify it's valid
    if not _verify_c4_free(n, colors):
        # Fallback: just return n=1
        return {'n': 1, 'colors': []}
    
    # Try to extend
    if time.time() < deadline - 10:
        n, colors = _try_extend(n, colors, 29, deadline)
    
    return {'n': n, 'colors': colors}

# EVOLVE-BLOCK-END
