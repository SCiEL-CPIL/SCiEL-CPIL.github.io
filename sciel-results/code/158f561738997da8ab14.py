import time
import random
import itertools

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    # Helper to check if a coloring is valid (no mono K2,2,2)
    # K2,2,2 is a complete tripartite graph with parts of size 2.
    # Vertices: {a,b}, {c,d}, {e,f}. Edges exist between any pair from different parts.
    # A monochromatic K2,2,2 means there exist 6 vertices partitioned into 3 pairs 
    # such that all 12 edges between these pairs are the same color.
    
    def is_valid(n, colors):
        # colors is a list of 0/1 for pairs (i,j) i<j
        # For efficiency, we can precompute adjacency or just check on the fly.
        # Given n <= 31, n^6 is too big for brute force, but we can be smart.
        # A K2,2,2 is defined by 3 disjoint pairs of vertices.
        # Let the pairs be (u1, v1), (u2, v2), (u3, v3).
        # The edges are (u1,u2), (u1,v2), (v1,u2), (v1,v2), ... etc.
        # All 12 edges must have the same color c.
        
        # To check efficiently:
        # Iterate over all combinations of 6 vertices? No, that's C(n,6) * 15 (ways to partition).
        # For n=20, C(20,6) is ~39k, which is fine.
        # For n=30, C(30,6) is ~593k. 593k * 15 * 12 checks is ~100M ops. Might be too slow in Python if done naively.
        
        # Optimized check:
        # For a fixed color c, we want to know if there exist 3 disjoint pairs 
        # (A,B), (C,D), (E,F) such that all cross edges are color c.
        
        # Let's use a bitset or simple loops.
        # For each pair of disjoint pairs (A,B) and (C,D), check if all 4 edges between them are color c.
        # If yes, mark that the "hyper-edge" between the two pairs is "c-complete".
        # Then check if there is a third pair (E,F) disjoint from A,B,C,D that is "c-complete" with both.
        
        # This is complex. Let's stick to a simpler heuristic for the inner loop if possible, 
        # or optimize the violation count.
        
        # Actually, for the repair loop, we need to find *which* edges to flip.
        # It's better to count violations or find one violation.
        
        # Let's try a direct check for validity first for small n, 
        # but for the search, we need a faster "violation detector".
        
        # A faster way to find a mono K2,2,2:
        # 1. Enumerate all pairs of vertices (u, v).
        # 2. For each color c, find all pairs (x, y) such that edges (u,x), (u,y), (v,x), (v,y) are all c.
        #    This forms a "c-rectangle" with sides u,v and x,y.
        # 3. If we have two such rectangles sharing no vertices? No, K2,2,2 is 3 parts.
        #    K2,2,2 is equivalent to a K4 in the line graph? No.
        #    K2,2,2 has vertices {1,2,3,4,5,6} with parts {1,2}, {3,4}, {5,6}.
        #    Edges are between parts.
        #    So, if we fix the first part {1,2}, we need to find two disjoint pairs {3,4} and {5,6}
        #    such that the subgraph induced by {1,2,3,4} is a K2,2 of color c,
        #    AND the subgraph induced by {1,2,5,6} is a K2,2 of color c,
        #    AND the subgraph induced by {3,4,5,6} is a K2,2 of color c.
        
        # Algorithm for finding a mono K2,2,2 of color c:
        # 1. Precompute for each pair (i,j) and color c, the set of vertices k such that 
        #    (i,k) is c and (j,k) is c. Let's call this S_c(i,j).
        # 2. For a pair (i,j) to be part of a K2,2,2 of color c, we need to find two disjoint pairs (k,l) and (m,n)
        #    from S_c(i,j) such that:
        #    - (k,l) is a pair where (k,m),(k,n),(l,m),(l,n) are all c.
        #    - (m,n) is a pair where (k,m),(k,n),(l,m),(l,n) are all c.
        #    - (k,m),(k,n),(l,m),(l,n) are all c (this is the condition for the third part).
        
        # This is still heavy. Let's try a randomized local search that minimizes the number of violations.
        
        pass 

    # Given the complexity of an exact validity check in Python for n=25+ within tight time limits,
    # and the fact that the seed is n=17, let's try to extend the Paley graph or use a simpler 
    # greedy construction that is known to work for slightly larger n, or a simple 
    # "random restart + greedy repair" that might stumble upon n=18 or 19.
    
    # However, R(K2,2,2) is known to be 31? The prompt says "record has 29". 
    # If the record is 29, finding 20 is a huge jump.
    
    # Let's try a simpler approach: 
    # The Paley graph of order 17 is a (17, 8, 4) graph. It is strongly regular.
    # It is known that Paley(17) is a Ramsey graph for K2,2,2? 
    # Actually, the Ramsey number R(K2,2,2) is 31. 
    # This means there EXISTS a coloring of K30 without mono K2,2,2.
    # The task is to find one.
    
    # Since finding one via search in 150s for n=30 is extremely hard (combinatorial explosion),
    # and the seed is 17, maybe there is a known construction for larger n.
    
    # Let's try the Paley graph of order 29?
    # 29 is a prime of the form 4k+1. 
    # Paley graph P(29) is a (29, 14, 7) graph.
    # Does it contain a mono K2,2,2?
    # Generally, Paley graphs are good candidates for Ramsey constructions.
    # Let's test P(29) validity.
    
    def check_paley(n):
        if n % 4 != 1:
            return False
        squares = {x * x % n for x in range(1, n)}
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append(0 if (j - i) % n in squares else 1)
        
        # Check for mono K2,2,2
        # Optimization: 
        # For each color c, build a graph G_c where edge (u,v) exists if color(u,v)=c.
        # We need to check if G_c contains K2,2,2.
        # K2,2,2 is the complete tripartite graph T_{2,2,2}.
        
        # A faster check for T_{2,2,2} in a graph G:
        # Iterate over all disjoint pairs of pairs?
        # Let's just brute force check for n=29 if it's fast enough.
        # C(29, 6) is ~475k. 
        # For each 6-set, check if it can be partitioned into 3 pairs such that all cross edges exist.
        # Number of partitions of 6 items into 3 pairs is 15.
        # 475k * 15 * 12 edge checks ~ 85M ops. 
        # In Python, 85M simple ops might take 5-10 seconds. 
        # We have 145 seconds. We can check multiple primes.
        
        # Let's write a fast checker.
        
        def has_mono_k222(n, colors):
            # Build adjacency bitmasks for each color
            # adj[c][i] is a bitmask of vertices j such that (i,j) is color c
            adj = [[0] * n for _ in range(2)]
            for i in range(n):
                for j in range(i + 1, n):
                    idx = i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)
                    # Wait, simpler to just loop i,j and fill
                    pass
            
            # Re-indexing colors for speed
            # colors is flat list of length n(n-1)/2
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    c = colors[idx]
                    adj[c][i] |= (1 << j)
                    adj[c][j] |= (1 << i)
                    idx += 1
            
            # Check for K2,2,2
            # Iterate over all 6-tuples? No, too slow.
            # Iterate over all pairs (u,v) and (x,y) disjoint.
            # If (u,v) and (x,y) form a K2,2 of color c, then check if there is a third pair (w,z)
            # disjoint from u,v,x,y that also forms K2,2 with both.
            
            # For a fixed color c:
            # Let S_c be the set of edges of color c.
            # We want to find 3 disjoint pairs P1, P2, P3 such that 
            # all edges between P1,P2, P1,P3, P2,P3 are in S_c.
            
            # Optimization:
            # For each pair of vertices (u, v), let N_c(u,v) = { k | (u,k) in S_c and (v,k) in S_c }.
            # If (u,v) is part of a K2,2,2 of color c, then there must exist two disjoint pairs (x,y) and (w,z)
            # in N_c(u,v) such that (x,y) is an edge in S_c, (w,z) is an edge in S_c, and 
            # all edges between {x,y} and {w,z} are in S_c.
            
            # Actually, if (x,y) is in N_c(u,v), it means u-x, u-y, v-x, v-y are all c.
            # For (x,y) to be a valid pair in the K2,2,2, we also need the edge (x,y) to be c? 
            # No! In K2,2,2, the parts are independent sets. 
            # So if {u,v} is one part, {x,y} is another, then edges between them exist.
            # Edges WITHIN {x,y} do NOT exist in K2,2,2.
            # So we need:
            # 1. (u,v) is a pair.
            # 2. (x,y) is a pair.
            # 3. (w,z) is a pair.
            # 4. All 4 edges between {u,v} and {x,y} are c.
            # 5. All 4 edges between {u,v} and {w,z} are c.
            # 6. All 4 edges between {x,y} and {w,z} are c.
            # 7. u,v,x,y,w,z are distinct.
            # Note: It is NOT required that (u,v) is an edge, (x,y) is an edge, etc. 
            # In fact, in a simple graph coloring, the "parts" of K2,2,2 are just sets of vertices.
            # The edges of K2,2,2 are the cross edges.
            # So we just need the cross edges to be monochromatic.
            
            # So, for color c:
            # For each pair (u,v):
            #   Let C = { k | k != u,v and adj[c][u] has k and adj[c][v] has k }
            #   This C is the set of vertices that are "c-connected" to both u and v.
            #   If |C| < 4, continue.
            #   We need to find two disjoint pairs (x,y) and (w,z) from C
            #   such that all 4 edges between {x,y} and {w,z} are c.
            #   This is equivalent to finding a K2,2 in the subgraph induced by C, 
            #   using only edges of color c.
            #   Wait, the edges between {x,y} and {w,z} must be c.
            #   So we need a K2,2 in the graph (C, E_c[C]).
            
            # So the problem reduces to:
            # For each color c and each pair (u,v), does the graph (C, E_c[C]) contain a K2,2?
            # K2,2 is a cycle of length 4.
            # So we just need to check if there is a 4-cycle in the "c-neighborhood" of (u,v).
            
            for c in [0, 1]:
                # Precompute C for each (u,v)? 
                # Or just iterate.
                for u in range(n):
                    for v in range(u + 1, n):
                        # Find C
                        # C = neighbors of u in color c AND neighbors of v in color c
                        # Mask
                        mask_u = adj[c][u]
                        mask_v = adj[c][v]
                        mask_c = mask_u & mask_v
                        # Remove u and v from mask_c (though they are not neighbors of themselves)
                        mask_c &= ~(1 << u)
                        mask_c &= ~(1 << v)
                        
                        # Get list of vertices in C
                        verts = [i for i in range(n) if (mask_c >> i) & 1]
                        if len(verts) < 4:
                            continue
                        
                        # Check if induced subgraph on verts has a K2,2 (4-cycle) with edges of color c.
                        # Since verts are small (max n-2), we can brute force 4-cycles.
                        # Or simpler: check all pairs (x,y) in verts.
                        # If (x,y) is an edge in E_c, and there exist (w,z) in verts\{x,y}
                        # such that (w,z) is an edge in E_c and all cross edges are c.
                        
                        # Actually, if (x,y) and (w,z) are edges in E_c, and 
                        # x-w, x-z, y-w, y-z are in E_c, then we have a K2,2.
                        # This is exactly a 4-cycle x-w-y-z-x in E_c.
                        
                        # So we just need to check if the graph (verts, E_c[verts]) contains a 4-cycle.
                        
                        # Efficient 4-cycle check for small graphs:
                        # For each edge (x,y) in E_c[verts]:
                        #   Check common neighbors in E_c[verts] \ {x,y}.
                        #   If there is a common neighbor w, and another common neighbor z?
                        #   No, a 4-cycle is x-w-y-z-x. 
                        #   Common neighbors of x and y in E_c[verts] must contain at least 2 vertices?
                        #   If x and y have common neighbors w and z, then x-w-y-z-x is a 4-cycle 
                        #   IF edges (x,z) and (y,w) exist? 
                        #   Wait. In a 4-cycle x-w-y-z-x, the edges are (x,w), (w,y), (y,z), (z,x).
                        #   So w and z are common neighbors of x and y?
                        #   Yes, w is connected to x and y. z is connected to x and y.
                        #   So if x and y have two common neighbors w and z in the graph E_c,
                        #   then we have a K2,2?
                        #   Let's check the edges:
                        #   (x,w) - yes
                        #   (w,y) - yes
                        #   (y,z) - yes (z is common neighbor of x,y? No, z is common neighbor means (x,z) and (y,z) exist)
                        #   (z,x) - yes
                        #   So yes, if two vertices x,y have at least 2 common neighbors in the graph G,
                        #   then G contains a K2,2 (specifically, the subgraph induced by {x,y,w,z} 
                        #   contains the edges of a 4-cycle, which is a K2,2 with parts {x,z} and {y,w}? 
                        #   No, K2,2 parts are independent. 
                        #   In the cycle x-w-y-z-x, the parts are {x,y} and {w,z}?
                        #   Edges in K2,2: (x,w), (x,z), (y,w), (y,z).
                        #   In the cycle: (x,w) yes, (y,w) yes, (y,z) yes, (x,z) yes.
                        #   So yes, a 4-cycle in G implies a K2,2 in G.
                        
                        # So we just need to check if any pair (x,y) in verts has >= 2 common neighbors in E_c[verts].
                        
                        # To speed this up:
                        # For each x in verts:
                        #   For each y in verts, y > x:
                        #     If (x,y) is NOT an edge in E_c? 
                        #     Wait, for the common neighbor logic, (x,y) doesn't need to be an edge.
                        #     The common neighbors are w,z such that (x,w),(y,w) and (x,z),(y,z) are edges.
                        #     This forms the K2,2 with parts {x,y} and {w,z}.
                        #     So we don't even need (x,y) to be an edge.
                        
                        # So: For each pair (x,y) in verts, count common neighbors in E_c[verts].
                        # If count >= 2, return True.
                        
                        # Implementation:
                        # Precompute common neighbor counts?
                        # For small verts (size ~ n), we can just do:
                        for i_idx in range(len(verts)):
                            x = verts[i_idx]
                            for j_idx in range(i_idx + 1, len(verts)):
                                y = verts[j_idx]
                                # Common neighbors of x and y in E_c
                                # mask_x = adj[c][x]
                                # mask_y = adj[c][y]
                                # common = mask_x & mask_y
                                # common &= mask_c (to ensure they are in verts)
                                common = adj[c][x] & adj[c][y] & mask_c
                                # Count bits
                                if common.bit_count() >= 2:
                                    return True
            return False

        return not has_mono_k222(n, colors)

    # Try Paley graphs for primes n = 17, 29, 37 (but n<=31)
    # Primes of form 4k+1: 5, 13, 17, 29.
    # 29 is the largest prime <= 31 of form 4k+1.
    
    # Let's test 29.
    if check_paley(29):
        # Construct the colors for 29
        n = 29
        squares = {x * x % n for x in range(1, n)}
        colors = [0 if (j - i) % n in squares else 1 for i in range(n) for j in range(i + 1, n)]
        return {'n': n, 'colors': colors}
    
    # If 29 fails, try 17 (seed)
    if check_paley(17):
        n = 17
        squares = {x * x % n for x in range(1, n)}
        colors = [0 if (j - i) % n in squares else 1 for i in range(n) for j in range(i + 1, n)]
        return {'n': n, 'colors': colors}
    
    # Fallback: Random search for n=18..20?
    # Given time constraints, return the best known valid one.
    # Since 17 is guaranteed valid (seed), return 17 if others fail.
    
    n = 17
    squares = {x * x % n for x in range(1, n)}
    colors = [0 if (j - i) % n in squares else 1 for i in range(n) for j in range(i + 1, n)]
    return {'n': n, 'colors': colors}

# EVOLVE-BLOCK-END
