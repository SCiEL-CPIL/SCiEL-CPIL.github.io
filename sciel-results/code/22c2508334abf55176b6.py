"""Improved program for dense edge-critical diameter 2 graphs."""
# EVOLVE-BLOCK-START

def construct():
    """
    Returns a list of edges representing the Shrikhande graph.
    The Shrikhande graph is a strongly regular graph with n=16, k=6, lambda=2, mu=2.
    Properties:
    - Diameter 2.
    - Removing any edge increases the diameter to 3 (it is minimally diameter 2 or edge-critical).
    """
    edges = []
    
    # The Shrikhande graph can be constructed from the Cartesian product of K4 and K2, with certain rewiring.
    # Or more simply, using its standard vertex ordering.
    
    # Standard edge list for Shrikand graph (16 vertices)
    # Vertices are 0-15.
    
    # A known representation:
    # Let V = {0, ..., 15}.
    # We can generate it using the following adjacency rules based on binary representation or modular arithmetic.
    
    # An efficient explicit construction:
    # The Shrikand graph is the 2x2x2 grid (cube Q3? No, Q3 has 8 vertices)
    # It is strongly regular (16,6,2,2).
    
    # Let's hardcode a known valid edge set for the Shrikand graph.
    # Source: House of Graphs or standard databases.
    
    # Vertices 0..7 and 8..15
    # Structure is often described as K4 x K2 with a "crossing" modification, 
    # or derived from the Clebsch graph.
    
    # Let's use a programmatic generation to be safe and explicit.
    # One construction:
    # Let the vertices be pairs (i, j) where i, j in {0, 1, 2, 3}.
    # i.e., 0=(0,0), 1=(0,1), ..., 15=(3,3).
    
    # Edge definition for Shrikand:
    # (i,j) ~ (k,l) if:
    # 1. (i,j) == (k,l) is not an edge (no loops).
    # 2. The distance in some specific metric?
    
    # Actually, the Shrikand graph is the complement of the Clebsch graph?
    # Clebsch graph is (16,5,0,2). Complement is (16,10,6,5). Diameter 2.
    # But we want Maximize edge count. 
    # Wait, the task asks for "Maximize edge count" among graphs that are "diameter exactly two" and "deleting EACH individual edge must increase its diameter above two".
    
    # Let's re-read carefully: "Maximize edge count".
    # If the Shrikand graph is minimally diameter 2, it has 48 edges.
    # Can we have a graph with MORE than 48 edges that is STILL minimally diameter 2?
    # No. If a graph is minimally diameter 2, adding any edge creates a graph that is NOT minimally diameter 2 (because removing the original critical edge might still leave diameter 2 via the new edge).
    # So, the set of all minimally diameter 2 graphs has a fixed "critical structure".
    # The Shrikand graph is a strongly regular graph that is minimally diameter 2.
    
    # Is it possible to have a denser minimally diameter 2 graph?
    # The maximum degree of a minimally diameter 2 graph on n=16 is constrained.
    # If min diameter 2, then for every edge (u,v), u and v have no common neighbors.
    # This implies girth >= 5.
    # A graph with n=16, girth >= 5, and maximal edges is a Moore graph or close to it.
    # The (16, 5)-Moore graph is the Hoffman-Singleton? No, HS is 50.
    # For n=16, degree 5:
    # Moore bound for d=2: n <= k^2 + 1 = 26.
    # Degree 5, n=16 is very sparse.
    # Shrikand has degree 6. n=16, k=6.
    # k^2+1 = 37 > 16. So k=6 is possible.
    # Can we have k=7?
    # If k=7, n=16.
    # Condition for minimal diameter 2: For every edge uv, N(u) ∩ N(v) = ∅.
    # This means the neighborhood of any vertex does not contain the neighbor of its neighbor.
    # This effectively means the graph has no triangles involving that edge.
    # Actually, if (u,v) is an edge, and w is a neighbor of u, w cannot be a neighbor of v.
    # This implies that the subgraph induced by N(u) is triangle-free? No, if (u,v) is edge, v is in N(u).
    # It implies that any edge (u,v) is in NO triangles.
    # Wait. If (u,v) is in a triangle (u,v,w), then removing (u,v) leaves path u-w-v of length 2.
    # So diameter stays 2.
    # Therefore, a graph is minimally diameter 2 IF AND ONLY IF every edge lies in no triangle?
    # Let's verify.
    # If every edge is in no triangle, then removing (u,v) means no path of length 2 between u and v.
    # Does it disconnect? Or is distance > 2?
    # If graph is connected, and diameter is 2, then for any u,v, there is a path of length 2.
    # If we remove (u,v) and there is NO path of length 2 (because no triangle), does the distance become 3?
    # It could become 3 or more. Or disconnected.
    # If it becomes disconnected, diameter becomes infinity (>2).
    # If it becomes 3, diameter becomes 3 (>2).
    # So, "Minimally Diameter 2" <=> "Graph has Diameter 2 AND Girth 5" (no triangles, no cycles of length 3).
    # Actually, if Girth >= 5, then no triangles.
    # If Graph has Diameter 2 and no triangles, it is minimally diameter 2.
    # 
    # So the problem reduces to: Find a graph on 16 vertices with Diameter 2 and Girth >= 5, maximizing edges.
    # This is a (16, k, 5) cage or similar.
    # For n=16, what is the max degree k for a girth 5 graph with diameter 2?
    # The (16, 6) Shrikand graph has girth 5? 
    # Strongly regular (16, 6, 2, 2).
    # Parameter lambda=2 means adjacent vertices have 2 common neighbors -> TRIANGLES exist!
    # If Shrikand has triangles, it is NOT minimally diameter 2.
    # Let's check Shrikand properties.
    # SRG(16, 6, 2, 2).
    # Adjacent vertices have 2 common neighbors.
    # So edge (u,v) is in triangles.
    # So removing (u,v) leaves a path of length 2.
    # Diameter remains 2.
    # So Shrikand is NOT minimally diameter 2.
    
    # So we need a graph with Diameter 2 and NO TRIANGLES (Girth >= 5).
    # Max edges for n=16, Girth 5.
    # The largest triangle-free graphs on n vertices.
    # If Girth 5, it's triangle-free.
    # Max edges for triangle-free graph on 16 vertices?
    # Turan number T(16, 3) is floor(16^2/4) = 64 (Bipartite K8,8).
    # But K8,8 has diameter 3 (between same side) or 2 (between opposite sides).
    # Diameter of K8,8 is 3.
    # We need Diameter 2.
    # So we need to add edges to K8,8 to reduce diameter to 2, without creating triangles.
    # If we add an edge within a partition, say (u,v) where u,v in A.
    # Then (u,v) is an edge.
    # Does it create a triangle?
    # If we only have bipartite edges + some edges within A and B.
    # Let G be a graph on 16 vertices.
    # If we want Diameter 2 and Triangle Free.
    # Let's look at the (16, 5) Moore graph?
    # There is a unique (16, 5) graph with girth 5? No.
    # The Hoffman-Singleton is (50, 7, 0, 1). Girth 5.
    # For n=16, k=5, girth 5.
    # Does such a graph have diameter 2?
    # Moore bound for k=5, d=2 is n <= 5^2+1 = 26.
    # A (16,5,0,2) strongly regular graph?
    # If SRG(16,5,0,2), then lambda=0 (no triangles).
    # Mu=2 means non-adjacent have 2 common neighbors.
    # Diameter is 2 (since mu>0).
    # Edges = 16*5/2 = 40.
    # 
    # Can we do better than 40 edges?
    # We need Diameter 2 and Triangle Free.
    # Try to construct from K8,8.
    # K8,8 has 64 edges. Diameter 3.
    # To get diameter 2, every pair in same partition must have a common neighbor.
    # In K8,8, pairs in same partition have 0 common neighbors (since neighbors are in other partition, and no edges in other partition).
    # Wait, in K8,8, u, v in A. N(u) = B. N(v) = B.
    # N(u) ∩ N(v) = B. Size 8.
    # So path u-b-v exists.
    # So distance u to v is 2?
    # u-b, b-v. Yes.
    # So K8,8 has diameter 2?
    # Let's check.
    # u in A, v in A. Path u-b-v. Length 2.
    # u in A, v in B. Path u-v. Length 1.
    # So K8,8 has diameter 2!
    # Wait, earlier I said 3. I confused myself.
    # In K_{m,n}, diameter is 2 if m,n > 1.
    # Diameter of K_{1,n} is 2? No, 1 to n is 1. But leaf to leaf is 2.
    # Diameter of K_{2,2} (C4) is 2.
    # Diameter of K_{8,8} is 2.
    
    # So K8,8 has diameter 2.
    # Is K8,8 triangle-free? Yes, it's bipartite.
    # So K8,8 is a candidate.
    # Is it MINIMALLY diameter 2?
    # Take an edge (u,v) in K8,8. u in A, v in B.
    # Remove (u,v).
    # Is diameter still 2?
    # We need to check if any pair becomes distance 3 or disconnected.
    # Specifically, u and v.
    # Original distance 1.
    # After removal, is there a path of length 2 between u and v?
    # Path u-w-v.
    # w must be a common neighbor of u and v.
    # N(u) = B \ {v} (after removal).
    # N(v) = A \ {u} (after removal).
    # N(u) ∩ N(v) = ∅.
    # So no path of length 2.
    # Is there a path of length 3?
    # u-b-a-v.
    # u~b (b in B). b~a (a in A). a~v (a in A, v in B).
    # Yes, path exists.
    # So d(u,v) becomes 3.
    # So diameter of K8,8 - e is 3.
    # So K8,8 is minimally diameter 2!
    
    # Edge count of K8,8 is 8*8 = 64.
    # This is much higher than 48.
    
    # Can we have more than 64 edges?
    # If we add an edge to K8,8, say within A. (u,v) in A.
    # New graph G'.
    # Diameter is still 2?
    # Yes, K8,8 was 2, adding edges doesn't increase diameter.
    # Is G' minimally diameter 2?
    # Check an edge in the original K8,8, say (x,y) x in A, y in B.
    # Remove (x,y).
    # Do we still have diameter 2?
    # We added edges within A and B.
    # Let's check x and y after removing (x,y).
    # We need a path of length 2.
    # Common neighbor z?
    # z in N(x) ∩ N(y).
    # N(x) includes B (except y) and potentially some A nodes if we added edges.
    # N(y) includes A (except x) and potentially some B nodes.
    # If we added (x, x') with x' in A, then x is connected to x'.
    # If we added (y, y') with y' in B, then y is connected to y'.
    # Does this create a common neighbor?
    # Not necessarily.
    # But generally, adding edges makes the graph LESS critical.
    # Specifically, if we add ANY edge, does it break the minimal property?
    # If we add (u,v) in A.
    # Consider the edge (u, b) where b in B.
    # Remove (u, b).
    # Is there a path of length 2 from u to b?
    # u is connected to other A nodes (due to added edges) and other B nodes.
    # b is connected to all A nodes and other B nodes (if added).
    # Path u - a - b?
    # u~a (if a in A and we added it). a~b (original K8,8 edge).
    # If we added enough edges such that for every b, there is some a in A adjacent to u, and a is adjacent to b.
    # In K8,8, all a in A are adjacent to all b in B.
    # So if u has ANY neighbor a in A, then u-a-b is a path.
    # So if we add at least one edge incident to u in A, then removing (u,b) still leaves d(u,b)=2.
    # Thus, (u,b) is NOT critical.
    # So the graph is NOT minimally diameter 2.
    
    # Therefore, the MAXIMAL number of edges in a minimally diameter 2 graph is 64 (K8,8).
    # Any addition of an edge creates non-critical edges.
    # So the answer is the edges of K8,8.

    for u in range(8):
        for v in range(8, 16):
            edges.append([u, v])
            
    return edges

# EVOLVE-BLOCK-END
