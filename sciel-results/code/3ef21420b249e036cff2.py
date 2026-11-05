import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Fallback solution: 8 disjoint K4s (n=32)
    def get_fallback():
        n = 32
        edges = []
        for b in range(8):
            base = b * 4
            for i in range(4):
                for j in range(i + 1, 4):
                    edges.append([base + i, base + j])
        return {'n': n, 'edges': edges}

    # Try to extend to n=33
    # Start with n=32 graph
    n_start = 32
    adj = [[0] * n_start for _ in range(n_start)]
    edges_start = []
    for b in range(8):
        base = b * 4
        for i in range(4):
            for j in range(i + 1, 4):
                u = base + i
                v = base + j
                adj[u][v] = 1
                adj[v][u] = 1
                edges_start.append([u, v])
                
    # Precompute distances in G_start (n=32)
    # We need to know which pairs (u, w) have distance 3 or 4.
    # Since G_start is disjoint union of K4s:
    # Within a K4, distance is 1.
    # Between different K4s, distance is infinity (disconnected).
    # Wait, if the graph is disconnected, distances are infinite.
    # The condition "no path of length 3" is vacuously true for disconnected components?
    # If u and w are in different components, there is no path. So no C5 can be formed involving u and w from different components?
    # Let's check.
    # If u in Comp1, w in Comp2.
    # Path u-...-w doesn't exist.
    # So we can connect v to vertices in different components freely regarding C5 formation?
    # Cycle v-u1-...-u4-v.
    # If u1 in Comp1 and u4 in Comp2, the path u1-...-u4 must cross components.
    # But G_start is disjoint union. No edges between components.
    # So no path exists between different components.
    # Therefore, we can pick neighbors from different components without creating a C5 via a path in G_start.
    # However, if we pick two neighbors u, w from the SAME component (a K4), their distance is 1.
    # Dist 1 is allowed (creates C3).
    # So, we can pick ANY set of neighbors from the 32 vertices?
    # Let's verify.
    # If we pick u, w from same K4, dist=1. No C5.
    # If we pick u from K4_1, w from K4_2. No path. No C5.
    # So the C5 constraint is trivially satisfied for ANY N(v) in this specific graph structure!
    # Because the only paths are of length 1 (within K4) or none (between K4s).
    # Length 3 or 4 paths do not exist in 8 disjoint K4s.
    
    # So the only constraint is the independence number.
    # We need to choose N(v) such that alpha(G \ N(v)) <= 7.
    # G is 8 disjoint K4s.
    # An independent set in G can pick at most 1 vertex from each K4.
    # Max independent set size is 8 (one from each).
    # We remove N(v) from G.
    # Let k_i be the number of neighbors of v in the i-th K4.
    # The remaining vertices in the i-th K4 form a subgraph.
    # If k_i = 0, the K4 remains intact. Max indep in it is 1.
    # If k_i = 1, 3 vertices remain (K3). Max indep is 1.
    # If k_i = 2, 2 vertices remain (K2). Max indep is 1.
    # If k_i = 3, 1 vertex remains. Max indep is 1.
    # If k_i = 4, 0 vertices remain. Max indep is 0.
    # So, for each K4, the contribution to alpha(G \ N(v)) is 1 if k_i < 4, and 0 if k_i = 4.
    # Total alpha = sum over i of (1 if k_i < 4 else 0).
    # We need this sum <= 7.
    # There are 8 K4s.
    # If we don't saturate any K4 (k_i < 4 for all i), sum is 8. Too high.
    # So we must saturate at least one K4 (k_i = 4 for some i).
    # If we saturate exactly one K4, sum is 7. Valid.
    # If we saturate two K4s, sum is 6. Valid.
    
    # So, to add vertex v, we must connect it to ALL 4 vertices of at least one K4.
    # Let's try to connect v to all 4 vertices of K4_0.
    # Then N(v) = {0, 1, 2, 3}.
    # Check C5: 
    # Neighbors are 0,1,2,3.
    # Distances between them are 1. No C5.
    # Check Alpha:
    # K4_0 is empty in G \ N(v).
    # K4_1..K4_7 are intact.
    # Max indep = 0 + 1*7 = 7.
    # Total alpha of new graph = max(7, 1 + 7) = 8?
    # Wait. Alpha(G_new) = max(Alpha(G \ N(v)), 1 + Alpha(G \ N(v)))? No.
    # Alpha(G_new) is the max independent set in G_new.
    # An independent set can either include v or not.
    # If it doesn't include v, it's an independent set in G. Max size 8.
    # If it includes v, it can include no neighbors of v.
    # So it's an independent set in G \ N(v).
    # Max size of indep set in G \ N(v) is 7.
    # So an indep set containing v has size at most 1 + 7 = 8.
    # So Alpha(G_new) = max(8, 8) = 8.
    # This is valid (alpha < 9).
    
    # So we can construct a valid graph on n=33!
    # Edges: All edges of 8 disjoint K4s, PLUS edges from new vertex 32 to 0,1,2,3.
    
    # Let's verify this logic.
    # Graph G_33:
    # Vertices 0..31: 8 disjoint K4s.
    # Vertex 32 connected to 0,1,2,3.
    # No other edges.
    
    # Check C5:
    # Any C5 must use vertex 32 or not.
    # If not, it's in G_32, which has no C5.
    # If yes, it uses 32.
    # Cycle: 32 - u - ... - w - 32.
    # u, w in {0,1,2,3}.
    # Path u-...-w in G_32.
    # u, w are in K4_0.
    # In K4_0, distance between any two is 1.
    # So path length is 1.
    # Cycle length: 32-u-w-32 is length 3 (Triangle).
    # Can we have a longer path?
    # No, because there are no other neighbors of 32.
    # So any cycle involving 32 must go 32-u-w-32.
    # This is a triangle.
    # So no C5.
    
    # Check Alpha:
    # We calculated it as 8.
    # So G_33 is valid.
    
    # Can we go to n=34?
    # Add vertex 33.
    # We need to connect 33 to some vertices.
    # Let N(33) be its neighbors.
    # We need alpha(G_33 \ N(33)) <= 7.
    # Current G_33 has alpha 8.
    # If we don't touch the structure that allows alpha 8, we might fail.
    # The alpha 8 comes from picking one vertex from each of the 7 intact K4s (K4_1..K4_7) and one vertex from... wait.
    # In G_33, K4_0 is "saturated" by 32.
    # But 32 is a vertex.
    # Independent set in G_33:
    # Option A: Include 32. Then can't include 0,1,2,3.
    # Can include one from each K4_1..K4_7. Total 1 + 7 = 8.
    # Option B: Exclude 32.
    # Can include one from each K4_0..K4_7. Total 8.
    # So alpha is 8.
    
    # To add 33, we need to reduce alpha of the rest to <= 7.
    # If we connect 33 to all 4 vertices of K4_1.
    # Then K4_1 is saturated by 33.
    # Alpha of G_33 \ N(33):
    # K4_0: intact (unless 33 connects to it, which it doesn't in this hypothesis).
    # K4_1: empty.
    # K4_2..K4_7: intact.
    # Max indep in G_33 \ N(33) is 1 (from K4_0) + 0 (K4_1) + 1*6 (K4_2..7) = 7.
    # So alpha(G_34) <= 1 + 7 = 8.
    # Also, independent sets not including 33 are in G_33. Alpha(G_33)=8.
    # So Alpha(G_34) = 8.
    # Valid!
    
    # Can we go to n=35?
    # Add vertex 34.
    # Connect 34 to all 4 vertices of K4_2.
    # Then K4_2 is saturated.
    # G_34 \ N(34):
    # K4_0: intact.
    # K4_1: saturated by 33.
    # K4_2: empty.
    # K4_3..K4_7: intact.
    # Max indep = 1 + 0 + 0 + 1*5 = 6.
    # Alpha(G_35) <= 1 + 6 = 7.
    # Independent sets not including 34 are in G_34. Alpha(G_34)=8.
    # So Alpha(G_35) = 8.
    # Valid!
    
    # So we can construct a graph on n=35!
    # Structure:
    # 8 disjoint K4s (vertices 0..31).
    # Vertex 32 connected to K4_0 (0,1,2,3).
    # Vertex 33 connected to K4_1 (4,5,6,7).
    # Vertex 34 connected to K4_2 (8,9,10,11).
    
    # Let's double check C5 for n=35.
    # Any C5 must involve 32, 33, or 34?
    # If not, it's in G_32, no C5.
    # If involves 32: Neighbors are 0,1,2,3.
    # Path between neighbors in G_32? Dist 1. Cycle length 3.
    # If involves 33: Neighbors 4,5,6,7.
    # Path between neighbors in G_32? Dist 1. Cycle length 3.
    # Note: 32 and 33 are not connected.
    # Can we have a cycle 32 - u - ... - w - 33 - ...?
    # No, 32 and 33 are not connected, so they can't be adjacent in a cycle.
    # Can they be part of a longer cycle?
    # 32 - u - x - y - w - 33 ...
    # This would require a path from u to w in the rest of the graph?
    # But 32 and 33 are not connected to each other.
    # A cycle containing both 32 and 33 would look like:
    # 32 - a - ... - b - 33 - c - ... - d - 32.
    # This requires edges (32,a), (a,...,b), (b,33), (33,c), (c,...,d), (d,32).
    # Length >= 6.
    # We only care about C5.
    # A C5 containing 32 and 33 would need 3 other vertices.
    # 32 - u - v - w - 33 - 32? No, 33-32 edge doesn't exist.
    # So 32 and 33 cannot be in the same C5 because they are not adjacent, and there isn't enough room for a path between them of length 3 or 4 that closes the cycle with length 5 total?
    # Cycle: 32 - u - v - w - 33 - 32 (Invalid, no edge 33-32).
    # Cycle: 32 - u - v - 33 - x - 32?
    # Edges: (32,u), (u,v), (v,33), (33,x), (x,32).
    # This is a C5.
    # Does this exist?
    # u is neighbor of 32 -> u in {0,1,2,3}.
    # x is neighbor of 32 -> x in {0,1,2,3}.
    # v is neighbor of 33 -> v in {4,5,6,7}.
    # We need edge (u,v) and (v,x)? No, (u,v) and (v,33) and (33,x).
    # Wait, the path is u-v.
    # u in K4_0, v in K4_1.
    # There is NO edge between K4_0 and K4_1.
    # So (u,v) does not exist.
    # Thus, no such C5.
    
    # Generally, any C5 involving the new vertices 32,33,34 would require paths between their neighbor sets.
    # Neighbor sets are disjoint and in different components of G_32.
    # No edges between components.
    # So no paths of length >= 1 between neighbors of different new vertices.
    # So no C5 involving multiple new vertices.
    # C5 involving only one new vertex (e.g., 32) is a triangle as shown.
    
    # So n=35 is valid!
    
    # Construct edges for n=35.
    n = 35
    edges = []
    # 8 K4s
    for b in range(8):
        base = b * 4
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
                
    # Vertex 32 connected to K4_0
    for i in range(4):
        edges.append([32, i])
        
    # Vertex 33 connected to K4_1
    for i in range(4):
        edges.append([33, 4 + i])
        
    # Vertex 34 connected to K4_2
    for i in range(4):
        edges.append([34, 8 + i])
        
    return {'n': n, 'edges': edges}

# The logic holds. We return n=35 directly.
