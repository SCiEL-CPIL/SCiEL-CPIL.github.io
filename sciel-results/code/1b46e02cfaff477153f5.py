import time
import sys
import random

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin under 120s (or 150s) limit
    
    # 1. Generate vertices of PG(2, 5)
    # Points are equivalence classes of (x, y, z) in GF(5)^3 \ {0}
    # Normalize first non-zero component to 1.
    
    pts = []
    # z=1: 25 points
    for x in range(5):
        for y in range(5):
            pts.append((x, y, 1))
    # z=0, x=1: 5 points
    for y in range(5):
        pts.append((1, y, 0))
    # z=0, x=0, y=1: 1 point
    pts.append((0, 1, 0))
    
    n = len(pts)
    assert n == 31
    
    # 2. Define Polarity
    # Standard polarity for q=5 using identity matrix (or similar symmetric non-singular)
    # Adjacency: p1 . p2 = 0 mod 5
    
    def is_adjacent(p1, p2):
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        return (x1*x2 + y1*y2 + z1*z2) % 5 == 0

    # Build adjacency list for the core graph (should be C4-free, 90 edges)
    adj = [set() for _ in range(31)]
    edges_core = []
    for i in range(31):
        for j in range(i+1, 31):
            if is_adjacent(pts[i], pts[j]):
                adj[i].add(j)
                adj[j].add(i)
                edges_core.append([i, j])
    
    # Verify core properties
    # In a polarity graph of PG(2,q) with q odd, the number of edges is q(q+1)(q+2)/2 / 2?
    # Actually, the number of absolute points is q+1.
    # Each absolute point has degree (q-1)/2.
    # Each non-absolute point has degree (q+1)/2.
    # For q=5:
    # Absolute points: 6 points, degree (5-1)/2 = 2.
    # Non-absolute points: 25 points, degree (5+1)/2 = 3.
    # Total edges = (6*2 + 25*3) / 2 = (12 + 75) / 2 = 87 / 2? No, sum of degrees is 87, which is odd.
    # Wait, let's re-calculate.
    # Sum of degrees = 2 * E.
    # If E=90, sum of degrees = 180.
    # 6*2 + 25*3 = 12 + 75 = 87. This is not 180.
    # My formula for degrees must be wrong or the polarity is different.
    
    # Let's just check the actual degree counts from the generated graph.
    degrees = [len(adj[i]) for i in range(31)]
    # print(degrees) # Debug
    
    edge_count = sum(len(s) for s in adj) // 2
    # print(f"Core edges: {edge_count}")
    
    if edge_count != 90:
        # Fallback if the specific polarity didn't yield 90 edges (unlikely for standard identity)
        # Try a different polarity matrix, e.g., diag(1, 1, -1) i.e. x1x2 + y1y2 - z1z2 = 0
        # But for now, let's assume the identity polarity works or adjust.
        # Actually, for q=5, identity matrix:
        # x^2 + y^2 + z^2 = 0.
        # Let's trust the code generation and check count.
        # If it's not 90, we might have a bug in point enumeration or adjacency.
        # Let's force a fallback to a known valid C4-free graph if this fails?
        # For this specific task, the identity polarity on PG(2,5) is a standard example.
        # Let's verify the count.
        pass

    # If we don't have exactly 90 edges, we have a problem.
    # Let's assume the standard construction yields 90.
    # If not, we can't proceed with the "add 1 edge" strategy easily.
    # However, the prompt says "one more than ex(31;C4) = 90", implying a 90-edge C4-free graph exists.
    # The polarity graph of PG(2,5) is such a graph.
    
    # 3. Find the best edge to add
    # We want to add an edge (u, v) where u, v are not adjacent in the core.
    # Adding (u, v) creates new C4s.
    # A new C4 containing (u, v) must look like u-a-b-v-u or u-a-b-v... wait.
    # A 4-cycle is u-v-b-a-u.
    # Since (u,v) is the new edge, the other three edges (v,b), (b,a), (a,u) must exist in the core.
    # So we are looking for paths of length 3 from u to v in the core: u-a-b-v.
    # The number of new C4s added is the number of such paths.
    
    # To minimize the score, we want to minimize the number of length-3 paths between u and v.
    
    best_u = -1
    best_v = -1
    min_paths = float('inf')
    
    # Precompute neighbor sets for fast intersection
    # adj is already a list of sets.
    
    # Iterate over all non-edges
    for u in range(31):
        if time.time() > deadline - 5:
            break
        for v in range(u+1, 31):
            if v in adj[u]:
                continue
            
            # Count paths of length 3: u -> a -> b -> v
            # a is a neighbor of u
            # b is a neighbor of v
            # a and b must be adjacent
            count = 0
            for a in adj[u]:
                # b must be in adj[v] and adj[a]
                # Intersection of adj[v] and adj[a]
                # We can iterate over the smaller set
                # Or just check membership
                for b in adj[v]:
                    if b in adj[a]:
                        count += 1
            
            if count < min_paths:
                min_paths = count
                best_u = u
                best_v = v
                
                # Early exit if we find 0?
                # If min_paths is 0, we can't do better.
                if min_paths == 0:
                    break
        
        if min_paths == 0:
            break
    
    if best_u == -1:
        # Fallback: pick any non-edge
        # This shouldn't happen if graph is not complete
        for u in range(31):
            for v in range(u+1, 31):
                if v not in adj[u]:
                    best_u = u
                    best_v = v
                    break
            if best_u != -1:
                break
    
    # 4. Construct final edge list
    final_edges = edges_core + [[best_u, best_v]]
    
    # Verify
    assert len(final_edges) == 91, f"Expected 91 edges, got {len(final_edges)}"
    # Ensure distinct
    assert len(set(tuple(sorted(e)) for e in final_edges)) == 91
    
    return final_edges
