import time
import random
import sys
import itertools

def construct():
    start_time = time.time()
    deadline = 115.0  # 5s safety margin
    
    # 1. Generate Paley 29
    n_29 = 29
    squares_29 = {x * x % 29 for x in range(1, 29)}
    colors_29 = [0 if (j - i) % 29 in squares_29 else 1 for i in range(n_29) for j in range(i + 1, n_29)]
    
    # Build adjacency sets for the fixed part (0..28)
    # E0: set of edges (i,j) with i<j and color 0
    # E1: set of edges (i,j) with i<j and color 1
    E0 = set()
    E1 = set()
    idx = 0
    for i in range(29):
        for j in range(i + 1, 29):
            if colors_29[idx] == 0:
                E0.add((i, j))
            else:
                E1.add((i, j))
            idx += 1
            
    # Precompute 6-tuples containing vertex 29
    # The tuple is (a,b,c,d,e,29) with a<b<c<d<e<29
    # We need to check if this 6-set forms a K2,2,2 in color 0 or 1.
    # The structure of K2,2,2 is 3 pairs.
    # Since 29 is one vertex, it must be in one of the pairs.
    # Let the pairs be P1, P2, P3.
    # One pair contains 29. Say P1 = {29, x}.
    # The other two pairs P2, P3 are chosen from the remaining 5 vertices {a,b,c,d,e}.
    # There are C(5,2) * C(3,2) / 2? No.
    # We partition the 5 vertices into two pairs and one singleton? No, K2,2,2 has 6 vertices.
    # If 29 is in the graph, it belongs to one of the 3 parts.
    # Let the parts be A, B, C.
    # Case 1: 29 is in A. A = {29, a}. B = {b, c}. C = {d, e}.
    # The edges required are between A-B, A-C, B-C.
    # Edges:
    # 29-b, 29-c, a-b, a-c
    # 29-d, 29-e, a-d, a-e
    # b-d, b-e, c-d, c-e
    # Total 12 edges.
    
    # We can precompute the "fixed" edges among {a,b,c,d,e}.
    # For a given assignment of edges from 29 to {0..28}, we can check violations.
    
    # Let's generate all possible partitions of 5 vertices into 2 pairs and 1 singleton?
    # No, the 6 vertices are {29, v1, v2, v3, v4, v5}.
    # 29 is in a pair with one of v1..v5.
    # Let 29 be paired with v1.
    # The remaining 4 vertices v2, v3, v4, v5 are partitioned into two pairs.
    # There are 3 ways to partition 4 items into 2 pairs.
    # So for each choice of v1 (5 choices) and each partition of the rest (3 choices), we have a candidate K2,2,2 structure.
    # Total structures to check per 6-tuple: 5 * 3 = 15?
    # Actually, the set of 6 vertices is fixed. The role of 29 is fixed (it's the new vertex).
    # But which vertex is paired with 29? Any of the other 5.
    # So yes, for a fixed set of 6 vertices including 29, there are 15 ways to form a K2,2,2 where 29 is in one part.
    # Wait, if 29 is in part A, A={29, v1}. B and C are pairs from remaining 4.
    # This covers all K2,2,2s containing 29?
    # Yes, because in any K2,2,2, 29 is in exactly one part, and that part has exactly one other vertex.
    
    # So, for each 5-tuple (v1, v2, v3, v4, v5) from 0..28:
    # For each v1 in 5:
    #   For each partition of {v2,v3,v4,v5} into 2 pairs (b,c) and (d,e):
    #     Check edges:
    #     (29, v1) is the edge inside A? No, (29, v1) is NOT required to be color c.
    #     The edges required are BETWEEN parts.
    #     A={29, v1}, B={b, c}, C={d, e}.
    #     Edges A-B: (29,b), (29,c), (v1,b), (v1,c)
    #     Edges A-C: (29,d), (29,e), (v1,d), (v1,e)
    #     Edges B-C: (b,d), (b,e), (c,d), (c,e)
    
    # We can precompute the "internal" edges among the 5 vertices.
    # Let's just iterate.
    
    tuples_5 = []
    for v1 in range(29):
        for v2 in range(v1+1, 29):
            for v3 in range(v2+1, 29):
                for v4 in range(v3+1, 29):
                    for v5 in range(v4+1, 29):
                        tuples_5.append((v1, v2, v3, v4, v5))
                        
    # There are C(29,5) = 118755 tuples.
    # For each, we check 15 configurations.
    # 1.78M checks per evaluation.
    
    def count_violations(edges_29):
        # edges_29 is a list of 29 ints (0 or 1) representing color of (i, 29)
        # We need to check if there is a K2,2,2 in color 0 or 1.
        # We only check those involving 29.
        count = 0
        for (v1, v2, v3, v4, v5) in tuples_5:
            verts = [v1, v2, v3, v4, v5]
            # Try all 5 choices for the partner of 29
            for i in range(5):
                partner = verts[i]
                remaining = [v for j, v in enumerate(verts) if j != i]
                # Partition remaining 4 into 2 pairs
                # (r0, r1) and (r2, r3)
                # 3 partitions:
                # 1: (r0, r1), (r2, r3)
                # 2: (r0, r2), (r1, r3)
                # 3: (r0, r3), (r1, r2)
                
                # Partition 1
                b, c = remaining[0], remaining[1]
                d, e = remaining[2], remaining[3]
                
                # Check Color 0
                # Edges: (29,b), (29,c), (partner,b), (partner,c),
                #        (29,d), (29,e), (partner,d), (partner,e),
                #        (b,d), (b,e), (c,d), (c,e)
                
                # Helper to check if edge (u,v) is in E0 or E1
                # Note: (29, x) is in edges_29[x]
                # (u, v) for u,v < 29 is in E0 or E1
                
                # We can precompute a matrix for the 29x29 part
                # But sets are fine.
                
                # Let's inline the checks for speed
                
                # Edges involving 29
                # (29, b) -> edges_29[b]
                # (29, c) -> edges_29[c]
                # (29, d) -> edges_29[d]
                # (29, e) -> edges_29[e]
                
                # Edges involving partner
                # (partner, b), (partner, c), (partner, d), (partner, e)
                
                # Edges between remaining
                # (b,d), (b,e), (c,d), (c,e)
                
                # We check if ALL are 0 or ALL are 1.
                
                # Get colors
                c29_b = edges_29[b]
                c29_c = edges_29[c]
                c29_d = edges_29[d]
                c29_e = edges_29[e]
                
                # Check if all 4 are 0
                if c29_b == 0 and c29_c == 0 and c29_d == 0 and c29_e == 0:
                    # Check partner edges
                    if (partner, b) in E0 and (partner, c) in E0 and (partner, d) in E0 and (partner, e) in E0:
                        # Check remaining edges
                        if (b, d) in E0 and (b, e) in E0 and (c, d) in E0 and (c, e) in E0:
                            count += 1
                            
                # Check if all 4 are 1
                elif c29_b == 1 and c29_c == 1 and c29_d == 1 and c29_e == 1:
                    if (partner, b) in E1 and (partner, c) in E1 and (partner, d) in E1 and (partner, e) in E1:
                        if (b, d) in E1 and (b, e) in E1 and (c, d) in E1 and (c, e) in E1:
                            count += 1
                            
                # Partition 2
                b, c = remaining[0], remaining[2]
                d, e = remaining[1], remaining[3]
                
                c29_b = edges_29[b]
                c29_c = edges_29[c]
                c29_d = edges_29[d]
                c29_e = edges_29[e]
                
                if c29_b == 0 and c29_c == 0 and c29_d == 0 and c29_e == 0:
                    if (partner, b) in E0 and (partner, c) in E0 and (partner, d) in E0 and (partner, e) in E0:
                        if (b, d) in E0 and (b, e) in E0 and (c, d) in E0 and (c, e) in E0:
                            count += 1
                            
                elif c29_b == 1 and c29_c == 1 and c29_d == 1 and c29_e == 1:
                    if (partner, b) in E1 and (partner, c) in E1 and (partner, d) in E1 and (partner, e) in E1:
                        if (b, d) in E1 and (b, e) in E1 and (c, d) in E1 and (c, e) in E1:
                            count += 1

                # Partition 3
                b, c = remaining[0], remaining[3]
                d, e = remaining[1], remaining[2]
                
                c29_b = edges_29[b]
                c29_c = edges_29[c]
                c29_d = edges_29[d]
                c29_e = edges_29[e]
                
                if c29_b == 0 and c29_c == 0 and c29_d == 0 and c29_e == 0:
                    if (partner, b) in E0 and (partner, c) in E0 and (partner, d) in E0 and (partner, e) in E0:
                        if (b, d) in E0 and (b, e) in E0 and (c, d) in E0 and (c, e) in E0:
                            count += 1
                            
                elif c29_b == 1 and c29_c == 1 and c29_d == 1 and c29_e == 1:
                    if (partner, b) in E1 and (partner, c) in E1 and (partner, d) in E1 and (partner, e) in E1:
                        if (b, d) in E1 and (b, e) in E1 and (c, d) in E1 and (c, e) in E1:
                            count += 1

        return count

    # Simulated Annealing on the 29 edges
    # Initial state: random
    current_edges = [random.randint(0, 1) for _ in range(29)]
    current_energy = count_violations(current_edges)
    
    best_edges = current_edges[:]
    best_energy = current_energy
    
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.99
    
    while time.time() < deadline - 2.0 and T > T_min:
        # Flip a random edge
        idx = random.randint(0, 28)
        current_edges[idx] = 1 - current_edges[idx]
        
        new_energy = count_violations(current_edges)
        delta = new_energy - current_energy
        
        if delta < 0 or random.random() < pow(2.718281828, -delta / T):
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_edges = current_edges[:]
                if best_energy == 0:
                    # Found valid coloring
                    # Construct full color list
                    full_colors = colors_29[:]
                    # Append edges to 29
                    for i in range(29):
                        full_colors.append(best_edges[i])
                    return {'n': 30, 'colors': full_colors}
        else:
            current_edges[idx] = 1 - current_edges[idx]
            
        T *= cooling_rate
        
    # If not found, return Paley 29
    return {'n': 29, 'colors': colors_29}
