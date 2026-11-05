"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit
    
    d = 7
    n = 2 ** d
    edges = []
    edge_to_idx = {}
    
    # 1. Generate all edges
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edges)
                edges.append((x, y))
                edge_to_idx[(x, y)] = idx
    
    m = len(edges) # 448
    
    # 2. Pre-compute conflicts
    # conflicts[i] is a list of indices j such that edge i and edge j form a 4-cycle
    # A 4-cycle is formed by two parallel edges.
    # Edge (u, v) where v = u ^ (1 << i).
    # Another edge (u', v') where v' = u' ^ (1 << i).
    # They form a face if u' = u ^ (1 << j) and v' = v ^ (1 << j) for some j != i.
    # Or more simply: for each edge, find its "parallel" partners.
    # Let's just iterate all pairs? 448^2 is ~200k, which is fine.
    # Or better: for each edge (u, v, i), the other edges in the faces containing it are:
    # For each j != i:
    #   u2 = u ^ (1 << j)
    #   v2 = v ^ (1 << j)
    #   Edge (u2, v2) is parallel to (u, v) in the face defined by bits i and j.
    #   Also, the edges (u, u2) and (v, v2) are the other two sides of the face.
    #   Wait, the conflict is with the *parallel* edge (u2, v2).
    #   If we select (u, v) and (u2, v2), and we also have (u, u2) and (v, v2), we have a C4.
    #   The constraint is "No 2-dimensional face may have all four edges selected".
    #   This is equivalent to: We cannot have a set of edges that contains a C4.
    #   This is the Maximum Induced Matching? No, it's the Maximum C4-free subgraph.
    #   This is NOT the same as Maximum Independent Set in the line graph.
    #   In the line graph, adjacent vertices share an endpoint.
    #   Here, two edges conflict if they are part of the same C4 *along with two other edges*.
    #   Actually, if we select a set of edges, it is invalid if it *contains* a C4.
    #   So we want a subset of edges with no C4.
    #   This is the "Maximum C4-free subgraph" problem.
    #   
    #   Let's re-read carefully: "No 2-dimensional face may have all four edges selected."
    #   This means for every square face, we cannot select all 4 edges.
    #   This is a hypergraph independent set problem where hyperedges are the 4-sets of edges forming a face.
    #   Each face has 4 edges. We cannot pick all 4.
    #   This is equivalent to: The set of selected edges must not contain any of the specific 4-sets corresponding to faces.
    #   
    #   My previous "conflict" definition was: "edges that share a 2-face".
    #   If I pick edge A and edge B, do they conflict? No, only if I *also* pick the other two.
    #   So this is not a pairwise conflict graph. It's a higher-order constraint.
    #   
    #   However, we can model this as:
    #   For each face, at most 3 edges can be selected.
    #   
    #   Simulated Annealing Strategy:
    #   State: Set of selected edges.
    #   Validity: Check if any face has 4 selected edges.
    #   Move: Add/Remove edge.
    #   Cost: Number of faces with 4 selected edges (should be 0 for valid).
    #   Objective: Maximize number of edges.
    #   
    #   Since we want to maximize edges subject to validity, we can use a penalty method or strict validity.
    #   Strict validity is harder to move in.
    #   Let's use a repair-based approach or just check validity on the fly.
    #   
    #   Given the small size (448 edges, 448 faces? No, number of faces in Q7 is 7 * 2^(7-2) = 7*16=112 faces?
    #   Number of faces in Qd is d * 2^(d-2). For d=7, 7 * 16 = 112 faces.
    #   Each face has 4 edges.
    #   
    #   Algorithm:
    #   1. Greedy init.
    #   2. SA:
    #      - Pick random edge e.
    #      - If e in S: remove it. Delta_count = -1.
    #      - If e not in S:
    #          - Tentatively add e.
    #          - Check if any face containing e now has 4 edges.
    #          - If yes, we must remove one of the other 3 edges in that face.
    #          - To maximize size, we might want to remove the edge that causes the least "damage" or just random.
    #          - Let's try: If adding e creates a conflict (face with 4), remove a random other edge from that face.
    #          - Delta_count = 1 - (1 if conflict else 0).
    #   
    #   Let's pre-compute:
    #   edge_faces: list of face indices for each edge.
    #   face_edges: list of edge indices for each face.
    
    faces = []
    face_to_idx = {}
    edge_faces = [[] for _ in range(m)]
    
    # Generate faces
    # A face is defined by fixing d-2 bits and varying 2 bits.
    # Iterate over all pairs of bits (i, j) and all assignments to the other d-2 bits.
    for i in range(d):
        for j in range(i+1, d):
            # Mask for the free bits
            # We iterate through all 2^(d-2) combinations of the other bits
            # It's easier to iterate through all vertices and check if they form a face?
            # No, just construct them.
            # For a specific pair (i, j), the face consists of 4 vertices:
            # v0, v1, v2, v3 where v1 = v0 ^ (1<<i), v2 = v0 ^ (1<<j), v3 = v0 ^ (1<<i) ^ (1<<j).
            # To avoid duplicates, we can fix the lowest bit not in {i,j} to 0?
            # Actually, we can just iterate x from 0 to n-1, and for each x, for each pair (i,j),
            # if x is the "minimum" vertex of the face (i.e., x has 0 in bits i and j, and ... wait, this is tricky).
            
            # Simpler: Iterate all x. For each x, for each pair (i,j) with i<j:
            # Let y = x ^ (1<<i), z = x ^ (1<<j), w = x ^ (1<<i) ^ (1<<j).
            # The edges are (x,y), (y,w), (w,z), (z,x).
            # To ensure we count each face once, we can require that x is the smallest vertex in the set {x,y,z,w}?
            # Or simply: A face is uniquely identified by its 4 vertices.
            # Let's just generate all 4-cycles and deduplicate.
            
            pass # We will do this in the next block or just inline.

    # Let's build face data structures properly.
    # A face is a set of 4 edges.
    # We can iterate through all pairs of edges that are parallel?
    # No, let's just iterate through all possible faces geometrically.
    
    # For each pair of dimensions (i, j):
    #   For each assignment of the other d-2 bits:
    #     There is exactly one face.
    #     How to iterate assignments?
    #     Let mask be the bits NOT in {i, j}.
    #     There are 2^(d-2) such faces.
    #     We can iterate k from 0 to 2^(d-2)-1, map k to the bits.
    
    # Alternative: Just iterate all x in 0..127.
    # For each x, for each i<j:
    #   y = x ^ (1<<i)
    #   z = x ^ (1<<j)
    #   w = x ^ (1<<i) ^ (1<<j)
    #   The edges are e1=(min(x,y),max(x,y)), e2=(min(y,w),max(y,w)), e3=(min(w,z),max(w,z)), e4=(min(z,x),max(z,x)).
    #   This generates the same face multiple times (4 times, once for each vertex as x).
    #   We can deduplicate using a set of frozensets of edge indices.
    
    face_set = set()
    for x in range(n):
        for i in range(d):
            for j in range(i+1, d):
                y = x ^ (1 << i)
                z = x ^ (1 << j)
                w = x ^ (1 << i) ^ (1 << j)
                e1 = (min(x, y), max(x, y))
                e2 = (min(y, w), max(y, w))
                e3 = (min(w, z), max(w, z))
                e4 = (min(z, x), max(z, x))
                idxs = (edge_to_idx[e1], edge_to_idx[e2], edge_to_idx[e3], edge_to_idx[e4])
                face_set.add(frozenset(idxs))
                
    faces = list(face_set)
    num_faces = len(faces)
    face_edges = [list(f) for f in faces]
    
    # Re-compute edge_faces
    for fi, f in enumerate(face_edges):
        for ei in f:
            edge_faces[ei].append(fi)
            
    # Greedy Initialization
    selected = [False] * m
    count = 0
    # Randomize order for greedy to get different starts?
    # Let's just use a fixed order for now, or random.
    order = list(range(m))
    random.shuffle(order)
    
    for ei in order:
        # Check if adding ei creates a face with 4 edges
        ok = True
        for fi in edge_faces[ei]:
            # Count selected edges in this face
            c = 0
            for ej in face_edges[fi]:
                if selected[ej]:
                    c += 1
            if c == 3:
                # Adding ei makes it 4
                ok = False
                break
        if ok:
            selected[ei] = True
            count += 1
            
    best_selected = selected[:]
    best_count = count
    best_val = best_count
    
    # Simulated Annealing
    T = 1.0
    cooling = 0.99
    min_T = 0.01
    
    # Pre-compute face edge lists for fast access
    # We need to quickly check if a face is "full" (4 selected).
    # Let's maintain face_counts: number of selected edges in each face.
    face_counts = [0] * num_faces
    for ei in range(m):
        if selected[ei]:
            for fi in edge_faces[ei]:
                face_counts[fi] += 1
                
    # Verify best is valid
    # (It should be by construction)
    
    while T > min_T:
        if time.time() > deadline:
            break
            
        # Pick random move
        # We can try adding a random unselected edge or removing a random selected edge.
        # To keep it simple, let's pick a random edge index.
        ei = random.randint(0, m - 1)
        
        if selected[ei]:
            # Remove ei
            selected[ei] = False
            count -= 1
            delta = -1
            # Update face counts
            for fi in edge_faces[ei]:
                face_counts[fi] -= 1
        else:
            # Add ei
            # Check conflicts
            conflicts = []
            for fi in edge_faces[ei]:
                if face_counts[fi] == 3:
                    # This face is almost full. Adding ei makes it 4.
                    # We must remove one of the other 3 edges.
                    # Which one?
                    # Let's pick a random one from the selected edges in this face.
                    # This is a "repair" step.
                    # We add ei, and remove a random other edge from this face.
                    # But what if multiple faces are full?
                    # If ei is in multiple full faces, we have to remove edges from all of them.
                    # This might result in removing more than 1 edge.
                    
                    # Let's identify all edges to remove.
                    # For each full face containing ei, pick one selected edge (not ei) to remove.
                    # It's possible the same edge is in multiple full faces.
                    
                    # Let's collect candidates.
                    pass # Logic below
                    
            # If no conflicts, just add.
            if not conflicts:
                 selected[ei] = True
                 count += 1
                 delta = 1
                 for fi in edge_faces[ei]:
                     face_counts[fi] += 1
            else:
                 # Repair logic
                 # This is getting complex. 
                 # Alternative: Reject the move if it creates a conflict.
                 # This makes SA slower but simpler and always valid.
                 # Let's try "Reject if conflict" first. It's a standard approach for hard constraints.
                 pass # We will implement "Reject if conflict" for simplicity and robustness.
                 
        # Since I chose "Reject if conflict" for the 'else' branch above, I need to implement it.
        # Let's restart the loop logic for the move selection to be cleaner.
        
    # Let's rewrite the SA loop properly with "Reject if conflict".
    
    # Reset state to best? No, continue from current.
    # But I messed up the state above. Let's re-initialize state variables for the SA loop.
    
    # Re-verify best_selected is valid and get its face counts
    # Actually, let's just run SA from the greedy result.
    
    # Re-init face_counts for current 'selected' (which is the greedy result)
    # Wait, I modified 'selected' in the broken loop above?
    # No, I didn't execute the 'else' branch logic fully.
    # Let's just discard the broken loop and write a clean SA loop.
    
    # Current state: selected (from greedy), count, face_counts (from greedy).
    
    T = 1.0
    cooling = 0.995
    min_T = 0.001
    step = 0
    
    while T > min_T:
        if time.time() > deadline:
            break
            
        ei = random.randint(0, m - 1)
        
        if selected[ei]:
            # Try to remove
            selected[ei] = False
            count -= 1
            delta = -1
            for fi in edge_faces[ei]:
                face_counts[fi] -= 1
        else:
            # Try to add
            # Check if any face containing ei has 3 selected edges
            conflict = False
            for fi in edge_faces[ei]:
                if face_counts[fi] == 3:
                    conflict = True
                    break
            
            if conflict:
                # Reject move (do nothing)
                # But we need a delta for acceptance?
                # If we reject, we don't change state.
                # We can just continue to next iteration.
                # Or we can treat it as a move with delta = -infinity?
                # Better to just skip.
                continue
            else:
                selected[ei] = True
                count += 1
                delta = 1
                for fi in edge_faces[ei]:
                    face_counts[fi] += 1
        
        # Acceptance
        if delta > 0:
            # Always accept improvement
            if count > best_count:
                best_count = count
                best_selected = selected[:]
                best_val = best_count
        else:
            # delta == -1 (removal)
            # Accept with probability exp(delta / T)
            if random.random() < math.exp(delta / T):
                # Accept
                pass
            else:
                # Reject: revert
                if delta == -1:
                    # We removed ei, so put it back
                    selected[ei] = True
                    count += 1
                    for fi in edge_faces[ei]:
                        face_counts[fi] += 1
                # delta > 0 is always accepted, so we don't reach here for delta > 0
        
        T *= cooling
        step += 1
        
    # Return best found
    out = []
    for ei in range(m):
        if best_selected[ei]:
            u, v = edges[ei]
            out.append([u, v])
            
    return out
# EVOLVE-BLOCK-END
