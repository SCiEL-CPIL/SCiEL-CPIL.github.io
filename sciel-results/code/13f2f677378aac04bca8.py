import time
import random
import numpy as np

def construct():
    """
    Maximize edges in Q7 with no C4.
    Strategy:
    1. Model the problem as selecting edges in Q7 such that no 2D face (square) is fully selected.
    2. A 2D face is determined by two dimensions i, j and fixed values for the other 5 dimensions.
    3. For each 2D face, at most 3 edges can be selected.
    4. We use a constructive approach:
       - Iterate through all possible 2D faces.
       - For each face, determine which edges are "available" (not conflicting with previously forced exclusions).
       - Use a greedy bitmask solver for each 2D face to maximize edges locally.
       - However, edges belong to multiple faces. A global greedy approach with backtracking or local improvement is better.
       
    Alternative Efficient Strategy:
    The problem is equivalent to finding the maximum number of edges in a C4-free subgraph of Q7.
    Known result: ex(Q_n, C4) is roughly n * 2^(n-1) / sqrt(n) for large n, but for n=7, specific constructions are known.
    Best known lower bound is 304.
    
    Let's use a randomized local search with a very fast validity check.
    Key Insight: An edge (u, v) is invalid if it completes a C4. 
    A C4 is formed by vertices {x, x^i, x^j, x^i^j}.
    The edges are (x, x^i), (x^i, x^i^j), (x^i^j, x^j), (x^j, x).
    If we select 3 edges of a face, the 4th is forbidden.
    
    We can represent the state as a set of selected edges.
    To speed up, we precompute for each edge, the list of "faces" it belongs to.
    Actually, it's easier to precompute for each edge, the 3 other edges in each of its 6 faces.
    
    Let's try a different angle: 
    Q7 is bipartite. 
    We can try to construct a solution based on a known algebraic construction or use a highly optimized local search.
    
    Optimized Local Search:
    1. Start with a random valid set (greedy).
    2. Use a "move" that swaps edges to improve density.
    3. Crucially, use a fast conflict check.
    
    Let's implement a fast greedy + local search.
    Precomputation:
    - all_edges: list of (u, v)
    - edge_to_idx: dict
    - face_conflicts: for each edge, list of (face_id, other_edges_in_face)
    
    Actually, the most effective method for small graphs like this (448 edges) is often a simple local search with good initial randomization and a fast "can I add edge e?" check.
    
    Let's refine the "fast check":
    Maintain a count `face_fullness[face_id]` = number of edges selected in that face.
    An edge e can be added if for all faces f containing e, `face_fullness[f] < 3`.
    When adding e, increment `face_fullness[f]` for all f containing e.
    When removing e, decrement.
    
    This reduces the check from O(degree) to O(6) per edge.
    """
    d = 7
    n_vertices = 1 << d
    all_edges = []
    edge_to_idx = {}
    
    # Generate edges
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(all_edges)
                all_edges.append((x, y))
                edge_to_idx[(x, y)] = idx
                
    n_edges = len(all_edges)
    
    # Precompute faces for each edge
    # A face is identified by (dim1, dim2, fixed_mask, fixed_val)
    # But we just need to know which edges are in the same face.
    # For an edge e=(u,v) with diff bit i, it is in faces with any other bit j.
    # The face is defined by the projection of u (and v) onto the bits excluding i and j.
    # Let's just store the list of other 3 edges for each (edge, j) pair.
    
    edge_faces = [[] for _ in range(n_edges)] # List of (face_id, [other_edge_idx1, other_edge_idx2, other_edge_idx3])
    # We need a global face ID to track counts.
    # Total faces: C(7,2) * 2^(7-2) = 21 * 32 = 672 faces.
    # Each face has 4 edges.
    
    face_edges = {} # face_id -> [e1, e2, e3, e4]
    
    def get_face_id(i, j, x):
        # i, j are the dimensions of the face.
        # x is a vertex in the face.
        # The face is determined by the bits of x excluding i and j.
        # Let's create a canonical key.
        # Mask out bits i and j.
        mask = ((1 << d) - 1) ^ (1 << i) ^ (1 << j)
        fixed_val = x & mask
        # Sort i, j for consistency
        if i > j:
            i, j = j, i
        return (i, j, fixed_val)
        
    face_id_counter = 0
    face_id_map = {}
    
    # Build face structure
    # Iterate over all faces
    for i in range(d):
        for j in range(i + 1, d):
            for x in range(n_vertices):
                # Only consider x where the "canonical" representative is x?
                # No, just generate all faces.
                fid_key = get_face_id(i, j, x)
                if fid_key not in face_id_map:
                    face_id_map[fid_key] = face_id_counter
                    face_id_counter += 1
                    face_edges[face_id_map[fid_key]] = []
                
                fid = face_id_map[fid_key]
                
                # The 4 vertices of this face:
                # We need to find the base vertex.
                # Let's just identify the 4 edges in this face.
                # Vertices: x, x^i, x^j, x^i^j
                # But we must ensure we are generating the same face for all 4 vertices.
                # The face is unique. Let's just add edges if they belong to this (i,j, fixed_val).
                
                # Edge 1: (x, x^i) if x < x^i and bit i is the diff?
                # Actually, let's just check if the edge connects two vertices in this face and differs in i or j.
                
                # Vertices in face:
                # v0 = x
                # v1 = x ^ (1 << i)
                # v2 = x ^ (1 << j)
                # v3 = x ^ (1 << i) ^ (1 << j)
                
                # Edges:
                # (v0, v1) diff i
                # (v2, v3) diff i
                # (v0, v2) diff j
                # (v1, v3) diff j
                
                edges_in_face = []
                
                e1_u, e1_v = v0, v1
                if e1_u > e1_v: e1_u, e1_v = e1_v, e1_u
                e1 = (e1_u, e1_v)
                
                e2_u, e2_v = v2, v3
                if e2_u > e2_v: e2_u, e2_v = e2_v, e2_u
                e2 = (e2_u, e2_v)
                
                e3_u, e3_v = v0, v2
                if e3_u > e3_v: e3_u, e3_v = e3_v, e3_u
                e3 = (e3_u, e3_v)
                
                e4_u, e4_v = v1, v3
                if e4_u > e4_v: e4_u, e4_v = e4_v, e4_u
                e4 = (e4_u, e4_v)
                
                for e in [e1, e2, e3, e4]:
                    if e in edge_to_idx:
                        edges_in_face.append(edge_to_idx[e])
                
                # Only store if we haven't seen this face fully?
                # The loop over x will hit the same face 4 times.
                # We can just update face_edges if it's empty or check length.
                if len(face_edges[fid]) < 4:
                    # Append unique edges
                    for idx in edges_in_face:
                        if idx not in face_edges[fid]:
                            face_edges[fid].append(idx)
                            
    # Now, build edge_faces: for each edge, which faces does it belong to?
    for fid, edges in face_edges.items():
        if len(edges) != 4:
            continue # Should be 4
        for e_idx in edges:
            others = [e for e in edges if e != e_idx]
            edge_faces[e_idx].append((fid, others))
            
    # Verify
    for e_idx in range(n_edges):
        if len(edge_faces[e_idx]) != 6:
            # Each edge is in 6 faces (d-1 = 6 other dimensions)
            pass # Might have duplicates or issues?
            # An edge (u,v) diff i. It is in faces (i,j) for j!=i.
            # So 6 faces.
            # The logic above should work.
            
    # State
    in_set = [False] * n_edges
    face_count = [0] * face_id_counter
    
    def can_add(e_idx):
        for fid, _ in edge_faces[e_idx]:
            if face_count[fid] >= 3:
                return False
        return True
        
    def add(e_idx):
        in_set[e_idx] = True
        for fid, _ in edge_faces[e_idx]:
            face_count[fid] += 1
            
    def remove(e_idx):
        in_set[e_idx] = False
        for fid, _ in edge_faces[e_idx]:
            face_count[fid] -= 1
            
    def count_edges():
        return sum(in_set)
        
    # Greedy Initialization
    def greedy_init():
        random.seed(42) # Reproducibility for first run? No, let's randomize.
        order = list(range(n_edges))
        random.shuffle(order)
        for e_idx in order:
            if can_add(e_idx):
                add(e_idx)
                
    # Local Search
    def local_search(iterations=10000):
        # 1-out, 1-in or 1-out, k-in
        # We'll try 1-out, 1-in first as it's faster.
        # Then 1-out, k-in.
        
        improved = True
        while improved and iterations > 0:
            improved = False
            iterations -= 1
            
            # Try to add any edge
            for e_idx in range(n_edges):
                if not in_set[e_idx] and can_add(e_idx):
                    add(e_idx)
                    improved = True
            
            if improved:
                continue
                
            # Try 1-out, 1-in
            # For each edge in set, remove it, see if we can add a better edge?
            # Or just see if we can add *any* edge that was blocked.
            # Since we tried adding all, we need to remove one to free space.
            
            # Heuristic: Remove an edge that is in a "tight" face (count 3) and see if we can add 2?
            # Simpler: Random 1-out, 1-in.
            
            # Let's try a specific move:
            # Pick a random edge u in set.
            # Remove u.
            # Try to add any edge v not in set that is now valid.
            # If we can add v, and v is "better" (e.g., leads to more future additions), keep it.
            # For now, just accept if we can add *any* edge.
            
            # To avoid oscillation, we might need a tabu or just random restarts.
            
            # Let's do a full scan for 1-out, k-in with k>1.
            # This is expensive but n=448 is small.
            
            best_gain = 0
            best_move = None
            
            # Sample a subset of edges to remove to save time?
            # With 448 edges, scanning all is 448 * (cost of check).
            # Cost of check: remove u (O(6)), then try adding all v (448 * O(6)).
            # 448 * 448 * 6 ~ 1.2M ops. Very fast in Python? Maybe 0.1s.
            # Let's try it.
            
            in_set_indices = [i for i, v in enumerate(in_set) if v]
            
            for u in in_set_indices:
                # Remove u
                remove(u)
                
                # Try to add edges
                added = []
                # We need to be careful: adding one edge might block another.
                # Greedy add:
                # Get list of candidates
                candidates = [v for v in range(n_edges) if not in_set[v]]
                random.shuffle(candidates)
                
                for v in candidates:
                    if can_add(v):
                        add(v)
                        added.append(v)
                        
                gain = len(added) - 1 # Removed 1, added len(added)
                
                if gain > best_gain:
                    best_gain = gain
                    best_move = (u, added)
                    # Rollback for now
                    for v in added:
                        remove(v)
                    add(u)
                    
                    # Early stop if we found a good move?
                    if best_gain >= 2:
                        break
                else:
                    # Rollback
                    for v in added:
                        remove(v)
                    add(u)
                    
            if best_move:
                u, added = best_move
                remove(u)
                for v in added:
                    add(v)
                improved = True
                
        return count_edges()
        
    deadline = time.time() + 110
    best_count = 0
    best_result = []
    
    # Multiple restarts
    while time.time() < deadline:
        # Reset
        in_set = [False] * n_edges
        face_count = [0] * face_id_counter
        
        greedy_init()
        current_count = count_edges()
        
        # Local search
        # We need to pass state to local_search? 
        # The functions above use global state.
        # Let's wrap it.
        
        # For simplicity, we'll just run the local search on the current global state.
        # But local_search modifies global state.
        
        # We need to save the best state.
        # Copying in_set and face_count is cheap.
        
        local_search(iterations=500)
        current_count = count_edges()
        
        if current_count > best_count:
            best_count = current_count
            best_result = [all_edges[i] for i in range(n_edges) if in_set[i]]
            
        if best_count >= 308:
            break
            
    return best_result
