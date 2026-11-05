import time
import random
import sys

def construct():
    """
    Constructs a C4-free subgraph of Q7 maximizing edge count.
    Strategy:
    1. Deterministic initialization using a parity-based coloring of edges.
       An edge (u, v) differing in bit i is included if popcount(u) % 2 == 0.
       This ensures that for any face (u, u^i, u^j, u^i^j), the edges are not all present.
       Specifically, the four edges of a face are:
       (u, u^i), (u, u^j), (u^i, u^i^j), (u^j, u^i^j).
       Let k = popcount(u).
       Edge 1: popcount(u)=k. Included if k even.
       Edge 2: popcount(u)=k. Included if k even.
       Edge 3: popcount(u^i)=k+1 (if bit i was 0) or k-1. Parity flips.
       Edge 4: popcount(u^j)=k+1 or k-1. Parity flips.
       Edge 5: popcount(u^i^j)=k+2 or k or k-2. Parity same as k.
       
       Let's check the condition "No 2-dimensional face may have all four edges selected".
       The four edges are:
       e1: u -- u^i (differs in i)
       e2: u -- u^j (differs in j)
       e3: u^i -- u^i^j (differs in j)
       e4: u^j -- u^i^j (differs in i)
       
       My rule: Include edge (a,b) if popcount(min(a,b)) % 2 == 0.
       Let's assume u < u^i and u < u^j for simplicity of min/max, though parity of popcount is what matters.
       Actually, let's just use the vertex u (the one with smaller value? No, parity is symmetric for the check? No.)
       
       Let's refine the deterministic rule:
       For an edge connecting x and y (x^y = 1<<i), include it if (x & y) has even popcount?
       Or simply: Include edge (x, x^(1<<i)) if popcount(x) % 2 == 0.
       
       Let's test this rule on a face with corners 0, 1, 2, 3 (bits 0,1).
       Vertices:
       0 (00): pop=0 (even)
       1 (01): pop=1 (odd)
       2 (10): pop=1 (odd)
       3 (11): pop=2 (even)
       
       Edges:
       (0,1): diff bit 0. Start at 0 (pop 0, even). Include? Yes.
       (0,2): diff bit 1. Start at 0 (pop 0, even). Include? Yes.
       (1,3): diff bit 1. Start at 1 (pop 1, odd). Include? No.
       (2,3): diff bit 0. Start at 2 (pop 1, odd). Include? No.
       
       Face 0-1-3-2 has edges (0,1) and (0,2) selected. (1,3) and (2,3) not.
       So not all 4 are selected. This is a valid C4-free set.
       
       How many edges?
       For each bit i, we count vertices x where bit i is 0 (so x < x^i usually, but let's just count based on x).
       Actually, for each dimension i, we include edges (x, x^i) where popcount(x) is even.
       Since for every pair {x, x^i}, exactly one has even popcount?
       Let x have parity p. x^i has parity 1-p.
       So exactly one of the two vertices in the pair has even popcount.
       Therefore, for every edge in the hypercube, exactly one endpoint has even popcount.
       So we include exactly 1 edge per pair?
       Wait. The edge is defined by the pair.
       If we say "Include edge (x, x^i) if popcount(x) is even", then for a specific edge, we check one endpoint.
       Since the edge connects x and x^i, and we define the edge uniquely, we just need to be consistent.
       Let's define: Edge e_i(x) is included if popcount(x) % 2 == 0.
       Total edges in Q7 = 448.
       For each dimension i, there are 64 edges.
       In each dimension i, the vertices are paired (x, x^i).
       One has even pop, one odd.
       So exactly 32 edges per dimension are included?
       Total edges = 7 * 32 = 224.
       This is a baseline of 224. The seed gets ~304.
       
       We can do better with a different linear rule.
       Rule: Include edge (x, x^i) if (x >> i) & 1 == 0? No, that's just half.
       
       Better approach: Use the structure of the 2-coloring of edges in a bipartite graph?
       Q7 is bipartite.
       
       Let's stick to the local search on a good initial guess.
       The current seed uses random greedy.
       A smarter initial guess:
       Start with all edges.
       Remove edges to break C4s.
       Greedy removal:
       While there is a C4, remove an edge from it.
       To maximize remaining edges, remove edges that are in many C4s.
       
       Algorithm:
       1. Start with all 448 edges.
       2. Compute for each edge the number of C4s it participates in.
          In Q_n, an edge (u, v) differing in bit i participates in (n-1) * (2^(n-2)) C4s?
          No, in Q7, an edge is in (7-1) * (2^(7-2)) = 6 * 16 = 96 C4s?
          Let's verify.
          Edge (u, v) diff in i.
          Choose another bit j (6 choices).
          The face is u, v, u^j, v^j.
          This forms a C4 with edges (u,v), (u, u^j), (v, v^j), (u^j, v^j).
          So each edge is in exactly 6 faces?
          Wait, a "2-dimensional face" is a square.
          Yes, each edge in Q7 is in exactly 6 squares (one for each other dimension j != i).
          
       3. So initially, every edge is in 6 C4s.
       4. We need to remove edges such that no C4 has all 4 edges.
       This is equivalent to finding a maximum induced subgraph without C4s.
       
       Local Search Strategy:
       1. Initialize with a random subset or the parity subset (224 edges).
       2. Try to add edges.
       3. Try to swap edges (remove one, add another) if it increases count or maintains count with better "potential".
       
       Actually, a known construction for C4-free subgraphs of hypercubes:
       For Q_7, a random subset of density p has no C4 if p is small enough.
       But we want max edges.
       
       Let's use a "Max-SAT" style local search.
       Variables: Edges (448 boolean vars).
       Clauses: For each C4 (face), NOT (e1 AND e2 AND e3 AND e4).
       Maximize sum(e_i).
       
       We can use a simple hill climbing:
       1. Start with the parity construction (224 edges).
       2. Iterate:
          - Try to add any edge that doesn't create a C4.
          - If stuck, try to remove an edge that is "bad" (part of many potential C4s) and then try to add others.
          
       To be faster and better, we can use a greedy algorithm that prioritizes edges by some heuristic.
       
       Let's try a different initialization:
       Randomly assign each edge a value 0 or 1 with probability 0.5.
       Then repair:
       While there is a C4 with all 4 edges:
           Remove one edge from the C4 randomly (or the one with highest "cost").
       
       Then improve:
       While possible:
           Try to add an edge that is currently missing.
           If it creates a C4, don't add.
           
       This "Repair and Add" is similar to the seed but starts from a dense random state and repairs.
       The seed starts from empty and adds.
       Starting from dense and removing might find denser solutions if the repair is smart.
       
       Let's implement:
       1. Precompute all C4s (faces). There are C(7,2) * 2^5 = 21 * 32 = 672 faces.
       2. For each face, store the 4 edges.
       3. Initialize: All edges present.
       4. Conflict resolution:
          - Count conflicts for each edge (number of C4s where all 4 edges are present).
          - While conflicts exist:
            - Pick an edge involved in the most conflicts.
            - Remove it.
            - Update conflict counts for edges in C4s containing the removed edge.
       5. Local Search (Add):
          - Try to add back removed edges.
          - If adding an edge creates a conflict, skip.
          
       6. Local Search (Swap):
          - Try to remove an edge e1 and add an edge e2.
          - If e1 is in no C4s (or few) and e2 is not in any C4s after adding...
          
       This should be fast because we only have 672 C4s.
       
    """
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    d = 7
    n = 2 ** d  # 128 vertices
    
    # 1. Precompute edges and faces
    # Edge index mapping
    # Edge is (u, v) with u < v.
    edge_list = []
    edge_to_idx = {}
    idx_to_edge = []
    
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edge_list)
                edge_list.append((x, y))
                edge_to_idx[(x, y)] = idx
                idx_to_edge.append((x, y))
    
    E = len(edge_list) # 448
    
    # Precompute faces (C4s)
    # A face is defined by two bits i, j and a base vertex u where bits i,j are 0?
    # No, just iterate all u and pairs i<j.
    # To avoid duplicates, we can canonicalize the face by its 4 vertices.
    # Or just generate all unique sets of 4 edges.
    
    faces = [] # List of lists of edge indices
    face_set = set()
    
    for u in range(n):
        for i in range(d):
            for j in range(i+1, d):
                v1 = u
                v2 = u ^ (1 << i)
                v3 = u ^ (1 << j)
                v4 = u ^ (1 << i) ^ (1 << j)
                
                # Get edges
                e1 = (min(v1, v2), max(v1, v2))
                e2 = (min(v1, v3), max(v1, v3))
                e3 = (min(v2, v4), max(v2, v4))
                e4 = (min(v3, v4), max(v3, v4))
                
                # Check if these are valid edges (they are, by construction)
                idx1 = edge_to_idx[e1]
                idx2 = edge_to_idx[e2]
                idx3 = edge_to_idx[e3]
                idx4 = edge_to_idx[e4]
                
                # Canonicalize face to avoid duplicates
                # A face is a set of 4 edge indices
                face_tuple = tuple(sorted([idx1, idx2, idx3, idx4]))
                
                if face_tuple not in face_set:
                    face_set.add(face_tuple)
                    faces.append([idx1, idx2, idx3, idx4])
    
    F = len(faces) # Should be 672
    
    # Precompute for each edge, which faces it belongs to
    edge_faces = [[] for _ in range(E)]
    for f_idx, face in enumerate(faces):
        for e_idx in face:
            edge_faces[e_idx].append(f_idx)
    
    # State: present[i] is True if edge i is selected
    # We want to maximize sum(present)
    # Constraint: For all faces, not all 4 edges are present.
    
    best_score = -1
    best_state = None
    
    # Strategy: Multiple restarts with different random initializations
    # Init 1: All edges present, then repair.
    # Init 2: Random 50%.
    # Init 3: Parity construction.
    
    def get_score(state):
        return sum(state)
    
    def is_valid(state):
        for face in faces:
            if state[face[0]] and state[face[1]] and state[face[2]] and state[face[3]]:
                return False
        return True
    
    def repair(state):
        # state is a list of bools
        # While there is a conflict, remove an edge
        # To be efficient, track conflict count per edge
        # conflict_count[e] = number of faces where all 4 edges are present
        
        conflict_count = [0] * E
        # Initialize conflict counts
        for face in faces:
            if state[face[0]] and state[face[1]] and state[face[2]] and state[face[3]]:
                for e in face:
                    conflict_count[e] += 1
        
        while True:
            # Find edge with max conflict count > 0
            max_conf = 0
            max_e = -1
            for e in range(E):
                if conflict_count[e] > max_conf:
                    max_conf = conflict_count[e]
                    max_e = e
            
            if max_conf == 0:
                break
            
            # Remove max_e
            state[max_e] = False
            
            # Update conflict counts for faces containing max_e
            for f_idx in edge_faces[max_e]:
                face = faces[f_idx]
                # Check if this face was a conflict (all 4 present)
                # It was, because max_e had conflict_count > 0
                # Now max_e is False, so this face is no longer a conflict
                # Decrement conflict_count for other 3 edges in this face
                for e in face:
                    if e != max_e:
                        conflict_count[e] -= 1
            
            # Note: Other faces not containing max_e are unaffected.
            # Faces containing max_e that were NOT conflicts?
            # If a face containing max_e was not a conflict, it means at least one other edge was False.
            # Removing max_e doesn't change the status of other edges, so it remains not a conflict.
            # So we only need to update faces that WERE conflicts.
            # The logic above decrements for all faces in edge_faces[max_e].
            # But we only should decrement if the face WAS a conflict.
            # How do we know?
            # conflict_count[max_e] was the number of conflicts.
            # We iterated through ALL faces in edge_faces[max_e].
            # We should only decrement for faces that were actually conflicts.
            # Let's re-evaluate.
            
            # Better update logic:
            # For each face f in edge_faces[max_e]:
            #   If before removal, state[f[0]] and ... and state[f[3]] was True (which implies conflict_count[f] > 0? No, conflict_count is per edge).
            #   Actually, if conflict_count[max_e] > 0, it means max_e is in at least one full face.
            #   We need to know WHICH faces were full.
            
            # Let's just re-calculate conflict for the specific faces?
            # Or maintain a count of "full edges" per face.
            
            # Revised Repair:
            # full_count[f] = number of edges in face f that are present.
            # Conflict if full_count[f] == 4.
            
            # Let's rewrite repair using full_count.
            pass # Placeholder, will implement below
            
    def repair_efficient(state):
        # full_count[f] for each face
        full_count = [0] * F
        for f_idx, face in enumerate(faces):
            count = 0
            for e in face:
                if state[e]:
                    count += 1
            full_count[f_idx] = count
            
        while True:
            # Find a face with full_count == 4
            target_face = -1
            for f_idx in range(F):
                if full_count[f_idx] == 4:
                    target_face = f_idx
                    break
                    
            if target_face == -1:
                break
                
            face = faces[target_face]
            # Pick an edge to remove from this face
            # Heuristic: Remove the edge with the highest "degree" in the conflict graph?
            # Or just random?
            # Let's pick the edge in this face that appears in the most other full faces?
            # Or simply the first one.
            # To maximize edges, maybe remove an edge that is "least useful".
            # For now, random from the 4.
            e_to_remove = random.choice(face)
            
            state[e_to_remove] = False
            
            # Update full_count for all faces containing e_to_remove
            for f_idx in edge_faces[e_to_remove]:
                full_count[f_idx] -= 1
                
            # Note: This might create new non-conflicts, but won't create new conflicts
            # because we only removed an edge.
            
    def local_search_add(state):
        # Try to add edges that are currently False
        # Check if adding them creates a conflict
        # We need to know if the face becomes full.
        
        # Precompute for each edge, the faces it belongs to.
        # When adding edge e, it becomes full for a face f if:
        # 1. e was not in f (impossible, e is in f)
        # 2. The other 3 edges in f are already present.
        
        # So for each missing edge e:
        #   Check all faces f containing e.
        #   If for any f, the other 3 edges are present, then adding e creates a conflict.
        
        # We can speed this up by tracking "deficit" per face.
        # deficit[f] = 4 - full_count[f].
        # If deficit[f] == 0, it's a conflict (shouldn't happen if state is valid).
        # If deficit[f] == 1, it means only 1 edge is missing.
        # If we add that missing edge, it becomes a conflict.
        
        # So, an edge e can be added if for ALL faces f containing e, deficit[f] > 1.
        # i.e., deficit[f] >= 2.
        
        # Let's maintain full_count to calculate deficit on the fly or maintain it.
        
        full_count = [0] * F
        for f_idx, face in enumerate(faces):
            count = 0
            for e in face:
                if state[e]:
                    count += 1
            full_count[f_idx] = count
            
        improved = True
        while improved:
            improved = False
            # Iterate through all missing edges
            # To make it deterministic or random? Random helps escape local optima?
            # But we want to add as many as possible.
            # Order doesn't matter much for "can add" check, but might affect which ones we add.
            
            # To be fast, we can just scan all missing edges.
            for e in range(E):
                if state[e]:
                    continue
                    
                # Check if e can be added
                can_add = True
                for f_idx in edge_faces[e]:
                    if full_count[f_idx] == 3:
                        can_add = False
                        break
                        
                if can_add:
                    state[e] = True
                    improved = True
                    # Update full_count for faces containing e
                    for f_idx in edge_faces[e]:
                        full_count[f_idx] += 1
                        
        return state
        
    # Main Loop
    random.seed(42)
    
    max_restarts = 5000
    restart = 0
    
    while restart < max_restarts:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
            
        restart += 1
        
        # Initialization
        # Option 1: All True, then repair
        # Option 2: Random 0.5
        # Option 3: Parity
        
        if restart % 3 == 1:
            state = [True] * E
        elif restart % 3 == 2:
            state = [random.random() < 0.5 for _ in range(E)]
        else:
            # Parity: edge (x, x^i) included if popcount(x) % 2 == 0
            state = [False] * E
            for idx, (u, v) in enumerate(edge_list):
                # u < v. Which one to use for parity?
                # In parity construction, we decided: Include if popcount(u) is even?
                # Let's check: u and v differ by 1 bit.
                # If we use u (smaller), it's consistent.
                if bin(u).count('1') % 2 == 0:
                    state[idx] = True
        
        # Repair
        repair_efficient(state)
        
        # Local Search Add
        local_search_add(state)
        
        # Optional: Local Search Swap
        # Try to remove an edge and add two others?
        # Or remove one, add one?
        # If we remove e1 and add e2, count stays same.
        # Only useful if e2 was blocked by e1?
        # If e1 is in a face that is full, removing e1 breaks that face.
        # Then e2 might be addable.
        
        # Let's do a simple swap local search
        # 1. Identify "blocked" edges: edges that cannot be added because they complete a face with 3 existing edges.
        # 2. For each blocked edge e2, identify the faces that block it.
        # 3. For each such face, pick an existing edge e1 in that face.
        # 4. If we remove e1, does e2 become addable?
        # 5. If yes, and removing e1 doesn't break other things (it only helps), do it?
        #    Wait, removing e1 reduces count by 1. Adding e2 increases by 1. Net 0.
        #    So this doesn't increase count.
        #    It only helps if it allows FUTURE additions.
        #    So we should do this if it leads to a state where we can add MORE edges.
        
        # Simpler: Just do "Add" phase again after a few random removals?
        
        score = sum(state)
        if score > best_score:
            best_score = score
            best_state = state[:]
            
            if best_score >= 308:
                break
                
        # Elapsed check
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
            
    if best_state is None:
        # Fallback: Empty set
        return []
        
    result = []
    for i in range(E):
        if best_state[i]:
            u, v = edge_list[i]
            result.append([u, v])
            
    return result

if __name__ == "__main__":
    # Quick test
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Edges: {len(res)}, Time: {end-start:.2f}s")
