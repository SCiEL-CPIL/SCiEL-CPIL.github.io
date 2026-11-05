import time
import sys
import random

def construct():
    """
    Constructs a C4-free subgraph of Q7 maximizing edge count.
    
    Strategy:
    1. Precompute all 448 edges and 672 faces (C4s) of Q7.
    2. Initialize with all edges present.
    3. Iteratively repair conflicts:
       - While any face has all 4 edges present:
         - Identify the "best" edge to remove from the conflicting face.
         - "Best" is defined as the edge whose removal allows the most other 
           currently missing edges to be added (minimizing regret).
         - Remove that edge.
    4. Greedy Addition Phase:
       - Add any missing edge that doesn't create a C4.
    5. Pair-Swap Local Search:
       - To escape local optima where no single addition is possible:
         - Try removing one edge e1 and adding another e2.
         - This is beneficial if removing e1 unblocks e2 and other edges, 
           leading to a net gain in edge count.
         - We implement a simplified version: 
           For each missing edge e2, find the faces blocking it.
           For each blocking face, try removing an edge e1 from that face.
           If removing e1 allows adding e2 (and potentially others), do it if it 
           leads to a better state after a subsequent greedy add phase.
           To keep it fast, we only accept swaps that immediately allow adding 
           more than 1 edge, or we use a limited lookahead.
           
    Given the time limit (150s), we focus on a high-quality deterministic 
    initialization and repair, followed by extensive local search.
    """
    start_time = time.time()
    time_limit = 145.0  # Safety margin

    d = 7
    n = 2 ** d  # 128 vertices

    # 1. Precompute edges
    edge_list = []
    edge_to_idx = {}
    
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edge_list)
                edge_list.append((x, y))
                edge_to_idx[(x, y)] = idx
    
    E = len(edge_list)  # 448

    # 2. Precompute faces
    faces = []
    face_set = set()
    
    for u in range(n):
        for i in range(d):
            for j in range(i + 1, d):
                v1 = u
                v2 = u ^ (1 << i)
                v3 = u ^ (1 << j)
                v4 = u ^ (1 << i) ^ (1 << j)
                
                e1 = (min(v1, v2), max(v1, v2))
                e2 = (min(v1, v3), max(v1, v3))
                e3 = (min(v2, v4), max(v2, v4))
                e4 = (min(v3, v4), max(v3, v4))
                
                idx1 = edge_to_idx[e1]
                idx2 = edge_to_idx[e2]
                idx3 = edge_to_idx[e3]
                idx4 = edge_to_idx[e4]
                
                face_tuple = tuple(sorted([idx1, idx2, idx3, idx4]))
                
                if face_tuple not in face_set:
                    face_set.add(face_tuple)
                    faces.append([idx1, idx2, idx3, idx4])
    
    F = len(faces)  # 672

    # 3. Precompute edge-to-face mapping
    edge_faces = [[] for _ in range(E)]
    for f_idx, face in enumerate(faces):
        for e_idx in face:
            edge_faces[e_idx].append(f_idx)

    # State representation: list of booleans
    # We use a helper class or just lists for speed.
    
    def full_count_calc(state):
        fc = [0] * F
        for f_idx, face in enumerate(faces):
            c = 0
            for e in face:
                if state[e]:
                    c += 1
            fc[f_idx] = c
        return fc

    def greedy_add(state, full_count):
        """Add all possible edges."""
        improved = True
        while improved:
            improved = False
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
                    for f_idx in edge_faces[e]:
                        full_count[f_idx] += 1
        return state, full_count

    def repair(state, full_count):
        """
        Remove edges to resolve conflicts.
        Heuristic: Remove the edge that is in the most "tight" faces 
        (faces with 3 other edges present).
        Actually, if a face is full (4 edges), we must remove one.
        To maximize future potential, we should remove an edge that 
        "blocks" the fewest other edges.
        An edge e blocks a missing edge m if they share a face f 
        where the other 2 edges in f are present.
        """
        while True:
            # Find a conflicting face
            conflict_face = -1
            for f_idx in range(F):
                if full_count[f_idx] == 4:
                    conflict_face = f_idx
                    break
            
            if conflict_face == -1:
                break
            
            face = faces[conflict_face]
            
            # Evaluate which edge to remove from this face
            best_e = -1
            best_score = -1
            
            for e in face:
                # Score: How many missing edges would be unblocked if we remove e?
                # Or simpler: How many faces does e belong to that are "almost full"?
                # Let's use: Number of missing edges m such that 
                # e and m are in a common face f, and f has 2 other present edges.
                # This is expensive to compute dynamically.
                
                # Simpler heuristic: Remove the edge with the highest degree 
                # in the subgraph of "conflict-adjacent" edges?
                # Or just the one that appears in the most faces total?
                # All edges in Q7 appear in 6 faces.
                
                # Let's try: Remove the edge that shares the most faces with 
                # other edges that are also in conflict?
                
                # For now, let's use a "regret" proxy:
                # Count how many missing edges m are blocked ONLY by this edge e 
                # in the context of the current conflict?
                
                # Actually, a very effective heuristic for this specific problem 
                # is to remove the edge that has the highest "load" in terms of 
                # being part of other faces that are close to being full.
                
                # Let's just pick the edge that allows the most immediate additions 
                # after removal. We can simulate this quickly.
                
                # Simulate removal of e
                # Decrement full_count for faces containing e
                affected_faces = edge_faces[e]
                original_fc = [0] * F # We don't want to copy the whole array, 
                                       # but we can track changes locally or 
                                       # just do a quick check.
                
                # Let's just pick the edge that is in the most faces with full_count >= 3
                # (excluding the current conflict face which is 4)
                score = 0
                for f_idx in affected_faces:
                    if f_idx != conflict_face:
                        if full_count[f_idx] >= 3:
                            score += 1
                
                if score > best_score:
                    best_score = score
                    best_e = e
            
            # Remove best_e
            state[best_e] = False
            for f_idx in edge_faces[best_e]:
                full_count[f_idx] -= 1

    def local_search_swap(state, full_count):
        """
        Try to improve by swapping edges.
        We look for a missing edge e2 that is blocked.
        If we remove a blocking edge e1, can we add e2 and others?
        """
        # This is complex. A simpler version:
        # Try all pairs of (remove e1, add e2) where e1 is present, e2 is missing.
        # There are 448*448 ~ 200k pairs.
        # For each, check validity.
        
        # Optimization:
        # e2 is blocked by faces where full_count == 3 and e2 is the missing edge.
        # Let blocking_faces(e2) = {f | e2 in f, full_count[f] == 3}
        # To add e2, we must remove at least one edge from each f in blocking_faces(e2).
        # If we remove e1, e2 becomes addable if e1 is in ALL blocking faces?
        # No, e1 must be in at least one blocking face, and removing it must reduce 
        # full_count of that face to 2. But if there are multiple blocking faces, 
        # removing one edge might not be enough unless e1 is in all of them.
        
        # So, only consider e2 if it is blocked by exactly 1 face.
        # If blocked by 1 face f, and e1 is in f, then removing e1 allows adding e2.
        # Net change: 0 (remove 1, add 1).
        # Is it useful? Only if it allows FUTURE additions.
        
        # Let's implement a "2-opt" style move:
        # Remove e1, Add e2.
        # Then run a quick greedy add to see if we gain more.
        
        # To keep it fast, we'll do this for a limited number of iterations.
        
        improved = True
        iterations = 0
        max_iterations = 10000
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            # Find a missing edge e2
            for e2 in range(E):
                if state[e2]:
                    continue
                    
                # Check if e2 is blocked
                blocking_faces = []
                for f_idx in edge_faces[e2]:
                    if full_count[f_idx] == 3:
                        blocking_faces.append(f_idx)
                
                if not blocking_faces:
                    # Should have been added in greedy phase
                    continue
                
                # If blocked by more than 1 face, removing a single edge won't fix it 
                # unless that edge is in all blocking faces.
                # For simplicity, only handle case where blocked by 1 face.
                if len(blocking_faces) > 1:
                    continue
                
                f_block = blocking_faces[0]
                
                # Try removing each edge e1 in f_block (except e2, which is missing)
                for e1 in faces[f_block]:
                    if e1 == e2:
                        continue
                    if not state[e1]:
                        continue
                    
                    # Simulate: Remove e1, Add e2
                    # Check if this is valid (it is, by construction of f_block)
                    # Check if it leads to improvement after greedy add
                    
                    # Save state
                    state[e1] = False
                    state[e2] = True
                    
                    # Update full_count
                    for f_idx in edge_faces[e1]:
                        full_count[f_idx] -= 1
                    for f_idx in edge_faces[e2]:
                        full_count[f_idx] += 1
                    
                    # Check if any face became full (conflict)
                    # This shouldn't happen if e1 was the only blocker for e2 
                    # and no other face was 3-full for e2.
                    # But removing e1 might have made some face 2-full, 
                    # and adding e2 might make it 3-full.
                    # It cannot make a face 4-full unless e2 was already in 3 other 
                    # full faces, which is not the case.
                    
                    # Run a quick greedy add to see if we gain
                    old_count = sum(state)
                    
                    # We don't want to run full greedy add every time (expensive).
                    # Let's just check if e2 was the only missing edge in some faces 
                    # that are now 3-full? No, e2 is present now.
                    
                    # Let's just accept the swap if it doesn't decrease count 
                    # (it doesn't) and hope it helps later?
                    # No, that's not a hill climb.
                    
                    # Let's check if removing e1 unblocks other edges.
                    # Specifically, are there other missing edges e3 such that 
                    # e1 was their only blocker?
                    gain = 0
                    for e3 in range(E):
                        if state[e3] or e3 == e2:
                            continue
                        # Check if e3 is blocked only by e1
                        # i.e., e3 is in a face f3 where full_count[f3] == 3 
                        # (after e1 removal, e2 addition) and e1 is in f3.
                        # This is getting complicated.
                        
                        # Simple heuristic: If we do the swap, do we gain anything 
                        # in a 1-step lookahead?
                        pass
                    
                    # For now, revert if no immediate gain found.
                    # Actually, let's just do the swap if it allows adding 
                    # at least one other edge immediately.
                    
                    can_gain = False
                    for e3 in range(E):
                        if state[e3]:
                            continue
                        # Check if e3 can be added now
                        ok = True
                        for f_idx in edge_faces[e3]:
                            if full_count[f_idx] == 3:
                                ok = False
                                break
                        if ok:
                            can_gain = True
                            break
                    
                    if can_gain:
                        # Keep the swap and add e3 (and others)
                        improved = True
                        # Run a small greedy add
                        # We'll just mark improved and let the main loop handle it
                        break
                    else:
                        # Revert
                        state[e1] = True
                        state[e2] = False
                        for f_idx in edge_faces[e1]:
                            full_count[f_idx] += 1
                        for f_idx in edge_faces[e2]:
                            full_count[f_idx] -= 1
                
                if improved:
                    break
            
            # If we broke out of e2 loop with improved=True, 
            # we should run greedy add to collect the gains
            if improved:
                state, full_count = greedy_add(state, full_count)

    # Main Execution
    random.seed(42)
    
    # Initialization: All edges
    state = [True] * E
    full_count = full_count_calc(state)
    
    # Repair
    repair(state, full_count)
    
    # Greedy Add
    state, full_count = greedy_add(state, full_count)
    
    current_score = sum(state)
    best_score = current_score
    best_state = state[:]
    
    # Local Search Swap
    # We run this for a while
    while time.time() - start_time < time_limit - 5:
        state, full_count = local_search_swap(state, full_count)
        
        # Greedy add again to clean up
        state, full_count = greedy_add(state, full_count)
        
        current_score = sum(state)
        if current_score > best_score:
            best_score = current_score
            best_state = state[:]
            # Optional: if we hit 308, break
            if best_score >= 308:
                break
    
    # Final check
    # Ensure validity
    final_full_count = full_count_calc(best_state)
    valid = True
    for fc in final_full_count:
        if fc == 4:
            valid = False
            break
    
    if not valid:
        # Fallback: Repair the best state
        state = best_state[:]
        full_count = final_full_count
        repair(state, full_count)
        state, full_count = greedy_add(state, full_count)
        best_state = state
        best_score = sum(state)

    result = []
    for i in range(E):
        if best_state[i]:
            u, v = edge_list[i]
            result.append([u, v])
            
    return result

if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Edges: {len(res)}, Time: {end-start:.2f}s")

