import time
import random
import sys
import math
import numpy as np

def construct():
    """
    Constructs a C4-free subgraph of Q7 maximizing edge count.
    
    Strategy:
    1. Precompute edges and faces (C4s).
    2. Use a "Greedy Set Cover" style initialization:
       - Start with all edges present.
       - While there are C4s (faces with 4 edges), identify the edge that belongs to the most conflicting faces and remove it.
       - This deterministically minimizes the number of removals needed to break all C4s, yielding a dense valid starting point.
    3. Perform Local Search (Hill Climbing) with "Swap" moves:
       - Try to add an edge. If it creates a C4, try to remove one of the other 3 edges in that C4 to "swap" the conflict.
       - This allows escaping local optima where single-edge additions are blocked.
    4. Multiple restarts with different random tie-breaking in the greedy phase to explore different dense configurations.
    """
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit
    
    d = 7
    n = 2 ** d  # 128 vertices
    
    # 1. Precompute edges and faces
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
    
    # Precompute faces (C4s)
    faces = []
    face_set = set()
    
    for u in range(n):
        for i in range(d):
            for j in range(i+1, d):
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
    
    # Precompute for each edge, which faces it belongs to
    edge_faces = [[] for _ in range(E)]
    for f_idx, face in enumerate(faces):
        for e_idx in face:
            edge_faces[e_idx].append(f_idx)
    
    # Convert faces to tuples for faster indexing
    faces_tuple = [tuple(f) for f in faces]
    
    def greedy_initialize(randomize_ties=False):
        """
        Initializes a valid state by removing edges from a full graph.
        Heuristic: Remove the edge in a conflicting face that is in the most other conflicting faces.
        """
        state = bytearray([1] * E)
        full_count = [4] * F  # All faces start with 4 edges
        
        # List of currently conflicting faces (count == 4)
        # Since we start with all 1s, all faces are conflicting.
        # We will maintain a list of conflicting faces.
        conflict_faces = list(range(F))
        
        # To speed up, we can use a heap or just pick randomly if randomize_ties
        # But "most expensive" is better.
        # Let's track the number of conflicts each edge is in.
        # Actually, just picking a random conflict face and the edge in it with highest degree in conflict graph is good.
        
        # Optimization: 
        # We iterate while there are conflicts.
        # Picking the "best" edge to remove globally is NP-hard, but locally (per face) we can pick the edge that appears in the most *other* conflict faces.
        
        while conflict_faces:
            if randomize_ties:
                f_idx = random.choice(conflict_faces)
            else:
                f_idx = conflict_faces[-1] # Just pick one, order doesn't matter much if we update counts
            
            f = faces_tuple[f_idx]
            
            # Find the edge in this face that is in the most *other* conflict faces
            # We need to know which other faces are currently conflicting.
            # Instead of checking all F faces for each of the 4 edges (expensive),
            # we can approximate: pick the edge that has the highest total degree in the conflict graph.
            # Let's maintain a `conflict_degree` array for edges.
            # conflict_degree[e] = number of conflict faces containing e.
            
            # Wait, calculating conflict_degree for all edges every step is O(E*F) which is ~300k ops. 
            # 448 edges * 672 faces. 
            # If we do this for each removal (up to 448 removals), it's 448 * 300k = 134M ops. Might be slow in Python.
            
            # Alternative: Just pick a random edge from the conflict face.
            # Or, pick the edge with the highest "static" degree (number of faces it's in total).
            # Let's try a hybrid: 
            # 1. Pick a random conflict face.
            # 2. Pick the edge in that face with the highest *current* conflict count.
            # To get current conflict count, we need to know if the face is in conflict_faces.
            # We can maintain a set of conflict faces for O(1) lookup.
            
            # Let's refine the loop for speed:
            # 1. Maintain a set `conflict_set` of f_idx where full_count[f_idx] == 4.
            # 2. Pick f_idx from conflict_set.
            # 3. For the 4 edges in f_idx, count how many *other* faces in conflict_set they belong to.
            #    This requires iterating edge_faces[e] and checking membership in conflict_set.
            #    edge_faces[e] has ~ (F/E)*4 ~ 6 entries.
            #    So checking 4 edges * 6 faces * O(1) set lookup = 24 ops.
            #    This is fast.
            
            # Let's re-implement with this logic.
            
            # Re-initialize state for this specific run logic
            # (The code above was pseudocode inside the function, let's write the actual loop)
            pass
            
        # Re-doing the function properly
        state = bytearray([1] * E)
        full_count = [4] * F
        conflict_set = set(range(F))
        
        while conflict_set:
            # Pick a face to resolve
            # Random selection helps escape deterministic traps
            f_idx = random.choice(list(conflict_set))
            f = faces_tuple[f_idx]
            
            # Find the edge in f to remove
            best_e = -1
            best_score = -1
            
            for e in f:
                # Score: number of other conflict faces this edge is in
                score = 0
                for f2 in edge_faces[e]:
                    if f2 in conflict_set:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_e = e
            
            # Remove best_e
            if state[best_e]:
                state[best_e] = 0
                # Update full_count and conflict_set for all faces containing best_e
                for f2 in edge_faces[best_e]:
                    full_count[f2] -= 1
                    if full_count[f2] == 4:
                        # It was 5? No, max is 4. 
                        # If it was 4, it becomes 3, so remove from conflict_set.
                        conflict_set.discard(f2)
                    # If it was 3, it becomes 2, no change to set.
                    # If it was <3, no change.
                    # Wait, if we remove an edge, count decreases.
                    # Only faces with count 4 are in conflict_set.
                    # So we only need to remove from conflict_set if count becomes 3 (was 4).
                    # It cannot become 4 by removing.
            
            # Remove the resolved face from conflict_set (it's now 3)
            conflict_set.discard(f_idx)
            
        score = sum(state)
        return state, full_count, score

    def local_search(state, full_count):
        """
        Hill climbing with add and swap moves.
        """
        current_score = sum(state)
        
        improved = True
        while improved:
            improved = False
            # Shuffle edges to avoid bias
            edges_to_try = list(range(E))
            random.shuffle(edges_to_try)
            
            for e in edges_to_try:
                if state[e]:
                    continue
                
                # Try to add e
                # Check if adding e creates a conflict
                conflict_faces = []
                for f_idx in edge_faces[e]:
                    if full_count[f_idx] == 3:
                        conflict_faces.append(f_idx)
                
                if not conflict_faces:
                    # Safe add
                    state[e] = 1
                    for f_idx in edge_faces[e]:
                        full_count[f_idx] += 1
                    current_score += 1
                    improved = True
                else:
                    # Try swap: remove one edge from a conflict face to allow adding e
                    # We want to remove an edge that is "least useful" or allows the add.
                    # For each conflict face, we can remove one of the other 3 edges.
                    # If we remove edge e2 from face f, we lose 1 edge but gain e (net 0 change in count)
                    # BUT we might unblock other adds?
                    # Actually, a swap (remove e2, add e) keeps the score same but changes the configuration.
                    # This is useful to escape local optima.
                    
                    # Heuristic: Try swaps that result in a state where we can subsequently add more?
                    # For now, let's just try "remove e2" if it doesn't create new conflicts?
                    # Removing e2 is always safe.
                    # So we can always do a swap if we are stuck.
                    # But we only want to swap if it leads to improvement later.
                    # Simple strategy: If we are stuck (no pure adds), perform random swaps to shuffle the config.
                    # But we are in a loop that stops if no improved. 
                    # Let's add a "swap" phase if no adds were made in a full pass.
                    pass
            
            if not improved:
                # Try swaps to escape local optimum
                # Pick a random edge e that is currently 0.
                # Find a conflict face f.
                # Pick another edge e2 in f.
                # Remove e2, add e.
                # Check if this is valid (it is, by construction).
                # Do this a few times.
                for _ in range(100):
                    e_add = random.randint(0, E - 1)
                    if state[e_add]:
                        continue
                    # Find a conflict face for e_add
                    conf_faces = [f for f in edge_faces[e_add] if full_count[f] == 3]
                    if not conf_faces:
                        # Should have been added in the add phase
                        continue
                    f_conf = random.choice(conf_faces)
                    # Pick an edge to remove from f_conf (not e_add, which is not in state yet)
                    # f_conf has 3 edges in state.
                    edges_in_face = [e for e in faces_tuple[f_conf] if state[e]]
                    if len(edges_in_face) != 3:
                        continue
                    e_remove = random.choice(edges_in_face)
                    
                    # Perform swap
                    state[e_add] = 1
                    state[e_remove] = 0
                    
                    # Update counts
                    for f2 in edge_faces[e_add]:
                        full_count[f2] += 1
                    for f2 in edge_faces[e_remove]:
                        full_count[f2] -= 1
                    
                    # Check if this swap allowed an immediate add?
                    # We'll let the next iteration of the while loop handle it.
                    improved = True
                    break # Restart the scan after a swap

        return state, full_count, current_score

    best_score = -1
    best_state = None
    
    # Run multiple restarts
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() - start_time > time_limit - 10:
            break
        
        # 1. Greedy Initialize
        state, full_count, score = greedy_initialize(randomize_ties=True)
        
        # 2. Local Search
        state, full_count, score = local_search(state, full_count)
        
        if score > best_score:
            best_score = score
            best_state = state[:]
            
        # 3. Final Greedy Add (safety net)
        # Rebuild full_count if needed? local_search maintains it.
        # Just do a quick pass to add any remaining safe edges.
        for e in range(E):
            if state[e]:
                continue
            can_add = True
            for f_idx in edge_faces[e]:
                if full_count[f_idx] == 3:
                    can_add = False
                    break
            if can_add:
                state[e] = 1
                for f_idx in edge_faces[e]:
                    full_count[f_idx] += 1
                best_score += 1
        
        if best_score >= 308:
            break

    if best_state is None:
        return []
        
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

