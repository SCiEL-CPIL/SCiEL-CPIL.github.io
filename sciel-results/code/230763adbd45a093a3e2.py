import time
import random
import sys
import math

def construct():
    """
    Constructs a C4-free subgraph of Q7 maximizing edge count.
    Strategy: WalkSAT-style stochastic local search on the 2-SAT encoding.
    1. Precompute all faces (C4s) and edge-face incidence.
    2. Initialize with a dense random state, repair to valid.
    3. WalkSAT loop:
       - If valid, try to add a random edge that keeps validity.
       - If invalid, pick a random violated face, flip a random edge in it
         with probability biased toward adding (to maximize edge count).
       - Track best valid state.
    4. Periodic greedy polish and restarts.
    5. Final greedy polish on best state.
    """
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit
    
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
    
    # Precompute faces (C4s)
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
    
    # Precompute edge-to-faces mapping
    edge_faces = [[] for _ in range(E)]
    for f_idx, face in enumerate(faces):
        for e_idx in face:
            edge_faces[e_idx].append(f_idx)
    
    def repair(state, face_counts):
        """Remove edges to eliminate all C4s."""
        while True:
            conflict_face = -1
            for f_idx in range(F):
                if face_counts[f_idx] == 4:
                    conflict_face = f_idx
                    break
            if conflict_face == -1:
                break
            face = faces[conflict_face]
            e_to_remove = random.choice(face)
            state[e_to_remove] = False
            for f_idx in edge_faces[e_to_remove]:
                face_counts[f_idx] -= 1
        return state, face_counts
    
    def greedy_polish(state, face_counts, deadline):
        """Try to add any edge that doesn't create a conflict. Repeat until no more can be added."""
        while time.time() < deadline:
            added = False
            for e in range(E):
                if not state[e]:
                    can_add = True
                    for f_idx in edge_faces[e]:
                        if face_counts[f_idx] == 3:
                            can_add = False
                            break
                    if can_add:
                        state[e] = True
                        for f_idx in edge_faces[e]:
                            face_counts[f_idx] += 1
                        added = True
            if not added:
                break
        return state, face_counts
    
    random.seed(42)
    
    best_score = -1
    best_state = None
    
    # Initial state: dense random, then repair
    state = [random.random() < 0.85 for _ in range(E)]
    face_counts = [0] * F
    for f_idx, face in enumerate(faces):
        count = 0
        for e in face:
            if state[e]:
                count += 1
        face_counts[f_idx] = count
    
    state, face_counts = repair(state, face_counts)
    current_score = sum(state)
    
    if current_score > best_score:
        best_score = current_score
        best_state = state[:]
    
    # WalkSAT parameters
    add_bias = 0.7  # Probability of trying to add vs remove when flipping in a violated face
    stagnation_counter = 0
    max_stagnation = 5000
    last_improve_time = start_time
    no_improve_limit = 8.0
    
    steps = 0
    
    while time.time() < time_limit - 5.0:
        # Check if state is valid
        is_valid = True
        violated_face = -1
        for f_idx in range(F):
            if face_counts[f_idx] == 4:
                is_valid = False
                violated_face = f_idx
                break
        
        if is_valid:
            # State is valid: try to add a random absent edge
            # Find a random absent edge
            e = random.randint(0, E - 1)
            if not state[e]:
                can_add = True
                for f_idx in edge_faces[e]:
                    if face_counts[f_idx] == 3:
                        can_add = False
                        break
                if can_add:
                    state[e] = True
                    current_score += 1
                    for f_idx in edge_faces[e]:
                        face_counts[f_idx] += 1
                    if current_score > best_score:
                        best_score = current_score
                        best_state = state[:]
                        stagnation_counter = 0
                        last_improve_time = time.time()
                    else:
                        stagnation_counter += 1
                else:
                    stagnation_counter += 1
            else:
                stagnation_counter += 1
        else:
            # State is invalid: pick a random violated face, flip an edge in it
            # First find all violated faces
            violated_faces = [f_idx for f_idx in range(F) if face_counts[f_idx] == 4]
            if violated_faces:
                f_idx = random.choice(violated_faces)
                face = faces[f_idx]
                e = random.choice(face)
                
                # Decide: add or remove?
                # If edge is absent, adding it to a face with count 4 is impossible (would make 5)
                # Actually if face_count is 4, all 4 edges are present. So we must remove one.
                # But we can also consider: remove this edge, which might fix this face
                # but could break other faces that were at count 3.
                
                if state[e]:
                    # Remove edge e
                    state[e] = False
                    current_score -= 1
                    for f_idx2 in edge_faces[e]:
                        face_counts[f_idx2] -= 1
                    stagnation_counter += 1
                else:
                    # Edge is absent but face has count 4? That means all 4 edges present.
                    # This shouldn't happen if face_counts is correct.
                    # Just skip.
                    stagnation_counter += 1
        
        # Stagnation handling
        if stagnation_counter > max_stagnation:
            # Perturbation: remove some random edges to create space
            num_remove = random.randint(10, 25)
            for _ in range(num_remove):
                e = random.randint(0, E - 1)
                if state[e]:
                    state[e] = False
                    for f_idx2 in edge_faces[e]:
                        face_counts[f_idx2] -= 1
                    current_score -= 1
            stagnation_counter = 0
            last_improve_time = time.time()
        
        if time.time() - last_improve_time > no_improve_limit:
            # More aggressive restart
            num_remove = random.randint(25, 45)
            for _ in range(num_remove):
                e = random.randint(0, E - 1)
                if state[e]:
                    state[e] = False
                    for f_idx2 in edge_faces[e]:
                        face_counts[f_idx2] -= 1
                    current_score -= 1
            # Greedy re-add
            state, face_counts = greedy_polish(state, face_counts, time.time() + 2.0)
            current_score = sum(state)
            if current_score > best_score:
                best_score = current_score
                best_state = state[:]
            stagnation_counter = 0
            last_improve_time = time.time()
        
        steps += 1
    
    # Final greedy polish on best state
    if best_state is not None:
        polish_state = best_state[:]
        polish_face_counts = [0] * F
        for f_idx, face in enumerate(faces):
            count = 0
            for e in face:
                if polish_state[e]:
                    count += 1
            polish_face_counts[f_idx] = count
        
        polish_state, polish_face_counts = greedy_polish(polish_state, polish_face_counts, time_limit - 1.0)
        polish_score = sum(polish_state)
        
        if polish_score > best_score:
            best_score = polish_score
            best_state = polish_state
    
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
