import time
import random
import sys
import math

def construct():
    """
    Constructs a C4-free subgraph of Q7 maximizing edge count.
    Strategy: Simulated Annealing with efficient conflict tracking.
    1. Precompute all faces (C4s) and edge-face incidence.
    2. Initialize with a dense random state or parity state.
    3. Repair initial state to remove any C4s.
    4. Run Simulated Annealing:
       - Move: Flip a random edge (add or remove).
       - Accept if it improves edge count (always).
       - Accept if it decreases edge count with probability exp(-delta/T).
       - This allows escaping local optima where greedy addition is stuck.
    5. Track best valid state found.
    6. Final greedy polish pass.
    """
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit
    
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
    
    # State representation:
    # present: list of bools
    # face_counts: list of ints (0-4), number of present edges in each face.
    # A face is a conflict if face_counts[f] == 4.
    
    def repair(state, face_counts):
        """Remove edges to eliminate all C4s. Returns updated state and face_counts."""
        # While there are conflicts, remove an edge from a conflicting face
        while True:
            # Find any conflict
            conflict_face = -1
            for f_idx in range(F):
                if face_counts[f_idx] == 4:
                    conflict_face = f_idx
                    break
            
            if conflict_face == -1:
                break
            
            face = faces[conflict_face]
            # Pick an edge to remove.
            e_to_remove = random.choice(face)
            
            state[e_to_remove] = False
            
            # Update face counts
            for f_idx in edge_faces[e_to_remove]:
                face_counts[f_idx] -= 1
                
        return state, face_counts
    
    def get_score(state):
        return sum(state)
    
    def greedy_polish(state, face_counts, remaining_time):
        """Try to add any edge that doesn't create a conflict. Repeat until no more can be added."""
        while time.time() - start_time < remaining_time:
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
    
    # Main Search Loop
    random.seed(42)
    
    best_score = -1
    best_state = None
    
    # Initial state: Parity construction
    state = [False] * E
    for idx, (u, v) in enumerate(edge_list):
        if bin(u).count('1') % 2 == 0:
            state[idx] = True
            
    face_counts = [0] * F
    for f_idx, face in enumerate(faces):
        count = 0
        for e in face:
            if state[e]:
                count += 1
        face_counts[f_idx] = count
    
    state, face_counts = repair(state, face_counts)
    current_score = get_score(state)
    
    if current_score > best_score:
        best_score = current_score
        best_state = state[:]
        
    # Simulated Annealing Parameters
    T_start = 2.0
    T_end = 0.005
    cooling_rate = 0.9998  # Slower cooling for more exploration
    
    # We want to perform as many steps as possible.
    
    T = T_start
    steps = 0
    stagnation_counter = 0
    max_stagnation = 8000  # Restart if stuck for this many steps
    
    # Track the last time we improved to monitor progress
    last_improve_time = start_time
    no_improve_limit = 10.0  # If no improvement for 10s, do a bigger restart
    
    while time.time() - start_time < time_limit:
        # Pick a random edge to flip
        e = random.randint(0, E - 1)
        
        if state[e]:
            # Try to remove edge e
            # Delta score = -1
            # Accept with probability exp(-1/T)
            
            p_accept = math.exp(-1.0 / T)
            
            if random.random() < p_accept:
                # Accept removal
                state[e] = False
                current_score -= 1
                for f_idx in edge_faces[e]:
                    face_counts[f_idx] -= 1
                stagnation_counter += 1
            else:
                stagnation_counter += 1
                
        else:
            # Try to add edge e
            # Check if adding creates a conflict
            can_add = True
            for f_idx in edge_faces[e]:
                if face_counts[f_idx] == 3:
                    can_add = False
                    break
            
            if can_add:
                # Accept addition (always improves score)
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
                # Cannot add without creating conflict
                # Reject
                stagnation_counter += 1
        
        # Cooling
        T *= cooling_rate
        if T < T_end:
            T = T_end
            
        # Restart if stagnation
        if stagnation_counter > max_stagnation:
            # Restart with a perturbation
            # Remove more edges to escape local optimum
            num_remove = random.randint(15, 30)
            for _ in range(num_remove):
                e = random.randint(0, E - 1)
                if state[e]:
                    state[e] = False
                    for f_idx in edge_faces[e]:
                        face_counts[f_idx] -= 1
                    current_score -= 1
            
            # Reset T and stagnation
            T = T_start * 0.5  # Restart with lower temperature
            stagnation_counter = 0
            last_improve_time = time.time()
        
        # If no improvement for a long time, do a bigger restart
        if time.time() - last_improve_time > no_improve_limit:
            # More aggressive restart
            num_remove = random.randint(30, 50)
            for _ in range(num_remove):
                e = random.randint(0, E - 1)
                if state[e]:
                    state[e] = False
                    for f_idx in edge_faces[e]:
                        face_counts[f_idx] -= 1
                    current_score -= 1
            
            T = T_start
            stagnation_counter = 0
            last_improve_time = time.time()
        
        steps += 1
    
    # Final greedy polish pass
    if best_state is not None:
        # Apply greedy polish to the best state
        polish_state = best_state[:]
        polish_face_counts = [0] * F
        for f_idx, face in enumerate(faces):
            count = 0
            for e in face:
                if polish_state[e]:
                    count += 1
            polish_face_counts[f_idx] = count
        
        polish_state, polish_face_counts = greedy_polish(polish_state, polish_face_counts, time_limit - 2.0)
        polish_score = get_score(polish_state)
        
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
