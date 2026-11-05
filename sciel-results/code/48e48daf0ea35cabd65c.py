import time
import random
import sys
import numpy as np

def construct():
    """
    Constructs a C4-free subgraph of Q7 maximizing edge count.
    Strategy:
    1. Deterministic precomputation of all edges and faces (C4s) using NumPy.
    2. Simulated Annealing global optimization.
       - State: Binary array of edges.
       - Move: Flip a single edge (add or remove).
       - Energy: Number of "full" faces (C4s with all 4 edges present).
       - Objective: Maximize edge count while maintaining Energy == 0.
       - This allows the search to temporarily accept invalid states (Energy > 0)
         to escape local optima, which is crucial for finding denser C4-free sets
         than simple hill-climbing.
    3. NumPy vectorization for fast energy evaluation and updates.
    """
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit
    
    d = 7
    n = 2 ** d  # 128 vertices
    
    # 1. Precompute edges and faces
    # Edge list: (u, v) where u < v and u^v is a power of 2
    edges = []
    edge_to_idx = {}
    
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edges)
                edges.append((x, y))
                edge_to_idx[(x, y)] = idx
    
    E = len(edges) # 448
    
    # Precompute faces (C4s)
    # Each face is defined by 4 edges.
    # We store faces as a 2D numpy array of shape (F, 4) containing edge indices.
    face_set = set()
    faces_list = []
    
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
                
                # Canonicalize to avoid duplicates
                # The set of 4 edges uniquely identifies the face
                face_tuple = tuple(sorted([idx1, idx2, idx3, idx4]))
                
                if face_tuple not in face_set:
                    face_set.add(face_tuple)
                    faces_list.append(face_tuple)
    
    F = len(faces_list) # 672
    faces = np.array(faces_list, dtype=np.int32) # Shape (F, 4)
    
    # Precompute for each edge, the indices of faces it belongs to
    # This allows fast update of face counts when an edge is flipped.
    edge_faces = [[] for _ in range(E)]
    for f_idx in range(F):
        for e_idx in faces[f_idx]:
            edge_faces[e_idx].append(f_idx)
    
    # Convert edge_faces to numpy arrays for faster iteration if needed, 
    # but lists are fine for small counts (6 faces per edge).
    
    # 2. Simulated Annealing Setup
    
    # State: 1 if edge present, 0 otherwise
    # Initial state: Random 50% or Parity
    # Parity construction is a good starting point (valid, 224 edges)
    # Let's start with a dense random state and let SA repair/optimize?
    # Or start with Parity and add?
    # SA works best if we can move freely. Let's start with a random state with p=0.5.
    
    rng = np.random.default_rng(42)
    
    # Initial state: Parity based
    # Edge (x, x^i) included if popcount(x) % 2 == 0
    state = np.zeros(E, dtype=np.uint8)
    for idx, (u, v) in enumerate(edges):
        # u < v. In parity, we check the "source" vertex.
        # To be consistent with the previous logic:
        # If we define the edge by the vertex with even popcount, it's always present in that construction.
        # Actually, simpler: just use the parity of the smaller vertex u?
        # The previous analysis showed:
        # Face 0-1-3-2: (0,1) yes, (0,2) yes, (1,3) no, (2,3) no.
        # 0 (00) pop 0 even. 1 (01) pop 1 odd.
        # Edge (0,1): u=0 (even) -> Yes.
        # Edge (0,2): u=0 (even) -> Yes.
        # Edge (1,3): u=1 (odd) -> No. (1 is min(1,3))
        # Edge (2,3): u=2 (odd) -> No. (2 is min(2,3))
        # So: Include if popcount(min(u,v)) % 2 == 0.
        if bin(u).count('1') % 2 == 0:
            state[idx] = 1
            
    # Calculate initial face counts
    # face_counts[f] = number of edges in face f that are present in state
    face_counts = np.zeros(F, dtype=np.int32)
    
    # Vectorized initialization of face_counts
    # faces is (F, 4). We want sum(state[faces[f, :]] for each f)
    # We can use fancy indexing
    # state[faces] gives shape (F, 4). Sum along axis 1.
    face_counts = np.sum(state[faces], axis=1)
    
    # Initial energy: number of faces with count 4
    energy = np.sum(face_counts == 4)
    # Initial score
    score = np.sum(state)
    
    best_score = score
    best_state = state.copy()
    
    # If initial state is invalid, we still track it, but SA will try to fix it.
    # However, the problem requires a VALID output.
    # So we must ensure we return a valid state.
    # If we end up with energy > 0, we must repair it deterministically before returning.
    
    # SA Parameters
    T_start = 1.0
    T_end = 0.01
    alpha = 0.999 # Cooling rate
    
    # We need to check time frequently.
    # NumPy operations are fast.
    
    current_T = T_start
    
    # Precompute list of edges for random selection
    edge_indices = np.arange(E)
    
    while time.time() - start_time < time_limit:
        # Check if we can break early if we hit the theoretical max?
        # Max is 448. Unlikely to reach.
        
        # Pick a random edge to flip
        e = rng.choice(edge_indices)
        
        # Calculate delta energy and delta score
        # Flipping edge e affects all faces containing e.
        # Let affected_faces = edge_faces[e]
        # For each face f in affected_faces:
        #   old_count = face_counts[f]
        #   new_count = old_count + 1 if state[e]==0 else old_count - 1
        #   old_full = 1 if old_count == 4 else 0
        #   new_full = 1 if new_count == 4 else 0
        #   delta_energy += new_full - old_full
        
        # We can vectorize this if we store face_counts in a numpy array.
        
        old_state_val = state[e]
        
        # Get indices of faces containing e
        f_indices = np.array(edge_faces[e], dtype=np.int32)
        
        # Get current counts for these faces
        old_counts = face_counts[f_indices]
        
        # New counts
        if old_state_val == 0:
            new_counts = old_counts + 1
        else:
            new_counts = old_counts - 1
            
        # Calculate change in "full" status (count == 4)
        old_full_mask = (old_counts == 4).astype(np.int32)
        new_full_mask = (new_counts == 4).astype(np.int32)
        
        delta_energy = np.sum(new_full_mask - old_full_mask)
        
        # Delta score
        delta_score = 1 if old_state_val == 0 else -1
        
        # Acceptance probability
        # We want to maximize score, minimize energy.
        # Let's define a combined cost: C = Energy * 1000 - Score
        # We want to minimize C.
        # Delta C = delta_energy * 1000 - delta_score
        
        delta_C = delta_energy * 1000 - delta_score
        
        accepted = False
        if delta_C <= 0:
            accepted = True
        else:
            # Metropolis criterion
            # P = exp(-delta_C / T)
            # If delta_C is large, P is small.
            # Note: delta_C is integer.
            # If current_T is very small, we only accept improvements.
            if current_T > 0:
                # Avoid overflow/underflow in exp
                # delta_C / T
                ratio = delta_C / current_T
                if ratio < 700: # exp(-700) is ~0
                    p = np.exp(-ratio)
                    if rng.random() < p:
                        accepted = True
        
        if accepted:
            # Apply move
            state[e] = 1 - state[e]
            
            # Update face_counts
            face_counts[f_indices] = new_counts
            
            # Update global energy and score
            energy += delta_energy
            score += delta_score
            
            # Track best valid state
            if energy == 0:
                if score > best_score:
                    best_score = score
                    best_state = state.copy()
                    
            # Also, if we are in a valid state, and we made a move that kept us valid (delta_energy 0, delta_score 1),
            # that's an improvement.
            
        # Cool down
        current_T *= alpha
        
        # If T is too low, restart? Or just keep going?
        # With alpha 0.999, it takes a long time to cool.
        # Maybe we should restart with a new random state if stuck?
        # For now, let's just let it run. The "random walk" at low T is still possible due to exp(-delta/T).
        
        # Optional: Periodic restart if no improvement for a long time?
        # To keep it simple and robust, we'll just let the SA run.
        
    # After time limit, if best_state is invalid, we must repair it.
    # The task requires: "No 2-dimensional face may have all four edges selected."
    # So we must return a valid set.
    
    final_state = best_state.copy()
    
    # Repair if necessary
    # Greedy removal of edges from full faces
    # Recalculate face counts for final_state to be safe
    final_face_counts = np.sum(final_state[faces], axis=1)
    
    # While there is a face with count 4
    while np.any(final_face_counts == 4):
        # Find a face with count 4
        full_face_indices = np.where(final_face_counts == 4)[0]
        if len(full_face_indices) == 0:
            break
        f_idx = full_face_indices[0]
        face = faces[f_idx]
        
        # Pick an edge to remove from this face
        # Heuristic: Remove the edge that is in the fewest other "almost full" (count 3) faces?
        # Or just random?
        # To be fast, just pick the first one.
        e_to_remove = face[0]
        
        # Remove it
        final_state[e_to_remove] = 0
        
        # Update counts for faces containing e_to_remove
        f_indices = np.array(edge_faces[e_to_remove], dtype=np.int32)
        final_face_counts[f_indices] -= 1
        
    # Final check and construct result
    result = []
    for i in range(E):
        if final_state[i]:
            u, v = edges[i]
            result.append([u, v])
            
    return result

if __name__ == "__main__":
    # Quick test
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Edges: {len(res)}, Time: {end-start:.2f}s")
    
    # Verify validity
    # Re-check faces
    d = 7
    n = 2 ** d
    edges_set = set()
    for e in res:
        edges_set.add(tuple(sorted(e)))
        
    valid = True
    for u in range(n):
        for i in range(d):
            for j in range(i+1, d):
                v1 = u
                v2 = u ^ (1 << i)
                v3 = u ^ (1 << j)
                v4 = u ^ (1 << i) ^ (1 << j)
                
                e1 = tuple(sorted((v1, v2)))
                e2 = tuple(sorted((v1, v3)))
                e3 = tuple(sorted((v2, v4)))
                e4 = tuple(sorted((v3, v4)))
                
                if e1 in edges_set and e2 in edges_set and e3 in edges_set and e4 in edges_set:
                    valid = False
                    print("Invalid C4 found!")
                    break
            if not valid: break
        if not valid: break
        
    print(f"Valid: {valid}")
