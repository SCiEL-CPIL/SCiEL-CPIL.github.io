import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    d = 7
    n = 1 << d
    all_edges = []
    edge_set = set()
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                all_edges.append((u, v))
                edge_set.add((u, v))
    
    # Precompute face constraints: for each edge, the 3 other edges in the 2D faces it belongs to
    # An edge (u,v) is in a face with (u,v^k) and (u^k, v^k) for k != i
    # Actually, a 2D face is defined by two bits i, j.
    # Edges in face (i,j) for base vertex u:
    # (u, u^i), (u, u^j), (u^i, u^i^j), (u^j, u^i^j)
    # We need to ensure not all 4 are present.
    
    # To speed up local search, we can precompute for each edge e, the list of "conflict sets"
    # If we add e, it conflicts if the other 3 edges of a face are present.
    # If we remove e, it frees up faces.
    
    # Let's use a simpler check for the local search steps, as n=128 is small.
    # Number of edges = 448.
    # Number of faces = C(7,2) * 2^(7-2) = 21 * 32 = 672.
    
    # Precompute faces as lists of 4 edges (tuples)
    faces = []
    for i in range(d):
        for j in range(i+1, d):
            for u in range(n):
                # Only consider u where bit i is 0 to avoid duplicates?
                # No, just generate all 4 vertices of the face.
                # Face vertices: u, u^i, u^j, u^i^j
                # To ensure unique faces, we can require u has bit i = 0 and bit j = 0?
                # Or just check if the set of 4 edges is new.
                v1 = u
                v2 = u ^ (1 << i)
                v3 = u ^ (1 << j)
                v4 = u ^ (1 << i) ^ (1 << j)
                
                e1 = (min(v1, v2), max(v1, v2))
                e2 = (min(v1, v3), max(v1, v3))
                e3 = (min(v2, v4), max(v2, v4))
                e4 = (min(v3, v4), max(v3, v4))
                
                face_edges = frozenset([e1, e2, e3, e4])
                faces.append(face_edges)
                
    # Remove duplicate faces (each face is generated 4 times, once for each vertex)
    unique_faces = list(set(faces))
    
    # For each edge, list the faces it belongs to
    edge_to_faces = {e: [] for e in all_edges}
    for idx, face in enumerate(unique_faces):
        for e in face:
            edge_to_faces[e].append(idx)
    
    # Initialize with a random valid set
    # Strategy: Randomly shuffle edges, add if valid
    random.seed(42)
    shuffled_edges = all_edges[:]
    random.shuffle(shuffled_edges)
    
    current_set = set()
    current_count = 0
    
    # Fast validity check helper
    def is_valid_add(edge, current_set):
        for face_idx in edge_to_faces[edge]:
            face = unique_faces[face_idx]
            # Check if other 3 edges are in current_set
            # face contains edge, so we check the others
            count_in_face = 0
            for e in face:
                if e in current_set:
                    count_in_face += 1
            # If edge is in current_set (it isn't yet), we'd add 1.
            # So if count_in_face == 3, adding edge makes it 4 -> invalid.
            if count_in_face == 3:
                return False
        return True

    for e in shuffled_edges:
        if is_valid_add(e, current_set):
            current_set.add(e)
            current_count += 1
            
    best_set = current_set.copy()
    best_count = current_count
    
    # Local Search
    # Temperature annealing
    T = 5.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Pick a random edge to flip
        e = random.choice(all_edges)
        
        if e in current_set:
            # Remove
            # Removing is always valid (reduces C4s)
            current_set.remove(e)
            current_count -= 1
            delta = -1
        else:
            # Add
            if not is_valid_add(e, current_set):
                continue # Skip invalid moves
            current_set.add(e)
            current_count += 1
            delta = 1
            
        # Acceptance
        if delta > 0:
            # Always accept improvement
            if current_count > best_count:
                best_count = current_count
                best_set = current_set.copy()
        else:
            # delta is -1
            # Accept with probability exp(-delta / T) = exp(1/T)
            # Wait, standard SA: accept if delta < 0 with prob exp(delta/T).
            # Here delta = -1. prob = exp(-1/T).
            if random.random() < (2.71828 ** (-1.0 / T)):
                pass
            else:
                # Reject: revert
                if delta < 0:
                    # We removed e, so add it back
                    current_set.add(e)
                    current_count += 1
                else:
                    # We added e, so remove it
                    current_set.remove(e)
                    current_count -= 1
        
        T *= cooling
        if T < 0.01:
            T = 0.01

    # Convert best_set to list of lists
    return [list(e) for e in best_set]
