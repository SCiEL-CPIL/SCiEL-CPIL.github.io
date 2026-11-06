import random
import time
import math

def construct():
    d = 7
    n_vertices = 2 ** d
    total_edges = n_vertices * d // 2  # 448

    # Precompute all edges
    all_edges = []
    edge_to_index = {}
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                e = (x, y)
                edge_to_index[e] = len(all_edges)
                all_edges.append(e)

    # Precompute faces
    # Face is defined by start vertex u and two dimensions i, j
    # Vertices: u, u^ei, u^ej, u^ei^ej
    faces = []
    face_id = 0
    for u in range(n_vertices):
        for i in range(d):
            for j in range(i + 1, d):
                v = u ^ (1 << i)
                w = u ^ (1 << j)
                z = u ^ (1 << i) ^ (1 << j)
                
                # The 4 edges of this face
                e1 = (min(u, v), max(u, v))
                e2 = (min(v, z), max(v, z))
                e3 = (min(z, w), max(z, w))
                e4 = (min(w, u), max(w, u))
                
                face_edges = [e1, e2, e3, e4]
                faces.append(face_edges)
                face_id += 1
    
    num_faces = len(faces) # 672

    # Map edge index to list of face indices
    edge_to_faces = [[] for _ in range(total_edges)]
    # Map face index to list of edge indices
    face_to_edges = [[] for _ in range(num_faces)]

    for f_idx, f in enumerate(faces):
        for e in f:
            e_idx = edge_to_index[e]
            edge_to_faces[e_idx].append(f_idx)
            face_to_edges[f_idx].append(e_idx)
            
    # Ensure face_to_edges contains unique edge indices (they should be)
    # And edge_to_faces contains the face indices

    def greedy_initialization():
        chosen = set()
        face_counts = [0] * num_faces
        indices = list(range(total_edges))
        random.shuffle(indices)
        
        for e_idx in indices:
            # Check validity
            valid = True
            for f_idx in edge_to_faces[e_idx]:
                if face_counts[f_idx] == 3:
                    valid = False
                    break
            if valid:
                chosen.add(e_idx)
                for f_idx in edge_to_faces[e_idx]:
                    face_counts[f_idx] += 1
                    
        return chosen, face_counts

    def try_add_all(chosen, face_counts):
        # Add any missing edge that is valid
        # To be deterministic or random? Random is better for diversity.
        missing = [i for i in range(total_edges) if i not in chosen]
        random.shuffle(missing)
        added = 0
        for e_idx in missing:
            valid = True
            for f_idx in edge_to_faces[e_idx]:
                if face_counts[f_idx] == 3:
                    valid = False
                    break
            if valid:
                chosen.add(e_idx)
                for f_idx in edge_to_faces[e_idx]:
                    face_counts[f_idx] += 1
                added += 1
        return added

    def simulated_annealing():
        start_time = time.time()
        deadline = start_time + 110 # 10s margin
        
        # Initial state
        current_chosen, current_counts = greedy_initialization()
        current_size = len(current_chosen)
        
        best_chosen = set(current_chosen)
        best_size = current_size
        
        # Temperature schedule
        T = 5.0
        cooling_rate = 0.9995
        
        # Kick size range
        min_kick = 5
        max_kick = 25
        
        while time.time() < deadline:
            # 1. Try to fill up (local optimum)
            try_add_all(current_chosen, current_counts)
            current_size = len(current_chosen)
            
            if current_size > best_size:
                best_size = current_size
                best_chosen = set(current_chosen)
                
            # 2. Generate neighbor via Kick
            kick_size = random.randint(min_kick, max_kick)
            if kick_size > current_size:
                kick_size = max(1, current_size // 2)
                
            # Select random edges to remove
            # Using a set for removal
            removal_candidates = list(current_chosen)
            random.shuffle(removal_candidates)
            to_remove = set(removal_candidates[:kick_size])
            
            # Apply removal
            next_chosen = set(current_chosen)
            next_counts = current_counts[:] # Copy
            
            for e_idx in to_remove:
                next_chosen.remove(e_idx)
                for f_idx in edge_to_faces[e_idx]:
                    next_counts[f_idx] -= 1
            
            # Greedy Repair
            try_add_all(next_chosen, next_counts)
            next_size = len(next_chosen)
            
            # 3. Acceptance criterion
            delta = next_size - current_size
            if delta >= 0:
                # Accept
                current_chosen = next_chosen
                current_counts = next_counts
                current_size = next_size
            else:
                # Accept with probability exp(delta / T)
                # delta is negative
                prob = math.exp(delta / T)
                if random.random() < prob:
                    current_chosen = next_chosen
                    current_counts = next_counts
                    current_size = next_size
            
            # Cool down
            T *= cooling_rate
            if T < 0.1:
                T = 0.1
                
            # Occasionally restart if stuck for too long?
            # The kick mechanism should handle this.
            
        return best_chosen

    best_set = simulated_annealing()
    
    # Convert to list of [u, v]
    result = []
    for e_idx in best_set:
        u, v = all_edges[e_idx]
        result.append([u, v])
        
    return result

if __name__ == "__main__":
    # Quick test
    edges = construct()
    print(f"Found {len(edges)} edges")
