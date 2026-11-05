import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    d = 7
    n_vertices = 1 << d
    n_edges = d * (1 << (d - 1)) # 448
    
    # Precompute edges
    # Edge index mapping
    edge_map = {}
    edges = []
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edges)
                edges.append((x, y))
                edge_map[(x, y)] = idx
    
    # Precompute faces
    # A face is defined by (base_vertex, i, j) where i < j
    # Vertices: x, x^2^i, x^2^j, x^2^(i+j)
    # Edges: (x, x^2^i), (x^2^i, x^2^(i+j)), (x^2^(i+j), x^2^j), (x^2^j, x)
    
    faces = []
    face_to_edges = []
    edge_to_faces = [[] for _ in range(n_edges)]
    
    for x in range(n_vertices):
        for i in range(d):
            for j in range(i + 1, d):
                f_idx = len(faces)
                v1 = x
                v2 = x ^ (1 << i)
                v3 = x ^ (1 << j)
                v4 = x ^ (1 << i) ^ (1 << j)
                
                # Ensure consistent ordering for edge lookup
                e1 = (min(v1, v2), max(v1, v2))
                e2 = (min(v2, v4), max(v2, v4))
                e3 = (min(v4, v3), max(v4, v3))
                e4 = (min(v3, v1), max(v3, v1))
                
                e_ids = [edge_map[e1], edge_map[e2], edge_map[e3], edge_map[e4]]
                faces.append(e_ids)
                face_to_edges.append(e_ids)
                
                for eid in e_ids:
                    edge_to_faces[eid].append(f_idx)
    
    n_faces = len(faces)
    
    def greedy_random():
        present = [False] * n_edges
        face_count = [0] * n_faces
        order = list(range(n_edges))
        random.shuffle(order)
        
        count = 0
        for e in order:
            # Check if adding e creates a full face
            can_add = True
            for f in edge_to_faces[e]:
                if face_count[f] >= 3:
                    can_add = False
                    break
            
            if can_add:
                present[e] = True
                count += 1
                for f in edge_to_faces[e]:
                    face_count[f] += 1
        
        return present, face_count, count

    def try_add(present, face_count, count):
        # Try to add any valid edge
        # We can shuffle or just scan
        # To be fast, we can keep a list of "candidate" edges? 
        # For now, just scan all missing edges in random order until one is found
        missing = [e for e in range(n_edges) if not present[e]]
        random.shuffle(missing)
        
        added = False
        for e in missing:
            can_add = True
            for f in edge_to_faces[e]:
                if face_count[f] >= 3:
                    can_add = False
                    break
            
            if can_add:
                present[e] = True
                count += 1
                for f in edge_to_faces[e]:
                    face_count[f] += 1
                added = True
                break # Only add one at a time to allow SA steps? 
                # Or add all possible? Adding all possible is a "greedy step".
                # Let's add all possible in this call to make progress.
                # But if we add all, we reach local max immediately.
                # Let's modify: try_add returns count of added.
        
        return count

    # Better Local Search Step:
    def local_search_step(present, face_count, count):
        # Try to add edges greedily
        changed = True
        while changed:
            changed = False
            # Randomize order to avoid bias
            order = list(range(n_edges))
            random.shuffle(order)
            for e in order:
                if not present[e]:
                    can_add = True
                    for f in edge_to_faces[e]:
                        if face_count[f] >= 3:
                            can_add = False
                            break
                    if can_add:
                        present[e] = True
                        count += 1
                        for f in edge_to_faces[e]:
                            face_count[f] += 1
                        changed = True
        return count

    best_count = 0
    best_present = None
    best_face_count = None
    
    # Phase 1: Random Greedy Restarts
    # We have ~110s.
    # Each greedy is fast.
    # Each local search step is fast.
    
    # Let's do a few greedy restarts, then SA on the best one.
    
    for _ in range(5): # 5 restarts
        if time.time() > deadline:
            break
        p, fc, c = greedy_random()
        c = local_search_step(p, fc, c)
        if c > best_count:
            best_count = c
            best_present = p[:]
            best_face_count = fc[:]
    
    # Phase 2: Simulated Annealing
    if best_present is None:
        p, fc, c = greedy_random()
        c = local_search_step(p, fc, c)
        best_count = c
        best_present = p[:]
        best_face_count = fc[:]
    
    current_present = best_present[:]
    current_face_count = best_face_count[:]
    current_count = best_count
    
    T = 0.1
    T_min = 0.001
    alpha = 0.99
    
    while time.time() < deadline:
        # Move: Pick random edge
        e = random.randint(0, n_edges - 1)
        
        if current_present[e]:
            # Remove edge
            # This always reduces count by 1
            # Accept with prob exp(-1/T)
            if random.random() < (1.0 / T): # exp(-1/T) approx 1/T for small T? No.
                # Correct: exp(-1/T)
                import math
                prob = math.exp(-1.0 / T)
                if random.random() < prob:
                    # Remove
                    current_present[e] = False
                    current_count -= 1
                    for f in edge_to_faces[e]:
                        current_face_count[f] -= 1
        else:
            # Add edge
            # Check validity
            can_add = True
            for f in edge_to_faces[e]:
                if current_face_count[f] >= 3:
                    can_add = False
                    break
            
            if can_add:
                # Always accept improvement
                current_present[e] = True
                current_count += 1
                for f in edge_to_faces[e]:
                    current_face_count[f] += 1
            
        # Update best
        if current_count > best_count:
            best_count = current_count
            best_present = current_present[:]
            best_face_count = current_face_count[:]
            
        # Cool down
        T *= alpha
        if T < T_min:
            # Reheat or restart?
            # If we stagnated, reheat
            T = 0.1
    
    # Final Greedy Polish
    final_count = local_search_step(best_present, best_face_count, best_count)
    
    # Construct output
    out = []
    for e in range(n_edges):
        if best_present[e]:
            u, v = edges[e]
            out.append([u, v])
            
    return out

if __name__ == "__main__":
    construct()
