import random
import time

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin

    d = 7
    n = 2 ** d
    
    # 1. Precompute edges and faces
    # Edge ID: (u, v) where u < v
    edges = []
    edge_id_map = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                edge_id = len(edges)
                edges.append((x, y))
                edge_id_map[(x, y)] = edge_id
                edge_id_map[(y, x)] = edge_id # just in case, though we keep u<v

    n_edges = len(edges)
    
    # Face ID: (base_vertex, i, j) where i < j
    # The face is defined by v, v^ei, v^ej, v^ei^ej
    # Edges in face: (v, v^ei), (v^ei, v^ei^ej), (v, v^ej), (v^ej, v^ei^ej)
    # We need to map these to edge IDs.
    
    face_edges = [] # list of lists of 4 edge IDs
    # To build edge_faces efficiently, we can just append
    
    # First, generate all faces
    face_list = [] # (v, i, j)
    for v in range(n):
        for i in range(d):
            for j in range(i+1, d):
                face_id = len(face_list)
                face_list.append((v, i, j))
                
                v_i = v ^ (1 << i)
                v_j = v ^ (1 << j)
                v_ij = v ^ (1 << i) ^ (1 << j)
                
                # Edges:
                # e1: (v, v_i)
                # e2: (v_i, v_ij)
                # e3: (v, v_j)
                # e4: (v_j, v_ij)
                
                e1 = edge_id_map[(min(v, v_i), max(v, v_i))]
                e2 = edge_id_map[(min(v_i, v_ij), max(v_i, v_ij))]
                e3 = edge_id_map[(min(v, v_j), max(v, v_j))]
                e4 = edge_id_map[(min(v_j, v_ij), max(v_j, v_ij))]
                
                face_edges.append([e1, e2, e3, e4])

    n_faces = len(face_edges)
    
    # Build edge_faces: for each edge, list of face indices
    edge_faces = [[] for _ in range(n_edges)]
    for f_idx, f_edges in enumerate(face_edges):
        for e_idx in f_edges:
            edge_faces[e_idx].append(f_idx)

    # 2. Greedy Hitting Set (Optimized with incremental benefit updates)
    def greedy_hitting_set(rng):
        covered = [False] * n_faces
        R = set()
        # benefit[e] = number of currently uncovered faces covered by edge e
        benefit = [len(edge_faces[e]) for e in range(n_edges)]
        
        for _ in range(n_faces):
            best_e = -1
            max_ben = -1
            # Find edge with max benefit
            # We can also collect ties for randomization
            max_edges = []
            for e in range(n_edges):
                if e not in R:
                    if benefit[e] > max_ben:
                        max_ben = benefit[e]
                        max_edges = [e]
                    elif benefit[e] == max_ben:
                        max_edges.append(e)
            
            if not max_edges or max_ben == 0:
                break
                
            chosen_e = rng.choice(max_edges)
            R.add(chosen_e)
            
            # Update covered status and benefits
            for f in edge_faces[chosen_e]:
                if not covered[f]:
                    covered[f] = True
                    # Decrement benefit for all other edges covering this face
                    for e in face_edges[f]:
                        if e != chosen_e and e not in R:
                            benefit[e] -= 1
            
            if all(covered):
                break
            if time.time() - start_time > time_limit - 10:
                break
                
        return R

    # 3. Local Search Minimization
    def minimize_hitting_set(R, rng):
        R = set(R)
        # Calculate cover counts
        cover_count = [0] * n_faces
        for e in R:
            for f in edge_faces[e]:
                cover_count[f] += 1
        
        improved = True
        while improved:
            improved = False
            # Try to remove each edge
            # Random order for non-determinism
            r_list = list(R)
            rng.shuffle(r_list)
            for e in r_list:
                if e not in R:
                    continue
                # Check if e is redundant
                redundant = True
                for f in edge_faces[e]:
                    if cover_count[f] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove e
                    R.remove(e)
                    for f in edge_faces[e]:
                        cover_count[f] -= 1
                    improved = True
                    if time.time() - start_time > time_limit - 5:
                        return R
        return R

    # 4. Initialization with multiple seeds
    best_R = None
    best_size = float('inf')
    
    # Use more seeds since greedy is faster
    seeds = list(range(1, 21))
    for seed in seeds:
        if time.time() - start_time > time_limit - 15:
            break
        R_candidate = greedy_hitting_set(random.Random(seed))
        R_candidate = minimize_hitting_set(R_candidate, random.Random(seed))
        if len(R_candidate) < best_size:
            best_size = len(R_candidate)
            best_R = R_candidate
            
    if best_R is None:
        best_R = greedy_hitting_set(random.Random(123))
        best_size = len(best_R)

    rng = random.Random(42)
    
    while time.time() - start_time < time_limit - 10:
        # Perturb
        current_R = set(best_R)
        
        # Remove 1-3 random edges
        num_remove = rng.randint(1, 3)
        to_remove = rng.sample(list(current_R), min(num_remove, len(current_R)))
        for e in to_remove:
            current_R.remove(e)
            
        # Repair
        # Calculate current cover status
        cover_count = [0] * n_faces
        for e in current_R:
            for f in edge_faces[e]:
                cover_count[f] += 1
                
        # While there are uncovered faces
        uncovered = [f for f in range(n_faces) if cover_count[f] == 0]
        while uncovered:
            f = rng.choice(uncovered)
            
            best_e = -1
            max_cover = -1
            # Candidates are edges in face f not in current_R
            # Since f is uncovered, no edge in face_edges[f] is in current_R.
            for e in face_edges[f]:
                count = 0
                for ff in edge_faces[e]:
                    if cover_count[ff] == 0:
                        count += 1
                if count > max_cover:
                    max_cover = count
                    best_e = e
                elif count == max_cover:
                    # Random tie-breaking
                    if rng.random() < 0.5:
                        best_e = e
            
            if best_e != -1:
                current_R.add(best_e)
                for ff in edge_faces[best_e]:
                    cover_count[ff] += 1
            else:
                # Fallback (should not happen)
                e = rng.choice(face_edges[f])
                current_R.add(e)
                for ff in edge_faces[e]:
                    cover_count[ff] += 1
                
            # Re-calculate uncovered
            uncovered = [ff for ff in range(n_faces) if cover_count[ff] == 0]
            
        # Minimize
        current_R = minimize_hitting_set(current_R, rng)
        
        if len(current_R) < best_size:
            best_size = len(current_R)
            best_R = current_R
            # Optional: break early if we hit theoretical lower bound?
            # Lower bound for hitting set size?
            # If we get 140, that's 308 edges.
            if best_size <= 140:
                # We found 308 edges! Very good.
                pass

    # Convert R (edge IDs) to list of [u, v]
    # We want to RETURN the edges that are NOT in R.
    # Wait, the task is to MAXIMIZE edges.
    # R is the set of REMOVED edges.
    # So the result is all edges EXCEPT those in R.
    
    result = []
    for e_id, (u, v) in enumerate(edges):
        if e_id not in best_R:
            result.append([u, v])
            
    return result
