import random
import time
import heapq

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

    # 2. Greedy Hitting Set (Priority queue for efficiency)
    def greedy_hitting_set(rng):
        covered = [False] * n_faces
        R = set()
        benefit = [len(edge_faces[e]) for e in range(n_edges)]
        
        # Priority queue: (-benefit, random_tiebreak, edge_id)
        heap = [(-benefit[e], rng.random(), e) for e in range(n_edges)]
        heapq.heapify(heap)
        
        n_uncovered = n_faces
        while n_uncovered > 0:
            # Pop until we find a valid edge
            chosen_e = -1
            chosen_ben = 0
            while heap:
                neg_b, _, e = heapq.heappop(heap)
                if e not in R and -neg_b == benefit[e] and benefit[e] > 0:
                    chosen_e = e
                    chosen_ben = benefit[e]
                    break
                elif e not in R and benefit[e] > 0:
                    # Re-push with updated benefit
                    heapq.heappush(heap, (-benefit[e], rng.random(), e))
            
            if chosen_e == -1:
                break
                
            R.add(chosen_e)
            
            # Update covered status and benefits
            for f in edge_faces[chosen_e]:
                if not covered[f]:
                    covered[f] = True
                    n_uncovered -= 1
                    for e in face_edges[f]:
                        if e != chosen_e and e not in R:
                            benefit[e] -= 1
                            if benefit[e] > 0:
                                heapq.heappush(heap, (-benefit[e], rng.random(), e))
            
            if time.time() - start_time > time_limit - 10:
                break
                
        return R

    # 3. Local Search Minimization with 1-removal and 2-opt swaps
    def minimize_hitting_set(R, rng):
        R = set(R)
        cover_count = [0] * n_faces
        for e in R:
            for f in edge_faces[e]:
                cover_count[f] += 1
        
        improved = True
        while improved:
            improved = False
            
            # Pass 1: Try to remove redundant edges
            r_list = list(R)
            rng.shuffle(r_list)
            for e in r_list:
                if e not in R:
                    continue
                redundant = True
                for f in edge_faces[e]:
                    if cover_count[f] == 1:
                        redundant = False
                        break
                if redundant:
                    R.remove(e)
                    for f in edge_faces[e]:
                        cover_count[f] -= 1
                    improved = True
                    if time.time() - start_time > time_limit - 5:
                        return R
            
            # Pass 2: Try 2-opt swaps (remove A, add B)
            # Only try if we have time
            if time.time() - start_time < time_limit - 10 and len(R) > 10:
                r_list2 = list(R)
                rng.shuffle(r_list2)
                for e_remove in r_list2:
                    if e_remove not in R:
                        continue
                    # What faces become uncovered if we remove e_remove?
                    newly_uncovered = []
                    for f in edge_faces[e_remove]:
                        if cover_count[f] == 1:
                            newly_uncovered.append(f)
                    if not newly_uncovered:
                        continue  # redundant, already handled in pass 1
                    
                    # Find best edge to add that covers most of newly_uncovered
                    # Candidate edges: those in newly_uncovered faces, not in R
                    best_e_add = -1
                    best_gain = 0
                    # Limit candidates to speed things up
                    candidate_edges = set()
                    for f in newly_uncovered:
                        for ce in face_edges[f]:
                            if ce not in R:
                                candidate_edges.add(ce)
                    
                    if len(candidate_edges) > 50:
                        candidate_edges = set(rng.sample(list(candidate_edges), 50))
                    
                    for e_add in candidate_edges:
                        gain = 0
                        for f in edge_faces[e_add]:
                            if cover_count[f] == 0:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_e_add = e_add
                    
                    if best_e_add != -1 and best_gain >= len(newly_uncovered):
                        # Perform swap: remove e_remove, add best_e_add
                        # Net effect: we lose coverage of newly_uncovered faces
                        # but gain coverage of best_gain previously uncovered faces
                        # Only do if net size doesn't increase (it's same size)
                        # and we cover at least as many previously uncovered faces
                        # Actually for minimization we just want same size or smaller
                        # This swap keeps size same, so only useful if it helps future removals
                        # Skip for now - just do removals
                        pass
                    
                    if time.time() - start_time > time_limit - 5:
                        return R
        
        return R

    # 4. Initialization with multiple seeds
    best_R = None
    best_size = float('inf')
    
    # Use more seeds since greedy is faster with priority queue
    seeds = list(range(1, 31))
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
        cover_count = [0] * n_faces
        for e in current_R:
            for f in edge_faces[e]:
                cover_count[f] += 1
                
        # While there are uncovered faces (maintain list incrementally)
        uncovered = [f for f in range(n_faces) if cover_count[f] == 0]
        while uncovered:
            f = rng.choice(uncovered)
            
            best_e = -1
            max_cover = -1
            for e in face_edges[f]:
                count = 0
                for ff in edge_faces[e]:
                    if cover_count[ff] == 0:
                        count += 1
                if count > max_cover:
                    max_cover = count
                    best_e = e
                elif count == max_cover:
                    if rng.random() < 0.5:
                        best_e = e
            
            if best_e != -1:
                current_R.add(best_e)
                # Update cover_count and uncovered incrementally
                for ff in edge_faces[best_e]:
                    if cover_count[ff] == 0:
                        # Remove from uncovered
                        uncovered.remove(ff)
                    cover_count[ff] += 1
            else:
                e = rng.choice(face_edges[f])
                current_R.add(e)
                for ff in edge_faces[e]:
                    if cover_count[ff] == 0:
                        uncovered.remove(ff)
                    cover_count[ff] += 1
            
        # Minimize
        current_R = minimize_hitting_set(current_R, rng)
        
        if len(current_R) < best_size:
            best_size = len(current_R)
            best_R = current_R
            if best_size <= 140:
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
