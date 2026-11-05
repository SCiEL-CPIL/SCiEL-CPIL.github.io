import random
import time
import heapq

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin

    d = 7
    n = 2 ** d
    
    # 1. Precompute edges and faces
    edges = []
    edge_id_map = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                edge_id = len(edges)
                edges.append((x, y))
                edge_id_map[(x, y)] = edge_id
                edge_id_map[(y, x)] = edge_id

    n_edges = len(edges)
    
    face_edges = []
    face_list = []
    for v in range(n):
        for i in range(d):
            for j in range(i+1, d):
                face_id = len(face_list)
                face_list.append((v, i, j))
                
                v_i = v ^ (1 << i)
                v_j = v ^ (1 << j)
                v_ij = v ^ (1 << i) ^ (1 << j)
                
                e1 = edge_id_map[(min(v, v_i), max(v, v_i))]
                e2 = edge_id_map[(min(v_i, v_ij), max(v_i, v_ij))]
                e3 = edge_id_map[(min(v, v_j), max(v, v_j))]
                e4 = edge_id_map[(min(v_j, v_ij), max(v_j, v_ij))]
                
                face_edges.append([e1, e2, e3, e4])

    n_faces = len(face_edges)
    
    edge_faces = [[] for _ in range(n_edges)]
    for f_idx, f_edges in enumerate(face_edges):
        for e_idx in f_edges:
            edge_faces[e_idx].append(f_idx)

    # 2. Greedy Hitting Set with heap-based selection
    def greedy_hitting_set(rng):
        covered = [False] * n_faces
        R = set()
        benefit = [len(edge_faces[e]) for e in range(n_edges)]
        
        # Max-heap: (-benefit, edge_id)
        heap = [(-benefit[e], e) for e in range(n_edges)]
        heapq.heapify(heap)
        
        in_R = [False] * n_edges
        valid_benefit = benefit[:]  # Track actual current benefit
        
        for _ in range(n_faces):
            # Find max benefit edge not in R
            while heap:
                neg_ben, e = heap[0]
                if in_R[e]:
                    heapq.heappop(heap)
                    continue
                if -neg_ben != valid_benefit[e]:
                    heapq.heappop(heap)
                    continue
                break
            
            if not heap:
                break
                
            neg_ben, best_e = heapq.heappop(heap)
            max_ben = -neg_ben
            
            if max_ben == 0:
                break
            
            # Collect ties for randomization
            max_edges = [best_e]
            while heap:
                neg_ben2, e2 = heap[0]
                if in_R[e2]:
                    heapq.heappop(heap)
                    continue
                if -neg_ben2 != valid_benefit[e2]:
                    heapq.heappop(heap)
                    continue
                if -neg_ben2 == max_ben:
                    heapq.heappop(heap)
                    max_edges.append(e2)
                else:
                    break
            
            chosen_e = rng.choice(max_edges)
            in_R[chosen_e] = True
            R.add(chosen_e)
            
            # Update covered status and benefits
            for f in edge_faces[chosen_e]:
                if not covered[f]:
                    covered[f] = True
                    for e in face_edges[f]:
                        if e != chosen_e and not in_R[e]:
                            valid_benefit[e] -= 1
                            # Push updated benefit
                            heapq.heappush(heap, (-valid_benefit[e], e))
            
            if all(covered):
                break
            if time.time() - start_time > time_limit - 10:
                break
                
        return R

    # 3. Local Search Minimization with simulated annealing
    def minimize_hitting_set(R, rng, temperature=1.0):
        R = set(R)
        cover_count = [0] * n_faces
        for e in R:
            for f in edge_faces[e]:
                cover_count[f] += 1
        
        improved = True
        while improved:
            improved = False
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
        return R

    # 4. Initialization with multiple seeds
    best_R = None
    best_size = float('inf')
    
    seeds = list(range(1, 11))
    for seed in seeds:
        if time.time() - start_time > time_limit - 20:
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
    
    # Track best for annealing
    best_size_anneal = best_size
    
    while time.time() - start_time < time_limit - 15:
        # Perturb
        current_R = set(best_R)
        
        num_remove = rng.randint(2, 6)
        to_remove = rng.sample(list(current_R), min(num_remove, len(current_R)))
        for e in to_remove:
            current_R.remove(e)
            
        # Repair
        cover_count = [0] * n_faces
        for e in current_R:
            for f in edge_faces[e]:
                cover_count[f] += 1
                
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
                for ff in edge_faces[best_e]:
                    cover_count[ff] += 1
            else:
                e = rng.choice(face_edges[f])
                current_R.add(e)
                for ff in edge_faces[e]:
                    cover_count[ff] += 1
                
            uncovered = [ff for ff in range(n_faces) if cover_count[ff] == 0]
            
        # Minimize
        current_R = minimize_hitting_set(current_R, rng)
        
        # Simulated annealing acceptance
        if len(current_R) < best_size:
            best_size = len(current_R)
            best_R = current_R
        else:
            # Accept worse solutions with probability exp(-delta/T)
            delta = len(current_R) - best_size
            T = max(0.5, 5.0 * (1.0 - (time.time() - start_time) / time_limit))
            if rng.random() < __import__('math').exp(-delta / T):
                best_R = current_R
                best_size = len(current_R)

    # Convert R (edge IDs) to list of [u, v]
    result = []
    for e_id, (u, v) in enumerate(edges):
        if e_id not in best_R:
            result.append([u, v])
            
    return result
