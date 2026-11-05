import random
import time

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
    for v in range(n):
        for i in range(d):
            for j in range(i+1, d):
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

    # 2. Greedy Hitting Set (Optimized)
    def greedy_hitting_set(rng):
        covered = [False] * n_faces
        R = set()
        benefit = [len(edge_faces[e]) for e in range(n_edges)]
        
        for _ in range(n_faces):
            best_e = -1
            max_ben = -1
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
            
            for f in edge_faces[chosen_e]:
                if not covered[f]:
                    covered[f] = True
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

    # 4. Repair
    def repair(R, rng):
        cover_count = [0] * n_faces
        for e in R:
            for f in edge_faces[e]:
                cover_count[f] += 1
        
        while True:
            uncovered = [f for f in range(n_faces) if cover_count[f] == 0]
            if not uncovered:
                break
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
                elif count == max_cover and rng.random() < 0.5:
                    best_e = e
            
            if best_e != -1:
                R.add(best_e)
                for ff in edge_faces[best_e]:
                    cover_count[ff] += 1
            else:
                e = rng.choice(face_edges[f])
                R.add(e)
                for ff in edge_faces[e]:
                    cover_count[ff] += 1
            if time.time() - start_time > time_limit - 5:
                break
        
        return R

    # 5. Simulated Annealing Function
    def simulated_annealing(initial_R, rng, time_budget):
        current_R = set(initial_R)
        current_size = len(current_R)
        best_R = set(current_R)
        best_size = current_size
        
        T_start = 5.0
        T_end = 0.1
        cooling = 0.995
        
        T = T_start
        no_improve = 0
        end_time = time.time() + time_budget
        
        while time.time() < end_time:
            if T < T_end:
                T = T_start
                no_improve = 0
            
            # Perturb: remove random edges
            num_remove = rng.randint(2, 8)
            to_remove = rng.sample(list(current_R), min(num_remove, len(current_R)))
            perturbed_R = set(current_R)
            for e in to_remove:
                perturbed_R.remove(e)
            
            # Repair
            perturbed_R = repair(perturbed_R, rng)
            perturbed_R = minimize_hitting_set(perturbed_R, rng)
            perturbed_size = len(perturbed_R)
            
            # Acceptance criterion
            delta = perturbed_size - current_size
            if delta <= 0:
                current_R = perturbed_R
                current_size = perturbed_size
                no_improve = 0
                if current_size < best_size:
                    best_size = current_size
                    best_R = set(current_R)
                    if best_size <= 140:
                        break
            else:
                if rng.random() < pow(2.718281828, -delta / T):
                    current_R = perturbed_R
                    current_size = perturbed_size
                no_improve += 1
            
            T *= cooling
            
        return best_R, best_size

    # 6. Multi-Start Strategy
    best_R = None
    best_size = float('inf')
    
    # Phase 1: Generate 5 diverse candidates
    candidates = []
    for seed in range(1, 6):
        if time.time() - start_time > time_limit - 50:
            break
        R_candidate = greedy_hitting_set(random.Random(seed))
        R_candidate = minimize_hitting_set(R_candidate, random.Random(seed))
        candidates.append((R_candidate, len(R_candidate)))
        
    if not candidates:
        R_candidate = greedy_hitting_set(random.Random(123))
        R_candidate = minimize_hitting_set(R_candidate, random.Random(123))
        candidates.append((R_candidate, len(R_candidate)))
        
    # Sort candidates by size
    candidates.sort(key=lambda x: x[1])
    
    # Phase 2: Run SA on the best 3 candidates for a fixed time slice
    # Total time remaining approx 115 - (time spent in phase 1)
    # Assume phase 1 took ~5s. Remaining ~110s.
    # Allocate 10s to each of top 3, then rest to the best overall.
    
    top_candidates = candidates[:3]
    time_per_sa = 10.0
    
    for idx, (R_init, size_init) in enumerate(top_candidates):
        if time.time() - start_time > time_limit - 20:
            break
        rng = random.Random(1000 + idx)
        best_R_sa, best_size_sa = simulated_annealing(R_init, rng, time_per_sa)
        if best_size_sa < best_size:
            best_size = best_size_sa
            best_R = best_R_sa
            
    # Phase 3: Run long SA on the best found so far
    if best_R is None:
        best_R, best_size = top_candidates[0]
        
    if time.time() - start_time < time_limit - 5:
        rng = random.Random(999)
        # Calculate remaining time
        remaining_time = time_limit - 5 - (time.time() - start_time)
        if remaining_time > 1:
            final_R, final_size = simulated_annealing(best_R, rng, remaining_time)
            if final_size < best_size:
                best_size = final_size
                best_R = final_R

    # Final minimization
    best_R = minimize_hitting_set(best_R, random.Random(99))
    
    # Convert R (edge IDs) to list of [u, v]
    result = []
    for e_id, (u, v) in enumerate(edges):
        if e_id not in best_R:
            result.append([u, v])
            
    return result
