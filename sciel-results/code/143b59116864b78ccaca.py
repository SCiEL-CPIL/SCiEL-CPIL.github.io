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

    # 2. Greedy Initialization of Kept Set (via Hitting Set)
    # We find a small hitting set R, then K = All \ R
    def greedy_hitting_set(rng):
        covered = [False] * n_faces
        R = set()
        benefit = [len(edge_faces[e]) for e in range(n_edges)]
        
        # Use a max-heap for faster selection
        # heapq is a min-heap, so negate benefit
        heap = [(-benefit[e], e) for e in range(n_edges)]
        heapq.heapify(heap)
        
        for _ in range(n_faces):
            # Pop until we find a valid edge not in R with positive benefit
            chosen_e = -1
            while heap:
                neg_ben, e = heapq.heappop(heap)
                if e not in R and -neg_ben > 0:
                    chosen_e = e
                    break
            
            if chosen_e == -1:
                break
                
            R.add(chosen_e)
            
            for f in edge_faces[chosen_e]:
                if not covered[f]:
                    covered[f] = True
                    for e in face_edges[f]:
                        if e != chosen_e and e not in R:
                            benefit[e] -= 1
                            heapq.heappush(heap, (-benefit[e], e))
            
            if all(covered):
                break
            if time.time() - start_time > time_limit - 10:
                break
                
        return R

    def get_kept_set(R):
        return set(e for e in range(n_edges) if e not in R)

    # Precompute for each edge, the faces it belongs to (as a list)
    # And for each face, a set of its edges for O(1) membership check
    face_edge_sets = [set(fe) for fe in face_edges]
    
    # 3. Local Search on Kept Set with Validity Preserving Moves
    def local_search_kept(K_init, rng, time_budget):
        K = set(K_init)
        # Initialize face counts
        face_counts = [0] * n_faces
        for e in K:
            for f in edge_faces[e]:
                face_counts[f] += 1
        
        # Verify validity (should be valid)
        for f in range(n_faces):
            if face_counts[f] == 4:
                e_to_remove = face_edges[f][0]
                K.remove(e_to_remove)
                for ff in edge_faces[e_to_remove]:
                    face_counts[ff] -= 1
                break

        best_K = set(K)
        best_size = len(K)
        
        T_start = 2.0
        T_end = 0.01
        cooling = 0.995
        T = T_start
        
        end_time = time.time() + time_budget
        current_size = len(K)
        no_improve_count = 0
        max_no_improve = 500
        
        while time.time() < end_time:
            # Check if we should restart temperature
            if T < T_end or no_improve_count > max_no_improve:
                T = T_start
                no_improve_count = 0
                # Optionally restart from best solution
                if no_improve_count > max_no_improve * 2:
                    K = set(best_K)
                    for f in range(n_faces):
                        face_counts[f] = 0
                    for e in K:
                        for f in edge_faces[e]:
                            face_counts[f] += 1
                    current_size = len(K)
                    no_improve_count = 0
                    max_no_improve = int(max_no_improve * 1.5)
            
            # Generate a move
            move_type = rng.random()
            if move_type < 0.6:
                # Try Add
                e = rng.randint(0, n_edges - 1)
                if e in K:
                    continue
                
                # Check if adding e is valid
                valid = True
                for f in edge_faces[e]:
                    if face_counts[f] == 3:
                        valid = False
                        break
                
                if valid:
                    K.add(e)
                    for f in edge_faces[e]:
                        face_counts[f] += 1
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_K = set(K)
                        no_improve_count = 0
                else:
                    pass
                    
            elif move_type < 0.85:
                # Try Remove
                if not K:
                    continue
                e = rng.choice(list(K))
                
                delta = -1
                # Accept worsening move with SA probability
                if rng.random() < pow(2.718281828, -delta / T):
                    K.remove(e)
                    for f in edge_faces[e]:
                        face_counts[f] -= 1
                    current_size -= 1
                    no_improve_count += 1
            else:
                # Try Swap: Remove e_out, Add e_in
                if not K:
                    continue
                e_out = rng.choice(list(K))
                e_in = rng.randint(0, n_edges - 1)
                while e_in in K:
                    e_in = rng.randint(0, n_edges - 1)
                
                # Check if K - {e_out} + {e_in} is valid
                valid = True
                for f in edge_faces[e_in]:
                    e_out_in_face = e_out in face_edge_sets[f]
                    new_count = face_counts[f] - (1 if e_out_in_face else 0)
                    if new_count == 3:
                        valid = False
                        break
                
                if valid:
                    # Perform Swap
                    K.remove(e_out)
                    for f in edge_faces[e_out]:
                        face_counts[f] -= 1
                    
                    K.add(e_in)
                    for f in edge_faces[e_in]:
                        face_counts[f] += 1
                    # Size remains same - accept with high probability for exploration
                    if rng.random() > 0.3:
                        # Undo Swap
                        K.remove(e_in)
                        for f in edge_faces[e_in]:
                            face_counts[f] -= 1
                        K.add(e_out)
                        for f in edge_faces[e_out]:
                            face_counts[f] += 1
                    else:
                        no_improve_count += 1

            T *= cooling
            
        return best_K, best_size

    # 4. Multi-Start
    best_K = set()
    best_size = 0
    
    # Phase 1: Get a good initial solution
    rng_init = random.Random(42)
    R_init = greedy_hitting_set(rng_init)
    K_init = get_kept_set(R_init)
    
    # Run Local Search with multiple restarts
    elapsed = time.time() - start_time
    remaining_time = time_limit - 5 - elapsed
    if remaining_time > 5:
        # Split time among multiple local search runs
        n_restarts = min(5, max(1, int(remaining_time / 15)))
        time_per_restart = remaining_time / n_restarts
        
        for i in range(n_restarts):
            if time.time() - start_time > time_limit - 3:
                break
            
            seed = 123 + i * 1000
            K_best, size_best = local_search_kept(K_init, random.Random(seed), time_per_restart)
            if size_best > best_size:
                best_K = K_best
                best_size = size_best
            
            # Try a different hitting set for next restart
            if i < n_restarts - 1 and time.time() - start_time < time_limit - 10:
                rng_next = random.Random(42 + i + 1)
                R_next = greedy_hitting_set(rng_next)
                K_init = get_kept_set(R_next)

    # Final polish: try to add any remaining edges that can be added
    # This is a deterministic improvement pass
    K_final = set(best_K)
    face_counts_final = [0] * n_faces
    for e in K_final:
        for f in edge_faces[e]:
            face_counts_final[f] += 1
    
    # Try all edges in order, add if valid
    for e in range(n_edges):
        if e not in K_final:
            valid = True
            for f in edge_faces[e]:
                if face_counts_final[f] == 3:
                    valid = False
                    break
            if valid:
                K_final.add(e)
                for f in edge_faces[e]:
                    face_counts_final[f] += 1
    
    # Convert K (edge IDs) to list of [u, v]
    result = []
    for e_id in K_final:
        u, v = edges[e_id]
        result.append([u, v])
        
    return result
