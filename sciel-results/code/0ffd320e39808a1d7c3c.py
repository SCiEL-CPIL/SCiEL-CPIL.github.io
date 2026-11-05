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

    def greedy_hitting_set(rng):
        benefit = [len(edge_faces[e]) for e in range(n_edges)]
        is_covered = [False] * n_faces
        R = set()
        
        heap = [(-benefit[e], e) for e in range(n_edges)]
        heapq.heapify(heap)
        
        uncovered_count = n_faces
        
        while uncovered_count > 0:
            if time.time() - start_time > time_limit - 10:
                break
                
            neg_b, e = heapq.heappop(heap)
            b = -neg_b
            
            if b != benefit[e]:
                continue
            if e in R:
                continue
            if b == 0:
                break
                
            R.add(e)
            
            for f in edge_faces[e]:
                if not is_covered[f]:
                    is_covered[f] = True
                    uncovered_count -= 1
                    
                    for e_prime in face_edges[f]:
                        if e_prime != e:
                            benefit[e_prime] -= 1
                            heapq.heappush(heap, (-benefit[e_prime], e_prime))
                            
        return R

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

    best_R = greedy_hitting_set(random.Random(123))
    best_size = len(best_R)
    
    rng = random.Random(42)
    
    while time.time() - start_time < time_limit - 10:
        current_R = set(best_R)
        
        num_remove = rng.randint(1, 3)
        to_remove = rng.sample(list(current_R), min(num_remove, len(current_R)))
        for e in to_remove:
            current_R.remove(e)
            
        cover_count = [0] * n_faces
        for e in current_R:
            for f in edge_faces[e]:
                cover_count[f] += 1
                
        uncovered = [f for f in range(n_faces) if cover_count[f] == 0]
        while uncovered:
            f = rng.choice(uncovered)
            e_candidates = face_edges[f]
            e = rng.choice(e_candidates)
            
            if e not in current_R:
                current_R.add(e)
                for ff in edge_faces[e]:
                    cover_count[ff] += 1
                
            uncovered = [ff for ff in range(n_faces) if cover_count[ff] == 0]
            
        current_R = minimize_hitting_set(current_R, rng)
        
        if len(current_R) < best_size:
            best_size = len(current_R)
            best_R = current_R

    result = []
    for e_id, (u, v) in enumerate(edges):
        if e_id not in best_R:
            result.append([u, v])
            
    return result
