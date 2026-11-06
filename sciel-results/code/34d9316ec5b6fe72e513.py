import time
import random
import math
import numpy as np

def construct():
    d = 7
    n = 1 << d  # 128 vertices
    deadline = time.time() + 118  # Safety margin

    # 1. Precompute edges
    all_edges = []
    edge_idx = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(all_edges)
                all_edges.append((x, y))
                edge_idx[(x, y)] = idx
    m = len(all_edges)  # 448

    # 2. Precompute faces and edge-face incidence
    face_edges = []
    edge_faces = [[] for _ in range(m)]
    
    for i in range(d):
        for j in range(i + 1, d):
            for x in range(n):
                if (x >> i) & 1 == 0 and (x >> j) & 1 == 0:
                    a = x
                    b = x ^ (1 << i)
                    c = x ^ (1 << j)
                    d2 = x ^ (1 << i) ^ (1 << j)
                    
                    e1 = edge_idx[(min(a, b), max(a, b))]
                    e2 = edge_idx[(min(b, d2), max(b, d2))]
                    e3 = edge_idx[(min(d2, c), max(d2, c))]
                    e4 = edge_idx[(min(c, a), max(c, a))]
                    
                    fid = len(face_edges)
                    face_edges.append([e1, e2, e3, e4])
                    for ei in (e1, e2, e3, e4):
                        edge_faces[ei].append(fid)
    
    n_faces = len(face_edges)  # 672

    # 3. Greedy hitting set initialization
    def greedy_init(seed=0):
        rng = random.Random(seed)
        R = np.zeros(m, dtype=np.uint8)
        face_count = np.zeros(n_faces, dtype=np.int32)
        uncovered = np.ones(n_faces, dtype=np.uint8)
        num_uncovered = n_faces
        
        # Precompute edge-face lists as numpy arrays for speed
        ef_lists = [np.array(edge_faces[e], dtype=np.int32) for e in range(m)]
        
        while num_uncovered > 0:
            best_e = -1
            best_gain = -1
            for e in range(m):
                if R[e]:
                    continue
                # Count uncovered faces covered by edge e
                faces_e = ef_lists[e]
                gain = np.sum(uncovered[faces_e])
                if gain > best_gain:
                    best_gain = gain
                    best_e = e
                    if gain == 0:
                        break
            
            if best_e == -1:
                break
            
            R[best_e] = 1
            faces_e = ef_lists[best_e]
            for f in faces_e:
                if uncovered[f]:
                    uncovered[f] = 0
                    num_uncovered -= 1
                face_count[f] += 1
        
        return R, face_count

    # 4. Local search with numpy
    def local_search(R_init, face_count_init, seed=0, time_budget=30):
        rng = random.Random(seed)
        R = R_init.copy()
        face_count = face_count_init.copy()
        
        current_size = int(np.sum(R))
        current_uncovered = int(np.sum(face_count == 0))
        
        best_size = current_size
        best_R = R.copy()
        best_uncovered = current_uncovered
        
        T = 1.5
        T_min = 0.01
        cooling = 0.9995
        t0 = time.time()
        
        # Precompute edge-face as list of lists for fast access
        ef = edge_faces
        
        while time.time() - t0 < time_budget:
            if T < T_min:
                T = T_min
            
            # Random move: flip one edge or swap
            if rng.random() < 0.6:
                # Single flip
                e = rng.randint(0, m - 1)
                if R[e]:
                    # Try removing e
                    faces_e = ef[e]
                    inc_unc = sum(1 for f in faces_e if face_count[f] == 1)
                    delta_cost = -1 + 10.0 * inc_unc
                    
                    if delta_cost < 0 or rng.random() < math.exp(-delta_cost / T):
                        R[e] = 0
                        for f in faces_e:
                            face_count[f] -= 1
                        current_size -= 1
                        current_uncovered += inc_unc
                else:
                    # Try adding e
                    faces_e = ef[e]
                    dec_unc = sum(1 for f in faces_e if face_count[f] == 0)
                    delta_cost = 1 - 10.0 * dec_unc
                    
                    if delta_cost < 0 or rng.random() < math.exp(-delta_cost / T):
                        R[e] = 1
                        for f in faces_e:
                            face_count[f] += 1
                        current_size += 1
                        current_uncovered -= dec_unc
            else:
                # Swap: remove one in R, add one not in R
                in_R = np.where(R == 1)[0]
                not_R = np.where(R == 0)[0]
                if len(in_R) == 0 or len(not_R) == 0:
                    T *= cooling
                    continue
                
                e_rem = in_R[rng.randint(0, len(in_R) - 1)]
                e_add = not_R[rng.randint(0, len(not_R) - 1)]
                
                faces_rem = ef[e_rem]
                faces_add = ef[e_add]
                
                # Compute delta correctly
                inc_unc = 0
                dec_unc = 0
                # Build set of affected faces
                affected = set(faces_rem) | set(faces_add)
                for f in affected:
                    old_c = face_count[f]
                    new_c = old_c
                    if f in faces_rem:
                        new_c -= 1
                    if f in faces_add:
                        new_c += 1
                    if old_c == 0 and new_c > 0:
                        dec_unc += 1
                    if old_c > 0 and new_c == 0:
                        inc_unc += 1
                
                delta_cost = 10.0 * (inc_unc - dec_unc)
                
                if delta_cost < 0 or rng.random() < math.exp(-delta_cost / T):
                    R[e_rem] = 0
                    R[e_add] = 1
                    for f in faces_rem:
                        face_count[f] -= 1
                    for f in faces_add:
                        face_count[f] += 1
                    current_uncovered += inc_unc - dec_unc
                    # current_size unchanged
            
            # Update best
            if current_uncovered == 0 and current_size < best_size:
                best_size = current_size
                best_R = R.copy()
            
            T *= cooling
        
        # If best is not valid, fall back to current if valid
        if best_uncovered > 0:
            # Check if current is valid
            if current_uncovered == 0:
                best_size = current_size
                best_R = R.copy()
            else:
                # Use the initial valid solution
                best_R = R_init.copy()
                best_size = int(np.sum(R_init))
        
        return best_R, best_size

    # 5. Greedy reduction: remove redundant edges from R
    def greedy_reduce(R):
        R = R.copy()
        face_count = np.zeros(n_faces, dtype=np.int32)
        for e in range(m):
            if R[e]:
                for f in edge_faces[e]:
                    face_count[f] += 1
        
        # Try to remove each edge if it doesn't make any face uncovered
        for e in range(m):
            if R[e]:
                faces_e = edge_faces[e]
                can_remove = all(face_count[f] > 1 for f in faces_e)
                if can_remove:
                    R[e] = 0
                    for f in faces_e:
                        face_count[f] -= 1
        
        return R

    # 6. Main search
    best_R_overall = None
    best_size_overall = m  # worst case: remove all edges
    
    # First, get a valid solution from greedy
    R_g, fc_g = greedy_init(seed=42)
    R_g = greedy_reduce(R_g)
    size_g = int(np.sum(R_g))
    best_R_overall = R_g
    best_size_overall = size_g
    
    # Verify validity
    fc_check = np.zeros(n_faces, dtype=np.int32)
    for e in range(m):
        if best_R_overall[e]:
            for f in edge_faces[e]:
                fc_check[f] += 1
    if not np.all(fc_check > 0):
        # Shouldn't happen, but fallback
        R_g2, fc_g2 = greedy_init(seed=123)
        best_R_overall = R_g2
        best_size_overall = int(np.sum(R_g2))
    
    # Multiple restarts with local search
    t_start = time.time()
    restart = 0
    while time.time() < deadline - 5:  # Leave 5s for final processing
        elapsed = time.time() - t_start
        remaining = deadline - 5 - time.time()
        
        # Allocate time for this restart
        if restart < 3:
            time_budget = min(30, remaining * 0.4)
        else:
            time_budget = min(20, remaining * 0.3)
        
        if time_budget < 2:
            break
        
        seed = 1000 + restart * 7919
        R_init, fc_init = greedy_init(seed=seed)
        R_init = greedy_reduce(R_init)
        size_init = int(np.sum(R_init))
        
        # Update best if greedy alone is better
        if size_init < best_size_overall:
            best_size_overall = size_init
            best_R_overall = R_init.copy()
        
        # Local search
        R_ls, size_ls = local_search(R_init, fc_init, seed=seed, time_budget=time_budget)
        
        if size_ls < best_size_overall:
            best_size_overall = size_ls
            best_R_overall = R_ls.copy()
        
        restart += 1
    
    # Final greedy reduction on best solution
    best_R_overall = greedy_reduce(best_R_overall)
    best_size_overall = int(np.sum(best_R_overall))
    
    # Final verification
    fc_final = np.zeros(n_faces, dtype=np.int32)
    for e in range(m):
        if best_R_overall[e]:
            for f in edge_faces[e]:
                fc_final[f] += 1
    if not np.all(fc_final > 0):
        # Fallback to a known valid solution
        R_fb, fc_fb = greedy_init(seed=999)
        R_fb = greedy_reduce(R_fb)
        best_R_overall = R_fb
        best_size_overall = int(np.sum(R_fb))
    
    # 7. Construct output: selected edges are those NOT in R
    out = []
    for e in range(m):
        if not best_R_overall[e]:
            x, y = all_edges[e]
            out.append([x, y])
    
    return out

if __name__ == "__main__":
    t0 = time.time()
    edges = construct()
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s, Edges: {len(edges)}")
    
    # Quick validation
    d = 7
    n = 1 << d
    selected_set = set()
    for u, v in edges:
        selected_set.add((min(u, v), max(u, v)))
    
    valid = True
    for i in range(d):
        for j in range(i + 1, d):
            for x in range(n):
                if (x >> i) & 1 == 0 and (x >> j) & 1 == 0:
                    a = x
                    b = x ^ (1 << i)
                    c = x ^ (1 << j)
                    d2 = x ^ (1 << i) ^ (1 << j)
                    
                    e1 = (min(a, b), max(a, b))
                    e2 = (min(b, d2), max(b, d2))
                    e3 = (min(d2, c), max(d2, c))
                    e4 = (min(c, a), max(c, a))
                    
                    cnt = sum(1 for e in [e1, e2, e3, e4] if e in selected_set)
                    if cnt == 4:
                        print(f"Invalid face at {x}, dims {i},{j}")
                        valid = False
                        break
            if not valid:
                break
        if not valid:
            break
    if valid:
        print("Valid!")
    else:
        print("INVALID!")
