import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    d = 7
    n = 2 ** d
    deadline = time.time() + 110.0

    edges = []
    edge_idx = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edges)
                edges.append((x, y))
                edge_idx[(x, y)] = idx
                edge_idx[(y, x)] = idx

    m = len(edges)

    face_edges = []
    edge_faces = [[] for _ in range(m)]

    for b in range(d):
        for c in range(b + 1, d):
            for a in range(n):
                if (a & (1 << b)) or (a & (1 << c)):
                    continue
                v0 = a
                v1 = a ^ (1 << b)
                v2 = a ^ (1 << c)
                v3 = a ^ (1 << b) ^ (1 << c)
                e0 = edge_idx[(min(v0, v1), max(v0, v1))]
                e1 = edge_idx[(min(v0, v2), max(v0, v2))]
                e2 = edge_idx[(min(v1, v3), max(v1, v3))]
                e3 = edge_idx[(min(v2, v3), max(v2, v3))]
                fid = len(face_edges)
                face_edges.append((e0, e1, e2, e3))
                for ei in (e0, e1, e2, e3):
                    edge_faces[ei].append(fid)

    num_faces = len(face_edges)
    edge_faces_arr = [np.array(f, dtype=np.int32) for f in edge_faces]

    def greedy_random(seed):
        rng = random.Random(seed)
        order = list(range(m))
        rng.shuffle(order)
        selected = np.zeros(m, dtype=bool)
        face_count = np.zeros(num_faces, dtype=np.int32)
        for ei in order:
            fids = edge_faces_arr[ei]
            if np.all(face_count[fids] < 3):
                selected[ei] = True
                face_count[fids] += 1
        return selected, face_count

    def complete(selected, face_count):
        changed = True
        while changed:
            changed = False
            for ei in range(m):
                if selected[ei]:
                    continue
                fids = edge_faces_arr[ei]
                if np.all(face_count[fids] < 3):
                    selected[ei] = True
                    face_count[fids] += 1
                    changed = True
        return selected, face_count

    def count_sel(selected):
        return int(np.sum(selected))

    best_selected, best_face_count = greedy_random(0)
    best_count = count_sel(best_selected)
    
    current_selected, current_face_count = best_selected.copy(), best_face_count.copy()
    current_count = best_count
    
    T = 10.0
    T_min = 0.5
    alpha = 0.99
    
    rng = random.Random(42)
    
    while time.time() < deadline - 5.0 and T > T_min:
        # Choose move type
        if rng.random() < 0.7:
            # 1-elimination
            sel_list = np.where(current_selected)[0]
            if len(sel_list) == 0: 
                continue
            ri = rng.choice(sel_list)
            
            old_selected = current_selected.copy()
            old_face_count = current_face_count.copy()
            old_count = current_count
            
            current_selected[ri] = False
            current_face_count[edge_faces_arr[ri]] -= 1
            
            current_selected, current_face_count = complete(current_selected, current_face_count)
            new_count = count_sel(current_selected)
            
            delta = new_count - old_count
            
            if delta > 0 or (T > 0 and rng.random() < np.exp(delta / T)):
                current_count = new_count
                if new_count > best_count:
                    best_count = new_count
                    best_selected = current_selected.copy()
                    best_face_count = current_face_count.copy()
            else:
                current_selected = old_selected
                current_face_count = old_face_count
                current_count = old_count
        else:
            # 2-elimination
            sel_list = np.where(current_selected)[0]
            if len(sel_list) < 2: 
                continue
            r1, r2 = rng.sample(sel_list.tolist(), 2)
            
            old_selected = current_selected.copy()
            old_face_count = current_face_count.copy()
            old_count = current_count
            
            current_selected[r1] = False
            current_face_count[edge_faces_arr[r1]] -= 1
            current_selected[r2] = False
            current_face_count[edge_faces_arr[r2]] -= 1
            
            current_selected, current_face_count = complete(current_selected, current_face_count)
            new_count = count_sel(current_selected)
            
            delta = new_count - old_count
            
            if delta > 0 or (T > 0 and rng.random() < np.exp(delta / T)):
                current_count = new_count
                if new_count > best_count:
                    best_count = new_count
                    best_selected = current_selected.copy()
                    best_face_count = current_face_count.copy()
            else:
                current_selected = old_selected
                current_face_count = old_face_count
                current_count = old_count

        T *= alpha

    out = []
    for ei in range(m):
        if best_selected[ei]:
            x, y = edges[ei]
            out.append([x, y])

    return out
# EVOLVE-BLOCK-END
