import random
import time

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin
    
    d = 7
    n = 2 ** d
    num_edges = d * (2 ** (d - 1))
    
    edges = []
    edge_index = {}
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(edges)
                edges.append((u, v))
                edge_index[(u, v)] = idx
                edge_index[(v, u)] = idx
    
    faces = []
    face_index = {}
    for i in range(d):
        for j in range(i + 1, d):
            for x in range(n):
                if (x & (1 << i)) == 0 and (x & (1 << j)) == 0:
                    f_idx = len(faces)
                    e1 = (x, x ^ (1 << i))
                    e2 = (x, x ^ (1 << j))
                    e3 = (x ^ (1 << i), x ^ (1 << i) ^ (1 << j))
                    e4 = (x ^ (1 << j), x ^ (1 << i) ^ (1 << j))
                    
                    def get_eidx(u, v):
                        if u > v: u, v = v, u
                        return edge_index[(u, v)]
                    
                    f_edges = [
                        get_eidx(*e1),
                        get_eidx(*e2),
                        get_eidx(*e3),
                        get_eidx(*e4)
                    ]
                    faces.append(f_edges)
                    face_index[(i, j, x)] = f_idx
                    
    edge_to_faces = [[] for _ in range(num_edges)]
    for f_idx, f_edges in enumerate(faces):
        for e_idx in f_edges:
            edge_to_faces[e_idx].append(f_idx)

    selected = [False] * num_edges
    face_count = [0] * len(faces)
    current_size = 0
    
    def add_edge(e_idx):
        nonlocal current_size
        if selected[e_idx]: return
        selected[e_idx] = True
        current_size += 1
        for f_idx in edge_to_faces[e_idx]:
            face_count[f_idx] += 1
            
    def remove_edge(e_idx):
        nonlocal current_size
        if not selected[e_idx]: return
        selected[e_idx] = False
        current_size -= 1
        for f_idx in edge_to_faces[e_idx]:
            face_count[f_idx] -= 1
            
    def can_add(e_idx):
        for f_idx in edge_to_faces[e_idx]:
            if face_count[f_idx] >= 3:
                return False
        return True

    # 1. Initial Greedy Construction
    edge_order = list(range(num_edges))
    random.shuffle(edge_order)
    
    for e_idx in edge_order:
        if can_add(e_idx):
            add_edge(e_idx)
            
    best_size = current_size
    best_selected = selected[:]
    
    # 2. Local Search
    all_edge_indices = list(range(num_edges))
    
    while time.time() - start_time < time_limit:
        e_rem = random.choice(all_edge_indices)
        
        if not selected[e_rem]:
            continue
            
        remove_edge(e_rem)
        
        candidate_faces = edge_to_faces[e_rem]
        candidates = set()
        for f_idx in candidate_faces:
            for e_idx in faces[f_idx]:
                if not selected[e_idx]:
                    candidates.add(e_idx)
        
        cand_list = list(candidates)
        random.shuffle(cand_list)
        
        added_edges = []
        for e_add in cand_list:
            if can_add(e_add):
                add_edge(e_add)
                added_edges.append(e_add)
        
        if current_size > best_size:
            best_size = current_size
            best_selected = selected[:]
        else:
            # Revert
            for e_add in added_edges:
                remove_edge(e_add)
            add_edge(e_rem)

    out = []
    for i in range(num_edges):
        if best_selected[i]:
            u, v = edges[i]
            out.append([u, v])
            
    return out
