import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    d = 7
    n_vertices = 1 << d
    n_edges = d * (1 << (d - 1))
    
    # Precompute edges
    edges = []
    edge_id = {}
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                eid = len(edges)
                edges.append((u, v))
                edge_id[(u, v)] = eid
                
    # Precompute faces
    face_edges = []
    edge_faces = [[] for _ in range(n_edges)]
    
    for i in range(d):
        for j in range(i + 1, d):
            bit_i = 1 << i
            bit_j = 1 << j
            for u in range(n_vertices):
                if (u & bit_i) == 0 and (u & bit_j) == 0:
                    v1 = u ^ bit_i
                    v2 = u ^ bit_j
                    v3 = u ^ bit_i ^ bit_j
                    
                    e1 = edge_id.get((min(u, v1), max(u, v1)))
                    e2 = edge_id.get((min(v1, v3), max(v1, v3)))
                    e3 = edge_id.get((min(u, v2), max(u, v2)))
                    e4 = edge_id.get((min(v2, v3), max(v2, v3)))
                    
                    if e1 is not None and e2 is not None and e3 is not None and e4 is not None:
                        fid = len(face_edges)
                        face_edges.append([e1, e2, e3, e4])
                        for e in [e1, e2, e3, e4]:
                            edge_faces[e].append(fid)
                            
    n_faces = len(face_edges)
    
    def can_add(eid, face_counts):
        for fid in edge_faces[eid]:
            if face_counts[fid] == 3:
                return False
        return True
        
    def add_edge(eid, selected, face_counts):
        selected[eid] = True
        for fid in edge_faces[eid]:
            face_counts[fid] += 1
            
    def remove_edge(eid, selected, face_counts):
        selected[eid] = False
        for fid in edge_faces[eid]:
            face_counts[fid] -= 1
            
    def get_edge_list(selected):
        return [[edges[i][0], edges[i][1]] for i in range(n_edges) if selected[i]]

    best_count = 0
    best_selected = None
    
    def greedy_random():
        selected = [False] * n_edges
        face_counts = [0] * n_faces
        order = list(range(n_edges))
        random.shuffle(order)
        
        count = 0
        for eid in order:
            if can_add(eid, face_counts):
                add_edge(eid, selected, face_counts)
                count += 1
        return selected, face_counts, count

    def local_search(selected, face_counts, count):
        current_selected = selected[:]
        current_face_counts = face_counts[:]
        current_count = count
        
        # Try to improve
        max_iter = 20
        for _ in range(max_iter):
            if time.time() > deadline:
                break
                
            improved = False
            rem_candidates = [i for i in range(n_edges) if current_selected[i]]
            random.shuffle(rem_candidates)
            
            for eid in rem_candidates:
                if time.time() > deadline:
                    break
                    
                saved_selected = current_selected[:]
                saved_face_counts = current_face_counts[:]
                saved_count = current_count
                
                remove_edge(eid, current_selected, current_face_counts)
                current_count -= 1
                
                candidates = set()
                for fid in edge_faces[eid]:
                    if current_face_counts[fid] == 2:
                        for e in face_edges[fid]:
                            if not current_selected[e]:
                                candidates.add(e)
                                
                cand_list = list(candidates)
                random.shuffle(cand_list)
                
                for ceid in cand_list:
                    if can_add(ceid, current_face_counts):
                        add_edge(ceid, current_selected, current_face_counts)
                        current_count += 1
                        
                if current_count > saved_count:
                    improved = True
                    break
                else:
                    current_selected = saved_selected
                    current_face_counts = saved_face_counts
                    current_count = saved_count
                    
            if not improved:
                break
                
        return current_selected, current_face_counts, current_count

    # Main loop
    while time.time() < deadline:
        selected, face_counts, count = greedy_random()
        
        if count > best_count:
            best_count = count
            best_selected = selected[:]
            
        if time.time() < deadline - 2.0:
            new_selected, new_face_counts, new_count = local_search(selected, face_counts, count)
            if new_count > best_count:
                best_count = new_count
                best_selected = new_selected[:]

    if best_selected is None:
        return []
        
    return get_edge_list(best_selected)
