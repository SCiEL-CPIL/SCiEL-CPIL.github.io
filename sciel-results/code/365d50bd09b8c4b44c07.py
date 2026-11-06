import time
import random

def construct():
    D = 7
    N = 2 ** D
    NUM_EDGES = D * (2 ** (D - 1))
    NUM_FACES = (D * (D - 1) // 2) * (2 ** (D - 2))

    edges = []
    edge_index = {}
    for x in range(N):
        for i in range(D):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edges)
                edges.append((x, y))
                edge_index[(x, y)] = idx
            else:
                # x > y, so y < x
                idx = len(edges)
                edges.append((y, x))
                edge_index[(y, x)] = idx
                # Note: The loop structure `if x < y` skips half.
                # Let's just iterate all and add if x < y.
                # Then lookup must use min/max.
                
    # Re-do edge construction cleanly
    edges = []
    edge_index = {}
    for x in range(N):
        for i in range(D):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edges)
                edges.append((x, y))
                edge_index[(x, y)] = idx

    edge_to_faces = [[] for _ in range(NUM_EDGES)]
    face_edges = [[] for _ in range(NUM_FACES)]
    
    face_idx = 0
    face_map = {}
    for i in range(D):
        for j in range(i + 1, D):
            for base in range(N):
                if (base & (1 << i)) == 0 and (base & (1 << j)) == 0:
                    face_map[(i, j, base)] = face_idx
                    face_idx += 1

    for i in range(D):
        for j in range(i + 1, D):
            for base in range(N):
                if (base & (1 << i)) == 0 and (base & (1 << j)) == 0:
                    f_idx = face_map[(i, j, base)]
                    v0 = base
                    v1 = base ^ (1 << i)
                    v2 = base ^ (1 << j)
                    v3 = base ^ (1 << i) ^ (1 << j)
                    
                    e1 = (min(v0, v1), max(v0, v1))
                    e2 = (min(v0, v2), max(v0, v2))
                    e3 = (min(v1, v3), max(v1, v3))
                    e4 = (min(v2, v3), max(v2, v3))
                    
                    e1_idx = edge_index[e1]
                    e2_idx = edge_index[e2]
                    e3_idx = edge_index[e3]
                    e4_idx = edge_index[e4]
                    
                    face_edges[f_idx] = [e1_idx, e2_idx, e3_idx, e4_idx]
                    for e_idx in [e1_idx, e2_idx, e3_idx, e4_idx]:
                        edge_to_faces[e_idx].append(f_idx)

    deadline = time.time() + 145
    best_count = 0
    best_selected = [False] * NUM_EDGES
    
    def can_add(e_idx, selected, face_load):
        for f_idx in edge_to_faces[e_idx]:
            if face_load[f_idx] >= 3:
                return False
        return True
        
    def add_edge(e_idx, selected, face_load):
        selected[e_idx] = True
        for f_idx in edge_to_faces[e_idx]:
            face_load[f_idx] += 1
            
    def remove_edge(e_idx, selected, face_load):
        selected[e_idx] = False
        for f_idx in edge_to_faces[e_idx]:
            face_load[f_idx] -= 1
            
    def greedy_fill(selected, face_load, order):
        count = 0
        for e_idx in order:
            if not selected[e_idx] and can_add(e_idx, selected, face_load):
                add_edge(e_idx, selected, face_load)
                count += 1
        return count
        
    def local_search(selected, face_load, current_count):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            addable = []
            for e_idx in range(NUM_EDGES):
                if not selected[e_idx] and can_add(e_idx, selected, face_load):
                    addable.append(e_idx)
                    
            if addable:
                e_idx = random.choice(addable)
                add_edge(e_idx, selected, face_load)
                current_count += 1
                improved = True
            else:
                num_remove = random.randint(1, 4)
                selected_edges = [e for e in range(NUM_EDGES) if selected[e]]
                if not selected_edges:
                    break
                removed = random.sample(selected_edges, num_remove)
                for e_idx in removed:
                    remove_edge(e_idx, selected, face_load)
                    current_count -= 1
                order = random.sample(range(NUM_EDGES), NUM_EDGES)
                current_count += greedy_fill(selected, face_load, order)
                improved = True
        return current_count

    while time.time() < deadline:
        selected = [False] * NUM_EDGES
        face_load = [0] * NUM_FACES
        current_count = 0
        order = random.sample(range(NUM_EDGES), NUM_EDGES)
        current_count = greedy_fill(selected, face_load, order)
        current_count = local_search(selected, face_load, current_count)
        
        if current_count > best_count:
            best_count = current_count
            best_selected = selected[:]
            
    result = []
    for e_idx in range(NUM_EDGES):
        if best_selected[e_idx]:
            result.append(list(edges[e_idx]))
            
    return result
