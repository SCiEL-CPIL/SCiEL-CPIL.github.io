import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    d = 7
    n = 1 << d
    num_edges = n * d // 2
    
    edge_to_u = [0] * num_edges
    edge_to_i = [0] * num_edges
    idx = 0
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_to_u[idx] = u
                edge_to_i[idx] = i
                idx += 1
                
    edge_idx = [[-1]*d for _ in range(n)]
    for k in range(num_edges):
        u = edge_to_u[k]
        i = edge_to_i[k]
        edge_idx[u][i] = k
        v = u ^ (1 << i)
        edge_idx[v][i] = k
        
    faces = []
    for i in range(d):
        for j in range(i+1, d):
            for u in range(n):
                if (u & (1<<i)) == 0 and (u & (1<<j)) == 0:
                    faces.append((u, i, j))
                    
    num_faces = len(faces)
    
    face_edges = []
    for u, i, j in faces:
        e1 = edge_idx[u][i]
        e2 = edge_idx[u ^ (1<<i)][j]
        e3 = edge_idx[u ^ (1<<j)][i]
        e4 = edge_idx[u][j]
        face_edges.append((e1, e2, e3, e4))
        
    edge_faces = [[] for _ in range(num_edges)]
    for f_idx, (e1, e2, e3, e4) in enumerate(face_edges):
        edge_faces[e1].append(f_idx)
        edge_faces[e2].append(f_idx)
        edge_faces[e3].append(f_idx)
        edge_faces[e4].append(f_idx)
        
    selected = [False] * num_edges
    face_counts = [0] * num_faces
    current_size = 0
    
    def is_valid_add(k):
        for f in edge_faces[k]:
            if face_counts[f] >= 3:
                return False
        return True
        
    def add_edge(k):
        nonlocal current_size
        selected[k] = True
        current_size += 1
        for f in edge_faces[k]:
            face_counts[f] += 1
            
    def remove_edge(k):
        nonlocal current_size
        selected[k] = False
        current_size -= 1
        for f in edge_faces[k]:
            face_counts[f] -= 1
            
    # Initialization
    edges_order = list(range(num_edges))
    random.shuffle(edges_order)
    for k in edges_order:
        if is_valid_add(k):
            add_edge(k)
            
    best_size = current_size
    best_selected = selected[:]
    
    T = 1.0
    cooling = 0.99995
    min_T = 0.01
    
    while time.time() < deadline:
        k = random.randint(0, num_edges - 1)
        
        if selected[k]:
            remove_edge(k)
            new_size = current_size
            if random.random() < math.exp(-1.0 / T):
                if new_size > best_size:
                    best_size = new_size
                    best_selected = selected[:]
            else:
                add_edge(k)
        else:
            if is_valid_add(k):
                add_edge(k)
                new_size = current_size
                if new_size > best_size:
                    best_size = new_size
                    best_selected = selected[:]
                    
        T *= cooling
        if T < min_T:
            T = min_T
            
    result = []
    for k in range(num_edges):
        if best_selected[k]:
            u = edge_to_u[k]
            i = edge_to_i[k]
            v = u ^ (1 << i)
            result.append([u, v])
            
    return result
