import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    d = 7
    n_vertices = 1 << d
    n_edges = d * (1 << (d - 1))
    
    edges = []
    for v in range(n_vertices):
        for i in range(d):
            u = v ^ (1 << i)
            if v < u:
                edges.append((v, u, i))
                
    pos_edge_idx = [[-1] * d for _ in range(n_vertices)]
    for idx, (v, u, i) in enumerate(edges):
        pos_edge_idx[v][i] = idx
        pos_edge_idx[u][i] = idx
        
    faces = []
    edge_faces = [[] for _ in range(n_edges)]
    
    for v in range(n_vertices):
        for i in range(d):
            for j in range(i + 1, d):
                e1 = pos_edge_idx[v][i]
                v2 = v ^ (1 << i)
                e2 = pos_edge_idx[v2][j]
                v3 = v ^ (1 << j)
                e3 = pos_edge_idx[v3][i]
                v4 = v2 ^ (1 << j)
                e4 = pos_edge_idx[v4][j]
                
                face_edges = [e1, e2, e3, e4]
                f_idx = len(faces)
                faces.append(face_edges)
                for e in face_edges:
                    edge_faces[e].append(f_idx)
                    
    n_faces = len(faces)
    
    def init_state():
        x = [random.randint(0, 1) for _ in range(n_edges)]
        S = [0] * n_faces
        for e in range(n_edges):
            if x[e]:
                for f in edge_faces[e]:
                    S[f] += 1
        return x, S
        
    P = 10.0
    
    best_valid_count = -1
    best_valid_x = None
    
    # Helper to check if valid and update best
    def try_update_best(x, S, violated_count, current_sum):
        nonlocal best_valid_count, best_valid_x
        if violated_count == 0:
            if current_sum > best_valid_count:
                best_valid_count = current_sum
                best_valid_x = x[:]
                
    # Simulated Annealing
    # We run multiple restarts until time runs out
    
    while time.time() < deadline:
        x, S = init_state()
        
        # Calculate initial stats
        current_sum = sum(x)
        violated_count = 0
        for s in S:
            if s > 3:
                violated_count += 1
                
        try_update_best(x, S, violated_count, current_sum)
        
        T = 5.0
        cooling = 0.9995
        steps = 0
        max_steps = 200000 # Dynamic based on time? No, fixed per restart.
        
        while steps < max_steps and time.time() < deadline:
            e = random.randint(0, n_edges - 1)
            
            delta = 0.0
            
            if x[e]:
                delta += 1.0
                # Removing edge
                for f in edge_faces[e]:
                    old_s = S[f]
                    new_s = old_s - 1
                    old_pen = max(0, old_s - 3)
                    new_pen = max(0, new_s - 3)
                    delta += P * (new_pen - old_pen)
                    
                    if old_s > 3 and new_s <= 3:
                        violated_count -= 1
                    elif old_s <= 3 and new_s > 3:
                        violated_count += 1
                        
            else:
                delta -= 1.0
                # Adding edge
                for f in edge_faces[e]:
                    old_s = S[f]
                    new_s = old_s + 1
                    old_pen = max(0, old_s - 3)
                    new_pen = max(0, new_s - 3)
                    delta += P * (new_pen - old_pen)
                    
                    if old_s > 3 and new_s <= 3:
                        violated_count -= 1
                    elif old_s <= 3 and new_s > 3:
                        violated_count += 1
                        
            if delta < 0 or random.random() < math.exp(-delta / T):
                x[e] = 1 - x[e]
                current_sum += 1 if x[e] else -1
                
                for f in edge_faces[e]:
                    S[f] += 1 if x[e] else -1
                    
                try_update_best(x, S, violated_count, current_sum)
                
            T *= cooling
            steps += 1
            
    if best_valid_x is None:
        # Fallback
        res = []
        for v in range(n_vertices):
            if v < (v ^ 1):
                res.append([v, v ^ 1])
        return res
    else:
        res = []
        for idx, val in enumerate(best_valid_x):
            if val:
                v, u, i = edges[idx]
                res.append([v, u])
        return res
