def construct():
    import time
    import random
    import math

    start_time = time.time()
    deadline = start_time + 115

    d = 7
    nverts = 1 << d  # 128

    # 1. Build edges
    edges = []
    edge_map = {}
    for u in range(nverts):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(edges)
                edges.append((u, v))
                edge_map[(u, v)] = idx
                edge_map[(v, u)] = idx
                
    num_edges = len(edges) # 448

    # 2. Build faces
    face_edges_list = []
    edge_faces = [[] for _ in range(num_edges)]
    
    for i in range(d):
        for j in range(i + 1, d):
            for u in range(nverts):
                if (u & (1 << i)) != 0: continue
                if (u & (1 << j)) != 0: continue
                
                v1 = u
                v2 = u ^ (1 << i)
                v3 = u ^ (1 << j)
                v4 = u ^ (1 << i) ^ (1 << j)
                
                e1 = edge_map[(min(v1,v2), max(v1,v2))]
                e2 = edge_map[(min(v1,v3), max(v1,v3))]
                e3 = edge_map[(min(v2,v4), max(v2,v4))]
                e4 = edge_map[(min(v3,v4), max(v3,v4))]
                
                f_idx = len(face_edges_list)
                face_edges_list.append((e1, e2, e3, e4))
                
                edge_faces[e1].append(f_idx)
                edge_faces[e2].append(f_idx)
                edge_faces[e3].append(f_idx)
                edge_faces[e4].append(f_idx)
                
    num_faces = len(face_edges_list)

    # Helper to check if edge e can be added
    def can_add(e, face_count):
        for f in edge_faces[e]:
            if face_count[f] == 3:
                return False
        return True

    def add_edge(e, selected, face_count):
        selected[e] = True
        for f in edge_faces[e]:
            face_count[f] += 1

    def remove_edge(e, selected, face_count):
        selected[e] = False
        for f in edge_faces[e]:
            face_count[f] -= 1

    # Initial random greedy
    def random_greedy():
        selected = [False] * num_edges
        face_count = [0] * num_faces
        order = list(range(num_edges))
        random.shuffle(order)
        count = 0
        for e in order:
            if can_add(e, face_count):
                add_edge(e, selected, face_count)
                count += 1
        return selected, face_count, count

    best_selected, best_face_count, best_count = None, None, 0
    
    # Find a good start
    t0 = time.time()
    while time.time() - t0 < 2.0:
        s, fc, c = random_greedy()
        if c > best_count:
            best_count = c
            best_selected = s[:]
            best_face_count = fc[:]

    selected = best_selected[:]
    face_count = best_face_count[:]
    count = best_count
    
    # Simulated Annealing
    # T starts high, decays
    T_start = 5.0
    T_end = 0.1
    alpha = 0.999 # cooling rate
    
    # We will run SA until deadline
    # To avoid getting stuck in a very bad local min, we can do random restarts if no improvement for a long time?
    # Or just rely on SA.
    
    T = T_start
    no_improve_steps = 0
    max_no_improve = 5000
    
    while time.time() < deadline:
        if no_improve_steps > max_no_improve:
            # Restart from best state with higher T? Or just continue?
            # Let's try to "shake" by doing a few random removals
            for _ in range(10):
                if count > 0:
                    e = random.randint(0, num_edges - 1)
                    if selected[e]:
                        remove_edge(e, selected, face_count)
                        count -= 1
            no_improve_steps = 0
            T = T_start * 0.5 # Reheat

        # Choose a move type
        r = random.random()
        
        if r < 0.5:
            # Try to add an edge
            # Pick a random unselected edge
            e = random.randint(0, num_edges - 1)
            if not selected[e]:
                if can_add(e, face_count):
                    add_edge(e, selected, face_count)
                    count += 1
                    if count > best_count:
                        best_count = count
                        best_selected = selected[:]
                        best_face_count = face_count[:]
                        no_improve_steps = 0
                    else:
                        no_improve_steps += 1
                else:
                    no_improve_steps += 1
        elif r < 0.8:
            # Try to remove an edge
            e = random.randint(0, num_edges - 1)
            if selected[e]:
                # Delta = -1
                # Accept with prob exp(1/T)
                if random.random() < math.exp(1.0 / T):
                    remove_edge(e, selected, face_count)
                    count -= 1
                    no_improve_steps += 1
                else:
                    no_improve_steps += 1
        else:
            # Try a swap: remove r, add a
            # Pick random selected r
            r_idx = random.randint(0, num_edges - 1)
            if selected[r_idx]:
                # Pick random unselected a
                a_idx = random.randint(0, num_edges - 1)
                if not selected[a_idx]:
                    # Temporarily remove r
                    remove_edge(r_idx, selected, face_count)
                    # Check if a can be added
                    if can_add(a_idx, face_count):
                        add_edge(a_idx, selected, face_count)
                        # Count is same
                        no_improve_steps += 1
                    else:
                        # Revert
                        add_edge(r_idx, selected, face_count)
                        no_improve_steps += 1

        T *= alpha
        if T < T_end:
            T = T_end

    # Return best found
    result = []
    for i in range(num_edges):
        if best_selected[i]:
            result.append(list(edges[i]))
            
    return result
