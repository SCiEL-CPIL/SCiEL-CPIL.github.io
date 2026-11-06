import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin

    d = 7
    nverts = 1 << d  # 128
    num_edges = d * (1 << (d - 1))  # 7 * 64 = 448

    # 1. Build edges and mapping
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

    # 2. Build faces
    # A face is defined by 2 dimensions (i, j) and a base vertex u with bits i,j = 0.
    # Vertices: u, u^i, u^j, u^i^j
    face_edges_list = []
    edge_faces = [[] for _ in range(num_edges)]
    
    # Precompute face edge sets for faster lookup?
    # We need to know which edges are in which faces.
    
    for i in range(d):
        for j in range(i + 1, d):
            mask_i = 1 << i
            mask_j = 1 << j
            for u in range(nverts):
                if (u & mask_i) != 0:
                    continue
                if (u & mask_j) != 0:
                    continue
                
                v1 = u
                v2 = u ^ mask_i
                v3 = u ^ mask_j
                v4 = u ^ mask_i ^ mask_j
                
                # The 4 edges of this face
                e1 = edge_map[(min(v1, v2), max(v1, v2))]
                e2 = edge_map[(min(v1, v3), max(v1, v3))]
                e3 = edge_map[(min(v2, v4), max(v2, v4))]
                e4 = edge_map[(min(v3, v4), max(v3, v4))]
                
                f_idx = len(face_edges_list)
                face_edges_list.append((e1, e2, e3, e4))
                
                edge_faces[e1].append(f_idx)
                edge_faces[e2].append(f_idx)
                edge_faces[e3].append(f_idx)
                edge_faces[e4].append(f_idx)
                
    num_faces = len(face_edges_list)
    face_edges_set = [set(f) for f in face_edges_list]

    # Helper to check if an edge can be added
    def can_add(e, face_count, selected):
        if selected[e]:
            return False
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

    # Greedy Initialization
    def greedy_init(seed=None):
        if seed is not None:
            random.seed(seed)
        selected = [False] * num_edges
        face_count = [0] * num_faces
        count = 0
        order = list(range(num_edges))
        random.shuffle(order)
        for e in order:
            if can_add(e, face_count, selected):
                add_edge(e, selected, face_count)
                count += 1
        return selected, face_count, count

    best_selected = None
    best_face_count = None
    best_count = 0
    
    # Phase 1: Multiple Greedy Starts
    t0 = time.time()
    while time.time() - t0 < 2.0:
        s, fc, c = greedy_init()
        if c > best_count:
            best_count = c
            best_selected = s
            best_face_count = fc

    selected = best_selected[:]
    face_count = best_face_count[:]
    count = best_count
    best_found = count
    best_selected_copy = selected[:]
    best_face_count_copy = face_count[:]

    # Local Search
    # We will use a loop that tries to improve the count.
    # Moves:
    # 1. Add any possible edge.
    # 2. If no add possible, try 1-swap: remove e1, add e2.
    # 3. If no 1-swap improves, try 2-swap: remove e1, e2, add e3, e4.
    
    no_improve = 0
    max_no_improve = 5000
    restart_count = 0
    max_restarts = 50

    while time.time() < deadline:
        # Try to add
        added = False
        # Randomly pick an unselected edge to try to add? 
        # Or iterate through all? Iterating is 448 checks. Fast.
        # But if we find one, we add it and restart the loop.
        for e in range(num_edges):
            if not selected[e] and can_add(e, face_count, selected):
                add_edge(e, selected, face_count)
                count += 1
                if count > best_found:
                    best_found = count
                    best_selected_copy = selected[:]
                    best_face_count_copy = face_count[:]
                added = True
                no_improve = 0
                break
        
        if added:
            continue
            
        # No edge can be added. Try 1-swap.
        # Pick a random selected edge to remove.
        sel_edges = [i for i in range(num_edges) if selected[i]]
        if not sel_edges:
            break
        
        e1 = random.choice(sel_edges)
        remove_edge(e1, selected, face_count)
        count -= 1
        
        # Try to find an edge to add that restores or increases count
        # Ideally, we want to add an edge that was blocked by e1.
        # But any edge that can be added is fine if it restores count.
        # If we add 1, count is same. If we can't add, we revert.
        # We want to find a state with count >= old_count.
        
        best_add_e = -1
        # Check all unselected edges
        for e2 in range(num_edges):
            if not selected[e2] and can_add(e2, face_count, selected):
                best_add_e = e2
                break
        
        if best_add_e != -1:
            add_edge(best_add_e, selected, face_count)
            count += 1
            # Count is restored. 
            # This is a neutral move. It changes the configuration.
            no_improve += 1
        else:
            # Revert
            add_edge(e1, selected, face_count)
            count += 1
            no_improve += 1
            
        # If we are stuck for too long, try a more aggressive 2-swap or restart
        if no_improve > max_no_improve:
            # Restart from best
            selected = best_selected_copy[:]
            face_count = best_face_count_copy[:]
            count = best_found
            no_improve = 0
            restart_count += 1
            
            if restart_count > max_restarts:
                 # Try a different strategy or break
                 break

    # Final Polish
    for e in range(num_edges):
        if not selected[e] and can_add(e, face_count, selected):
            add_edge(e, selected, face_count)
            count += 1
            if count > best_found:
                best_found = count
                best_selected_copy = selected[:]
                best_face_count_copy = face_count[:]

    result = []
    for i in range(num_edges):
        if best_selected_copy[i]:
            result.append(list(edges[i]))

    return result
