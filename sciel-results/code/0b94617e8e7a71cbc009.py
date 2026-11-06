import time
import numpy as np
import random

# EVOLVE-BLOCK-START
def construct():
    d = 7
    n = 1 << d
    deadline = time.time() + 145.0  # Safety margin under 150s

    # Build all edges of Q7
    edges = []
    edge_index = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edges)
                edges.append((x, y))
                edge_index[(x, y)] = idx
    m = len(edges)  # 448

    # Build all 2D faces
    faces = []
    for i in range(d):
        for j in range(i + 1, d):
            bi = 1 << i
            bj = 1 << j
            for a in range(n):
                if (a & bi) == 0 and (a & bj) == 0:
                    v1 = a
                    v2 = a ^ bi
                    v3 = a ^ bj
                    v4 = a ^ bi ^ bj
                    e1 = edge_index[(min(v1, v2), max(v1, v2))]
                    e2 = edge_index[(min(v3, v4), max(v3, v4))]
                    e3 = edge_index[(min(v1, v3), max(v1, v3))]
                    e4 = edge_index[(min(v2, v4), max(v2, v4))]
                    faces.append((e1, e2, e3, e4))
    
    f_count = len(faces)

    # For each edge, list of faces containing it
    edge_faces = [[] for _ in range(m)]
    for fi, face in enumerate(faces):
        for e in face:
            edge_faces[e].append(fi)

    # State variables
    selected = np.zeros(m, dtype=bool)
    face_counts = np.zeros(f_count, dtype=np.int32)
    
    def add_edge(e):
        selected[e] = True
        face_counts[edge_faces[e]] += 1

    def remove_edge(e):
        selected[e] = False
        face_counts[edge_faces[e]] -= 1

    def can_add(e):
        return np.all(face_counts[edge_faces[e]] < 3)

    def greedy_fill(order):
        added = 0
        for e in order:
            if not selected[e] and can_add(e):
                add_edge(e)
                added += 1
        return added

    def local_search():
        nonlocal selected, face_counts
        improved = True
        while improved:
            improved = False
            # 1. Try to add any edge
            # We scan all edges not selected
            for e in range(m):
                if not selected[e] and can_add(e):
                    add_edge(e)
                    improved = True
                    break 
            if improved:
                continue
            
            # 2. No add possible. Try a beneficial swap.
            blocked_edges = [e for e in range(m) if not selected[e] and not can_add(e)]
            if not blocked_edges:
                break
            
            random.shuffle(blocked_edges)
            for e in blocked_edges:
                crit_faces = [fi for fi in edge_faces[e] if face_counts[fi] == 3]
                if not crit_faces:
                    continue
                random.shuffle(crit_faces)
                for fi in crit_faces:
                    face = faces[fi]
                    sel_in_face = [ee for ee in face if selected[ee] and ee != e]
                    if not sel_in_face:
                        continue
                    random.shuffle(sel_in_face)
                    for r in sel_in_face:
                        remove_edge(r)
                        add_edge(e)
                        
                        addable = [ee for ee in range(m) if not selected[ee] and can_add(ee)]
                        if addable:
                            add_edge(addable[0])
                            improved = True
                            break
                        else:
                            remove_edge(e)
                            add_edge(r)
                        
                    if improved:
                        break
                if improved:
                    break
        return np.sum(selected)

    best_count = 0
    best_selected = np.zeros(m, dtype=bool)
    
    rng = random.Random(42)
    while time.time() < deadline:
        selected[:] = False
        face_counts[:] = 0
        
        order = list(range(m))
        rng.shuffle(order)
        greedy_fill(order)
        
        current_count = local_search()
        
        if current_count > best_count:
            best_count = current_count
            best_selected = selected.copy()

    out = []
    for e in range(m):
        if best_selected[e]:
            out.append(list(edges[e]))
            
    return out
# EVOLVE-BLOCK-END
