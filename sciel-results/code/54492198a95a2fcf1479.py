<<<<<<< SEARCH
def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 600
    
    # 1. Initial configuration
    pts, lns = _initial_grid_diagonal()
    
    # 2. Refine points
    if time.time() < deadline - 30:
        pts = _refine_points(pts, lns, deadline - 30)
    
    # 3. Refine lines
    if time.time() < deadline - 30:
        lns = _refine_lines(pts, lns, deadline - 30)
    
    # 4. Local search with simulated annealing
    if time.time() < deadline - 10:
        pts, lns = _local_search(pts, lns, deadline - 5)
    
    # 5. Final refinement
    if time.time() < deadline - 5:
        pts = _refine_points(pts, lns, deadline - 5)
        lns = _refine_lines(pts, lns, deadline - 2)
    
    # 6. Validation
    pts = np.array(pts, dtype=np.int64)
    lns = np.array(lns, dtype=np.int64)
    
    if not _is_valid(pts, lns):
        # Fallback
        xs = np.arange(6)
        ys = np.arange(9)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        pts = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        lines = []
        for x in range(6):
            lines.append((1, 0, -x))
        for y in range(9):
            lines.append((0, 1, -y))
        for c in range(0, 12):
            lines.append((1, 1, -c))
        lns = np.array(lines, dtype=np.int64)
    
    return pts, lns
=======
def _final_polish(pts, lns, deadline):
    """
    Final high-precision polish: try all single-point moves and single-line replacements
    with exhaustive local search until no improvement is found.
    """
    n_pts, n_lines = 54, 27
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        
        # Try moving each point to any nearby position
        for i in range(n_pts):
            best_pos = tuple(pts[i])
            best_delta = 0
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    new_pos = (int(pts[i, 0]) + dx, int(pts[i, 1]) + dy)
                    if new_pos == best_pos:
                        continue
                    point_set = set(tuple(p) for p in pts)
                    point_set.discard(best_pos)
                    if new_pos in point_set:
                        continue
                    delta = 0
                    for j in range(n_lines):
                        A, B, C = lns[j]
                        was = (A * best_pos[0] + B * best_pos[1] + C == 0)
                        now = (A * new_pos[0] + B * new_pos[1] + C == 0)
                        delta += (1 if now else 0) - (1 if was else 0)
                    if delta > best_delta:
                        best_delta = delta
                        best_pos = new_pos
            if best_delta > 0:
                pts[i] = best_pos
                improved = True
        
        # Try replacing each line with the best candidate
        for j in range(n_lines):
            best_line = _normalize_line(int(lns[j, 0]), int(lns[j, 1]), int(lns[j, 2]))
            best_delta = 0
            # Generate candidates based on point distribution
            x_min = int(np.min(pts[:, 0])) - 5
            x_max = int(np.max(pts[:, 0])) + 5
            y_min = int(np.min(pts[:, 1])) - 5
            y_max = int(np.max(pts[:, 1])) + 5
            
            for A in range(-15, 16):
                for B in range(-15, 16):
                    if A == 0 and B == 0:
                        continue
                    for C in range(y_min - 10, y_max + 11):
                        new_line = _normalize_line(A, B, C)
                        if not new_line or new_line == best_line:
                            continue
                        line_set = set(_normalize_line(int(l[0]), int(l[1]), int(l[2])) for l in lns)
                        line_set.discard(best_line)
                        if new_line in line_set:
                            continue
                        delta = 0
                        A2, B2, C2 = new_line
                        for i in range(n_pts):
                            was = (lns[j, 0] * pts[i, 0] + lns[j, 1] * pts[i, 1] + lns[j, 2] == 0)
                            now = (A2 * pts[i, 0] + B2 * pts[i, 1] + C2 == 0)
                            delta += (1 if now else 0) - (1 if was else 0)
                        if delta > best_delta:
                            best_delta = delta
                            best_line = new_line
            if best_delta > 0:
                lns[j] = best_line
                improved = True
    
    return pts, lns

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 600
    
    # 1. Initial configuration
    pts, lns = _initial_grid_diagonal()
    
    # 2. Refine points
    if time.time() < deadline - 40:
        pts = _refine_points(pts, lns, deadline - 40)
    
    # 3. Refine lines
    if time.time() < deadline - 40:
        lns = _refine_lines(pts, lns, deadline - 40)
    
    # 4. Local search with simulated annealing
    if time.time() < deadline - 20:
        pts, lns = _local_search(pts, lns, deadline - 10)
    
    # 5. Final polish with exhaustive local search
    if time.time() < deadline - 5:
        pts, lns = _final_polish(pts, lns, deadline - 3)
    
    # 6. Validation
    pts = np.array(pts, dtype=np.int64)
    lns = np.array(lns, dtype=np.int64)
    
    if not _is_valid(pts, lns):
        # Fallback
        xs = np.arange(6)
        ys = np.arange(9)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        pts = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        lines = []
        for x in range(6):
            lines.append((1, 0, -x))
        for y in range(9):
            lines.append((0, 1, -y))
        for c in range(0, 12):
            lines.append((1, 1, -c))
        lns = np.array(lines, dtype=np.int64)
    
    return pts, lns
>>>>>>> REPLACE
