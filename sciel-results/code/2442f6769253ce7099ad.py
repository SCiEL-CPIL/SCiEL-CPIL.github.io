<<<<<<< SEARCH
def _local_polish(points, max_time=5.0):
    """Local search: try moving each point to reduce ordinary lines."""
    if not _valid(points):
        return points
    rng = random.Random(42)
    cur = [p[:] for p in points]
    cur_obj = _objective(cur)
    best = [p[:] for p in points]
    best_obj = cur_obj
    start_time = time.time()
    
    improved = True
    while improved and time.time() - start_time < max_time:
        improved = False
        for i in range(12):
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    new_pts = [p[:] for p in cur]
                    new_pts[i][0] += dx
                    new_pts[i][1] += dy
                    if not _valid(new_pts):
                        continue
                    new_obj = _objective(new_pts)
                    if new_obj < cur_obj:
                        cur = new_pts
                        cur_obj = new_obj
                        if cur_obj < best_obj:
                            best_obj = cur_obj
                            best = [p[:] for p in cur]
                        improved = True
                        break
                else:
                    continue
                break
    return best
=======
def _local_polish(points, max_time=5.0):
    """Local search: try moving each point to reduce ordinary lines."""
    if not _valid(points):
        return points
    rng = random.Random(42)
    cur = [p[:] for p in points]
    cur_obj = _objective(cur)
    best = [p[:] for p in points]
    best_obj = cur_obj
    start_time = time.time()
    
    improved = True
    while improved and time.time() - start_time < max_time:
        improved = False
        for i in range(12):
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    new_pts = [p[:] for p in cur]
                    new_pts[i][0] += dx
                    new_pts[i][1] += dy
                    if not _valid(new_pts):
                        continue
                    new_obj = _objective(new_pts)
                    if new_obj < cur_obj:
                        cur = new_pts
                        cur_obj = new_obj
                        if cur_obj < best_obj:
                            best_obj = cur_obj
                            best = [p[:] for p in cur]
                        improved = True
                        break
                else:
                    continue
                break
    return best

def _search_4triple_configurations(max_time=30.0):
    """
    Search for configurations with 4 collinear triples.
    We define 4 lines and place 3 points on each.
    The key is to choose lines such that intersections create the right structure.
    """
    rng = random.Random(12345)
    best_pts = None
    best_score = float('inf')
    start_time = time.time()
    
    # Strategy: Use lines with small integer slopes and intercepts
    # Line representation: y = mx + b, or x = c (vertical)
    # We'll parameterize lines by (type, param1, param2)
    # type 0: horizontal y = c
    # type 1: vertical x = c  
    # type 2: slope 1: y = x + c
    # type 3: slope -1: y = -x + c
    # type 4: slope 2: y = 2x + c
    # type 5: slope -2: y = -2x + c
    # type 6: slope 1/2: y = x/2 + c (i.e., 2y = x + 2c)
    
    def line_to_points(line_spec, num_points=3, exclude=set()):
        """Get integer points on a line."""
        lt, p1, p2 = line_spec
        pts = []
        if lt == 0:  # horizontal y = p1
            for x in range(-20, 21):
                pt = [x, p1]
                if (pt[0], pt[1]) not in exclude and abs(pt[0]) <= 10**9 and abs(pt[1]) <= 10**9:
                    pts.append(pt)
                    if len(pts) >= num_points:
                        break
        elif lt == 1:  # vertical x = p1
            for y in range(-20, 21):
                pt = [p1, y]
                if (pt[0], pt[1]) not in exclude and abs(pt[0]) <= 10**9 and abs(pt[1]) <= 10**9:
                    pts.append(pt)
                    if len(pts) >= num_points:
                        break
        elif lt == 2:  # y = x + p1
            for x in range(-20, 21):
                pt = [x, x + p1]
                if (pt[0], pt[1]) not in exclude and abs(pt[0]) <= 10**9 and abs(pt[1]) <= 10**9:
                    pts.append(pt)
                    if len(pts) >= num_points:
                        break
        elif lt == 3:  # y = -x + p1
            for x in range(-20, 21):
                pt = [x, -x + p1]
                if (pt[0], pt[1]) not in exclude and abs(pt[0]) <= 10**9 and abs(pt[1]) <= 10**9:
                    pts.append(pt)
                    if len(pts) >= num_points:
                        break
        elif lt == 4:  # y = 2x + p1
            for x in range(-20, 21):
                pt = [x, 2*x + p1]
                if (pt[0], pt[1]) not in exclude and abs(pt[0]) <= 10**9 and abs(pt[1]) <= 10**9:
                    pts.append(pt)
                    if len(pts) >= num_points:
                        break
        elif lt == 5:  # y = -2x + p1
            for x in range(-20, 21):
                pt = [x, -2*x + p1]
                if (pt[0], pt[1]) not in exclude and abs(pt[0]) <= 10**9 and abs(pt[1]) <= 10**9:
                    pts.append(pt)
                    if len(pts) >= num_points:
                        break
        elif lt == 6:  # 2y = x + p1, i.e., y = (x + p1)/2
            for x in range(-20, 21):
                if (x + p1) % 2 == 0:
                    y = (x + p1) // 2
                    pt = [x, y]
                    if (pt[0], pt[1]) not in exclude and abs(pt[0]) <= 10**9 and abs(pt[1]) <= 10**9:
                        pts.append(pt)
                        if len(pts) >= num_points:
                            break
        return pts
    
    # Try many random combinations of 4 lines
    num_attempts = 0
    while time.time() - start_time < max_time:
        num_attempts += 1
        
        # Choose 4 distinct lines
        lines = []
        used_specs = set()
        for _ in range(4):
            for _attempt in range(50):
                lt = rng.randint(0, 6)
                p1 = rng.randint(-15, 15)
                p2 = 0  # not used currently
                spec = (lt, p1, p2)
                if spec not in used_specs:
                    used_specs.add(spec)
                    lines.append(spec)
                    break
        
        if len(lines) < 4:
            continue
        
        # Try to place 3 points on each line
        all_points = []
        used = set()
        success = True
        for line_spec in lines:
            candidates = line_to_points(line_spec, 3, exclude=used)
            if len(candidates) < 3:
                success = False
                break
            for pt in candidates[:3]:
                all_points.append(pt)
                used.add((pt[0], pt[1]))
        
        if not success or len(all_points) != 12:
            continue
        if not _valid(all_points):
            continue
        
        obj = _objective(all_points)
        if obj < best_score:
            best_score = obj
            best_pts = [p[:] for p in all_points]
        
        # If we found a good solution, try to refine
        if best_score <= 6000:  # 6 ordinary lines
            # Try local polish
            polished = _local_polish(best_pts, max_time=2.0)
            if polished:
                pol_obj = _objective(polished)
                if pol_obj < best_score:
                    best_score = pol_obj
                    best_pts = polished
    
    return best_pts, best_score
>>>>>>> REPLACE

