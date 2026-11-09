import numpy as np
import time
import random
from math import gcd
from itertools import combinations

def normalize_line(A, B, C):
    if A == 0 and B == 0:
        return None
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g == 0:
        g = 1
    A //= g
    B //= g
    C //= g
    if A < 0 or (A == 0 and B < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def count_incidences(points, lines):
    if len(points) == 0 or len(lines) == 0:
        return 0
    P = points.astype(np.int64)
    L = lines.astype(np.int64)
    px = P[:, 0].reshape(-1, 1)
    py = P[:, 1].reshape(-1, 1)
    A = L[:, 0].reshape(1, -1)
    B = L[:, 1].reshape(1, -1)
    C = L[:, 2].reshape(1, -1)
    vals = A * px + B * py + C
    count = np.sum(vals == 0)
    return int(count)

def line_from_two_points(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    A = dy
    B = -dx
    C = dx * p1[1] - dy * p1[0]
    return normalize_line(A, B, C)

def line_intersection(l1, l2):
    """Return the intersection point of two lines as (x, y) or None if parallel/coincident."""
    A1, B1, C1 = l1
    A2, B2, C2 = l2
    det = A1 * B2 - A2 * B1
    if det == 0:
        return None  # parallel or coincident
    x = (B1 * C2 - B2 * C1) / det
    y = (C1 * A2 - C2 * A1) / det
    # Check if integer
    if abs(x - round(x)) > 1e-9 or abs(y - round(y)) > 1e-9:
        return None
    return (int(round(x)), int(round(y)))

def build_incidence_from_lines(lines):
    """
    Given a list of lines, find all intersection points (points where >= 2 lines meet).
    Return (points, incidence_count) where points are the intersection points.
    """
    point_to_lines = {}  # point -> list of line indices
    
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            pt = line_intersection(lines[i], lines[j])
            if pt is not None:
                if pt not in point_to_lines:
                    point_to_lines[pt] = []
                point_to_lines[pt].append(i)
                point_to_lines[pt].append(j)
    
    # Each point's incidence = number of lines passing through it
    # We need to compute for each point how many of the 27 lines pass through it
    point_incidence = {}
    for pt, line_indices in point_to_lines.items():
        # Count unique lines
        point_incidence[pt] = len(set(line_indices))
    
    # Sort by incidence (descending)
    sorted_points = sorted(point_incidence.items(), key=lambda x: (-x[1], x[0]))
    
    return sorted_points

def greedy_select_lines(base_points, n_lines=27, max_iterations=50):
    """
    Greedily select lines to maximize the number of high-incidence intersection points.
    Start with lines through pairs of base_points, then greedily add lines that create
    new high-incidence intersections.
    """
    # Generate all lines through pairs of base_points
    all_candidate_lines = set()
    n = len(base_points)
    for i in range(n):
        for j in range(i + 1, n):
            l = line_from_two_points(base_points[i], base_points[j])
            if l is not None:
                all_candidate_lines.add(l)
    
    candidate_list = list(all_candidate_lines)
    random.shuffle(candidate_list)
    
    # Greedy selection
    selected_lines = []
    for line in candidate_list:
        if len(selected_lines) >= n_lines:
            break
        # Check if this line adds value: compute new intersections
        new_lines = selected_lines + [line]
        sorted_pts = build_incidence_from_lines(new_lines)
        
        # Score: sum of incidences of top 54 points
        score = sum(inc for _, inc in sorted_pts[:54])
        
        if not selected_lines or score > 0:
            selected_lines.append(line)
    
    # If we don't have enough, fill with random lines
    while len(selected_lines) < n_lines:
        A = random.randint(-10, 10)
        B = random.randint(-10, 10)
        if A == 0 and B == 0:
            A = 1
        C = random.randint(-30, 30)
        nl = normalize_line(A, B, C)
        if nl and nl not in selected_lines:
            selected_lines.append(nl)
    
    return selected_lines[:n_lines]

def local_search_lines(lines, base_points, time_limit, seed=42):
    """
    Local search: try replacing one line at a time with a better one from the candidate pool.
    """
    random.seed(seed)
    start_time = time.time()
    
    current_lines = list(lines)
    
    # Generate candidate pool
    candidate_pool = set()
    n = len(base_points)
    for i in range(n):
        for j in range(i + 1, n):
            l = line_from_two_points(base_points[i], base_points[j])
            if l is not None:
                candidate_pool.add(l)
    
    # Add some random lines
    for _ in range(200):
        A = random.randint(-15, 15)
        B = random.randint(-15, 15)
        if A == 0 and B == 0:
            A = 1
        C = random.randint(-50, 50)
        nl = normalize_line(A, B, C)
        if nl:
            candidate_pool.add(nl)
    
    candidate_list = list(candidate_pool)
    random.shuffle(candidate_list)
    
    def score_lines(line_set):
        sorted_pts = build_incidence_from_lines(list(line_set))
        if len(sorted_pts) < 54:
            return sum(inc for _, inc in sorted_pts)
        return sum(inc for _, inc in sorted_pts[:54])
    
    current_score = score_lines(current_lines)
    best_lines = list(current_lines)
    best_score = current_score
    
    no_improve = 0
    
    while time.time() - start_time < time_limit:
        # Try to improve by replacing one line
        improved = False
        for i in range(len(current_lines)):
            if time.time() - start_time > time_limit:
                break
            
            old_line = current_lines[i]
            
            # Try a random candidate line not already in the set
            for _ in range(50):
                new_line = random.choice(candidate_list)
                if new_line in current_lines:
                    continue
                
                trial_lines = current_lines.copy()
                trial_lines[i] = new_line
                trial_score = score_lines(trial_lines)
                
                if trial_score > current_score:
                    current_lines = trial_lines
                    current_score = trial_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_lines = list(current_lines)
                    break
            
            if improved:
                no_improve = 0
                break
            else:
                no_improve += 1
        
        if no_improve > 20:
            # Perturb: replace 2 random lines
            if len(current_lines) >= 2:
                idx1, idx2 = random.sample(range(len(current_lines)), 2)
                current_lines[idx1] = random.choice(candidate_list)
                current_lines[idx2] = random.choice(candidate_list)
                current_score = score_lines(current_lines)
                no_improve = 0
    
    return best_lines, best_score

def szemeredi_trotter_incidences():
    deadline = time.time() + 580  # Safety margin before 600s
    
    best_I_global = -1
    best_points_global = None
    best_lines_global = None
    
    # Step 1: Generate diverse base point sets
    base_point_sets = []
    
    # Set 1: Small grid
    grid1 = []
    for x in range(-5, 6):
        for y in range(-5, 6):
            grid1.append((x, y))
    base_point_sets.append(grid1)
    
    # Set 2: Wider grid
    grid2 = []
    for x in range(-8, 9):
        for y in range(-8, 9):
            grid2.append((x, y))
    base_point_sets.append(grid2)
    
    # Set 3: Random points
    random.seed(123)
    grid3 = []
    for _ in range(100):
        x = random.randint(-15, 15)
        y = random.randint(-15, 15)
        if (x, y) not in grid3:
            grid3.append((x, y))
    base_point_sets.append(grid3)
    
    # Set 4: Structured - lines y = mx + c for small m, c
    grid4 = []
    for m in range(-2, 3):
        for c in range(-10, 11):
            for x in range(-5, 6):
                y = m * x + c
                grid4.append((x, y))
    # Deduplicate
    grid4 = list(set(grid4))
    base_point_sets.append(grid4)
    
    # Step 2: For each base set, run greedy + local search
    for bs_idx, base_pts in enumerate(base_point_sets):
        if time.time() > deadline - 60:
            break
        
        # Greedy line selection
        selected_lines = greedy_select_lines(base_pts, n_lines=27, max_iterations=20)
        
        # Compute points and score
        sorted_pts = build_incidence_from_lines(selected_lines)
        if len(sorted_pts) >= 54:
            top_points = [pt for pt, _ in sorted_pts[:54]]
        else:
            top_points = [pt for pt, _ in sorted_pts]
            # Fill with random points if needed
            while len(top_points) < 54:
                x = random.randint(-20, 20)
                y = random.randint(-20, 20)
                if (x, y) not in top_points:
                    top_points.append((x, y))
        
        points_arr = np.array(top_points, dtype=np.int64)
        lines_arr = np.array(selected_lines, dtype=np.int64)
        
        I = count_incidences(points_arr, lines_arr)
        
        if I > best_I_global:
            best_I_global = I
            best_points_global = points_arr.copy()
            best_lines_global = lines_arr.copy()
        
        # Local search to improve
        if time.time() < deadline - 30:
            remaining = min(120, deadline - time.time() - 20)
            if remaining > 10:
                improved_lines, _ = local_search_lines(selected_lines, base_pts, 
                                                        time_limit=remaining, 
                                                        seed=100 + bs_idx)
                sorted_pts2 = build_incidence_from_lines(improved_lines)
                if len(sorted_pts2) >= 54:
                    top_points2 = [pt for pt, _ in sorted_pts2[:54]]
                else:
                    top_points2 = [pt for pt, _ in sorted_pts2]
                    while len(top_points2) < 54:
                        x = random.randint(-20, 20)
                        y = random.randint(-20, 20)
                        if (x, y) not in top_points2:
                            top_points2.append((x, y))
                
                points_arr2 = np.array(top_points2, dtype=np.int64)
                lines_arr2 = np.array(improved_lines, dtype=np.int64)
                I2 = count_incidences(points_arr2, lines_arr2)
                
                if I2 > best_I_global:
                    best_I_global = I2
                    best_points_global = points_arr2.copy()
                    best_lines_global = lines_arr2.copy()
    
    # Step 3: Try a different approach - start from known good configurations
    # 6x9 grid with diagonal lines
    grid_pts = []
    for x in range(6):
        for y in range(9):
            grid_pts.append((x, y))
    
    # Lines: horizontal, vertical, and diagonals
    lines_set = set()
    # Horizontal lines y = c
    for c in range(9):
        lines_set.add((0, 1, -c))
    # Vertical lines x = c  
    for c in range(6):
        lines_set.add((1, 0, -c))
    # Diagonals
    for d in range(-8, 9):
        lines_set.add((1, -1, d))
    
    lines_list = list(lines_set)
    random.shuffle(lines_list)
    selected = lines_list[:27]
    
    sorted_pts = build_incidence_from_lines(selected)
    if len(sorted_pts) >= 54:
        top_points = [pt for pt, _ in sorted_pts[:54]]
    else:
        top_points = [pt for pt, _ in sorted_pts]
        while len(top_points) < 54:
            x = random.randint(0, 10)
            y = random.randint(0, 10)
            if (x, y) not in top_points:
                top_points.append((x, y))
    
    points_arr = np.array(top_points, dtype=np.int64)
    lines_arr = np.array(selected, dtype=np.int64)
    I = count_incidences(points_arr, lines_arr)
    
    if I > best_I_global:
        best_I_global = I
        best_points_global = points_arr.copy()
        best_lines_global = lines_arr.copy()
    
    # Local search on this
    if time.time() < deadline - 30:
        remaining = min(100, deadline - time.time() - 20)
        if remaining > 10:
            improved_lines, _ = local_search_lines(selected, grid_pts, 
                                                    time_limit=remaining, 
                                                    seed=500)
            sorted_pts2 = build_incidence_from_lines(improved_lines)
            if len(sorted_pts2) >= 54:
                top_points2 = [pt for pt, _ in sorted_pts2[:54]]
            else:
                top_points2 = [pt for pt, _ in sorted_pts2]
                while len(top_points2) < 54:
                    x = random.randint(0, 15)
                    y = random.randint(0, 15)
                    if (x, y) not in top_points2:
                        top_points2.append((x, y))
            
            points_arr2 = np.array(top_points2, dtype=np.int64)
            lines_arr2 = np.array(improved_lines, dtype=np.int64)
            I2 = count_incidences(points_arr2, lines_arr2)
            
            if I2 > best_I_global:
                best_I_global = I2
                best_points_global = points_arr2.copy()
                best_lines_global = lines_arr2.copy()
    
    # Step 4: Random restarts with intersection-based construction
    for rs in range(8):
        if time.time() > deadline - 30:
            break
        
        random.seed(2000 + rs)
        
        # Generate random base points
        n_base = random.randint(20, 40)
        base_pts = []
        for _ in range(n_base):
            x = random.randint(-10, 10)
            y = random.randint(-10, 10)
            if (x, y) not in base_pts:
                base_pts.append((x, y))
        
        # Greedy line selection
        selected_lines = greedy_select_lines(base_pts, n_lines=27)
        
        sorted_pts = build_incidence_from_lines(selected_lines)
        if len(sorted_pts) >= 54:
            top_points = [pt for pt, _ in sorted_pts[:54]]
        else:
            top_points = [pt for pt, _ in sorted_pts]
            while len(top_points) < 54:
                x = random.randint(-20, 20)
                y = random.randint(-20, 20)
                if (x, y) not in top_points:
                    top_points.append((x, y))
        
        points_arr = np.array(top_points, dtype=np.int64)
        lines_arr = np.array(selected_lines, dtype=np.int64)
        I = count_incidences(points_arr, lines_arr)
        
        if I > best_I_global:
            best_I_global = I
            best_points_global = points_arr.copy()
            best_lines_global = lines_arr.copy()
        
        # Quick local search
        if time.time() < deadline - 20:
            remaining = min(60, deadline - time.time() - 15)
            if remaining > 5:
                improved_lines, _ = local_search_lines(selected_lines, base_pts,
                                                        time_limit=remaining,
                                                        seed=3000 + rs)
                sorted_pts2 = build_incidence_from_lines(improved_lines)
                if len(sorted_pts2) >= 54:
                    top_points2 = [pt for pt, _ in sorted_pts2[:54]]
                else:
                    top_points2 = [pt for pt, _ in sorted_pts2]
                    while len(top_points2) < 54:
                        x = random.randint(-20, 20)
                        y = random.randint(-20, 20)
                        if (x, y) not in top_points2:
                            top_points2.append((x, y))
                
                points_arr2 = np.array(top_points2, dtype=np.int64)
                lines_arr2 = np.array(improved_lines, dtype=np.int64)
                I2 = count_incidences(points_arr2, lines_arr2)
                
                if I2 > best_I_global:
                    best_I_global = I2
                    best_points_global = points_arr2.copy()
                    best_lines_global = lines_arr2.copy()
    
    # Final validation
    if best_points_global is None or best_lines_global is None:
        # Fallback
        lines = np.array([(a, -1, b) for a in range(1, 5) for b in range(1, 8)][:27], dtype=np.int64)
        pts = []
        for x in range(9):
            for y in range(6):
                pts.append((x, y))
        best_points_global = np.array(pts[:54], dtype=np.int64)
        best_lines_global = lines
    
    # Ensure exactly 54 unique points
    unique_points, indices = np.unique(best_points_global, axis=0, return_index=True)
    if len(unique_points) != 54:
        # Pad or trim
        if len(unique_points) > 54:
            best_points_global = unique_points[:54]
        else:
            # Add random points
            existing = set(map(tuple, unique_points))
            while len(best_points_global) < 54:
                x = random.randint(-20, 20)
                y = random.randint(-20, 20)
                if (x, y) not in existing:
                    best_points_global = np.vstack([best_points_global, [[x, y]]])
                    existing.add((x, y))
            best_points_global = best_points_global[:54]
    else:
        best_points_global = unique_points[indices]
    
    # Ensure exactly 27 unique lines
    unique_lines = []
    for line in best_lines_global:
        nl = normalize_line(int(line[0]), int(line[1]), int(line[2]))
        if nl and nl not in unique_lines:
            unique_lines.append(nl)
    
    if len(unique_lines) < 27:
        # Fill with random lines
        while len(unique_lines) < 27:
            A = random.randint(-10, 10)
            B = random.randint(-10, 10)
            if A == 0 and B == 0:
                A = 1
            C = random.randint(-30, 30)
            nl = normalize_line(A, B, C)
            if nl and nl not in unique_lines:
                unique_lines.append(nl)
    
    best_lines_global = np.array(unique_lines[:27], dtype=np.int64)
    
    # Final incidence count
    final_I = count_incidences(best_points_global, best_lines_global)
    
    return best_points_global, best_lines_global

# EVOLVE-BLOCK-END
