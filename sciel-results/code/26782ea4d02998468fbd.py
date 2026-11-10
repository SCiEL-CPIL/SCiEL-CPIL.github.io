import numpy as np
import time
import random
from math import gcd
from itertools import combinations, product

def normalize_line(A, B, C):
    """Normalize line coefficients to ensure uniqueness and integer validity."""
    if A == 0 and B == 0:
        return None
    
    # Handle zero C case for gcd
    abs_g = gcd(abs(A), abs(B))
    if C != 0:
        abs_g = gcd(abs_g, abs(C))
    if abs_g == 0:
        abs_g = 1
        
    A //= abs_g
    B //= abs_g
    C //= abs_g
    
    # Make first non-zero entry positive
    if A < 0 or (A == 0 and B < 0):
        A = -A
        B = -B
        C = -C
        
    return (A, B, C)

def count_incidences(points, lines):
    """
    points: (n, 2) array
    lines: (m, 3) array (A, B, C)
    Returns number of (p, l) such that A*x + B*y + C == 0
    """
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

def get_best_points_for_lines(lines, candidate_points, k=54):
    """
    Given a set of lines and a pool of candidate points,
    find the top k points that maximize incidences.
    """
    if len(candidate_points) < k:
        return candidate_points.copy(), 0
    
    L = lines.astype(np.int64)
    P = candidate_points.astype(np.int64)
    
    px = P[:, 0].reshape(-1, 1)
    py = P[:, 1].reshape(-1, 1)
    
    A = L[:, 0].reshape(1, -1)
    B = L[:, 1].reshape(1, -1)
    C = L[:, 2].reshape(1, -1)
    
    vals = A * px + B * py + C
    counts = np.sum(vals == 0, axis=1)
    
    # Get top k indices
    top_indices = np.argsort(counts)[::-1][:k]
    best_points = candidate_points[top_indices]
    
    return best_points, int(counts[top_indices].max())

def generate_candidate_points(size=50, extra=2000):
    """Generate a grid of candidate points plus some random perturbations."""
    xs = np.arange(-size, size + 1)
    ys = np.arange(-size, size + 1)
    XX, YY = np.meshgrid(xs, ys)
    grid_pts = np.column_stack([XX.ravel(), YY.ravel()])
    
    # Add some random points in a slightly larger range for diversity
    rng = np.random.RandomState(123)
    rand_pts = rng.randint(-size-5, size+6, size=(extra, 2))
    
    all_pts = np.vstack([grid_pts, rand_pts])
    # Remove duplicates
    all_pts, _ = np.unique(all_pts, axis=0, return_index=True)
    return all_pts

def greedy_select_lines_from_points(points, n_lines=27, candidate_pool=None):
    """
    Greedily select lines that maximize total incidences with the given points.
    Uses a smarter approach: prioritize lines through 3+ points.
    """
    P = points.astype(np.int64)
    px = P[:, 0].reshape(-1, 1)
    py = P[:, 1].reshape(-1, 1)
    
    n_pts = len(points)
    candidate_lines = []
    seen_lines = set()
    
    # Generate lines from pairs
    for i in range(n_pts):
        for j in range(i+1, n_pts):
            dx = points[j, 0] - points[i, 0]
            dy = points[j, 1] - points[i, 1]
            A = dy
            B = -dx
            C = dx * points[i, 1] - dy * points[i, 0]
            nl = normalize_line(int(A), int(B), int(C))
            if nl and nl not in seen_lines:
                seen_lines.add(nl)
                candidate_lines.append(nl)
    
    if not candidate_lines:
        for i in range(27):
            candidate_lines.append((1, 0, -i))
        return np.array(candidate_lines[:27], dtype=np.int64)
    
    # Pre-compute incidence counts for all candidate lines
    line_counts = {}
    for nl in candidate_lines:
        A, B, C = nl
        vals = A * px + B * py + C
        cnt = int(np.sum(vals == 0))
        line_counts[nl] = cnt
    
    # Sort candidates by count descending for faster greedy
    sorted_candidates = sorted(candidate_lines, key=lambda l: line_counts[l], reverse=True)
    
    # Greedy selection with marginal gain consideration
    selected_lines = []
    selected_set = set()
    
    for _ in range(n_lines):
        best_line = None
        best_score = -1
        for nl in sorted_candidates:
            if nl in selected_set:
                continue
            score = line_counts[nl]
            if score > best_score:
                best_score = score
                best_line = nl
        if best_line is None:
            break
        selected_lines.append(best_line)
        selected_set.add(best_line)
    
    # Fill if needed with random lines
    rng = random.Random(42)
    while len(selected_lines) < n_lines:
        A = rng.randint(-7, 7)
        B = rng.randint(-7, 7)
        C = rng.randint(-35, 35)
        nl = normalize_line(A, B, C)
        if nl and nl not in selected_set:
            selected_lines.append(nl)
            selected_set.add(nl)
    
    return np.array(selected_lines[:n_lines], dtype=np.int64)

def generate_initial_lines(config_id):
    """Generate diverse initial line configurations."""
    lines = []
    if config_id == 0:
        # Grid + diagonals
        for i in range(9):
            lines.append((1, 0, -i))
        for j in range(6):
            lines.append((0, 1, -j))
        for c in range(-8, 4):
            lines.append((1, -1, -c))
    elif config_id == 1:
        # Various slopes
        a_vals = [0, 1, -1, 2, -2]
        b_vals = [0, 1, -1, 2, -2, 3, -3]
        for a in a_vals:
            for b in b_vals:
                lines.append((a, -1, b))
                if len(lines) >= 27:
                    break
            if len(lines) >= 27:
                break
        while len(lines) < 27:
            lines.append((1, 0, len(lines)))
    elif config_id == 2:
        # Pencil through origin
        pairs = [(1,0), (0,1), (1,1), (1,-1), (2,1), (1,2), (3,1), (1,3), (3,2), (2,3), (3,4), (4,3), (5,1), (1,5)]
        for A, B in pairs:
            lines.append((A, -B, 0))
        for i in range(27 - len(lines)):
            lines.append((1, 1, i+1))
    elif config_id == 3:
        # Random
        random.seed(42)
        seen = set()
        while len(lines) < 27:
            A = random.randint(-5, 5)
            B = random.randint(-5, 5)
            C = random.randint(-10, 10)
            if A == 0 and B == 0:
                continue
            nl = normalize_line(A, B, C)
            if nl and nl not in seen:
                seen.add(nl)
                lines.append(nl)
                
    return np.array(lines[:27], dtype=np.int64)

def improve_configuration(lines, candidate_points, time_limit, seed=42):
    """
    Joint optimization of points and lines using simulated annealing.
    Improved: better move generation, faster evaluation, smarter restarts.
    """
    random.seed(seed)
    current_lines = lines.copy()
    current_points, _ = get_best_points_for_lines(current_lines, candidate_points)
    current_I = count_incidences(current_points, current_lines)
    
    best_lines = current_lines.copy()
    best_points = current_points.copy()
    best_I = current_I
    
    start_time = time.time()
    T = 15.0
    cooling = 0.999
    min_T = 0.3
    
    no_improve = 0
    n_cand_pts = len(candidate_points)
    
    # Precompute which candidate points are on which lines for faster evaluation
    # Use a lookup structure for point swaps
    
    while time.time() - start_time < time_limit:
        if no_improve > 300:
            T = max(T, 15.0)
            no_improve = 0
            
        move_type = random.random()
        
        if move_type < 0.40:
            # Replace a line with a line through a pair of current points (prioritize high-incidence pairs)
            idx = random.randint(0, 26)
            old_line = current_lines[idx].copy()
            
            # Sample pairs, weight toward pairs that are already on many lines
            p1_idx, p2_idx = random.sample(range(54), 2)
            p1 = current_points[p1_idx]
            p2 = current_points[p2_idx]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            A = dy
            B = -dx
            C = dx * p1[1] - dy * p1[0]
            
            new_line = normalize_line(int(A), int(B), int(C))
            if new_line is None:
                continue
            
            new_line_tup = tuple(new_line)
            if any(tuple(l) == new_line_tup for l in current_lines):
                continue
                
            current_lines[idx] = new_line
            new_points, _ = get_best_points_for_lines(current_lines, candidate_points)
            new_I = count_incidences(new_points, current_lines)
            
            delta_I = new_I - current_I
            
            if delta_I > 0 or (T > 0 and random.random() < np.exp(delta_I / T)):
                current_I = new_I
                current_points = new_points
                if current_I > best_I:
                    best_I = current_I
                    best_lines = current_lines.copy()
                    best_points = current_points.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                current_lines[idx] = old_line
                no_improve += 1
                
        elif move_type < 0.55:
            # Replace a line with a random line from small coefficients
            idx = random.randint(0, 26)
            old_line = current_lines[idx].copy()
            
            A = random.randint(-8, 8)
            B = random.randint(-8, 8)
            C = random.randint(-40, 40)
            
            new_line = normalize_line(A, B, C)
            if new_line is None:
                continue
            
            new_line_tup = tuple(new_line)
            if any(tuple(l) == new_line_tup for l in current_lines):
                continue
                
            current_lines[idx] = new_line
            new_points, _ = get_best_points_for_lines(current_lines, candidate_points)
            new_I = count_incidences(new_points, current_lines)
            
            delta_I = new_I - current_I
            
            if delta_I > 0 or (T > 0 and random.random() < np.exp(delta_I / T)):
                current_I = new_I
                current_points = new_points
                if current_I > best_I:
                    best_I = current_I
                    best_lines = current_lines.copy()
                    best_points = current_points.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                current_lines[idx] = old_line
                no_improve += 1
                
        elif move_type < 0.80:
            # Swap a point - try multiple candidates and pick best
            pt_idx = random.randint(0, 53)
            old_pt = current_points[pt_idx].copy()
            
            # Try a few random candidates and keep the best
            best_new_I = current_I
            best_new_pt = None
            for _ in range(5):
                cand_idx = random.randint(0, n_cand_pts-1)
                new_pt = candidate_points[cand_idx].copy()
                
                if any(np.array_equal(current_points[i], new_pt) for i in range(54)):
                    continue
                
                current_points[pt_idx] = new_pt
                trial_I = count_incidences(current_points, current_lines)
                if trial_I > best_new_I:
                    best_new_I = trial_I
                    best_new_pt = new_pt.copy()
                current_points[pt_idx] = old_pt
            
            if best_new_pt is not None:
                current_points[pt_idx] = best_new_pt
                new_I = best_new_I
            else:
                new_I = current_I
            
            delta_I = new_I - current_I
            
            if delta_I > 0 or (T > 0 and random.random() < np.exp(delta_I / T)):
                current_I = new_I
                if current_I > best_I:
                    best_I = current_I
                    best_lines = current_lines.copy()
                    best_points = current_points.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
                
        else:
            # Rebuild lines greedily from current points
            new_lines = greedy_select_lines_from_points(current_points, 27)
            
            unique_lines = []
            for l in new_lines:
                nl = normalize_line(int(l[0]), int(l[1]), int(l[2]))
                if nl and tuple(nl) not in [tuple(u) for u in unique_lines]:
                    unique_lines.append(nl)
            
            if len(unique_lines) >= 27:
                new_lines = np.array(unique_lines[:27], dtype=np.int64)
                new_points, _ = get_best_points_for_lines(new_lines, candidate_points)
                new_I = count_incidences(new_points, new_lines)
                
                delta_I = new_I - current_I
                
                if delta_I > 0 or (T > 0 and random.random() < np.exp(delta_I / (T * 0.5))):
                    current_I = new_I
                    current_lines = new_lines
                    current_points = new_points
                    if current_I > best_I:
                        best_I = current_I
                        best_lines = current_lines.copy()
                        best_points = current_points.copy()
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
                
        T *= cooling
        T = max(T, min_T)
        
    return best_lines, best_points, best_I

def szemeredi_trotter_incidences():
    deadline = time.time() + 580
    
    # Generate candidate points - larger grid for more diversity
    candidate_points = generate_candidate_points(size=50, extra=2000)
    
    best_I_global = -1
    best_points_global = None
    best_lines_global = None
    
    # Try multiple initial configurations
    for config_id in range(4):
        if time.time() > deadline - 40:
            break
            
        lines = generate_initial_lines(config_id)
        points, _ = get_best_points_for_lines(lines, candidate_points)
        I = count_incidences(points, lines)
        
        if I > best_I_global:
            best_I_global = I
            best_points_global = points.copy()
            best_lines_global = lines.copy()
            
        if time.time() < deadline - 50:
            remaining_time = min(130, deadline - time.time() - 40)
            if remaining_time > 15:
                bl, bp, bi = improve_configuration(lines, candidate_points, 
                                                   time_limit=remaining_time, 
                                                   seed=200 + config_id * 17)
                if bi > best_I_global:
                    best_I_global = bi
                    best_points_global = bp.copy()
                    best_lines_global = bl.copy()
    
    # Final refinement: do one more round of greedy line selection on best points
    if best_points_global is not None and time.time() < deadline - 30:
        refined_lines = greedy_select_lines_from_points(best_points_global, 27)
        unique_refined = []
        for l in refined_lines:
            nl = normalize_line(int(l[0]), int(l[1]), int(l[2]))
            if nl and tuple(nl) not in [tuple(u) for u in unique_refined]:
                unique_refined.append(nl)
        if len(unique_refined) >= 27:
            refined_lines = np.array(unique_refined[:27], dtype=np.int64)
            refined_points, _ = get_best_points_for_lines(refined_lines, candidate_points)
            refined_I = count_incidences(refined_points, refined_lines)
            if refined_I > best_I_global:
                best_I_global = refined_I
                best_points_global = refined_points.copy()
                best_lines_global = refined_lines.copy()
    
    # Validation
    if best_points_global is None or best_lines_global is None:
        lines = np.array([(1, 0, -i) for i in range(9)] + [(0, 1, -j) for j in range(6)] + [(1, -1, -c) for c in range(-8, 4)], dtype=np.int64)
        points, _ = get_best_points_for_lines(lines, candidate_points)
        best_points_global = points
        best_lines_global = lines
        
    # Ensure unique points
    unique_points, indices = np.unique(best_points_global, axis=0, return_index=True)
    if len(unique_points) != 54:
        lines = np.array([(1, 0, -i) for i in range(9)] + [(0, 1, -j) for j in range(6)] + [(1, -1, -c) for c in range(-8, 4)], dtype=np.int64)
        points, _ = get_best_points_for_lines(lines, candidate_points)
        best_points_global = points
        best_lines_global = lines
    else:
        best_points_global = unique_points[indices]
        
    # Ensure unique lines
    unique_lines = []
    for line in best_lines_global:
        n_line = normalize_line(int(line[0]), int(line[1]), int(line[2]))
        if n_line and n_line not in unique_lines:
            unique_lines.append(n_line)
            
    if len(unique_lines) < 27:
        lines = np.array([(1, 0, -i) for i in range(9)] + [(0, 1, -j) for j in range(6)] + [(1, -1, -c) for c in range(-8, 4)], dtype=np.int64)
        points, _ = get_best_points_for_lines(lines, candidate_points)
        best_points_global = points
        best_lines_global = lines
    else:
        best_lines_global = np.array(unique_lines[:27], dtype=np.int64)
        
    return best_points_global, best_lines_global

# EVOLVE-BLOCK-END
