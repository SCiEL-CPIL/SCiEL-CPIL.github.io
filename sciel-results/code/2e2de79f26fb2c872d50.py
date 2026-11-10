import numpy as np
import time
import random
from math import gcd

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
        A = -A
        B = -B
        C = -C
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
    return int(np.sum(vals == 0))

def get_best_points_for_lines(lines, candidate_points, k=54):
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
    top_indices = np.argsort(counts)[::-1][:k]
    return candidate_points[top_indices], int(counts[top_indices].max())

def generate_candidate_points(size=40):
    xs = np.arange(-size, size + 1)
    ys = np.arange(-size, size + 1)
    XX, YY = np.meshgrid(xs, ys)
    return np.column_stack([XX.ravel(), YY.ravel()])

def greedy_select_lines_from_points(points, n_lines=27):
    """Greedily select lines by marginal gain: each new line must cover points not yet covered."""
    P = points.astype(np.int64)
    n_pts = len(points)
    
    # Precompute all candidate lines from point pairs
    candidate_lines = []
    seen_lines = set()
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
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
        lines = [(1, 0, -i) for i in range(27)]
        return np.array(lines, dtype=np.int64)
    
    # For each candidate line, precompute which points it covers
    px = P[:, 0]
    py = P[:, 1]
    line_point_masks = []
    for nl in candidate_lines:
        A, B, C = nl
        vals = A * px + B * py + C
        mask = (vals == 0)
        line_point_masks.append(mask)
    
    selected_lines = []
    selected_set = set()
    covered = np.zeros(n_pts, dtype=bool)
    
    for _ in range(n_lines):
        best_line = None
        best_gain = -1
        for idx, nl in enumerate(candidate_lines):
            if nl in selected_set:
                continue
            # Marginal gain = number of points on this line not yet covered
            gain = int(np.sum(line_point_masks[idx] & ~covered))
            if gain > best_gain:
                best_gain = gain
                best_line = (idx, nl)
        if best_line is None or best_gain == 0:
            break
        idx, nl = best_line
        selected_lines.append(nl)
        selected_set.add(nl)
        covered |= line_point_masks[idx]
    
    # Fill with random lines if needed
    while len(selected_lines) < n_lines:
        A = random.randint(-5, 5)
        B = random.randint(-5, 5)
        C = random.randint(-20, 20)
        nl = normalize_line(A, B, C)
        if nl and nl not in selected_set:
            selected_lines.append(nl)
            selected_set.add(nl)
    
    return np.array(selected_lines[:n_lines], dtype=np.int64)

def generate_initial_lines(config_id):
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
        # Small slopes
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
            lines.append((1, 1, i + 1))
    elif config_id == 3:
        # Random small coefficients
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
    elif config_id == 4:
        # Finer grid
        for i in range(11):
            lines.append((1, 0, -i))
        for j in range(5):
            lines.append((0, 1, -j))
        for c in range(-10, 2):
            lines.append((1, -1, -c))
    elif config_id == 5:
        # Different slope set
        a_vals = [0, 1, -1, 2, -2, 3]
        b_vals = [0, 1, -1, 2]
        for a in a_vals:
            for b in b_vals:
                lines.append((a, -1, b))
                if len(lines) >= 27:
                    break
            if len(lines) >= 27:
                break
        while len(lines) < 27:
            lines.append((0, 1, len(lines)))
    elif config_id == 6:
        # Near-pencil: most lines through (0,0), a few offset
        pairs = [(1,0), (0,1), (1,1), (1,-1), (2,1), (1,2), (3,1), (1,3), (3,2), (2,3), (3,4), (4,3), (5,1), (1,5), (4,1), (1,4), (5,2), (2,5), (5,3), (3,5), (4,5), (5,4), (6,1), (1,6), (6,5), (5,6)]
        for A, B in pairs:
            lines.append((A, -B, 0))
        for i in range(27 - len(lines)):
            lines.append((1, 0, i + 1))
    elif config_id == 7:
        # Two-direction grid: horizontal and vertical with wider spacing
        for i in range(14):
            lines.append((1, 0, -i * 2))
        for j in range(13):
            lines.append((0, 1, -j * 2))
    
    return np.array(lines[:27], dtype=np.int64)

def improve_configuration(lines, candidate_points, time_limit, seed=42):
    random.seed(seed)
    current_lines = lines.copy()
    current_points, _ = get_best_points_for_lines(current_lines, candidate_points)
    current_I = count_incidences(current_points, current_lines)
    
    best_lines = current_lines.copy()
    best_points = current_points.copy()
    best_I = current_I
    
    start_time = time.time()
    T = 10.0
    cooling = 0.9995
    min_T = 0.5
    no_improve = 0
    n_candidates = len(candidate_points)
    
    # Precompute candidate point incidence matrix for fast point-swap scoring
    L = current_lines.astype(np.int64)
    CP = candidate_points.astype(np.int64)
    cpx = CP[:, 0].reshape(-1, 1)
    cpy = CP[:, 1].reshape(-1, 1)
    LA = L[:, 0].reshape(1, -1)
    LB = L[:, 1].reshape(1, -1)
    LC = L[:, 2].reshape(1, -1)
    cand_incidence = np.sum((LA * cpx + LB * cpy + LC) == 0, axis=1)  # shape (n_candidates,)
    
    while time.time() - start_time < time_limit:
        if no_improve > 400:
            T = max(T, 10.0)
            no_improve = 0
        
        move_type = random.random()
        
        if move_type < 0.30:
            # Replace a line with a line through a pair of current points
            idx = random.randint(0, 26)
            old_line = current_lines[idx].copy()
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
            if delta_I > 0 or random.random() < np.exp(delta_I / T):
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
                
        elif move_type < 0.50:
            # Replace a line with a random line from small coefficients
            idx = random.randint(0, 26)
            old_line = current_lines[idx].copy()
            A = random.randint(-8, 8)
            B = random.randint(-8, 8)
            C = random.randint(-30, 30)
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
            if delta_I > 0 or random.random() < np.exp(delta_I / T):
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
            # Swap a point: use precomputed incidence for fast scoring
            pt_idx = random.randint(0, 53)
            old_pt = current_points[pt_idx].copy()
            cand_idx = random.randint(0, n_candidates - 1)
            new_pt = candidate_points[cand_idx].copy()
            if any(np.array_equal(current_points[i], new_pt) for i in range(54)):
                continue
            # Fast delta: new point's incidence minus old point's incidence
            # Compute old point's incidence on current lines
            L_cur = current_lines.astype(np.int64)
            old_inc = int(np.sum(L_cur[:, 0] * old_pt[0] + L_cur[:, 1] * old_pt[1] + L_cur[:, 2] == 0))
            new_inc = int(cand_incidence[cand_idx])
            delta_I = new_inc - old_inc
            if delta_I > 0 or random.random() < np.exp(delta_I / T):
                current_points[pt_idx] = new_pt
                current_I += delta_I
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
            # Rebuild: greedily re-select lines from current points
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
                if delta_I > 0 or random.random() < np.exp(delta_I / (T * 0.5)):
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
    deadline = time.time() + 570  # 30s safety margin
    
    candidate_points = generate_candidate_points(size=40)
    
    best_I_global = -1
    best_points_global = None
    best_lines_global = None
    
    for config_id in range(8):
        if time.time() > deadline - 30:
            break
        
        lines = generate_initial_lines(config_id)
        points, _ = get_best_points_for_lines(lines, candidate_points)
        I = count_incidences(points, lines)
        
        if I > best_I_global:
            best_I_global = I
            best_points_global = points.copy()
            best_lines_global = lines.copy()
        
        if time.time() < deadline - 40:
            remaining_time = min(80, deadline - time.time() - 30)
            if remaining_time > 10:
                bl, bp, bi = improve_configuration(lines, candidate_points,
                                                   time_limit=remaining_time,
                                                   seed=300 + config_id * 23)
                if bi > best_I_global:
                    best_I_global = bi
                    best_points_global = bp.copy()
                    best_lines_global = bl.copy()
    
    # Fallback
    if best_points_global is None or best_lines_global is None:
        lines = np.array([(1, 0, -i) for i in range(9)] + [(0, 1, -j) for j in range(6)] + [(1, -1, -c) for c in range(-8, 4)], dtype=np.int64)
        points, _ = get_best_points_for_lines(lines, candidate_points)
        best_points_global = points
        best_lines_global = lines
    
    # Validate points: 54 unique
    unique_points, indices = np.unique(best_points_global, axis=0, return_index=True)
    if len(unique_points) != 54:
        lines = np.array([(1, 0, -i) for i in range(9)] + [(0, 1, -j) for j in range(6)] + [(1, -1, -c) for c in range(-8, 4)], dtype=np.int64)
        points, _ = get_best_points_for_lines(lines, candidate_points)
        best_points_global = points
        best_lines_global = lines
    else:
        best_points_global = unique_points[indices]
    
    # Validate lines: 27 unique
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
