import numpy as np
import time
import random
from math import gcd

def normalize_line(A, B, C):
    """Normalize line coefficients to ensure uniqueness and integer validity."""
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
    """Count (point, line) pairs with A*x + B*y + C == 0."""
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
    """Select k points that maximize incidences with the given lines."""
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
    best_points = candidate_points[top_indices]
    return best_points, int(counts[top_indices].max())

def generate_candidate_points(size=50):
    """Generate a grid of candidate points."""
    xs = np.arange(-size, size + 1)
    ys = np.arange(-size, size + 1)
    XX, YY = np.meshgrid(xs, ys)
    return np.column_stack([XX.ravel(), YY.ravel()])

def greedy_select_lines_from_points(points, n_lines=27):
    """Greedily select lines that maximize total incidences with the given points."""
    P = points.astype(np.int64)
    px = P[:, 0].reshape(-1, 1)
    py = P[:, 1].reshape(-1, 1)
    
    candidate_lines = []
    seen_lines = set()
    n_pts = len(points)
    
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            dx = points[j, 0] - points[i, 0]
            dy = points[j, 1] - points[i, 1]
            A = int(dy)
            B = int(-dx)
            C = int(dx * points[i, 1] - dy * points[i, 0])
            nl = normalize_line(A, B, C)
            if nl and nl not in seen_lines:
                seen_lines.add(nl)
                candidate_lines.append(nl)
    
    if not candidate_lines:
        for i in range(27):
            candidate_lines.append((1, 0, -i))
        return np.array(candidate_lines[:27], dtype=np.int64)
    
    selected_lines = []
    selected_set = set()
    line_counts = {}
    
    for nl in candidate_lines:
        A, B, C = nl
        vals = A * px + B * py + C
        line_counts[nl] = int(np.sum(vals == 0))
    
    for _ in range(n_lines):
        best_line = None
        best_score = -1
        for nl in candidate_lines:
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
    
    while len(selected_lines) < n_lines:
        A = random.randint(-10, 10)
        B = random.randint(-10, 10)
        C = random.randint(-30, 30)
        nl = normalize_line(A, B, C)
        if nl and nl not in selected_set:
            selected_lines.append(nl)
            selected_set.add(nl)
    
    return np.array(selected_lines[:n_lines], dtype=np.int64)

def generate_grid_based_lines():
    """
    Construct lines from a 9x9 grid structure:
    - 9 vertical lines (x = c for c in -4..4)
    - 9 horizontal lines (y = c for c in -4..4)  
    - 9 diagonal lines (x - y = c for c in -4..4)
    This gives a strong baseline of 27 lines.
    """
    lines = []
    # 9 vertical lines
    for c in range(-4, 5):
        lines.append((1, 0, -c))
    # 9 horizontal lines
    for c in range(-4, 5):
        lines.append((0, 1, -c))
    # 9 diagonal lines (x - y = c)
    for c in range(-4, 5):
        lines.append((1, -1, -c))
    return np.array(lines, dtype=np.int64)

def generate_grid_points():
    """
    Generate 54 points from a 9x9 grid by selecting a subset.
    Use the full 9x9 grid (81 points) but select 54 via greedy.
    """
    grid_size = 9
    center = grid_size // 2
    xs = np.arange(-center, center + 1)
    ys = np.arange(-center, center + 1)
    XX, YY = np.meshgrid(xs, ys)
    all_points = np.column_stack([XX.ravel(), YY.ravel()])
    
    # Select 54 points: prefer center, then expand outward
    distances = np.sqrt(XX.ravel()**2 + YY.ravel()**2)
    order = np.argsort(distances)
    selected = all_points[order[:54]]
    return selected

def improve_configuration(lines, candidate_points, time_limit, seed=42):
    """
    Joint optimization using simulated annealing.
    """
    random.seed(seed)
    current_lines = lines.copy()
    current_points, _ = get_best_points_for_lines(current_lines, candidate_points)
    current_I = count_incidences(current_points, current_lines)
    
    best_lines = current_lines.copy()
    best_points = current_points.copy()
    best_I = current_I
    
    start_time = time.time()
    T = 10.0
    cooling = 0.9997
    min_T = 0.5
    
    no_improve = 0
    
    while time.time() - start_time < time_limit:
        if no_improve > 500:
            T = max(T, 10.0)
            no_improve = 0
            
        move_type = random.random()
        
        if move_type < 0.40:
            # Replace a line with a line through a pair of current points
            idx = random.randint(0, 26)
            old_line = current_lines[idx].copy()
            
            p1_idx, p2_idx = random.sample(range(54), 2)
            p1 = current_points[p1_idx]
            p2 = current_points[p2_idx]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            A = int(dy)
            B = int(-dx)
            C = int(dx * p1[1] - dy * p1[0])
            
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
                
        elif move_type < 0.65:
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
                
        elif move_type < 0.85:
            # Re-optimize points for current lines
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
                
                if delta_I > 0 or random.random() < np.exp(delta_I / (T * 0.3)):
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

def polish_configuration(points, lines, candidate_points, time_limit):
    """Final polish: local search focusing on improving worst lines."""
    random.seed(12345)
    current_points = points.copy()
    current_lines = lines.copy()
    current_I = count_incidences(current_points, current_lines)
    
    best_points = current_points.copy()
    best_lines = current_lines.copy()
    best_I = current_I
    
    start_time = time.time()
    
    def compute_line_counts(pts, lns):
        P = pts.astype(np.int64)
        L = lns.astype(np.int64)
        px = P[:, 0].reshape(-1, 1)
        py = P[:, 1].reshape(-1, 1)
        A = L[:, 0].reshape(1, -1)
        B = L[:, 1].reshape(1, -1)
        C = L[:, 2].reshape(1, -1)
        vals = A * px + B * py + C
        return np.sum(vals == 0, axis=0)
    
    def compute_point_counts(pts, lns):
        P = pts.astype(np.int64)
        L = lns.astype(np.int64)
        px = P[:, 0].reshape(-1, 1)
        py = P[:, 1].reshape(-1, 1)
        A = L[:, 0].reshape(1, -1)
        B = L[:, 1].reshape(1, -1)
        C = L[:, 2].reshape(1, -1)
        vals = A * px + B * py + C
        return np.sum(vals == 0, axis=1)
    
    while time.time() - start_time < time_limit:
        line_counts = compute_line_counts(current_points, current_lines)
        point_counts = compute_point_counts(current_points, current_lines)
        
        worst_line_idx = np.argmin(line_counts)
        sorted_pt_idx = np.argsort(point_counts)[::-1]
        top_half = sorted_pt_idx[:30]
        
        improved = False
        for _ in range(30):
            p1_idx, p2_idx = random.sample(list(top_half), 2)
            p1 = current_points[p1_idx]
            p2 = current_points[p2_idx]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            A = int(dy)
            B = int(-dx)
            C = int(dx * p1[1] - dy * p1[0])
            
            new_line = normalize_line(A, B, C)
            if new_line is None:
                continue
            
            new_line_tup = tuple(new_line)
            if any(tuple(l) == new_line_tup for l in current_lines):
                continue
            
            old_line = current_lines[worst_line_idx].copy()
            current_lines[worst_line_idx] = new_line
            new_I = count_incidences(current_points, current_lines)
            
            if new_I > best_I:
                best_I = new_I
                best_lines = current_lines.copy()
                best_points = current_points.copy()
            else:
                current_lines[worst_line_idx] = old_line
            
            if new_I > current_I:
                current_I = new_I
                improved = True
                break
        
        if not improved:
            break
        
        new_points, _ = get_best_points_for_lines(current_lines, candidate_points)
        new_I = count_incidences(new_points, current_lines)
        if new_I >= current_I:
            current_points = new_points
            current_I = new_I
            if current_I > best_I:
                best_I = current_I
                best_lines = current_lines.copy()
                best_points = current_points.copy()
    
    return best_lines, best_points, best_I


def szemeredi_trotter_incidences():
    deadline = time.time() + 570  # 60s safety margin
    
    # Generate candidate points with larger grid
    candidate_points = generate_candidate_points(size=50)
    
    best_I_global = -1
    best_points_global = None
    best_lines_global = None
    
    # Strategy 1: Grid-based initialization (9 vertical + 9 horizontal + 9 diagonal)
    lines_grid = generate_grid_based_lines()
    points_grid, _ = get_best_points_for_lines(lines_grid, candidate_points)
    I_grid = count_incidences(points_grid, lines_grid)
    
    if I_grid > best_I_global:
        best_I_global = I_grid
        best_points_global = points_grid.copy()
        best_lines_global = lines_grid.copy()
    
    # Strategy 2: Pure grid points
    points_pure_grid = generate_grid_points()
    lines_from_points = greedy_select_lines_from_points(points_pure_grid, 27)
    I_pure = count_incidences(points_pure_grid, lines_from_points)
    
    if I_pure > best_I_global:
        best_I_global = I_pure
        best_points_global = points_pure_grid.copy()
        best_lines_global = lines_from_points.copy()
    
    # Run optimization from grid-based start
    if time.time() < deadline - 30:
        remaining = min(150, deadline - time.time() - 30)
        if remaining > 20:
            bl, bp, bi = improve_configuration(lines_grid, candidate_points, 
                                               time_limit=remaining, seed=100)
            if bi > best_I_global:
                best_I_global = bi
                best_points_global = bp.copy()
                best_lines_global = bl.copy()
    
    # Run optimization from pure grid start
    if time.time() < deadline - 30:
        remaining = min(150, deadline - time.time() - 30)
        if remaining > 20:
            bl, bp, bi = improve_configuration(lines_from_points, candidate_points,
                                               time_limit=remaining, seed=200)
            if bi > best_I_global:
                best_I_global = bi
                best_points_global = bp.copy()
                best_lines_global = bl.copy()
    
    # Run optimization from a mixed initial config
    if time.time() < deadline - 20:
        mixed_lines = []
        for c in range(-3, 4):  # 7 vertical
            mixed_lines.append((1, 0, -c))
        for c in range(-3, 4):  # 7 horizontal
            mixed_lines.append((0, 1, -c))
        for c in range(-3, 4):  # 7 diagonal (x-y=c)
            mixed_lines.append((1, -1, -c))
        for c in range(-2, 3):  # 5 anti-diagonal (x+y=c)
            mixed_lines.append((1, 1, -c))
        for c in range(-1, 2):  # 3 more diagonals (2x-y=c)
            mixed_lines.append((2, -1, -c))
        
        mixed_lines_arr = np.array(mixed_lines[:27], dtype=np.int64)
        points_mixed, _ = get_best_points_for_lines(mixed_lines_arr, candidate_points)
        
        remaining = min(100, deadline - time.time() - 20)
        if remaining > 15:
            bl, bp, bi = improve_configuration(mixed_lines_arr, candidate_points,
                                               time_limit=remaining, seed=300)
            if bi > best_I_global:
                best_I_global = bi
                best_points_global = bp.copy()
                best_lines_global = bl.copy()
    
    # Validation and cleanup
    if best_points_global is None or best_lines_global is None:
        lines_fallback = generate_grid_based_lines()
        points_fallback, _ = get_best_points_for_lines(lines_fallback, candidate_points)
        best_points_global = points_fallback
        best_lines_global = lines_fallback
    
    # Ensure 54 unique points
    unique_points, indices = np.unique(best_points_global, axis=0, return_index=True)
    if len(unique_points) != 54:
        lines_fallback = generate_grid_based_lines()
        points_fallback, _ = get_best_points_for_lines(lines_fallback, candidate_points)
        best_points_global = points_fallback
        best_lines_global = lines_fallback
    else:
        best_points_global = unique_points[indices]
    
    # Ensure 27 unique lines
    unique_lines = []
    for line in best_lines_global:
        n_line = normalize_line(int(line[0]), int(line[1]), int(line[2]))
        if n_line and n_line not in unique_lines:
            unique_lines.append(n_line)
    
    if len(unique_lines) < 27:
        lines_fallback = generate_grid_based_lines()
        points_fallback, _ = get_best_points_for_lines(lines_fallback, candidate_points)
        best_points_global = points_fallback
        best_lines_global = lines_fallback
    else:
        best_lines_global = np.array(unique_lines[:27], dtype=np.int64)
    
    # Final polish
    if time.time() < deadline - 10:
        polish_time = min(40, deadline - time.time() - 10)
        if polish_time > 5:
            pl, pp, pi = polish_configuration(best_points_global, best_lines_global, 
                                              candidate_points, polish_time)
            if pi > best_I_global:
                best_I_global = pi
                best_points_global = pp.copy()
                best_lines_global = pl.copy()
    
    # Final validation
    best_points_global = best_points_global.astype(np.int64)
    best_lines_global = best_lines_global.astype(np.int64)
    
    return best_points_global, best_lines_global

# EVOLVE-BLOCK-END
