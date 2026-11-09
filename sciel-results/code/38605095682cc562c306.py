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

def get_best_points_for_lines(lines, candidate_points):
    """
    Given a set of lines and a pool of candidate points,
    find the top 54 points that maximize incidences.
    Returns (best_points, max_incidence_count)
    """
    L = lines
    if len(candidate_points) == 0:
        return np.zeros((54, 2), dtype=np.int64), 0
    
    px = candidate_points[:, 0].reshape(-1, 1)
    py = candidate_points[:, 1].reshape(-1, 1)
    
    A = L[:, 0].reshape(1, -1)
    B = L[:, 1].reshape(1, -1)
    C = L[:, 2].reshape(1, -1)
    
    vals = A * px + B * py + C
    counts = np.sum(vals == 0, axis=1)
    
    # Get top 54
    top_indices = np.argsort(counts)[::-1][:54]
    best_points = candidate_points[top_indices]
    max_inc = int(counts[top_indices].max()) if len(counts) > 0 else 0
    
    return best_points, max_inc

def generate_candidate_lines(points, n_lines=27, max_candidates=2000):
    """
    Generate a diverse pool of candidate lines from the point set.
    Lines are defined by pairs of points, plus some random lines.
    """
    lines_set = set()
    n = len(points)
    
    # Lines through pairs of points
    if n >= 2:
        # Use a subset of pairs to keep it manageable
        num_pairs = min(n * (n - 1) // 2, max_candidates)
        indices = list(range(n))
        random.shuffle(indices)
        
        pair_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if pair_count >= num_pairs:
                    break
                p1 = points[i]
                p2 = points[j]
                
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                
                A = dx
                B = -dy
                C = dy * p1[0] - dx * p1[1]
                
                nl = normalize_line(A, B, C)
                if nl:
                    lines_set.add(nl)
                    pair_count += 1
            if pair_count >= num_pairs:
                break
    
    # Add some random lines for diversity
    for _ in range(100):
        A = random.randint(-20, 20)
        B = random.randint(-20, 20)
        if A == 0 and B == 0:
            continue
        C = random.randint(-100, 100)
        nl = normalize_line(A, B, C)
        if nl:
            lines_set.add(nl)
    
    lines_list = list(lines_set)
    random.shuffle(lines_list)
    
    # Return up to n_lines candidates
    if len(lines_list) > n_lines * 5:
        return np.array(lines_list[:n_lines * 5], dtype=np.int64)
    return np.array(lines_list, dtype=np.int64)

def simulated_annealing(lines, points, candidate_points, time_limit, seed=42):
    """
    Simulated annealing to jointly optimize lines and points.
    State: (lines, points)
    Moves: 
      1. Replace a line with a new candidate line
      2. Swap a point with a candidate point
    Energy: -I (we minimize -I, i.e., maximize I)
    """
    random.seed(seed)
    
    current_lines = lines.copy()
    current_points = points.copy()
    current_I = count_incidences(current_points, current_lines)
    
    best_lines = current_lines.copy()
    best_points = current_points.copy()
    best_I = current_I
    
    start_time = time.time()
    T = 10.0  # Initial temperature
    cooling = 0.995
    
    no_improve = 0
    
    while time.time() - start_time < time_limit:
        # Check for stagnation
        if no_improve > 500:
            T = max(T, 5.0)  # Reheat
            no_improve = 0
        
        # Choose a move
        move_type = random.random()
        
        if move_type < 0.7:
            # Replace a line
            idx = random.randint(0, len(current_lines) - 1)
            old_line = current_lines[idx].copy()
            
            # Generate new line from random pair of points
            if len(current_points) >= 2:
                p1_idx, p2_idx = random.sample(range(len(current_points)), 2)
                p1 = current_points[p1_idx]
                p2 = current_points[p2_idx]
                
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                
                A = dx
                B = -dy
                C = dy * p1[0] - dx * p1[1]
                
                new_line = normalize_line(A, B, C)
            else:
                new_line = None
            
            if new_line is None:
                no_improve += 1
                continue
            
            # Check if already in lines
            new_line_tup = tuple(new_line)
            if any(tuple(l) == new_line_tup for l in current_lines):
                no_improve += 1
                continue
            
            current_lines[idx] = new_line
            
            # Re-optimize points
            new_points, _ = get_best_points_for_lines(current_lines, candidate_points)
            new_I = count_incidences(new_points, current_lines)
            
            delta_I = new_I - current_I
            
            if delta_I > 0 or random.random() < np.exp(delta_I / T):
                current_points = new_points
                current_I = new_I
                
                if current_I > best_I:
                    best_I = current_I
                    best_lines = current_lines.copy()
                    best_points = current_points.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Revert
                current_lines[idx] = old_line
                no_improve += 1
                
        else:
            # Swap a point with a candidate
            if len(candidate_points) > 54:
                idx = random.randint(0, 53)
                old_point = current_points[idx].copy()
                
                # Pick a random candidate point not already in current_points
                cand_idx = random.randint(0, len(candidate_points) - 1)
                new_point = candidate_points[cand_idx]
                
                # Check if new_point is already in current_points
                if any(np.array_equal(p, new_point) for p in current_points):
                    no_improve += 1
                    continue
                    
                current_points[idx] = new_point
                new_I = count_incidences(current_points, current_lines)
                
                delta_I = new_I - current_I
                
                if delta_I > 0 or random.random() < np.exp(delta_I / T):
                    current_I = new_I
                    
                    if current_I > best_I:
                        best_I = current_I
                        best_lines = current_lines.copy()
                        best_points = current_points.copy()
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    # Revert
                    current_points[idx] = old_point
                    no_improve += 1
        
        # Cool down
        T *= cooling
        T = max(T, 0.1)
    
    return best_lines, best_points, best_I

def szemeredi_trotter_incidences():
    deadline = time.time() + 600  # 600s limit, keep margin
    best_I_global = -1
    best_points_global = None
    best_lines_global = None
    
    # Step 1: Build a large candidate point pool
    # Use a grid of points from -50 to 50
    grid_size = 101  # -50 to 50
    xs = np.arange(-grid_size // 2, grid_size // 2 + 1)
    ys = np.arange(-grid_size // 2, grid_size // 2 + 1)
    XX, YY = np.meshgrid(xs, ys)
    candidate_points = np.column_stack([XX.ravel(), YY.ravel()])
    
    # Step 2: Generate initial line configurations from different families
    initial_configs = []
    
    # Config 1: Grid lines y = a*x + b
    for a in range(1, 8):
        for b in range(1, 8):
            initial_configs.append((a, -1, b))
    # Trim to 27
    initial_configs = initial_configs[:27]
    
    # Config 2: Vertical + horizontal
    initial_configs2 = []
    for i in range(1, 15):
        initial_configs2.append((1, 0, -i))
    for j in range(1, 14):
        initial_configs2.append((0, 1, -j))
    initial_configs2 = initial_configs2[:27]
    
    # Config 3: Diagonals
    initial_configs3 = []
    for c in range(-20, 7):
        initial_configs3.append((1, 1, -c))
    for c in range(-20, 7):
        initial_configs3.append((1, -1, -c))
    initial_configs3 = initial_configs3[:27]
    
    all_initial = [initial_configs, initial_configs2, initial_configs3]
    
    # Step 3: For each initial config, run simulated annealing
    for config_idx, init_lines in enumerate(all_initial):
        if time.time() > deadline - 30:
            break
            
        lines = np.array(init_lines, dtype=np.int64)
        
        # Get best points for these lines
        points, _ = get_best_points_for_lines(lines, candidate_points)
        I = count_incidences(points, lines)
        
        if I > best_I_global:
            best_I_global = I
            best_points_global = points.copy()
            best_lines_global = lines.copy()
        
        # Run simulated annealing
        if time.time() < deadline - 60:
            remaining_time = min(120, deadline - time.time() - 30)
            if remaining_time > 10:
                bl, bp, bi = simulated_annealing(lines, points, candidate_points, 
                                                  time_limit=remaining_time, 
                                                  seed=1000 + config_idx)
                if bi > best_I_global:
                    best_I_global = bi
                    best_points_global = bp.copy()
                    best_lines_global = bl.copy()
    
    # Step 4: Additional random starts
    num_random_starts = 3
    for rs in range(num_random_starts):
        if time.time() > deadline - 30:
            break
            
        # Generate random initial lines
        random_lines = []
        for _ in range(27):
            A = random.randint(-15, 15)
            B = random.randint(-15, 15)
            if A == 0 and B == 0:
                A = 1
            C = random.randint(-50, 50)
            nl = normalize_line(A, B, C)
            if nl and nl not in [tuple(l) for l in random_lines]:
                random_lines.append(nl)
        
        # Ensure we have 27 unique lines
        while len(random_lines) < 27:
            A = random.randint(-15, 15)
            B = random.randint(-15, 15)
            if A == 0 and B == 0:
                A = 1
            C = random.randint(-50, 50)
            nl = normalize_line(A, B, C)
            if nl and nl not in [tuple(l) for l in random_lines]:
                random_lines.append(nl)
        
        lines = np.array(random_lines[:27], dtype=np.int64)
        points, _ = get_best_points_for_lines(lines, candidate_points)
        I = count_incidences(points, lines)
        
        if I > best_I_global:
            best_I_global = I
            best_points_global = points.copy()
            best_lines_global = lines.copy()
        
        if time.time() < deadline - 60:
            remaining_time = min(60, deadline - time.time() - 30)
            if remaining_time > 10:
                bl, bp, bi = simulated_annealing(lines, points, candidate_points,
                                                  time_limit=remaining_time,
                                                  seed=2000 + rs)
                if bi > best_I_global:
                    best_I_global = bi
                    best_points_global = bp.copy()
                    best_lines_global = bl.copy()
    
    # Final Safety Check
    if best_points_global is None or best_lines_global is None:
        # Fallback to a simple grid
        lines = np.array([(a, -1, b) for a in range(1, 5) for b in range(1, 8)][:27], dtype=np.int64)
        points, _ = get_best_points_for_lines(lines, candidate_points)
        best_points_global = points
        best_lines_global = lines
    
    # Validate Points
    unique_points, indices = np.unique(best_points_global, axis=0, return_index=True)
    if len(unique_points) != 54:
        # Fallback
        lines = np.array([(a, -1, b) for a in range(1, 5) for b in range(1, 8)][:27], dtype=np.int64)
        points, _ = get_best_points_for_lines(lines, candidate_points)
        best_points_global = points
        best_lines_global = lines
    else:
        best_points_global = unique_points[indices]
        
    # Validate Lines
    unique_lines = []
    for line in best_lines_global:
        n_line = normalize_line(line[0], line[1], line[2])
        if n_line and n_line not in unique_lines:
            unique_lines.append(n_line)
            
    if len(unique_lines) < 27:
        # Fallback
        lines = np.array([(a, -1, b) for a in range(1, 5) for b in range(1, 8)][:27], dtype=np.int64)
        points, _ = get_best_points_for_lines(lines, candidate_points)
        best_points_global = points
        best_lines_global = lines
    else:
        best_lines_global = np.array(unique_lines[:27], dtype=np.int64)

    return best_points_global, best_lines_global

# EVOLVE-BLOCK-END
