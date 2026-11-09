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
    P = candidate_points
    
    if len(P) == 0 or len(L) == 0:
        return np.zeros((54, 2), dtype=np.int64), 0

    px = P[:, 0].reshape(-1, 1)
    py = P[:, 1].reshape(-1, 1)
    
    A = L[:, 0].reshape(1, -1)
    B = L[:, 1].reshape(1, -1)
    C = L[:, 2].reshape(1, -1)
    
    # Calculate incidence count for each candidate point with each line
    vals = A * px + B * py + C
    counts = np.sum(vals == 0, axis=1)
    
    # Get top 54 points
    # If we have fewer than 54 candidates, we might need to pad, but our pool is large.
    if len(counts) < 54:
         # Pad with zeros if necessary (should not happen with large pool)
         pad = np.zeros(54 - len(counts), dtype=np.int64)
         counts = np.concatenate([counts, pad])
         # This logic is tricky if we need to return actual points.
         # Assuming pool size >= 54.
         
    top_indices = np.argsort(counts)[::-1][:54]
    best_points = P[top_indices]
    max_inc = int(np.sum(counts[top_indices]))
    
    return best_points, max_inc

def generate_candidate_lines(points):
    """
    Generate all unique lines defined by pairs of points.
    points: (N, 2) array
    Returns: list of normalized (A, B, C) tuples
    """
    lines_set = set()
    N = len(points)
    for i in range(N):
        x1, y1 = points[i]
        for j in range(i + 1, N):
            x2, y2 = points[j]
            
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            
            nl = normalize_line(A, B, C)
            if nl:
                lines_set.add(nl)
                
    return list(lines_set)

def szemeredi_trotter_incidences():
    deadline = time.time() + 580 # Keep 20s safety margin for 600s limit
    
    # 1. Generate a pool of candidate points
    # A grid of 20x20 = 400 points centered at (0,0)
    # Range: -10 to 9
    grid_size = 20
    coords = np.arange(-grid_size // 2, grid_size // 2)
    xs, ys = np.meshgrid(coords, coords)
    candidate_points = np.column_stack((xs.ravel(), ys.ravel()))
    # Ensure int64
    candidate_points = candidate_points.astype(np.int64)
    
    # 2. Generate a pool of candidate lines
    # Lines from all pairs of candidate points might be too many (400*399/2 ~ 80k lines)
    # Storing 80k lines and checking incidences (80k * 54) is ~4M ops, which is fine.
    # However, selecting 27 from 80k is a huge search space.
    
    # To make search tractable, we can restrict the line pool.
    # Let's use a slightly smaller grid for line generation or subsample.
    # Actually, let's generate lines from a dense grid, but maybe limit the pool size to top "interesting" lines?
    # No, let's just generate all lines from a 10x10 grid (100 points -> 4950 lines).
    # 4950 lines is a manageable pool.
    
    small_grid_size = 10
    coords_small = np.arange(-small_grid_size // 2, small_grid_size // 2)
    xs_s, ys_s = np.meshgrid(coords_small, coords_small)
    small_points = np.column_stack((xs_s.ravel(), ys_s.ravel())).astype(np.int64)
    
    line_pool = generate_candidate_lines(small_points)
    line_pool = np.array(line_pool, dtype=np.int64)
    num_lines_pool = len(line_pool)
    
    if num_lines_pool == 0:
         # Fallback
         return np.zeros((54, 2), dtype=np.int64), np.zeros((27, 3), dtype=np.int64)

    # 3. Simulated Annealing to select 27 lines
    # State: A set of 27 line indices from line_pool
    # Objective: Maximize I (incidences of best 54 points)
    
    # Initialize with a random set of 27 lines
    # Or a greedy initialization: pick lines that cover many points in candidate_points
    # Greedy init:
    # Calculate incidence count of each line in pool against ALL candidate_points
    # This gives a proxy for "density"
    
    P_all = candidate_points
    px_all = P_all[:, 0].reshape(-1, 1)
    py_all = P_all[:, 1].reshape(-1, 1)
    
    A_pool = line_pool[:, 0].reshape(1, -1)
    B_pool = line_pool[:, 1].reshape(1, -1)
    C_pool = line_pool[:, 2].reshape(1, -1)
    
    # vals_all: (NumCandidates, NumLinesPool)
    vals_all = A_pool * px_all + B_pool * py_all + C_pool
    line_counts_all = np.sum(vals_all == 0, axis=0) # Count of candidate points on each line
    
    # Select top 27 lines by this count for initial state
    # This is a strong heuristic start
    initial_indices = np.argsort(line_counts_all)[::-1][:27]
    current_indices = set(initial_indices.tolist())
    
    # Get initial score
    current_lines = line_pool[list(current_indices)]
    current_points, current_score = get_best_points_for_lines(current_lines, candidate_points)
    
    best_score = current_score
    best_lines = current_lines.copy()
    best_points = current_points.copy()
    
    # Annealing parameters
    T = 10.0
    T_min = 0.1
    alpha = 0.9995 # Cooling rate
    
    start_time = time.time()
    steps = 0
    
    while T > T_min and time.time() < deadline:
        # Propose a move: swap one line with another
        # Pick a random line in current set to remove
        remove_idx = random.choice(list(current_indices))
        
        # Pick a random line from pool to add
        # To avoid adding a line already in the set, we can pick randomly and check
        # Or maintain a list of available lines
        # For simplicity, pick random index from pool
        add_idx = random.randint(0, num_lines_pool - 1)
        
        if add_idx == remove_idx:
            continue
            
        if add_idx in current_indices:
            continue
            
        # Try the swap
        new_indices = current_indices.copy()
        new_indices.remove(remove_idx)
        new_indices.add(add_idx)
        
        new_lines = line_pool[list(new_indices)]
        new_points, new_score = get_best_points_for_lines(new_lines, candidate_points)
        
        delta = new_score - current_score
        
        # Accept if better, or with probability exp(-delta/T) if worse
        if delta > 0 or random.random() < np.exp(-delta / T):
            current_indices = new_indices
            current_lines = new_lines
            current_points = new_points
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_lines = current_lines.copy()
                best_points = current_points.copy()
        else:
            # Revert (do nothing, current state remains)
            pass
            
        T *= alpha
        steps += 1
        
        # Every 1000 steps, do a "restart" or "kick" if stuck?
        # Or just let it cool.
        
    # Final optimization: Local search on the best found set
    # Try swapping each line in best_lines with any line in pool to see if it improves
    # This is O(27 * PoolSize * CostOfEval)
    # PoolSize ~ 5000. 27 * 5000 = 135k evals.
    # CostOfEval is small (54*27 ops).
    # 135k * ~1000 ops = 135M ops. Might be too slow in Python loop.
    # We can vectorize the evaluation for a single swap.
    
    if time.time() < deadline - 10:
        best_lines_arr = best_lines
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(27):
                # Try replacing line i with every line in pool
                # Vectorized check:
                # For each line in pool, replace line i, compute score.
                # This is hard to vectorize fully across the 27 lines because the "best points" change.
                # We can approximate: keep points fixed, just check if new line has more incidences with current points?
                # No, that's not optimal.
                
                # Let's just do a few random swaps locally
                for _ in range(50):
                    j = random.randint(0, num_lines_pool - 1)
                    if j == i: continue
                    if j in [idx for idx in range(27)]: # This logic is flawed, indices are into pool
                        pass
                    
                    # Construct new lines
                    new_lines = best_lines_arr.copy()
                    new_lines[i] = line_pool[j]
                    
                    # Check duplicates? The pool has unique lines.
                    # But if j is already in best_lines, we'd have a duplicate.
                    # We need to ensure j is not in the current set of best_lines.
                    # best_lines is an array. Checking membership is O(27).
                    if any(np.array_equal(new_lines[i], best_lines_arr[k]) for k in range(27) if k != i):
                         continue
                         
                    new_points, new_score = get_best_points_for_lines(new_lines, candidate_points)
                    if new_score > best_score:
                        best_score = new_score
                        best_lines_arr = new_lines
                        best_points = new_points
                        improved = True
                        break # Restart inner loop or just continue? Break inner, continue outer.
                    if improved:
                        break
                if improved:
                    break

    # Final validation and formatting
    # Ensure 54 unique points
    unique_points, indices = np.unique(best_points, axis=0, return_index=True)
    if len(unique_points) < 54:
        # Fallback to grid points
        lines = np.array([
            [1, 0, -1], [1, 0, -2], [1, 0, -3], [1, 0, -4], [1, 0, -5], [1, 0, -6], [1, 0, -7], [1, 0, -8], [1, 0, -9],
            [0, 1, -1], [0, 1, -2], [0, 1, -3], [0, 1, -4], [0, 1, -5], [0, 1, -6], [0, 1, -7], [0, 1, -8], [0, 1, -9],
            [1, 1, -2], [1, 1, -4], [1, 1, -6], [1, 1, -8], [1, -1, -2], [1, -1, -4], [1, -1, -6], [1, -1, -8],
            [2, 1, -3]
        ], dtype=np.int64)
        points = np.array([
            [1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [1, 9],
            [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8], [2, 9],
            [3, 1], [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [3, 7], [3, 8], [3, 9],
            [4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [4, 6], [4, 7], [4, 8], [4, 9],
            [5, 1], [5, 2], [5, 3], [5, 4], [5, 5], [5, 6], [5, 7], [5, 8], [5, 9],
            [6, 1], [6, 2], [6, 3], [6, 4], [6, 5], [6, 6], [6, 7], [6, 8], [6, 9]
        ], dtype=np.int64)
        return points, lines
    else:
        best_points = unique_points[indices][:54]
        
    # Ensure 27 unique lines
    unique_lines = []
    for line in best_lines_arr:
        nl = normalize_line(line[0], line[1], line[2])
        if nl and nl not in unique_lines:
            unique_lines.append(nl)
            
    if len(unique_lines) < 27:
         # Pad with fallback lines
         fallback_lines = [
            [1, 0, -1], [1, 0, -2], [1, 0, -3], [1, 0, -4], [1, 0, -5], [1, 0, -6], [1, 0, -7], [1, 0, -8], [1, 0, -9],
            [0, 1, -1], [0, 1, -2], [0, 1, -3], [0, 1, -4], [0, 1, -5], [0, 1, -6], [0, 1, -7], [0, 1, -8], [0, 1, -9],
            [1, 1, -2], [1, 1, -4], [1, 1, -6], [1, 1, -8], [1, -1, -2], [1, -1, -4], [1, -1, -6], [1, -1, -8],
            [2, 1, -3]
         ]
         for fl in fallback_lines:
             if len(unique_lines) >= 27: break
             if fl not in unique_lines:
                 unique_lines.append(fl)
                 
    best_lines_final = np.array(unique_lines[:27], dtype=np.int64)
    
    return best_points, best_lines_final

# EVOLVE-BLOCK-END
