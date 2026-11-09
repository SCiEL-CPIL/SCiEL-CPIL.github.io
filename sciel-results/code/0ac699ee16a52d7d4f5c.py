# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random

def _normalize_line(A, B, C):
    A, B, C = int(A), int(B), int(C)
    g = math.gcd(abs(A), abs(B))
    g = math.gcd(g, abs(C))
    if g == 0:
        return (0, 0, 0) # Should not happen for valid lines from points
    A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _get_best_lines(points):
    n = len(points)
    # Generate all lines from pairs of points
    lines_set = set()
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            # Line through (x1,y1) and (x2,y2)
            # A*x + B*y + C = 0
            # A = y2 - y1
            # B = x1 - x2
            # C = x2*y1 - x1*y2
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            if A == 0 and B == 0:
                continue
            norm = _normalize_line(A, B, C)
            lines_set.add(norm)
    
    # If we don't have enough lines, add some arbitrary ones?
    # With 54 points, we should have plenty.
    # But if all points are collinear, we only have 1 line.
    # In that case, we need to add other lines.
    # Let's add a few generic lines if needed.
    if len(lines_set) < 27:
        # Add vertical, horizontal, diagonal lines with small coeffs
        for c in range(-10, 11):
            lines_set.add(_normalize_line(1, 0, -c))
            lines_set.add(_normalize_line(0, 1, -c))
            lines_set.add(_normalize_line(1, 1, -c))
            lines_set.add(_normalize_line(1, -1, -c))
    
    lines_list = list(lines_set)
    
    # Compute incidences
    # points: (n, 2)
    # lines: (m, 3)
    px = points[:, 0:1]
    py = points[:, 1:2]
    
    # Vectorized computation is tricky with variable length lines, 
    # but m is ~1431, n is 54.
    # Looping over lines is fine.
    line_counts = []
    for line in lines_list:
        A, B, C = line
        vals = A * points[:, 0] + B * points[:, 1] + C
        count = np.sum(vals == 0)
        line_counts.append((count, line))
        
    # Sort by count descending
    line_counts.sort(key=lambda x: -x[0])
    
    # Select top 27
    selected = [lc[1] for lc in line_counts[:27]]
    
    # If we have less than 27, pad with zeros (should not happen with padding above)
    while len(selected) < 27:
         selected.append((0, 1, 0)) # Placeholder, shouldn't be used if padding works
    
    return np.array(selected[:27], dtype=np.int64)

def _incidence_count(points, lines):
    # points: (N, 2), lines: (M, 3)
    # We need a matrix of shape (M, N) where entry (i, j) is A_i*x_j + B_i*y_j + C_i
    # px: (N, 1) -> T -> (1, N)
    # A:  (M, 1)
    # Result: (M, N)
    px = points[:, 0:1].T
    py = points[:, 1:2].T
    A = lines[:, 0:1]
    B = lines[:, 1:2]
    C = lines[:, 2:3]
    total = A * px + B * py + C
    return np.sum(total == 0)

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 590
    
    # Initial points: Grid
    k = 3
    xs = np.arange(1, k + 1)
    ys = np.arange(1, 2 * k * k + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # Initial lines
    lines = _get_best_lines(points)
    current_score = _incidence_count(points, lines)
    
    best_points = points.copy()
    best_lines = lines.copy()
    best_score = current_score
    
    # Local Search
    # We will try to move points to improve the score.
    # To speed up, we can keep the set of candidate lines from the previous step?
    # No, moving a point changes all lines passing through it.
    # So we must recompute best lines.
    
    # To make it faster, we can restrict the search space of points.
    # Points are integers.
    
    n_points = len(points)
    
    # Pre-compute a pool of "generic" lines to ensure we always have 27 candidates
    # even if the point configuration is degenerate.
    # This is handled in _get_best_lines.
    
    # Hill Climbing
    # We try to move one point at a time.
    # Number of iterations depends on time.
    # Each iteration: 
    # 1. Pick random point
    # 2. Pick random new pos (small step)
    # 3. Check validity (no overlap)
    # 4. Compute new best lines and score
    # 5. Accept/Reject
    
    # To improve efficiency, we can batch process or use smarter acceptance.
    # But 1431 lines * 54 points is small.
    
    # Let's run for a fixed number of iterations or until deadline.
    # Estimate cost of one iteration:
    # _get_best_lines: ~1431 pairs, unique set, then 1431 * 54 ops. ~100us?
    # So 10,000 iterations in 1 second.
    # We have 590 seconds. We can do millions of iterations.
    
    # However, Python overhead in loops is high.
    # Let's optimize _get_best_lines slightly.
    
    # Actually, let's just run a loop.
    
    # Random seed for reproducibility? Not required, but good for debugging.
    # random.seed(42)
    
    # To avoid getting stuck, we can use a "temperature" or just simple hill climbing with random restarts if stuck?
    # Simple hill climbing with "sideways" moves (accept if score equal) is usually good.
    
    # Let's add a check for timeout.
    
    # Optimization: Instead of recomputing ALL lines from scratch, 
    # we can maintain the set of lines.
    # But lines change when points move.
    
    # Let's just do the naive recompute.
    
    # To speed up the inner loop of _get_best_lines, we can use numpy more.
    # But the set logic is in Python.
    
    # Let's try to optimize _get_best_lines by using a list and then unique-ing.
    
    # Revised _get_best_lines for speed:
    
    def _get_best_lines_fast(points):
        n = len(points)
        # Generate lines
        # Using list comprehension for pairs
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                A = y2 - y1
                B = x1 - x2
                C = x2 * y1 - x1 * y2
                if A != 0 or B != 0:
                    pairs.append((A, B, C))
        
        # Normalize and unique
        # This part is slow in Python.
        # Let's try to vectorize normalization? Hard with gcd.
        # But 1431 items is small.
        
        seen = set()
        unique_lines = []
        for A, B, C in pairs:
            g = math.gcd(abs(A), abs(B))
            g = math.gcd(g, abs(C))
            if g == 0: continue
            A, B, C = A // g, B // g, C // g
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A, B, C = -A, -B, -C
            key = (A, B, C)
            if key not in seen:
                seen.add(key)
                unique_lines.append(key)
                
        # If not enough, add generic
        if len(unique_lines) < 27:
             for c in range(-20, 21):
                 for A, B in [(1,0), (0,1), (1,1), (1,-1)]:
                     key = _normalize_line(A, B, -c)
                     if key not in seen:
                         seen.add(key)
                         unique_lines.append(key)
                         
        lines_arr = np.array(unique_lines, dtype=np.int64)
        
        # Compute counts
        # lines_arr: (m, 3)
        # points: (n, 2)
        # We want sum over points for each line.
        # L @ P^T ?
        # A*x + B*y + C
        # (lines_arr[:, 0:1] * points[:, 0:1].T + lines_arr[:, 1:2] * points[:, 1:2].T + lines_arr[:, 2:3].T)
        # This creates an (m, n) matrix.
        # m ~ 1431, n = 54. 1431 * 54 ~ 77k elements. Very fast.
        
        px = points[:, 0:1].T # (1, n)
        py = points[:, 1:2].T # (1, n)
        A = lines_arr[:, 0:1] # (m, 1)
        B = lines_arr[:, 1:2] # (m, 1)
        C = lines_arr[:, 2:3] # (m, 1)
        
        vals = A * px + B * py + C.T # (m, n)
        counts = np.sum(vals == 0, axis=1) # (m,)
        
        # Find top 27
        # argsort
        if len(counts) > 27:
             # We want indices of top 27
             # Use argpartition for speed? Or just sort.
             # Sorting 1431 elements is fast.
             idx = np.argsort(-counts)
             top_idx = idx[:27]
        else:
             top_idx = np.arange(len(counts))
             
        return lines_arr[top_idx]
    
    # Replace the inner function definition in the loop? No, define it outside or use the global one.
    # Let's redefine _get_best_lines to use the fast version logic.
    
    # Update global _get_best_lines to use the fast method
    def _get_best_lines_optimized(points):
        n = len(points)
        pairs = []
        # Generate all lines defined by pairs of points
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i + 1, n):
                x2, y2 = points[j]
                A = y2 - y1
                B = x1 - x2
                C = x2 * y1 - x1 * y2
                if A != 0 or B != 0:
                    pairs.append((A, B, C))
        
        seen = set()
        unique_lines = []
        for A, B, C in pairs:
            g = math.gcd(abs(A), abs(B))
            g = math.gcd(g, abs(C))
            if g == 0: continue
            A, B, C = A // g, B // g, C // g
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A, B, C = -A, -B, -C
            key = (A, B, C)
            if key not in seen:
                seen.add(key)
                unique_lines.append(key)
                
        if len(unique_lines) < 27:
             for c in range(-20, 21):
                 for A, B in [(1,0), (0,1), (1,1), (1,-1)]:
                     key = _normalize_line(A, B, -c)
                     if key not in seen:
                         seen.add(key)
                         unique_lines.append(key)
                         
        lines_arr = np.array(unique_lines, dtype=np.int64)
        
        # Correct broadcasting:
        # points: (N, 2)
        # lines: (M, 3)
        # We want M x N matrix of values A*x + B*y + C
        # px: (N,) -> (1, N)
        # A: (M, 1)
        # Result: (M, N)
        px = points[:, 0:1].T
        py = points[:, 1:2].T
        A = lines_arr[:, 0:1]
        B = lines_arr[:, 1:2]
        C = lines_arr[:, 2:3]
        
        vals = A * px + B * py + C
        counts = np.sum(vals == 0, axis=1)
        
        if len(counts) > 27:
             idx = np.argsort(-counts)
             top_idx = idx[:27]
        else:
             top_idx = np.arange(len(counts))
             
        return lines_arr[top_idx]

    # Use the optimized version
    lines = _get_best_lines_optimized(points)
    current_score = _incidence_count(points, lines)
    
    best_points = points.copy()
    best_lines = lines.copy()
    best_score = current_score
    
    # Hill Climbing Loop
    # We will try to move points.
    # To allow exploration, we accept equal scores with prob 0.5.
    # We also try larger jumps occasionally.
    
    # To speed up, we can pre-allocate some arrays?
    # No, numpy handles it.
    
    # Let's run the loop.
    # We need to be careful with the time limit.
    
    # Check time every 1000 iterations.
    
    # To improve the search, we can also try to "swap" points?
    # No, moving one point is sufficient to reach any configuration.
    
    # Let's add a "restart" mechanism if we get stuck for a long time?
    # Or just keep going.
    
    # One issue: The points might drift far away.
    # We can constrain the search space?
    # Or just let it drift.
    
    # Let's try to bias the search towards the center?
    # No, just random walk.
    
    # To make the search more effective, we can try multiple random points per iteration?
    # No, one at a time is standard.
    
    # Let's run.
    
    # We will use a counter to check time.
    iter_count = 0
    # Use a smaller step size (1) to allow more frequent updates and finer tuning,
    # or vary it. Step 1 is safer for integer grid.
    while time.time() < deadline - 5:
        iter_count += 1
        
        # Pick a random point
        i = random.randint(0, n_points - 1)
        
        # Pick a new position
        # Try small steps first
        dx = random.randint(-1, 1)
        dy = random.randint(-1, 1)
        
        new_x = points[i, 0] + dx
        new_y = points[i, 1] + dy
        
        # Check if new position is valid (not occupied)
        # Vectorized check
        if np.any((points[:, 0] == new_x) & (points[:, 1] == new_y) & (np.arange(n_points) != i)):
            continue
            
        # Create new points array
        new_points = points.copy()
        new_points[i, 0] = new_x
        new_points[i, 1] = new_y
        
        # Compute new best lines and score
        new_lines = _get_best_lines_optimized(new_points)
        new_score = _incidence_count(new_points, new_lines)
        
        # Accept/Reject
        if new_score > current_score:
            points = new_points
            lines = new_lines
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
                best_lines = lines.copy()
                
        elif new_score == current_score:
            if random.random() < 0.5:
                points = new_points
                lines = new_lines
                # current_score stays same
                
        else:
            # Reject
            pass
            
    # Return best found
    return best_points, best_lines


# EVOLVE-BLOCK-END
