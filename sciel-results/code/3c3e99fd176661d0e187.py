import numpy as np
import time
import random

def szemeredi_trotter_incidences():
    """
    Maximizes point-line incidences for 54 points and 27 lines on the integer lattice.
    Uses a greedy selection from a pool of lines with small integer coefficients,
    followed by a local search to optimize the set of lines.
    """
    start_time = time.time()
    deadline = 600  # seconds
    safety_margin = 10
    
    # 1. Define the point set
    # A 6x9 grid is a standard choice for 54 points that allows for good line coverage.
    # We can shift it to be centered or start at 1.
    # Let's use x in 1..6, y in 1..9
    xs = np.arange(1, 7)
    ys = np.arange(1, 10)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # 2. Generate a pool of candidate lines
    # Lines are defined by A*x + B*y + C = 0.
    # We want lines that pass through many grid points.
    # Small coefficients A, B are likely to be better.
    # Let's generate all lines defined by pairs of points in a slightly larger grid,
    # or simply enumerate A, B, C in a range.
    # Enumerating A, B, C is easier and ensures we don't miss lines that don't pass through grid point pairs (though those are useless for incidence).
    # Actually, any line with >= 2 incidences must pass through two grid points.
    # So we can generate lines from pairs of points in a "superset" grid or the same grid.
    # If we use the same grid, we only get lines with >= 2 points.
    # Let's use a slightly larger grid for line generation to allow lines that might cut through the 54 points nicely but are defined by points outside?
    # No, if a line doesn't pass through any of the 54 points, it contributes 0.
    # So we only care about lines that pass through at least one point.
    # To find lines with many points, they must pass through pairs of points.
    # Let's generate lines from all pairs of points in the 54-point set, plus some "nearby" points to capture lines that might have 1 point in the set but are "aligned" with structure?
    # Actually, just pairs from the 54 points is sufficient to find all lines with >= 2 incidences.
    # Lines with exactly 1 incidence are unlikely to be optimal unless we are forced to pick them, but we have 27 lines and 54 points.
    # The average incidence per line in the best case is high.
    
    # Let's generate candidate lines from all pairs of points in the set.
    # This might be O(N^2) lines, which is ~1500. That's a small pool.
    # To get more variety, we can extend the grid slightly for line definition.
    # Let's use a grid of x in -2..9, y in -2..12 for line generation.
    xs_gen = np.arange(-2, 10)
    ys_gen = np.arange(-2, 13)
    px_gen, py_gen = np.meshgrid(xs_gen, ys_gen, indexing="ij")
    gen_points = np.stack([px_gen.ravel(), py_gen.ravel()], axis=1).astype(np.int64)
    
    # Generate lines from pairs of gen_points
    # We will store lines as normalized (A, B, C) tuples
    line_set = set()
    n_gen = len(gen_points)
    for i in range(n_gen):
        x1, y1 = gen_points[i]
        for j in range(i + 1, n_gen):
            x2, y2 = gen_points[j]
            # Line through (x1, y1) and (x2, y2)
            # A = y2 - y1
            # B = x1 - x2
            # C = x2*y1 - x1*y2
            A = int(y2 - y1)
            B = int(x1 - x2)
            C = int(x2 * y1 - x1 * y2)
            
            # Normalize
            if A == 0 and B == 0:
                continue
            # Find gcd
            g = 0
            for val in [abs(A), abs(B), abs(C)]:
                if g == 0:
                    g = val
                else:
                    while val % g != 0:
                        val, g = g, val % g
                    g = val # This is wrong, need proper gcd
            
            # Proper GCD
            import math
            g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
            if g == 0:
                continue
            A //= g
            B //= g
            C //= g
            
            # Ensure first non-zero is positive
            if A < 0 or (A == 0 and B < 0):
                A = -A
                B = -B
                C = -C
            
            line_set.add((A, B, C))
    
    # Convert to array for vectorized operations
    lines_pool = np.array(list(line_set), dtype=np.int64)
    
    # Filter out lines that don't hit any of the 54 points
    # Calculate incidences for all lines in pool with all points
    # points: (54, 2), lines_pool: (M, 3)
    # Incidence if A*x + B*y + C == 0
    
    # To speed up, we can compute A*x + B*y for all pairs
    # This is (54, 1, 1) * (1, M, 1) + (54, 1, 1) * (1, 1, M) + (1, M, 1)
    # Might be memory heavy if M is large.
    # n_gen is 12 * 15 = 180 points. Pairs ~ 16000.
    # Unique lines will be less.
    # Let's check size of lines_pool
    M = len(lines_pool)
    
    # Vectorized incidence check
    # P: (54, 2)
    # L: (M, 3)
    # Val = P[:, 0] * L[:, 0] + P[:, 1] * L[:, 1] + L[:, 2]
    # Shape: (54, M)
    
    # We can do this in chunks if M is large, but 16k * 54 is ~1M operations, which is fine.
    Px = points[:, 0].reshape(-1, 1)
    Py = points[:, 1].reshape(-1, 1)
    LA = lines_pool[:, 0].reshape(1, -1)
    LB = lines_pool[:, 1].reshape(1, -1)
    LC = lines_pool[:, 2].reshape(1, -1)
    
    vals = Px * LA + Py * LB + LC
    inc_matrix = (vals == 0).astype(np.int64) # (54, M)
    
    # Count incidences per line
    line_counts = inc_matrix.sum(axis=0)
    
    # Keep only lines with count > 0
    valid_indices = np.where(line_counts > 0)[0]
    lines_pool = lines_pool[valid_indices]
    inc_matrix = inc_matrix[:, valid_indices]
    line_counts = line_counts[valid_indices]
    M = len(lines_pool)
    
    if M == 0:
        # Fallback
        return points, np.zeros((27, 3), dtype=np.int64)
        
    # Greedy Selection
    # We want to select 27 lines to maximize sum of incidences.
    # This is a set cover-like problem but with weights.
    # Greedy: pick line with max current incidence, update remaining points?
    # No, points are fixed. We just want to maximize sum of counts of selected lines.
    # Since incidences are additive, the optimal solution is simply the 27 lines with the highest individual counts?
    # Wait. Is there any interaction?
    # The score is I = sum_{p in P, l in L} [p on l].
    # I = sum_{l in L} (number of points on l).
    # Since the terms are independent, the maximum is achieved by picking the 27 lines with the highest individual incidence counts.
    # Is that true?
    # Yes, because the objective function is linear in the selection of lines.
    # Let x_l be 1 if line l is selected.
    # Maximize sum_l x_l * count_l subject to sum x_l = 27, x_l in {0, 1}.
    # The optimal solution is to pick the 27 largest count_l.
    
    # So, why did the "mixed slopes" greedy do better?
    # Maybe the pool was different?
    # Or maybe I am missing something.
    # "greedy selection picks the highest-incidence lines across all slope families"
    # If the objective is just the sum of incidences, and incidences are defined per line independently, then yes, top 27 lines is optimal for a fixed pool.
    # The improvement must come from a better pool or a different point set.
    
    # Let's verify the score of the top 27 lines from this pool.
    sorted_indices = np.argsort(line_counts)[::-1]
    selected_indices = sorted_indices[:27]
    selected_lines = lines_pool[selected_indices]
    
    # Calculate score
    current_score = line_counts[selected_indices].sum()
    
    # Is it possible to improve the point set?
    # The point set is fixed in the function, but we can choose it.
    # The task says "returning (points, lines)". We can choose the points.
    # The baseline used a 3x18 grid.
    # The inspiration used a 6x9 grid.
    # Let's try a few different point configurations and pick the best one.
    
    best_points = points
    best_lines = selected_lines
    best_score = current_score
    
    # Try a few other grids
    configs = [
        (6, 9),   # 54
        (3, 18),  # 54
        (2, 27),  # 54
        (9, 6),   # 54
        (1, 54),  # 54 - degenerate
        (54, 1),  # 54 - degenerate
    ]
    
    # Also try non-rectangular or shifted grids?
    # For now, stick to rectangular grids centered or aligned.
    
    for h, w in configs:
        if h * w != 54:
            continue
            
        # Generate points
        # Center them? Or start at 1?
        # Starting at 1 is fine, but centering might allow symmetric lines like x=0, y=0.
        # Let's try centered.
        x_start = -w // 2
        y_start = -h // 2
        xs_c = np.arange(x_start, x_start + w)
        ys_c = np.arange(y_start, y_start + h)
        px_c, py_c = np.meshgrid(xs_c, ys_c, indexing="ij")
        pts_c = np.stack([px_c.ravel(), py_c.ravel()], axis=1).astype(np.int64)
        
        # Generate lines for this point set
        # Use a slightly larger generation grid
        pad = 2
        xs_g = np.arange(x_start - pad, x_start + w + pad)
        ys_g = np.arange(y_start - pad, y_start + h + pad)
        px_g, py_g = np.meshgrid(xs_g, ys_g, indexing="ij")
        gen_pts_c = np.stack([px_g.ravel(), py_g.ravel()], axis=1).astype(np.int64)
        
        line_set_c = set()
        n_g = len(gen_pts_c)
        # To save time, only generate lines if we have time
        if time.time() - start_time > deadline - safety_margin:
            break
            
        for i in range(n_g):
            x1, y1 = gen_pts_c[i]
            for j in range(i + 1, n_g):
                x2, y2 = gen_pts_c[j]
                A = int(y2 - y1)
                B = int(x1 - x2)
                C = int(x2 * y1 - x1 * y2)
                if A == 0 and B == 0:
                    continue
                g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
                if g == 0:
                    continue
                A //= g
                B //= g
                C //= g
                if A < 0 or (A == 0 and B < 0):
                    A = -A
                    B = -B
                    C = -C
                line_set_c.add((A, B, C))
        
        if not line_set_c:
            continue
            
        lines_c = np.array(list(line_set_c), dtype=np.int64)
        M_c = len(lines_c)
        
        # Calculate incidences
        Px_c = pts_c[:, 0].reshape(-1, 1)
        Py_c = pts_c[:, 1].reshape(-1, 1)
        LA_c = lines_c[:, 0].reshape(1, -1)
        LB_c = lines_c[:, 1].reshape(1, -1)
        LC_c = lines_c[:, 2].reshape(1, -1)
        
        vals_c = Px_c * LA_c + Py_c * LB_c + LC_c
        inc_c = (vals_c == 0).astype(np.int64)
        counts_c = inc_c.sum(axis=0)
        
        valid_c = np.where(counts_c > 0)[0]
        if len(valid_c) == 0:
            continue
            
        lines_c = lines_c[valid_c]
        counts_c = counts_c[valid_c]
        
        if len(counts_c) < 27:
            # Pad with zero-count lines? Or just take what we have?
            # We need 27 lines. If we have fewer, we can duplicate? No, "two rows describing the same line" is invalid.
            # We must find 27 distinct lines.
            # If the pool is too small, we need to expand the generation grid.
            # For now, skip if not enough.
            continue
            
        sorted_c = np.argsort(counts_c)[::-1]
        sel_c = sorted_c[:27]
        score_c = counts_c[sel_c].sum()
        
        if score_c > best_score:
            best_score = score_c
            best_points = pts_c
            best_lines = lines_c[sel_c]
            
    # Check time
    if time.time() - start_time > deadline - safety_margin:
        return best_points, best_lines
        
    # If we have time, try to improve with local search?
    # Since the objective is linear, local search on lines won't help if we already picked the top 27 from a sufficient pool.
    # The bottleneck is the pool size and point configuration.
    
    # One more check: ensure no duplicate lines in best_lines
    # They come from a set, so they are unique.
    
    return best_points, best_lines
