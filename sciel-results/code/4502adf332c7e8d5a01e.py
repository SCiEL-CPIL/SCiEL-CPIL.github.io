# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from math import gcd
from itertools import product

def _normalize_line(A, B, C):
    """Normalize line coefficients (A, B, C) to a canonical form."""
    if A == 0 and B == 0:
        return (0, 0, 0)
    
    # Find GCD of A, B, C
    g = gcd(abs(A), abs(B))
    g = gcd(g, abs(C))
    
    if g > 1:
        A //= g
        B //= g
        C //= g
        
    # Ensure first non-zero entry is positive
    if A < 0:
        A = -A
        B = -B
        C = -C
    elif A == 0 and B < 0:
        B = -B
        C = -C
        
    return (A, B, C)

def _count_incidences(points, top_k=27):
    """
    Count the number of points on each line defined by pairs of points.
    Returns the sum of the counts of the top_k lines.
    Also returns the top_k lines themselves.
    """
    n = len(points)
    if n < 2:
        return 0, []

    line_counts = {}
    
    # Iterate over all pairs to find lines
    # O(N^2) for N=54 is ~1431 pairs, which is fast.
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            # Calculate line coefficients for (x1, y1) and (x2, y2)
            # A*x + B*y + C = 0
            # (y2-y1)x - (x2-x1)y + (x2-x1)y1 - (y2-y1)x2 = 0
            A = y2 - y1
            B = -(x2 - x1)
            C = (x2 - x1) * y1 - (y2 - y1) * x2
            
            if A == 0 and B == 0:
                continue # Duplicate points, shouldn't happen if valid
                
            key = _normalize_line(A, B, C)
            line_counts[key] = line_counts.get(key, 0) + 1
            
    # Convert pair counts to point counts
    line_point_counts = {}
    for key, pc in line_counts.items():
        # Solve k(k-1)/2 = pc
        # k^2 - k - 2*pc = 0
        # k = (1 + sqrt(1 + 8*pc)) / 2
        disc = 1 + 8 * pc
        sqrt_disc = int(np.sqrt(disc))
        if sqrt_disc * sqrt_disc == disc:
            k = (1 + sqrt_disc) // 2
            if k * (k - 1) // 2 == pc:
                line_point_counts[key] = k
            else:
                line_point_counts[key] = 2 # Fallback
        else:
            # This case implies the pair counts don't form a perfect triangle number,
            # which is impossible if the counts are correct for distinct lines.
            line_point_counts[key] = 2

    # Get top 27 lines
    sorted_lines = sorted(line_point_counts.items(), key=lambda item: item[1], reverse=True)
    
    total_inc = 0
    top_lines = []
    for i in range(min(top_k, len(sorted_lines))):
        key, count = sorted_lines[i]
        total_inc += count
        top_lines.append(key)
        
    return total_inc, top_lines

def _generate_padding_lines(points, existing_keys, num_needed):
    """Generate valid lines that are not in existing_keys."""
    padding = []
    existing_set = set(existing_keys)
    
    # Try to find lines passing through single points or just valid lines
    # We can generate lines x=c, y=c, x+y=c, etc.
    
    coords = points
    min_x = np.min(coords[:, 0])
    max_x = np.max(coords[:, 0])
    min_y = np.min(coords[:, 1])
    max_y = np.max(coords[:, 1])
    
    candidates = []
    
    # Generate a pool of simple lines
    for c in range(min_x - 10, max_x + 11):
        key = _normalize_line(1, 0, -c)
        if key not in existing_set:
            candidates.append(key)
    for c in range(min_y - 10, max_y + 11):
        key = _normalize_line(0, 1, -c)
        if key not in existing_set:
            candidates.append(key)
    for c in range(min_x + min_y - 20, max_x + max_y + 21):
        key = _normalize_line(1, 1, -c)
        if key not in existing_set:
            candidates.append(key)
        key = _normalize_line(1, -1, -c)
        if key not in existing_set:
            candidates.append(key)
            
    random.shuffle(candidates)
    for k in candidates:
        if len(padding) >= num_needed:
            break
        padding.append(k)
        existing_set.add(k)
        
    # If still not enough, generate random garbage lines (score 0)
    while len(padding) < num_needed:
        A = random.randint(-10, 10)
        B = random.randint(-10, 10)
        C = random.randint(-100, 100)
        key = _normalize_line(A, B, C)
        if key not in existing_set and key != (0,0,0):
            padding.append(key)
            existing_set.add(key)
            
    return padding

def _score_and_get_lines(points):
    """
    Compute the score (sum of top 27 line incidences) and the actual 27 lines.
    """
    inc, top_keys = _count_incidences(points, 27)
    
    # We need exactly 27 lines.
    if len(top_keys) < 27:
        num_needed = 27 - len(top_keys)
        padding = _generate_padding_lines(points, top_keys, num_needed)
        top_keys.extend(padding)
        
    # Convert keys to array
    lines_arr = np.array(top_keys, dtype=np.int64)
    
    return inc, lines_arr

def _is_valid_points(points):
    """Check for duplicate points."""
    pts = [tuple(p) for p in points]
    if len(set(pts)) != len(pts):
        return False
    return True

def _generate_grid_points(rows, cols, start_x=0, start_y=0, dx=1, dy=1):
    """Generate a simple rectangular grid of points."""
    pts = []
    for r in range(rows):
        for c in range(cols):
            x = start_x + c * dx
            y = start_y + r * dy
            pts.append([x, y])
    return np.array(pts, dtype=np.int64)

def _generate_fat_grid_points(rows, cols, spread=1):
    """
    Generate a 'fat' grid where each cell is a small cluster.
    This increases incidences by creating local collinearities.
    """
    pts = []
    offsets = [(0,0), (1,0), (0,1), (1,1)] # 2x2 cluster
    # We need 54 points. 
    # If we use a 3x2 grid of clusters, that's 6 clusters * 9 points = 54.
    # Or 2x3 grid of clusters.
    # Let's try to fit 54 points into a grid structure with some flexibility.
    
    # For this specific function, let's just generate a standard grid first.
    # The "fat" concept is better handled by the affine transform or specific patterns.
    return _generate_grid_points(rows, cols)

def _apply_affine_transform(points, m11, m12, m21, m22, tx, ty):
    """
    Apply an affine transformation to the point set.
    P' = M * P + T
    Must result in integer coordinates.
    """
    transformed = np.empty_like(points)
    for i in range(len(points)):
        x, y = points[i]
        new_x = m11 * x + m12 * y + tx
        new_y = m21 * x + m22 * y + ty
        transformed[i] = [new_x, new_y]
    return transformed

def _evaluate_candidate(points):
    """
    Evaluate a candidate point set.
    Returns (score, points, lines) if valid, else (0, None, None).
    """
    if not _is_valid_points(points):
        return 0, None, None
    
    # Check for integer coordinates (should be true by construction)
    if not np.all(np.equal(points, points.astype(np.int64))):
        return 0, None, None
        
    score, lines = _score_and_get_lines(points)
    return score, points, lines

def szemeredi_trotter_incidences():
    """
    Szemeredi-Trotter incidence maximization.
    Uses a deterministic structural search over geometric families.
    """
    start_time = time.time()
    timeout = 590  # Safety margin
    
    best_score = 0
    best_points = None
    best_lines = None
    
    # List of candidate generators
    # We will try various grid shapes and affine transforms.
    
    # 1. Standard Grids
    # 54 = 1*54, 2*27, 3*18, 6*9, 9*6, 18*3, 27*2, 54*1
    grid_shapes = [
        (1, 54), (2, 27), (3, 18), (6, 9), (9, 6), (18, 3), (27, 2), (54, 1)
    ]
    
    # 2. Affine Transformations
    # We'll test a set of small integer matrices with determinant +/- 1 (unimodular)
    # to preserve area and density, and also some with det > 1 to stretch.
    # Actually, stretching reduces density, so unimodular is best for grids.
    # But we can also try rotating (if possible) or shearing.
    
    # Unimodular matrices (det = 1 or -1)
    unimodular_matrices = [
        [[1, 0], [0, 1]],   # Identity
        [[1, 1], [0, 1]],   # Shear X
        [[1, 0], [1, 1]],   # Shear Y
        [[0, 1], [-1, 0]],  # Rotate 90
        [[1, -1], [0, 1]],  # Shear X neg
        [[1, 0], [-1, 1]],  # Shear Y neg
        [[-1, 0], [0, 1]],  # Flip X
        [[1, 0], [0, -1]],  # Flip Y
        [[-1, 0], [0, -1]], # Invert
        [[0, -1], [1, 0]],  # Rotate -90
        [[2, 1], [1, 1]],   # Det 1, stretch
        [[1, 2], [1, 1]],   # Det 1, stretch
        [[2, -1], [1, 0]],  # Det 1
        [[-1, 2], [0, 1]],  # Det -1
    ]
    
    # Translation ranges to keep points in a reasonable bounding box
    # and to avoid trivial symmetries
    translations = list(product(range(-10, 11), range(-10, 11)))
    
    # We will evaluate all combinations of:
    # Shape x Matrix x Translation
    # This might be too many. 8 shapes * 14 matrices * 441 translations = ~50k evaluations.
    # Each evaluation is O(N^2) ~ 1500 ops. 50k * 1500 = 75M ops.
    # Python is slow. 75M ops might take ~10-20s. This is feasible.
    
    # However, we can prune. 
    # Many translations just shift the set. The score is invariant under translation.
    # So we only need to try ONE translation per matrix/shape, e.g., (0,0).
    # Wait, does the score depend on absolute position?
    # Incidences are geometric. Translation does not change collinearity.
    # So score is invariant under translation.
    # Therefore, we only need to try (0,0) translation for each matrix/shape.
    
    # Revised plan:
    # For each shape in grid_shapes:
    #   For each matrix in unimodular_matrices:
    #     Generate base grid (centered at 0 or starting at 0)
    #     Apply matrix
    #     Evaluate
    
    # Let's center the grid to avoid large numbers? 
    # _normalize_line handles large numbers fine with gcd.
    # But let's start at 0,0 for simplicity.
    
    candidates = []
    
    for rows, cols in grid_shapes:
        base_points = _generate_grid_points(rows, cols)
        
        for m in unimodular_matrices:
            m11, m12 = m[0]
            m21, m22 = m[1]
            
            transformed = _apply_affine_transform(base_points, m11, m12, m21, m22, 0, 0)
            
            # Check validity (unimodular transform of distinct points yields distinct points)
            # But let's double check just in case of integer overflow or bugs.
            if _is_valid_points(transformed):
                score, pts, lines = _evaluate_candidate(transformed)
                if score > best_score:
                    best_score = score
                    best_points = pts
                    best_lines = lines
    
    # 3. "Fat" Grids / Clusters
    # Try a 3x2 grid of 3x3 clusters? 6*9=54.
    # Or 2x3 grid of 3x3 clusters.
    # Or 1x2 grid of 5x... no.
    
    # Let's try a 3x2 grid of points, where each "point" is actually a small cluster.
    # This is hard to generate generically. 
    # Instead, let's try a specific known good configuration: 
    # A 9x6 grid is the seed. 
    # What about a 6x9 grid? (Same as 9x6 rotated)
    
    # Let's try "broken" grids.
    # Take a 9x6 grid, and shift every other row by 1 unit.
    # This creates a hexagonal-like packing which might have different incidence properties.
    
    base_9x6 = _generate_grid_points(9, 6) # 9 rows, 6 cols
    # Shift rows 1, 3, 5, 7 by +1 in x
    shifted_9x6 = base_9x6.copy()
    for r in range(1, 9, 2):
        shifted_9x6[r*6:(r+1)*6, 0] += 1
        
    score, pts, lines = _evaluate_candidate(shifted_9x6)
    if score > best_score:
        best_score = score
        best_points = pts
        best_lines = lines
        
    # Try shifting every row by alternating amounts
    shifted_9x6_v2 = base_9x6.copy()
    for r in range(9):
        if r % 2 == 0:
            shifted_9x6_v2[r*6:(r+1)*6, 0] += 1
        else:
            shifted_9x6_v2[r*6:(r+1)*6, 0] -= 1
            
    score, pts, lines = _evaluate_candidate(shifted_9x6_v2)
    if score > best_score:
        best_score = score
        best_points = pts
        best_lines = lines

    # 4. Local Search / Refinement
    # If we have time, do a few steps of local search from the best structural candidate.
    # This combines the structural insight with fine-tuning.
    
    if best_points is not None and time.time() - start_time < timeout - 100:
        # Run a short SA from the best structural candidate
        # This is a hybrid: Structural Initialization + Local Refinement
        
        current_points = best_points.copy()
        current_score = best_score
        current_lines = best_lines
        
        n = len(current_points)
        steps = 2000 # Short run
        
        for step in range(steps):
            if time.time() - start_time > timeout - 50:
                break
                
            # Pick a random point to move
            idx = random.randint(0, n - 1)
            old_x, old_y = current_points[idx]
            
            # Propose a small move
            new_x = old_x + random.randint(-2, 2)
            new_y = old_y + random.randint(-2, 2)
            
            # Check validity
            is_dup = False
            for j in range(n):
                if j == idx:
                    continue
                if current_points[j][0] == new_x and current_points[j][1] == new_y:
                    is_dup = True
                    break
                    
            if is_dup:
                continue
                
            # Apply move
            current_points[idx] = [new_x, new_y]
            
            # Calculate new score
            new_score, new_lines = _score_and_get_lines(current_points)
            
            # Accept if better (greedy hill climbing)
            # Or use SA acceptance? Greedy is safer for limited time.
            if new_score > current_score:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_points = current_points.copy()
                    best_lines = new_lines
            else:
                # Revert
                current_points[idx] = [old_x, old_y]
                
    # Final validation and formatting
    # Ensure points are int64
    points_out = best_points.astype(np.int64)
    lines_out = best_lines.astype(np.int64)
    
    # Double check validity of returned objects
    # 1. Shapes
    assert points_out.shape == (54, 2)
    assert lines_out.shape == (27, 3)
    
    # 2. No duplicate points
    pts_set = set([tuple(p) for p in points_out])
    assert len(pts_set) == 54
    
    # 3. No invalid lines (A=B=0)
    for l in lines_out:
        assert not (l[0] == 0 and l[1] == 0)
        
    # 4. No duplicate lines
    lines_set = set([tuple(l) for l in lines_out])
    assert len(lines_set) == 27
    
    return points_out, lines_out

# EVOLVE-BLOCK-END
