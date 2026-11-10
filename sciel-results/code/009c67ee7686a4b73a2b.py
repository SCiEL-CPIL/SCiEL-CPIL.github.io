import numpy as np
import time
from math import gcd
from itertools import combinations

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = 620.0  # safety margin before 630s kill
    
    # 7x8 grid (56 points), remove 2 to get 54
    nx, ny = 7, 8
    xs = np.arange(nx, dtype=np.int64)
    ys = np.arange(ny, dtype=np.int64)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    base_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    n_pts = len(base_points)
    
    # Generate candidate lines
    candidates = set()
    
    # Vertical lines
    for c in range(nx):
        candidates.add((1, 0, -c))
        
    # Horizontal lines
    for c in range(ny):
        candidates.add((0, 1, -c))
        
    # Diagonal lines with various slopes
    # Line: A*x + B*y + C = 0 where A=dy, B=-dx, gcd(dx,|dy|)=1
    for dx in range(1, nx + 1):
        for dy in range(-(ny - 1), ny):
            if dy == 0:
                continue
            if gcd(dx, abs(dy)) != 1:
                continue
            # C = dx*y - dy*x
            C_vals = dx * base_points[:, 1] - dy * base_points[:, 0]
            for C in np.unique(C_vals):
                candidates.add((dy, -dx, int(C)))
                
    # Also try slightly larger slopes that might still hit 2+ points
    for dx in range(nx + 1, nx + 5):
        for dy in range(-(ny + 3), ny + 4):
            if dy == 0:
                continue
            if gcd(dx, abs(dy)) != 1:
                continue
            C_vals = dx * base_points[:, 1] - dy * base_points[:, 0]
            for C in np.unique(C_vals):
                candidates.add((dy, -dx, int(C)))

    # Also try steeper slopes (swap roles: A=-dy, B=dx)
    for dy in range(1, ny + 5):
        for dx in range(-(nx + 3), nx + 4):
            if dx == 0:
                continue
            if gcd(abs(dx), dy) != 1:
                continue
            C_vals = -dy * base_points[:, 0] + dx * base_points[:, 1]
            for C in np.unique(C_vals):
                candidates.add((-dy, dx, int(C)))
                
    # Normalize lines
    def normalize_line(A, B, C):
        if A == 0 and B == 0:
            return None
        g = gcd(gcd(abs(A), abs(B)), abs(C))
        if g > 0:
            A, B, C = A // g, B // g, C // g
        if A < 0 or (A == 0 and B < 0):
            A, B, C = -A, -B, -C
        return (A, B, C)
        
    unique_lines = []
    seen = set()
    for line in candidates:
        norm = normalize_line(*line)
        if norm is not None and norm not in seen:
            seen.add(norm)
            unique_lines.append(norm)
            
    unique_lines = np.array(unique_lines, dtype=np.int64)
    n_lines = len(unique_lines)
    
    # Precompute point-line incidence matrix
    # incidence[k, m] = 1 if point k lies on line m
    A_arr = unique_lines[:, 0]
    B_arr = unique_lines[:, 1]
    C_arr = unique_lines[:, 2]
    
    # Compute all values: vals[k, m] = A[m]*x[k] + B[m]*y[k] + C[m]
    x_all = base_points[:, 0]
    y_all = base_points[:, 1]
    vals = A_arr[None, :] * x_all[:, None] + B_arr[None, :] * y_all[:, None] + C_arr[None, :]
    incidence = (vals == 0).astype(np.int8)
    
    # Total incidence per line (sum over all 56 points)
    total_inc = incidence.sum(axis=0)  # shape (n_lines,)
    
    best_score = -1
    best_points = None
    best_lines = None
    best_removed = None
    
    # Iterate over pairs of points to remove
    for i, j in combinations(range(n_pts), 2):
        if time.time() > deadline - 10:  # leave 10s for refinement
            break
            
        # Incidence per line after removing points i and j
        line_inc = total_inc - incidence[i] - incidence[j]
        
        # Select top 27 lines
        if n_lines >= 27:
            top_indices = np.argsort(-line_inc)[:27]
            score = int(line_inc[top_indices].sum())
        else:
            score = int(line_inc.sum())
            top_indices = np.arange(n_lines)
            
        if score > best_score:
            best_score = score
            mask = np.ones(n_pts, dtype=bool)
            mask[i] = False
            mask[j] = False
            best_points = base_points[mask].copy()
            best_lines = unique_lines[top_indices].copy()
            best_removed = (i, j)
            
    # Fallback
    if best_points is None:
        best_points = base_points[:54]
        best_removed = (54, 55)
        x54 = best_points[:, 0]
        y54 = best_points[:, 1]
        vals54 = A_arr[None, :] * x54[:, None] + B_arr[None, :] * y54[:, None] + C_arr[None, :]
        inc54 = (vals54 == 0).astype(np.int8)
        line_inc54 = inc54.sum(axis=0)
        top_idx = np.argsort(-line_inc54)[:27]
        best_lines = unique_lines[top_idx].copy()
        best_score = int(line_inc54[top_idx].sum())
    
    # Local refinement: try swapping one removed point with one kept point
    if time.time() < deadline - 30 and best_removed is not None:
        removed_set = set(best_removed)
        kept_indices = [k for k in range(n_pts) if k not in removed_set]
        
        current_score = best_score
        
        # For each removed point r and kept point k, try swapping
        for r in best_removed:
            if time.time() > deadline - 30:
                break
            for k in kept_indices:
                if time.time() > deadline - 30:
                    break
                # New removed set: {r replaced by k}
                # line_inc = total_inc - incidence[r'] - incidence[k']
                # where r' is the other removed point
                other = best_removed[0] if best_removed[1] == r else best_removed[1]
                line_inc = total_inc - incidence[other] - incidence[k]
                if n_lines >= 27:
                    top_idx = np.argsort(-line_inc)[:27]
                    new_score = int(line_inc[top_idx].sum())
                else:
                    new_score = int(line_inc.sum())
                    top_idx = np.arange(n_lines)
                
                if new_score > current_score:
                    current_score = new_score
                    mask = np.ones(n_pts, dtype=bool)
                    mask[other] = False
                    mask[k] = False
                    best_points = base_points[mask].copy()
                    best_lines = unique_lines[top_idx].copy()
                    best_removed = (other, k)
        
        best_score = current_score
    
    # Second refinement pass: try more 1-swaps after the first pass
    if time.time() < deadline - 15 and best_removed is not None:
        kept_indices = [k for k in range(n_pts) if k not in set(best_removed)]
        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            for r in list(best_removed):
                if time.time() > deadline - 15:
                    break
                other = best_removed[0] if best_removed[1] == r else best_removed[1]
                for k in kept_indices:
                    if time.time() > deadline - 15:
                        break
                    line_inc = total_inc - incidence[other] - incidence[k]
                    if n_lines >= 27:
                        top_idx = np.argsort(-line_inc)[:27]
                        new_score = int(line_inc[top_idx].sum())
                    else:
                        new_score = int(line_inc.sum())
                        top_idx = np.arange(n_lines)
                    
                    if new_score > best_score:
                        best_score = new_score
                        mask = np.ones(n_pts, dtype=bool)
                        mask[other] = False
                        mask[k] = False
                        best_points = base_points[mask].copy()
                        best_lines = unique_lines[top_idx].copy()
                        best_removed = (other, k)
                        kept_indices = [kk for kk in range(n_pts) if kk not in set(best_removed)]
                        improved = True
                        break
    
    # 3-point swap: try replacing one kept point with one removed point AND
    # adjusting the other removed point
    if time.time() < deadline - 10 and best_removed is not None:
        kept_indices = [k for k in range(n_pts) if k not in set(best_removed)]
        # Try: keep r1, remove k1, and also try different r2
        r1, r2 = best_removed
        for k1 in kept_indices:
            if time.time() > deadline - 10:
                break
            # Remove k1, keep r1: removed set = {r2, k1}
            line_inc = total_inc - incidence[r2] - incidence[k1]
            if n_lines >= 27:
                top_idx = np.argsort(-line_inc)[:27]
                new_score = int(line_inc[top_idx].sum())
            else:
                new_score = int(line_inc.sum())
                top_idx = np.arange(n_lines)
            if new_score > best_score:
                best_score = new_score
                mask = np.ones(n_pts, dtype=bool)
                mask[r2] = False
                mask[k1] = False
                best_points = base_points[mask].copy()
                best_lines = unique_lines[top_idx].copy()
                best_removed = (r2, k1)
    
    # Final: re-optimize line selection given the settled point set
    if time.time() < deadline - 5:
        x54 = best_points[:, 0]
        y54 = best_points[:, 1]
        vals54 = A_arr[None, :] * x54[:, None] + B_arr[None, :] * y54[:, None] + C_arr[None, :]
        inc54 = (vals54 == 0).astype(np.int8)
        line_inc54 = inc54.sum(axis=0)
        if n_lines >= 27:
            top_idx = np.argsort(-line_inc54)[:27]
            final_score = int(line_inc54[top_idx].sum())
        else:
            top_idx = np.arange(n_lines)
            final_score = int(line_inc54.sum())
        if final_score >= best_score:
            best_lines = unique_lines[top_idx].copy()
            best_score = final_score
    
    return best_points, best_lines
