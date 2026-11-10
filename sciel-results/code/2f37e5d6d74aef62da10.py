# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations


def szemeredi_trotter_incidences():
    """
    Construct a set of 54 points and 27 lines in the integer plane to maximize
    point-line incidences.
    
    Strategy: Use a 7x8 grid of points, removing 2 points to get 54 points.
    We search for the 2 points to remove and the 12 diagonal lines to keep
    that maximizes the total incidence count.
    """
    # Generate full 7x8 grid
    xs = np.arange(1, 8)
    ys = np.arange(1, 9)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    full_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # Pre-compute diagonal lines (slope 1 and -1)
    # Slope 1: y = x + b => x - y + b = 0
    b_values = list(range(-6, 8)) # y-x range
    # Slope -1: x + y = s => x + y - s = 0
    s_values = list(range(2, 16)) # x+y range
    
    # Calculate lengths of each diagonal in the full grid
    len_b = {}
    for b in b_values:
        # Points where y - x == b
        mask = (full_points[:, 1] - full_points[:, 0]) == b
        len_b[b] = np.sum(mask)
        
    len_s = {}
    for s in s_values:
        # Points where x + y == s
        mask = (full_points[:, 0] + full_points[:, 1]) == s
        len_s[s] = np.sum(mask)
        
    # We need to select 6 b's and 6 s's.
    # And 2 points to remove.
    # To be efficient, we first select the top 6 b's and top 6 s's based on full length.
    # Then we adjust for removals.
    
    top_6_b = sorted(b_values, key=lambda b: len_b[b], reverse=True)[:6]
    top_6_s = sorted(s_values, key=lambda s: len_s[s], reverse=True)[:6]
    
    best_score = -1
    best_points = None
    best_lines = None
    
    # Iterate over all pairs of points to remove
    # 56 points -> 56*55/2 = 1540 pairs. Fast.
    for i in range(len(full_points)):
        for j in range(i+1, len(full_points)):
            p1 = full_points[i]
            p2 = full_points[j]
            
            # Calculate new lengths for the top diagonals
            # A point is on a diagonal if it satisfies the equation.
            
            # Check which of the top_6_b lines contain p1 or p2
            b_losses = [0]*6
            for idx, b in enumerate(top_6_b):
                if p1[1] - p1[0] == b: b_losses[idx] += 1
                if p2[1] - p2[0] == b: b_losses[idx] += 1
                
            s_losses = [0]*6
            for idx, s in enumerate(top_6_s):
                if p1[0] + p1[1] == s: s_losses[idx] += 1
                if p2[0] + p2[1] == s: s_losses[idx] += 1
                
            new_len_b = [len_b[top_6_b[k]] - b_losses[k] for k in range(6)]
            new_len_s = [len_s[top_6_s[k]] - s_losses[k] for k in range(6)]
            
            diag_score = sum(new_len_b) + sum(new_len_s)
            
            # Calculate V and H scores
            # V: x=i. Loss 1 if p1 or p2 has x=i.
            v_scores = [0]*7
            for idx, x_val in enumerate(xs):
                loss = 0
                if p1[0] == x_val: loss += 1
                if p2[0] == x_val: loss += 1
                v_scores[idx] = 8 - loss
                
            h_scores = [0]*8
            for idx, y_val in enumerate(ys):
                loss = 0
                if p1[1] == y_val: loss += 1
                if p2[1] == y_val: loss += 1
                h_scores[idx] = 7 - loss
                
            vh_score = sum(v_scores) + sum(h_scores)
            
            total_score = vh_score + diag_score
            
            if total_score > best_score:
                best_score = total_score
                # Construct points
                mask = np.ones(len(full_points), dtype=bool)
                mask[i] = False
                mask[j] = False
                best_points = full_points[mask]
                
                # Construct lines
                lines = []
                for y_val in ys:
                    lines.append((0, 1, -y_val))
                for x_val in xs:
                    lines.append((1, 0, -x_val))
                for b in top_6_b:
                    lines.append((1, -1, b))
                for s in top_6_s:
                    lines.append((1, 1, -s))
                best_lines = np.array(lines, dtype=np.int64)
                
    return best_points, best_lines

# EVOLVE-BLOCK-END
