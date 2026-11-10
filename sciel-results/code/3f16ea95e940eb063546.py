import numpy as np
import time
import random
from math import gcd

def _normalize_line(A, B, C):
    if A == 0 and B == 0 and C == 0:
        return (0, 0, 0)
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g > 0:
        A //= g
        B //= g
        C //= g
    if A < 0:
        A, B, C = -A, -B, -C
    elif A == 0 and B < 0:
        A, B, C = -A, -B, -C
    return (A, B, C)

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 580.0
    
    n_points = 54
    n_lines = 27
    
    # 1. Optimized Static Initialization
    # 6x9 Grid
    k = 6
    m = 9
    xs = np.arange(1, k+1)
    ys = np.arange(1, m+1)
    px, py = np.meshgrid(xs, ys, indexing='ij')
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    lines_list = []
    
    # 6 Vertical lines: x = i
    for i in range(1, k+1):
        lines_list.append((1, 0, -i))
        
    # 9 Horizontal lines: y = j
    for j in range(1, m+1):
        lines_list.append((0, 1, -j))
        
    # 4 Slope 1 lines with 6 points: y = x + c, c in {0, 1, 2, 3}
    for c in [0, 1, 2, 3]:
        lines_list.append((1, -1, c))
        
    # 4 Slope -1 lines with 6 points: x + y = c, c in {7, 8, 9, 10}
    for c in [7, 8, 9, 10]:
        lines_list.append((1, 1, -c))
        
    # 4 more lines with 5 points
    # Slope 1: c in {-1, 4}
    lines_list.append((1, -1, -1))
    lines_list.append((1, -1, 4))
    # Slope -1: c in {6, 11}
    lines_list.append((1, 1, -6))
    lines_list.append((1, 1, -11))
    
    lines = np.array(lines_list, dtype=np.int64)
    
    # Verify uniqueness
    norm_lines = [_normalize_line(int(A), int(B), int(C)) for A, B, C in lines]
    if len(set(norm_lines)) < n_lines:
        # Fallback if duplicates (should not happen)
        # But we assume the static construction is valid.
        pass

    # Compute initial incidence matrix
    M = (lines[:, 0][np.newaxis, :] * points[:, 0][:, np.newaxis] + 
         lines[:, 1][np.newaxis, :] * points[:, 1][:, np.newaxis] + 
         lines[:, 2][np.newaxis, :] == 0)
    
    I = np.sum(M)
    best_I = I
    best_points = points.copy()
    best_lines = lines.copy()
    
    # Simulated Annealing
    T = 2.0
    T_min = 0.01
    cooling_rate = 0.9999
    
    point_set = set(map(tuple, points.tolist()))
    line_set = set(norm_lines)
    
    while time.time() < deadline and T > T_min:
        # Randomly choose to move a point or a line
        if random.random() < 0.5:
            # Move a point
            i = random.randint(0, n_points - 1)
            old_px, old_py = points[i, 0], points[i, 1]
            
            # Try a few random nearby positions
            best_delta = -1000
            best_pos = None
            for _ in range(10):
                new_px = old_px + random.randint(-2, 2)
                new_py = old_py + random.randint(-2, 2)
                if (new_px, new_py) in point_set and (new_px, new_py) != (old_px, old_py):
                    continue
                # Calculate delta
                new_M_i = (lines[:, 0] * new_px + lines[:, 1] * new_py + lines[:, 2] == 0)
                delta = np.sum(new_M_i) - np.sum(M[i])
                if delta > best_delta:
                    best_delta = delta
                    best_pos = (new_px, new_py)
            
            if best_pos is None:
                continue
                
            new_px, new_py = best_pos
            if (new_px, new_py) in point_set:
                continue
                
            new_M_i = (lines[:, 0] * new_px + lines[:, 1] * new_py + lines[:, 2] == 0)
            delta_I = np.sum(new_M_i) - np.sum(M[i])
            
            if delta_I > 0 or random.random() < np.exp(delta_I / T):
                M[i] = new_M_i
                point_set.discard((old_px, old_py))
                points[i, 0] = new_px
                points[i, 1] = new_py
                point_set.add((new_px, new_py))
                I += delta_I
                
                if I > best_I:
                    best_I = I
                    best_points = points.copy()
                    best_lines = lines.copy()
        else:
            # Change a line
            j = random.randint(0, n_lines - 1)
            old_norm = norm_lines[j]
            
            best_delta = -1000
            best_line = None
            best_norm = None
            
            # Try retargeting to random pairs of points
            for _ in range(20):
                a = random.randint(0, n_points - 1)
                b = random.randint(0, n_points - 1)
                while b == a:
                    b = random.randint(0, n_points - 1)
                
                x_a, y_a = points[a]
                x_b, y_b = points[b]
                A = y_b - y_a
                B = x_a - x_b
                C = -(A * x_a + B * y_a)
                
                if A == 0 and B == 0:
                    continue
                
                norm = _normalize_line(int(A), int(B), int(C))
                
                # Check if this line is already used by another line index
                is_dup = False
                for l_idx in range(n_lines):
                    if l_idx != j and norm_lines[l_idx] == norm:
                        is_dup = True
                        break
                if is_dup:
                    continue
                
                new_M_j = (A * points[:, 0] + B * points[:, 1] + C == 0)
                delta = np.sum(new_M_j) - np.sum(M[:, j])
                if delta > best_delta:
                    best_delta = delta
                    best_line = [A, B, C]
                    best_norm = norm
            
            if best_line is None:
                continue
            
            A_new, B_new, C_new = best_line
            new_M_j = (A_new * points[:, 0] + B_new * points[:, 1] + C_new == 0)
            delta_I = np.sum(new_M_j) - np.sum(M[:, j])
            
            if delta_I > 0 or random.random() < np.exp(delta_I / T):
                M[:, j] = new_M_j
                lines[j, 0] = A_new
                lines[j, 1] = B_new
                lines[j, 2] = C_new
                norm_lines[j] = best_norm
                I += delta_I
                
                if I > best_I:
                    best_I = I
                    best_points = points.copy()
                    best_lines = lines.copy()
                    
        T *= cooling_rate
        
    return best_points, best_lines
