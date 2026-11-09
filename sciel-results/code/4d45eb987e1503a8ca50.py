# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from math import gcd

def _normalize_line(A, B, C):
    """Normalize line coefficients (A, B, C) such that gcd(|A|,|B|,|C|)=1 and first non-zero is positive."""
    if A == 0 and B == 0 and C == 0:
        return (0, 0, 0) # Invalid, but handled elsewhere
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g > 0:
        A //= g
        B //= g
        C //= g
    # Make first non-zero entry positive
    if A < 0:
        A, B, C = -A, -B, -C
    elif A == 0 and B < 0:
        A, B, C = -A, -B, -C
    return (A, B, C)

def _is_same_line(l1, l2):
    """Check if two normalized lines are the same."""
    return l1 == l2

def _count_incidences(points, lines):
    """Count incidences between points and lines."""
    # points: (N, 2), lines: (M, 3)
    # We want A*x + B*y + C == 0
    # Use broadcasting
    # points[:, 0] is x, points[:, 1] is y
    # lines[:, 0] is A, lines[:, 1] is B, lines[:, 2] is C
    # Result is (N, M) boolean matrix
    M = (lines[:, 0][np.newaxis, :] * points[:, 0][:, np.newaxis] + 
         lines[:, 1][np.newaxis, :] * points[:, 1][:, np.newaxis] + 
         lines[:, 2][np.newaxis, :] == 0)
    return M

def szemeredi_trotter_incidences():
    """
    Szemeredi-Trotter incidence maximization.
    Uses Simulated Annealing with local moves to maximize incidences.
    """
    start_time = time.time()
    deadline = start_time + 590.0 # Safety margin
    
    # Initial state: Grid construction
    k = 3
    xs = np.arange(1, k + 1)
    ys = np.arange(1, 2 * k * k + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    rows = []
    for a in range(1, k + 1):
        for b in range(1, k * k + 1):
            rows.append((a, -1, b))
    lines = np.array(rows, dtype=np.int64)
    
    # Normalize lines for duplicate checking
    norm_lines = []
    for A, B, C in lines:
        norm_lines.append(_normalize_line(A, B, C))
    
    # Initial incidence matrix
    M = _count_incidences(points, lines)
    I = np.sum(M)
    
    best_I = I
    best_points = points.copy()
    best_lines = lines.copy()
    best_norm_lines = norm_lines.copy()
    
    # Simulated Annealing parameters
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.9999
    
    n_points = 54
    n_lines = 27
    
    while time.time() < deadline:
        # Generate a move
        if random.random() < 0.5:
            # Move a point
            i = random.randint(0, n_points - 1)
            dx = random.randint(-3, 3)
            dy = random.randint(-3, 3)
            if dx == 0 and dy == 0:
                continue
            new_px = points[i, 0] + dx
            new_py = points[i, 1] + dy
            
            # Check for duplicate point
            # We can check against all other points
            # To speed up, we could use a set of tuples, but 54 is small
            is_dup = False
            for j in range(n_points):
                if j == i:
                    continue
                if points[j, 0] == new_px and points[j, 1] == new_py:
                    is_dup = True
                    break
            if is_dup:
                continue
                
            # Calculate new incidences for point i
            # new_M_i[j] = 1 if new point is on line j
            # lines[:, 0] * new_px + lines[:, 1] * new_py + lines[:, 2] == 0
            new_M_i = (lines[:, 0] * new_px + lines[:, 1] * new_py + lines[:, 2] == 0)
            
            delta_I = np.sum(new_M_i) - np.sum(M[i])
            
            # Acceptance criterion
            if delta_I > 0 or random.random() < np.exp(delta_I / T):
                # Update state
                M[i] = new_M_i
                points[i, 0] = new_px
                points[i, 1] = new_py
                I += delta_I
                
                if I > best_I:
                    best_I = I
                    best_points = points.copy()
                    best_lines = lines.copy()
                    best_norm_lines = norm_lines.copy()
                    
        else:
            # Move a line: Retarget to pass through two random points
            j = random.randint(0, n_lines - 1)
            # Pick two distinct points
            a = random.randint(0, n_points - 1)
            b = random.randint(0, n_points - 1)
            while b == a:
                b = random.randint(0, n_points - 1)
                
            x_a, y_a = points[a]
            x_b, y_b = points[b]
            
            # Line through (x_a, y_a) and (x_b, y_b)
            # A(x-x_a) + B(y-y_a) = 0
            # A = y_b - y_a, B = x_a - x_b, C = -(A*x_a + B*y_a)
            A = y_b - y_a
            B = x_a - x_b
            C = -(A * x_a + B * y_a)
            
            # Normalize
            new_norm = _normalize_line(A, B, C)
            
            # Check for duplicate line
            is_dup = False
            for l_idx in range(n_lines):
                if l_idx == j:
                    continue
                if best_norm_lines[l_idx] == new_norm: # Using best is slightly wrong, should use current norm_lines
                    is_dup = True
                    break
            # Corrected: check against current norm_lines
            is_dup = False
            for l_idx in range(n_lines):
                if l_idx == j:
                    continue
                if norm_lines[l_idx] == new_norm:
                    is_dup = True
                    break
            if is_dup:
                continue
            
            # Calculate new incidences for line j
            # new_M_j[i] = 1 if point i is on new line
            # new_norm is (A', B', C'). Use it for calculation? 
            # Using normalized A,B,C is fine for == 0 check.
            A_new, B_new, C_new = new_norm
            new_M_j = (A_new * points[:, 0] + B_new * points[:, 1] + C_new == 0)
            
            delta_I = np.sum(new_M_j) - np.sum(M[:, j])
            
            if delta_I > 0 or random.random() < np.exp(delta_I / T):
                # Update state
                M[:, j] = new_M_j
                lines[j, 0] = A_new
                lines[j, 1] = B_new
                lines[j, 2] = C_new
                norm_lines[j] = new_norm
                I += delta_I
                
                if I > best_I:
                    best_I = I
                    best_points = points.copy()
                    best_lines = lines.copy()
                    best_norm_lines = norm_lines.copy()
                    
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min

    return best_points, best_lines

# EVOLVE-BLOCK-END
