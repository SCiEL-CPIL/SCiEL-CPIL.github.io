import numpy as np
from scipy.optimize import minimize, linprog
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_score = -np.inf
    best_circles = None
    
    # Function to validate a configuration
    def validate(circles, w, h, tol=1e-7):
        if w <= 0 or h <= 0:
            return False
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        if np.any(r < -tol):
            return False
            
        # Boundary check
        if np.any(x < r - tol) or np.any(x > w - r + tol) or np.any(y < r - tol) or np.any(y > h - r + tol):
            return False
            
        # Overlap check
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                if dist < r[i] + r[j] - tol:
                    return False
        return True

    # Successive Convex Relaxation (SCR) Solver
    def solve_scr(init_circles, w_init, h_init, max_iters=50, tol=1e-8):
        # init_circles: (N, 3) array of [x, y, r]
        # w_init, h_init: initial dimensions
        # Note: w + h = 2. We optimize w, h is derived.
        # However, SCR linearizes around current positions. 
        # It's often better to fix aspect ratio or optimize it separately.
        # Let's fix w and h for the SCR phase to ensure stability, 
        # then maybe do a final adjustment? 
        # Actually, the task allows variable rectangle. 
        # Let's optimize w as well in the LP? 
        # The constraint x <= w - r becomes x + r <= w. This is linear.
        # So we can include w in the LP variables.
        
        # Variables: x (N), y (N), r (N), w (1). Total 3N + 1 = 64 vars.
        # Objective: Maximize sum(r) -> Minimize -sum(r)
        
        x = init_circles[:, 0].copy()
        y = init_circles[:, 1].copy()
        r = init_circles[:, 2].copy()
        w = w_init
        h = 2.0 - w
        
        # Pre-calculate indices for pairs
        i_idx, j_idx = np.triu_indices(N, k=1)
        
        for it in range(max_iters):
            if time.time() > deadline - 2:
                break
                
            # 1. Compute linearization vectors u_ij
            # u_ij = (x_i - x_j, y_i - y_j) / dist
            # If dist is 0, pick random direction
            u_x = np.zeros(len(i_idx))
            u_y = np.zeros(len(i_idx))
            for k in range(len(i_idx)):
                i, j = i_idx[k], j_idx[k]
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 1e-9:
                    angle = np.random.uniform(0, 2*np.pi)
                    u_x[k] = np.cos(angle)
                    u_y[k] = np.sin(angle)
                else:
                    u_x[k] = dx / dist
                    u_y[k] = dy / dist
            
            # 2. Formulate LP
            # Minimize c^T v
            # v = [x_0...x_N-1, y_0...y_N-1, r_0...r_N-1, w]
            c = np.zeros(3*N + 1)
            c[2*N : 3*N] = -1.0  # Maximize sum(r)
            
            # Constraints: A_ub v <= b_ub
            # A_eq v == b_eq
            
            # Boundary Constraints:
            # x_i >= r_i  =>  x_i - r_i >= 0  =>  -x_i + r_i <= 0
            # x_i <= w - r_i => x_i + r_i <= w => x_i + r_i - w <= 0
            # y_i >= r_i  =>  -y_i + r_i <= 0
            # y_i <= h - r_i => y_i + r_i <= h => y_i + r_i - (2-w) <= 0 => y_i + r_i + w <= 2
            
            # Non-overlap Constraints (Linearized):
            # u_x[i,j] * (x_i - x_j) + u_y[i,j] * (y_i - y_j) >= r_i + r_j
            # -u_x[i,j] x_i + u_x[i,j] x_j - u_y[i,j] y_i + u_y[i,j] y_j + r_i + r_j <= 0
            
            A_ub = []
            b_ub = []
            
            # Boundary: -x_i + r_i <= 0
            for i in range(N):
                row = np.zeros(3*N + 1)
                row[i] = -1.0
                row[2*N + i] = 1.0
                A_ub.append(row)
                b_ub.append(0.0)
                
                row = np.zeros(3*N + 1)
                row[N + i] = -1.0
                row[2*N + i] = 1.0
                A_ub.append(row)
                b_ub.append(0.0)
                
            # Boundary: x_i + r_i - w <= 0
            for i in range(N):
                row = np.zeros(3*N + 1)
                row[i] = 1.0
                row[2*N + i] = 1.0
                row[3*N] = -1.0
                A_ub.append(row)
                b_ub.append(0.0)
                
            # Boundary: y_i + r_i + w <= 2
            for i in range(N):
                row = np.zeros(3*N + 1)
                row[N + i] = 1.0
                row[2*N + i] = 1.0
                row[3*N] = 1.0
                A_ub.append(row)
                b_ub.append(2.0)
                
            # Non-overlap: -u_x x_i + u_x x_j - u_y y_i + u_y y_j + r_i + r_j <= 0
            for k in range(len(i_idx)):
                i, j = i_idx[k], j_idx[k]
                row = np.zeros(3*N + 1)
                row[i] = -u_x[k]
                row[j] = u_x[k]
                row[N + i] = -u_y[k]
                row[N + j] = u_y[k]
                row[2*N + i] = 1.0
                row[2*N + j] = 1.0
                A_ub.append(row)
                b_ub.append(0.0)
                
            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)
            
            # Bounds: r_i >= 0, w in [0.5, 1.5] (to keep aspect ratio reasonable)
            # x, y are bounded by constraints, but we can add loose bounds to help solver
            # Actually, x and y are effectively bounded by r and w/h.
            # Let's just bound r and w.
            bounds = []
            for i in range(N):
                bounds.append((None, None)) # x
            for i in range(N):
                bounds.append((None, None)) # y
            for i in range(N):
                bounds.append((0, None))   # r
            bounds.append((0.5, 1.5))      # w
            
            # Solve LP
            try:
                res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
                if not res.success:
                    break
                    
                v = res.x
                x_new = v[0:N]
                y_new = v[N:2*N]
                r_new = v[2*N:3*N]
                w_new = v[3*N]
                h_new = 2.0 - w_new
                
                # Check convergence
                if np.max(np.abs(x_new - x)) < tol and np.max(np.abs(r_new - r)) < tol:
                    x, y, r, w, h = x_new, y_new, r_new, w_new, h_new
                    break
                    
                x, y, r, w, h = x_new, y_new, r_new, w_new, h_new
                
            except Exception:
                break
                
        return np.column_stack((x, y, r)), w, h

    # Initial Guesses
    guesses = []
    
    # 1. Square Grid
    side = 5
    # 5x5 grid = 25 circles, we take 21.
    # Better: 4x5 grid + 1? Or just 5x5 and remove 4?
    # Let's try a few grid layouts.
    
    def make_grid(cols, rows):
        # Returns list of (x, y, r) for N circles
        # We need to fit N circles.
        # If cols*rows >= N, we can place them.
        # We will optimize positions, so initial placement just needs to be distinct.
        pts = []
        for i in range(cols):
            for j in range(rows):
                if len(pts) >= N:
                    break
                x = (i + 0.5) / cols
                y = (j + 0.5) / rows
                # Initial radius estimate
                r = 0.5 / max(cols, rows) * 0.9
                pts.append((x, y, r))
        return pts

    guesses.append(make_grid(5, 5))
    guesses.append(make_grid(4, 6))
    guesses.append(make_grid(6, 4))
    guesses.append(make_grid(3, 7))
    
    # 2. Random
    np.random.seed(42)
    for _ in range(3):
        x = np.random.rand(N)
        y = np.random.rand(N)
        r = np.full(N, 0.05)
        guesses.append(list(zip(x, y, r)))

    for guess in guesses:
        if time.time() > deadline - 5:
            break
            
        init_circles = np.array(guess)
        # Initial w, h
        w_init = 1.0
        h_init = 1.0
        
        # Run SCR
        try:
            circles, w, h = solve_scr(init_circles, w_init, h_init, max_iters=30)
            
            # Validate
            if validate(circles, w, h):
                score = np.sum(circles[:, 2])
                if score > best_score:
                    best_score = score
                    best_circles = circles
        except Exception:
            continue

    if best_circles is None:
        # Fallback: Simple grid
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side * 0.95
                cells.append([x, y, r])
        best_circles = np.array(cells)
        
    return best_circles
