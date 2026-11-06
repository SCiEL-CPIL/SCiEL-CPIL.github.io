import time
import math
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # Directions for neighbors on triangular lattice
    dirs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    
    def dist_sq(p1, p2):
        da = p1[0] - p2[0]
        db = p1[1] - p2[1]
        return da*da + da*db + db*db
        
    def count_edges(points):
        count = 0
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if dist_sq(points[i], points[j]) == 1:
                    count += 1
        return count
        
    def hex_dist(a, b):
        return (abs(a) + abs(b) + abs(a+b)) // 2

    # 1. Try ILP
    try:
        # Use Radius 4 for safety
        max_rad = 4
        candidates = []
        for a in range(-max_rad, max_rad + 1):
            for b in range(-max_rad, max_rad + 1):
                if hex_dist(a, b) <= max_rad:
                    candidates.append((a, b))
        
        n_cand = len(candidates)
        point_to_idx = {p: i for i, p in enumerate(candidates)}
        
        edges = []
        for i in range(n_cand):
            for j in range(i+1, n_cand):
                if dist_sq(candidates[i], candidates[j]) == 1:
                    edges.append((i, j))
                    
        n_edges = len(edges)
        n_vars = n_cand + n_edges
        
        c = np.zeros(n_vars)
        c[n_cand:] = -1.0
        
        n_cons = 1 + 3 * n_edges
        A = np.zeros((n_cons, n_vars))
        lb = np.full(n_cons, -np.inf)
        ub = np.full(n_cons, np.inf)
        
        A[0, :n_cand] = 1.0
        lb[0] = 32
        ub[0] = 32
        
        for k, (u, v) in enumerate(edges):
            idx_y = n_cand + k
            row_base = 1 + 3 * k
            
            A[row_base, idx_y] = 1.0
            A[row_base, u] = -1.0
            ub[row_base] = 0
            
            A[row_base + 1, idx_y] = 1.0
            A[row_base + 1, v] = -1.0
            ub[row_base + 1] = 0
            
            A[row_base + 2, idx_y] = -1.0
            A[row_base + 2, u] = 1.0
            A[row_base + 2, v] = 1.0
            ub[row_base + 2] = 1
            
        integrality = np.ones(n_vars)
        bounds = Bounds(lb=0, ub=1)
        
        res = milp(c=c, constraints=LinearConstraint(A, lb, ub), integrality=integrality, bounds=bounds)
        
        if res.success:
            x = res.x[:n_cand]
            selected_idx = np.where(x > 0.5)[0]
            if len(selected_idx) == 32:
                selected_points = [candidates[i] for i in selected_idx]
                edges_count = count_edges(selected_points)
                if edges_count >= 76:
                    return [list(p) for p in selected_points]
                # If less than 76, maybe we should still return it if it's better than heuristic?
                # But heuristic might get 76.
                # We'll store it and compare later.
                ilp_result = selected_points
                ilp_score = edges_count
            else:
                ilp_result = None
                ilp_score = 0
        else:
            ilp_result = None
            ilp_score = 0
            
    except Exception:
        ilp_result = None
        ilp_score = 0

    # 2. Fallback Heuristic
    # ... (Original code) ...
    
    # Compare and return best
    if ilp_result and ilp_score > current_score:
        return [list(p) for p in ilp_result]
        
    return [list(p) for p in current_points]
