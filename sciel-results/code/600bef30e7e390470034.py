"""Rewritten program for the suite task; uses SLP with barycentric coordinates."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog
import itertools

def construct():
    """
    Place 11 points in the equilateral triangle with vertices (0,0), (1,0), (0.5, sqrt(3)/2).
    Maximize the minimum area of any triangle formed by three points.
    Uses a Sequence of Linear Programs (SLP) approach.
    """
    sqrt3 = np.sqrt(3)
    
    def bary_to_cart(a, b, c):
        x = b + 0.5 * c
        y = (sqrt3 / 2) * c
        return x, y
    
    def cart_to_bary(x, y):
        c = 2 * y / sqrt3
        b = x - 0.5 * c
        a = 1 - b - c
        return a, b, c
    
    # Initial configuration: symmetric arrangement
    # Use a known good starting point - mix of vertices, edge midpoints, and interior
    # Barycentric coordinates for initial points
    initial_bary = [
        (0.9, 0.05, 0.05),   # near vertex A
        (0.05, 0.9, 0.05),   # near vertex B
        (0.05, 0.05, 0.9),   # near vertex C
        (0.5, 0.25, 0.25),   # center-ish
        (0.25, 0.5, 0.25),   # edge AB midpoint-ish
        (0.25, 0.25, 0.5),   # edge AC midpoint-ish
        (0.5, 0.05, 0.45),   # edge BC midpoint-ish
        (0.4, 0.3, 0.3),     # interior
        (0.3, 0.4, 0.3),     # interior
        (0.3, 0.3, 0.4),     # interior
        (0.35, 0.35, 0.3),   # interior
    ]
    
    points = np.array([bary_to_cart(a, b, c) for a, b, c in initial_bary])
    
    def compute_min_area(pts):
        """Compute minimum triangle area."""
        min_a = float('inf')
        for i, j, k in itertools.combinations(range(N), 3):
            pi, pj, pk = pts[i], pts[j], pts[k]
            area = 0.5 * abs((pj[0] - pi[0]) * (pk[1] - pi[1]) - (pk[0] - pi[0]) * (pj[1] - pi[1]))
            if area < min_a:
                min_a = area
        return min_a
    
    best_points = points.copy()
    best_min = compute_min_area(points)
    
    triples = list(itertools.combinations(range(N), 3))
    n_triples = len(triples)
    
    # SLP iterations
    for iteration in range(80):
        # Compute signed areas and their signs at current point
        signed_areas = np.zeros(n_triples)
        signs = np.ones(n_triples)
        for idx, (i, j, k) in enumerate(triples):
            pi, pj, pk = points[i], points[j], points[k]
            sa = 0.5 * ((pj[0] - pi[0]) * (pk[1] - pi[1]) - (pk[0] - pi[0]) * (pj[1] - pi[1]))
            signed_areas[idx] = sa
            if sa < 0:
                signs[idx] = -1
        
        # Build LP: maximize t
        # Variables: x_0,...,x_10, y_0,...,y_10, t => 23 variables
        n_vars = 2 * N + 1
        
        c = np.zeros(n_vars)
        c[-1] = -1  # minimize -t => maximize t
        
        A_ub = []
        b_ub = []
        
        # Domain constraints (using Cartesian form):
        # y_i >= 0
        for i in range(N):
            row = np.zeros(n_vars)
            row[N + i] = -1
            A_ub.append(row)
            b_ub.append(0)
        
        # y_i <= sqrt(3) * x_i
        for i in range(N):
            row = np.zeros(n_vars)
            row[i] = -sqrt3
            row[N + i] = 1
            A_ub.append(row)
            b_ub.append(0)
        
        # y_i <= sqrt(3) * (1 - x_i)
        for i in range(N):
            row = np.zeros(n_vars)
            row[i] = sqrt3
            row[N + i] = 1
            A_ub.append(row)
            b_ub.append(sqrt3)
        
        # Triangle area constraints (linearized):
        # For each triple, sgn * S_linear >= t
        # S_ijk = 0.5 * (x_j*y_k - x_j*y_i - x_i*y_k - x_k*y_j + x_k*y_i + x_i*y_j)
        # Linearize: S_lin = S_cur + sum_m (dS/dx_m * (x_m - x_m0) + dS/dy_m * (y_m - y_m0))
        # Constraint: sgn * S_lin >= t
        # => sgn * S_cur + sgn * sum(grad * delta) - t >= 0
        # => -sgn * S_cur - sgn * sum(grad * delta) + t <= 0
        
        for idx, (i, j, k) in enumerate(triples):
            sa = signed_areas[idx]
            sgn = signs[idx]
            
            row = np.zeros(n_vars)
            b_val = -sgn * sa
            
            # Gradients of S_ijk with respect to coordinates
            # dS/dx_i = 0.5 * (y_j - y_k)
            row[i] = -sgn * 0.5 * (points[j, 1] - points[k, 1])
            # dS/dx_j = 0.5 * (y_k - y_i)
            row[j] = -sgn * 0.5 * (points[k, 1] - points[i, 1])
            # dS/dx_k = 0.5 * (y_i - y_j)
            row[k] = -sgn * 0.5 * (points[i, 1] - points[j, 1])
            # dS/dy_i = 0.5 * (x_k - x_j)
            row[N + i] = -sgn * 0.5 * (points[k, 0] - points[j, 0])
            # dS/dy_j = 0.5 * (x_i - x_k)
            row[N + j] = -sgn * 0.5 * (points[i, 0] - points[k, 0])
            # dS/dy_k = 0.5 * (x_j - x_i)
            row[N + k] = -sgn * 0.5 * (points[j, 0] - points[i, 0])
            # Coefficient of t
            row[-1] = 1
            
            A_ub.append(row)
            b_ub.append(b_val)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        # Bounds
        bounds = []
        for i in range(N):
            bounds.append((0, 1))
        for i in range(N):
            bounds.append((0, sqrt3 / 2))
        bounds.append((0, 0.5))  # t upper bound (area of unit triangle is sqrt(3)/4 ~ 0.433, but min will be much less)
        
        # Solve LP
        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if result.success:
                new_points = np.column_stack([result.x[:N], result.x[N:2*N]])
                new_t = result.x[-1]
                
                # Validate: check all points are in the triangle
                valid = True
                for i in range(N):
                    x, y = new_points[i]
                    if y < -1e-9 or y > sqrt3 * x + 1e-9 or y > sqrt3 * (1 - x) + 1e-9:
                        valid = False
                        break
                
                if valid:
                    new_min = compute_min_area(new_points)
                    if new_min > best_min - 1e-12:
                        best_points = new_points.copy()
                        best_min = new_min
                        points = new_points
                    else:
                        break
                else:
                    break
            else:
                break
        except Exception:
            break
    
    return best_points

# EVOLVE-BLOCK-END
