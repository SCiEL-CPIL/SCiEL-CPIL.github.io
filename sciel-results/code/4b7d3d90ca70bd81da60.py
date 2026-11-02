"""Heilbronn 11-point problem: maximize minimum triangle area in unit equilateral triangle.
Strategy: Linear Programming (LP) formulation.
The area of a triangle formed by three points in barycentric coordinates is a linear function
of the coordinates (specifically, a determinant which is multilinear, but for fixed signs 
or using absolute value constraints, it can be handled). 
Actually, the area is 0.5 * |det|. To maximize the minimum area, we can set up an LP:
Maximize t
Subject to:
  For all triples (i, j, k):
    det(P_i, P_j, P_k) >= 2*t  OR  det(P_i, P_j, P_k) <= -2*t
This is not a standard LP because of the OR. However, we can use the fact that we can 
orient the points. Or, simpler: we can just maximize the minimum of the absolute values.
But scipy's linprog requires linear constraints.
Wait, the area is linear in each coordinate, but the absolute value makes it non-linear.
However, we can split the problem into two LPs? No, that's not exhaustive.
Let's look at the structure. The area of triangle (i,j,k) is 0.5 * | (x_j-x_i)(y_k-y_i) - (y_j-y_i)(x_k-x_i) |.
In barycentric coordinates (u, v) where x = u + 0.5v, y = S3H*v:
The determinant is linear in the coordinates of each point.
Let D(i,j,k) be the determinant. It is a linear function of u_i, v_i, u_j, v_j, u_k, v_k.
We want max t such that |D(i,j,k)| >= 2*t for all i,j,k.
This is equivalent to:
D(i,j,k) >= 2*t  OR  D(i,j,k) <= -2*t
This is a disjunctive constraint.
However, for small N, we might be able to guess the signs?
Alternatively, we can use a different approach:
We can formulate this as a semi-definite program or use a solver that handles this.
But we only have standard scipy.
Let's try a different formulation.
Maximize t
Subject to:
  D(i,j,k)^2 >= (2t)^2
This is non-convex.

Let's reconsider the "SOCP" hint in the directive.
Actually, there is a simpler way. The area is proportional to the absolute value of a linear form.
We can introduce a variable for each triangle's "signed area" s_ijk.
s_ijk = D(i,j,k)
We want to maximize t such that s_ijk >= 2t OR s_ijk <= -2t.
This is hard for a single LP.

However, we can observe that for the optimal configuration, the points will be in a "general position" and the signs of the determinants will be consistent with some ordering? No.

Let's try a different structural change: **Simulated Annealing** with a very specific cooling schedule and a move set that specifically targets the smallest triangles, but using a much more robust global search mechanism than the current nudge. Or, even better, **Particle Swarm Optimization (PSO)** which is often better than DE for this kind of multimodal problem.

Wait, the directive suggests SOCP/LP.
Let's look at the LP approach again.
If we fix the signs of the determinants, it becomes an LP.
There are 165 triangles. $2^{165}$ sign combinations is too many.
But maybe we can solve it as a sequence of LPs?
Or, we can use `scipy.optimize.linprog` to maximize the minimum of a set of linear functions, IF we can express the area as a linear function without absolute value.
But area is always positive.
The determinant can be positive or negative.
We can define two sets of constraints:
1. Assume all determinants are positive. Maximize t subject to D_ijk >= 2t.
2. Assume all determinants are negative. Maximize t subject to -D_ijk >= 2t.
These are convex LPs.
The global optimum must satisfy one of these? No, the signs can be mixed.
However, if we start from a good initial guess (e.g. from the current program or a random start), we can determine the signs of the determinants for the current configuration. Then we solve the LP that enforces those signs. This is a local linearization.
We can iterate:
1. Start with a configuration.
2. Determine signs of D_ijk.
3. Solve LP: Maximize t subject to sign_ijk * D_ijk >= 2t, and points in triangle.
4. Update configuration.
5. Repeat.

This is a "Sign-Linearization" strategy. It is structurally different from gradient descent on min-area. It uses the convexity of the feasible region defined by fixed signs.

Let's implement this.
"""
import numpy as np
import time
from scipy.optimize import linprog, minimize
from itertools import combinations

N = 11
S3H = np.sqrt(3) / 2
TRI_AREA = 0.5 * 1.0 * S3H

# Precompute all (i,j,k) triples
_TRIPLES = list(combinations(range(N), 3))
_N_TRIPLES = len(_TRIPLES)

# Coefficients for the determinant in terms of barycentric coordinates (u, v)
# Point p = (u + 0.5v, S3H*v)
# D(p_i, p_j, p_k) = 0.5 * | (x_j-x_i)(y_k-y_i) - (y_j-y_i)(x_k-x_i) |
# Let's derive the linear coefficients for D (without 0.5 and abs)
# x = u + 0.5v, y = S3H*v
# x_j - x_i = (u_j - u_i) + 0.5(v_j - v_i)
# y_k - y_i = S3H(v_k - v_i)
# (x_j-x_i)(y_k-y_i) = S3H [ (u_j-u_i)(v_k-v_i) + 0.5(v_j-v_i)(v_k-v_i) ]
# Similarly for the other term.
# The determinant is bilinear in u,v. It is NOT linear in the variables if we consider u,v as variables.
# Wait. The determinant of 3 points in 2D is linear in the coordinates of each point individually, 
# but it is a multilinear function of all 6 coordinates.
# D(u_i, v_i, u_j, v_j, u_k, v_k) is linear in u_i, v_i, etc.
# Specifically, D = c_u_i u_i + c_v_i v_i + ...
# Yes, it is linear in each coordinate. So it is a linear function of the vector [u_i, v_i, u_j, v_j, u_k, v_k].
# So we CAN use LP!

# Let's compute the coefficients for each triple.
# D = (x_j-x_i)(y_k-y_i) - (y_j-y_i)(x_k-x_i)
# Substitute x = u + 0.5v, y = S3H*v
# D = [ (u_j-u_i) + 0.5(v_j-v_i) ] [ S3H(v_k-v_i) ] - [ S3H(v_j-v_i) ] [ (u_k-u_i) + 0.5(v_k-v_i) ]
# D = S3H [ (u_j-u_i)(v_k-v_i) + 0.5(v_j-v_i)(v_k-v_i) - (v_j-v_i)(u_k-u_i) - 0.5(v_j-v_i)(v_k-v_i) ]
# The 0.5 terms cancel!
# D = S3H [ (u_j-u_i)(v_k-v_i) - (v_j-v_i)(u_k-u_i) ]
# Let's expand this to get linear coefficients.
# Term 1: u_j v_k - u_j v_i - u_i v_k + u_i v_i
# Term 2: - (v_j u_k - v_j u_i - v_i u_k + v_i u_i)
# D = S3H [ u_j v_k - u_j v_i - u_i v_k + u_i v_i - v_j u_k + v_j u_i + v_i u_k - v_i u_i ]
# D = S3H [ u_j v_k - u_j v_i - u_i v_k + v_j u_i - v_j u_k + v_i u_k ]
# The u_i v_i and v_i u_i terms cancel.
# This is still a product of variables! It is NOT linear in the variables u_i, v_i, etc.
# My previous assumption was wrong. The area is NOT a linear function of the coordinates in the sense of a single linear form c^T x.
# It is a quadratic form (specifically, a determinant).
# So LP is not directly applicable to the raw coordinates.

# However, we can use a different variable transformation or a different solver.
# But the directive says "Formulate ... as SOCP ... leveraging ... linear function".
# Maybe I am missing something.
# "the area of a triangle formed by points in barycentric coordinates is a linear function of the coordinates"
# Is it?
# If the triangle vertices are fixed, the area of a sub-triangle formed by points on the edges is linear? No.
# The area of a triangle with vertices P_i, P_j, P_k is 0.5 |det(P_j-P_i, P_k-P_i)|.
# This is a polynomial of degree 2 in the coordinates.
# So the premise in the directive might be slightly loose, or refers to a different formulation.
# However, we can use **Second-Order Cone Programming (SOCP)** concepts via `scipy.optimize`?
# Scipy does not have a native SOCP solver.
# But we can use `cvxpy` if available? No, only standard lib, numpy, scipy.

# Let's pivot to a different structural change: **Evolutionary Strategy with Covariance Matrix Adaptation (CMA-ES)**?
# SciPy doesn't have CMA-ES.
# How about **Basin Hopping** with a robust local optimizer?
# The current program uses DE + L-BFGS-B + Nudge.
# Basin Hopping is different: it performs global random walks with local minimization at each step.
# This is a different search paradigm.

# Let's implement Basin Hopping.
# 1. Start with a random point.
# 2. Local minimize (using L-BFGS-B on -min_area^2 or similar).
# 3. Random kick.
# 4. Local minimize again.
# 5. Accept/reject based on energy (score).

# To make it structurally different, we will also use a **multi-start** strategy with diverse initializations (grid-based, random, hexagonal) and a **simulated annealing** acceptance criterion for the kicks.

def _min_area(points):
    """Compute minimum triangle area among all triples. points: (N,2) array."""
    p_i = points[_I_IDX] if '_I_IDX' in globals() else points[np.array([t[0] for t in _TRIPLES])]
    p_j = points[_J_IDX] if '_J_IDX' in globals() else points[np.array([t[1] for t in _TRIPLES])]
    p_k = points[_K_IDX] if '_K_IDX' in globals() else points[np.array([t[2] for t in _TRIPLES])]
    dx1 = p_j[:, 0] - p_i[:, 0]
    dy1 = p_j[:, 1] - p_i[:, 1]
    dx2 = p_k[:, 0] - p_i[:, 0]
    dy2 = p_k[:, 1] - p_i[:, 1]
    areas = 0.5 * np.abs(dx1 * dy2 - dy1 * dx2)
    return np.min(areas)

# Precompute indices for speed
_I_IDX = np.array([t[0] for t in _TRIPLES])
_J_IDX = np.array([t[1] for t in _TRIPLES])
_K_IDX = np.array([t[2] for t in _TRIPLES])

def _project_to_triangle(pts):
    """Project points to the unit equilateral triangle."""
    pts = pts.copy()
    for i in range(len(pts)):
        x, y = pts[i]
        # Barycentric coords
        v = y / S3H
        u = x - 0.5 * v
        # Clamp
        if u < 0:
            if v < 0:
                # Corner (0,0)
                u, v = 0, 0
            elif u + v > 1:
                # Edge x=0? No, u<0, v>0.
                # Project to u=0
                u = 0
                if v > 1: v = 1
            else:
                u = 0
        elif v < 0:
            if u < 0:
                u, v = 0, 0
            elif u + v > 1:
                # Edge y=0? No, v<0.
                v = 0
                if u > 1: u = 1
            else:
                v = 0
        else:
            if u + v > 1:
                s = u + v
                u /= s
                v /= s
        # Reconstruct
        pts[i, 0] = u + 0.5 * v
        pts[i, 1] = S3H * v
    return pts

def _objective(x):
    """Objective for local minimization: negative min area."""
    u = x[0::2]
    v = x[1::2]
    pts = np.empty((N, 2))
    pts[:, 0] = u + 0.5 * v
    pts[:, 1] = S3H * v
    pts = _project_to_triangle(pts)
    # Re-extract u,v from projected pts to ensure consistency?
    # Actually, L-BFGS-B works on x. If we project inside the objective, the gradient is wrong.
    # Better to use bounds in L-BFGS-B.
    # But the triangle is not a box.
    # We can use a penalty method or just rely on bounds for u,v in [0,1] and u+v<=1.
    # L-BFGS-B doesn't support linear inequality constraints directly.
    # We can use a penalty for u+v>1.
    pen = 1000.0 * np.sum(np.maximum(0, u + v - 1.0)**2)
    areas = _min_area(pts)
    return -areas + pen

def _local_optimize(x0, time_start, time_limit):
    """Run L-BFGS-B from x0."""
    if time.time() > time_limit:
        return x0, _min_area(_project_to_triangle(np.column_stack([x0[0::2] + 0.5*x0[1::2], S3H*x0[1::2]])))
    
    bounds = [(0.0, 1.0)] * (2 * N)
    try:
        res = minimize(
            _objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12}
        )
        return res.x, res.fun
    except Exception:
        return x0, _objective(x0)

def construct():
    start_time = time.time()
    TIME_LIMIT = 140  # safety margin
    
    best_points = None
    best_score = -float('inf')
    
    # Initial candidates
    candidates = []
    
    # 1. Random starts
    rng = np.random.RandomState(42)
    for _ in range(10):
        pts = rng.random((N, 2)) * np.array([1.0, S3H])
        pts = _project_to_triangle(pts)
        candidates.append(pts)
        
    # 2. Hexagonal grid approximation
    # 11 points: 1 center, 6 ring, 4 more?
    # Let's just use a few structured starts.
    
    # Basin Hopping
    # We will run basin hopping from each candidate.
    # To save time, we will limit the number of steps.
    
    T = 0.1  # Initial temperature
    T_min = 1e-6
    alpha = 0.9  # Cooling rate
    max_kicks = 50
    
    for cand_pts in candidates:
        if time.time() > TIME_LIMIT:
            break
            
        # Convert to barycentric x
        u_init = np.zeros(N)
        v_init = np.zeros(N)
        for i in range(N):
            px, py = cand_pts[i]
            v_init[i] = py / S3H
            u_init[i] = px - 0.5 * v_init[i]
        x_current = np.column_stack([u_init, v_init]).flatten()
        
        # Initial local opt
        x_current, _ = _local_optimize(x_current, start_time, TIME_LIMIT)
        pts_current = _project_to_triangle(np.column_stack([x_current[0::2] + 0.5*x_current[1::2], S3H*x_current[1::2]]))
        score_current = _min_area(pts_current)
        
        if score_current > best_score:
            best_score = score_current
            best_points = pts_current.copy()
            
        # Basin Hopping loop
        for kick_iter in range(max_kicks):
            if time.time() > TIME_LIMIT:
                break
                
            # Random kick
            step_size = 0.05
            kick = rng.normal(0, step_size, (N, 2))
            pts_kicked = pts_current + kick
            pts_kicked = _project_to_triangle(pts_kicked)
            
            # Convert to barycentric
            u_kick = np.zeros(N)
            v_kick = np.zeros(N)
            for i in range(N):
                px, py = pts_kicked[i]
                v_kick[i] = py / S3H
                u_kick[i] = px - 0.5 * v_kick[i]
            x_kicked = np.column_stack([u_kick, v_kick]).flatten()
            
            # Local opt
            x_new, _ = _local_optimize(x_kicked, start_time, TIME_LIMIT)
            pts_new = _project_to_triangle(np.column_stack([x_new[0::2] + 0.5*x_new[1::2], S3H*x_new[1::2]]))
            score_new = _min_area(pts_new)
            
            # Acceptance criterion (Metropolis)
            delta = score_new - score_current
            if delta > 0:
                pts_current = pts_new
                score_current = score_new
                x_current = x_new
                if score_current > best_score:
                    best_score = score_current
                    best_points = pts_new.copy()
            else:
                if rng.random() < np.exp(delta / T):
                    pts_current = pts_new
                    score_current = score_new
                    x_current = x_new
                    
            T *= alpha
            if T < T_min:
                T = T_min
                
    if best_points is None:
        # Fallback
        best_points = np.zeros((N, 2))
        
    return best_points

