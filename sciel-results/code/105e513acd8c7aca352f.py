# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def squares_in_square10() -> tuple:
    """
    Solve the 10-squares-in-a-square problem via MILP.
    Maximize sum of side lengths subject to non-overlap and containment.
    """
    t0 = time.time()
    deadline = t0 + 580.0
    
    n = 10
    tol = 1e-7
    
    # Pairs
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            pairs.append((i, j))
    m = len(pairs)  # 45
    nbin = 2 * m  # 90 binary vars
    nv = 3 * n + nbin  # 30 + 90 = 120
    
    def x_idx(i): return i
    def y_idx(i): return n + i
    def s_idx(i): return 2*n + i
    def a_idx(k): return 3*n + 2*k
    def b_idx(k): return 3*n + 2*k + 1
    
    # Objective: minimize -sum(s_i)
    c = np.zeros(nv)
    for i in range(n):
        c[s_idx(i)] = -1.0
    
    A = []
    b_lower = []
    b_upper = []
    
    # Boundary constraints
    for i in range(n):
        # cx_i - 0.5*s_i >= 0
        row = np.zeros(nv); row[x_idx(i)] = 1.0; row[s_idx(i)] = -0.5
        A.append(row); b_lower.append(0.0); b_upper.append(np.inf)
        # cx_i + 0.5*s_i <= 1
        row = np.zeros(nv); row[x_idx(i)] = 1.0; row[s_idx(i)] = 0.5
        A.append(row); b_lower.append(-np.inf); b_upper.append(1.0)
        # cy_i - 0.5*s_i >= 0
        row = np.zeros(nv); row[y_idx(i)] = 1.0; row[s_idx(i)] = -0.5
        A.append(row); b_lower.append(0.0); b_upper.append(np.inf)
        # cy_i + 0.5*s_i <= 1
        row = np.zeros(nv); row[y_idx(i)] = 1.0; row[s_idx(i)] = 0.5
        A.append(row); b_lower.append(-np.inf); b_upper.append(1.0)
    
    # Non-overlap constraints
    M = 2.0
    for k, (i, j) in enumerate(pairs):
        ai = a_idx(k)
        bi = b_idx(k)
        
        # a + b >= 1
        row = np.zeros(nv); row[ai] = 1.0; row[bi] = 1.0
        A.append(row); b_lower.append(1.0); b_upper.append(np.inf)
        
        # cx_j - cx_i - 0.5*(s_i+s_j) >= -M*(1-a)
        row = np.zeros(nv)
        row[x_idx(j)] = 1.0; row[x_idx(i)] = -1.0
        row[s_idx(i)] = -0.5; row[s_idx(j)] = -0.5
        row[ai] = M
        A.append(row); b_lower.append(-M); b_upper.append(np.inf)
        
        # cy_j - cy_i - 0.5*(s_i+s_j) >= -M*(1-b)
        row = np.zeros(nv)
        row[y_idx(j)] = 1.0; row[y_idx(i)] = -1.0
        row[s_idx(i)] = -0.5; row[s_idx(j)] = -0.5
        row[bi] = M
        A.append(row); b_lower.append(-M); b_upper.append(np.inf)
    
    A = np.array(A)
    b_lower = np.array(b_lower)
    b_upper = np.array(b_upper)
    
    lb = np.zeros(nv)
    ub = np.ones(nv)
    for i in range(n):
        lb[x_idx(i)] = 0.0; ub[x_idx(i)] = 1.0
        lb[y_idx(i)] = 0.0; ub[y_idx(i)] = 1.0
        lb[s_idx(i)] = 1e-9; ub[s_idx(i)] = 1.0
    for k in range(nbin):
        lb[3*n+k] = 0.0; ub[3*n+k] = 1.0
    
    integrality = np.zeros(nv)
    integrality[3*n:] = 1
    
    constraints = LinearConstraint(A, b_lower, b_upper)
    bounds = Bounds(lb, ub)
    
    result = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds,
                  options={"time_limit": 500, "mip_gap": 1e-4})
    
    if time.time() > deadline:
        k = int(np.ceil(np.sqrt(n)))
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1).astype(np.float64)
        sides = np.full(n, side, dtype=np.float64)
        return centers, sides
    
    if result.success and result.x is not None:
        x = result.x
        cx = x[0:n].copy()
        cy = x[n:2*n].copy()
        s = x[2*n:3*n].copy()
        
        s = np.clip(s, 1e-9, 1.0)
        cx = np.clip(cx, 0.0, 1.0)
        cy = np.clip(cy, 0.0, 1.0)
        
        centers = np.stack([cx, cy], axis=1)
        valid = True
        for i in range(n):
            if centers[i,0] - s[i]/2 < -tol or centers[i,0] + s[i]/2 > 1+tol:
                valid = False
            if centers[i,1] - s[i]/2 < -tol or centers[i,1] + s[i]/2 > 1+tol:
                valid = False
        for i in range(n):
            for j in range(i+1, n):
                t = (s[i] + s[j])/2 - tol
                if abs(centers[i,0]-centers[j,0]) < t and abs(centers[i,1]-centers[j,1]) < t:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            return centers.astype(np.float64), s.astype(np.float64)
    
    # Fallback
    k = int(np.ceil(np.sqrt(n)))
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    centers = np.stack([cx, cy], axis=1).astype(np.float64)
    sides = np.full(n, side, dtype=np.float64)
    return centers, sides

# EVOLVE-BLOCK-END
