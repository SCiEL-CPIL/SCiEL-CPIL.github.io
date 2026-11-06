import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds
import sys

def construct():
    start_time = time.time()
    # Safety margin: Evaluator kills at 150s. We use 140s.
    deadline = start_time + 140.0
    
    N = 256
    
    # 1. Precompute all 4-term arithmetic progressions (0-indexed)
    ap_list = []
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(0, max_a):
            ap_list.append((a, a+d, a+2*d, a+3*d))
            
    if not ap_list:
        return [0] * N
        
    aps = np.array(ap_list, dtype=np.int32)
    num_aps = len(aps)
    
    # 2. Build incidence matrix A (num_aps x N)
    # A[ap_idx, pos] = 1 if pos is in AP ap_idx
    A = np.zeros((num_aps, N), dtype=np.int8)
    for j in range(num_aps):
        a, b, c_idx, d_idx = aps[j]
        A[j, a] = 1
        A[j, b] = 1
        A[j, c_idx] = 1
        A[j, d_idx] = 1
        
    # 3. Define the MILP problem
    # Variables:
    # x_i for i in 0..N-1: binary variable for color of i (0 or 1)
    # y_j for j in 0..num_aps-1: binary variable indicating if AP j is monochromatic
    
    # Total variables: N + num_aps
    M = N + num_aps
    c_vec = np.zeros(M, dtype=np.float64)
    # Minimize sum of y_j
    c_vec[N:] = 1.0
    
    # Constraints:
    # For each AP j with indices i1, i2, i3, i4:
    # We need y_j >= 1 if x_i1 == x_i2 == x_i3 == x_i4
    # This can be modeled as:
    # y_j >= x_i1 + x_i2 + x_i3 + x_i4 - 3
    # y_j >= 4 - (x_i1 + x_i2 + x_i3 + x_i4)
    
    # Let's build the constraint matrix.
    # Row 1 for each AP: y_j - x_i1 - x_i2 - x_i3 - x_i4 >= -3
    # Row 2 for each AP: y_j + x_i1 + x_i2 + x_i3 + x_i4 >= 4
    
    # Wait, standard form for scipy.optimize.milp is:
    # A_ub @ x <= b_ub
    # A_eq @ x == b_eq
    
    # Let's use A_ub @ x <= b_ub.
    # y_j - x_i1 - x_i2 - x_i3 - x_i4 >= -3  =>  -y_j + x_i1 + x_i2 + x_i3 + x_i4 <= 3
    # y_j + x_i1 + x_i2 + x_i3 + x_i4 >= 4   =>  -y_j - x_i1 - x_i2 - x_i3 - x_i4 <= -4
    
    # Total constraints: 2 * num_aps
    num_constraints = 2 * num_aps
    A_ub = np.zeros((num_constraints, M), dtype=np.float64)
    b_ub = np.zeros(num_constraints, dtype=np.float64)
    
    for j in range(num_aps):
        i1, i2, i3, i4 = aps[j]
        y_idx = N + j
        
        # Constraint 1: -y_j + x_i1 + x_i2 + x_i3 + x_i4 <= 3
        row1 = 2 * j
        A_ub[row1, y_idx] = -1.0
        A_ub[row1, i1] = 1.0
        A_ub[row1, i2] = 1.0
        A_ub[row1, i3] = 1.0
        A_ub[row1, i4] = 1.0
        b_ub[row1] = 3.0
        
        # Constraint 2: -y_j - x_i1 - x_i2 - x_i3 - x_i4 <= -4
        row2 = 2 * j + 1
        A_ub[row2, y_idx] = -1.0
        A_ub[row2, i1] = -1.0
        A_ub[row2, i2] = -1.0
        A_ub[row2, i3] = -1.0
        A_ub[row2, i4] = -1.0
        b_ub[row2] = -4.0

    # Bounds
    lb = np.zeros(M, dtype=np.float64)
    ub = np.ones(M, dtype=np.float64)
    bounds = Bounds(lb, ub)
    
    # Integrality: all variables are integer
    integrality = np.ones(M, dtype=np.int8)
    
    # 4. Solve using scipy.optimize.milp
    # We need to be careful with the time limit.
    # scipy.optimize.milp does not have a direct time limit parameter in all versions,
    # but we can rely on the solver stopping or we can run it in a way that allows interruption?
    # Actually, scipy's milp uses HiGHS. HiGHS supports time limits.
    # However, scipy.optimize.milp options are limited.
    # Let's try running it. If it takes too long, we might need a fallback.
    # But for N=256, the problem size is ~3000 variables and ~3000 constraints.
    # HiGHS should be able to handle this reasonably well, or at least find a good solution.
    
    # To be safe, let's add a timeout mechanism if possible, or just trust it.
    # Since we can't easily interrupt scipy.optimize.milp, we will assume it finishes or we have a fallback.
    # Actually, let's check if we can pass options.
    # scipy.optimize.milp(..., options={'time_limit': 100}) might work if supported.
    
    try:
        # Try to set a time limit in options if supported
        # Note: HiGHS option is 'time_limit'
        options = {'time_limit': 120} # Leave 20s for overhead and fallback
        
        # Run MILP
        result = milp(c=c_vec, constraints=[LinearConstraint(A_ub, -np.inf, b_ub)], 
                      integrality=integrality, bounds=bounds, options=options)
        
        if result.success and result.x is not None:
            # Extract x variables (first N)
            x = result.x[:N].astype(int)
            # Ensure binary
            x = (x > 0.5).astype(int)
            
            # Verify and return
            coloring = x.tolist()
            return coloring
            
    except Exception as e:
        print(f"MILP failed: {e}")
        
    # Fallback: Local Search if MILP fails or times out without solution
    # This ensures we always return a valid coloring.
    
    # Precompute neighbor lists for local search
    index_to_others = [[] for _ in range(N)]
    for j in range(num_aps):
        a, b, c_idx, d_idx = aps[j]
        index_to_others[a].append((b, c_idx, d_idx))
        index_to_others[b].append((a, c_idx, d_idx))
        index_to_others[c_idx].append((a, b, d_idx))
        index_to_others[d_idx].append((a, b, c_idx))
        
    def calc_score(c):
        ca = c[aps[:, 0]]
        cb = c[aps[:, 1]]
        cc = c[aps[:, 2]]
        cd = c[aps[:, 3]]
        is_mono = (ca == cb) & (cb == cc) & (cc == cd)
        return np.sum(is_mono)
        
    def calc_delta(c, i):
        val_i = c[i]
        delta = 0
        for o1, o2, o3 in index_to_others[i]:
            v1 = c[o1]
            v2 = c[o2]
            v3 = c[o3]
            if v1 == v2 and v2 == v3:
                if val_i == v1:
                    delta -= 1
                else:
                    delta += 1
        return delta

    # Simple local search fallback
    rng = np.random.default_rng(42)
    best_score = np.inf
    best_coloring = None
    
    # Check time remaining
    if time.time() > deadline - 5:
        # Not enough time for local search, return random
        c = rng.integers(0, 2, size=N, dtype=np.int8)
        return c.tolist()
        
    # Run a few restarts of local search
    for restart in range(10):
        if time.time() > deadline - 2:
            break
            
        c = rng.integers(0, 2, size=N, dtype=np.int8)
        current_score = calc_score(c)
        
        if current_score < best_score:
            best_score = current_score
            best_coloring = c.copy()
            
        # Local search
        for _ in range(50): # Max sweeps
            changed = False
            for i in range(N):
                d = calc_delta(c, i)
                if d < 0:
                    c[i] = 1 - c[i]
                    current_score += d
                    changed = True
                    if current_score < best_score:
                        best_score = current_score
                        best_coloring = c.copy()
            if not changed:
                break
                
    if best_coloring is None:
        return [0] * N
        
    return best_coloring.tolist()
