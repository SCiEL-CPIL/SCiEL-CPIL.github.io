# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds


def squares_in_square10() -> tuple:
    """
    Solve the "10 squares in a unit square" packing problem via MILP.
    
    Variables per square i: x_i, y_i (center), s_i (side length).
    For each pair (i,j), binary variables for relative position:
      a_ij: i left of j, b_ij: j left of i, c_ij: i below j, d_ij: j below i
    
    Objective: maximize sum(s_i)
    """
    n = 10
    # Variable ordering: x_0..x_9, y_0..y_9, s_0..s_9, then binaries
    # Total continuous: 3*n = 30
    # For each pair (i,j) with i<j: 4 binary vars
    npairs = n * (n - 1) // 2
    nbin = 4 * npairs
    nvars = 3 * n + nbin
    
    # Objective: maximize sum(s_i) -> minimize -sum(s_i)
    c = np.zeros(nvars)
    for i in range(n):
        c[2 * n + i] = -1.0  # s_i is at index 2n + i
    
    # Bounds
    lb = np.zeros(nvars)
    ub = np.ones(nvars)
    # x, y in [0, 1]
    # s in (0, 1], use small positive lower bound
    lb[2*n:3*n] = 1e-9
    ub[2*n:3*n] = 1.0
    # Binary vars
    lb[3*n:] = 0.0
    ub[3*n:] = 1.0
    
    # Integer mask: last nbin variables are integer (binary)
    integrality = np.zeros(nvars, dtype=int)
    integrality[3*n:] = 1
    
    # Constraints
    A_ub_list = []
    b_ub_list = []
    A_eq_list = []
    b_eq_list = []
    
    M = 2.0  # Big-M constant (max possible separation is 1)
    
    def add_ineq(A_row, b_val):
        A_ub_list.append(A_row)
        b_ub_list.append(b_val)
    
    # Square i inside [0,1]^2:
    # x_i - s_i/2 >= 0  ->  -x_i + s_i/2 <= 0
    # x_i + s_i/2 <= 1  ->   x_i - s_i/2 <= 1
    # y_i - s_i/2 >= 0  ->  -y_i + s_i/2 <= 0
    # y_i + s_i/2 <= 1  ->   y_i - s_i/2 <= 1
    for i in range(n):
        # -x_i + s_i/2 <= 0
        row = np.zeros(nvars)
        row[i] = -1.0
        row[2*n + i] = 0.5
        add_ineq(row, 0.0)
        
        # x_i - s_i/2 <= 1
        row = np.zeros(nvars)
        row[i] = 1.0
        row[2*n + i] = -0.5
        add_ineq(row, 1.0)
        
        # -y_i + s_i/2 <= 0
        row = np.zeros(nvars)
        row[n + i] = -1.0
        row[2*n + i] = 0.5
        add_ineq(row, 0.0)
        
        # y_i - s_i/2 <= 1
        row = np.zeros(nvars)
        row[n + i] = 1.0
        row[2*n + i] = -0.5
        add_ineq(row, 1.0)
    
    # Pair constraints
    pair_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            # Binary variable indices for pair (i,j)
            a_idx = 3 * n + pair_idx      # i left of j
            b_idx = 3 * n + pair_idx + 1  # j left of i
            c_idx = 3 * n + pair_idx + 2  # i below j
            d_idx = 3 * n + pair_idx + 3  # j below i
            
            # a_ij + b_ij + c_ij + d_ij >= 1
            row = np.zeros(nvars)
            row[a_idx] = 1.0
            row[b_idx] = 1.0
            row[c_idx] = 1.0
            row[d_idx] = 1.0
            # >= 1 -> -row <= -1
            add_ineq(-row, -1.0)
            
            # a_ij + b_ij <= 1
            row = np.zeros(nvars)
            row[a_idx] = 1.0
            row[b_idx] = 1.0
            add_ineq(row, 1.0)
            
            # c_ij + d_ij <= 1
            row = np.zeros(nvars)
            row[c_idx] = 1.0
            row[d_idx] = 1.0
            add_ineq(row, 1.0)
            
            # Separation constraints with big-M:
            # If a_ij=1: x_i + s_i/2 <= x_j - s_j/2
            #   x_i - x_j + s_i/2 + s_j/2 <= M(1 - a_ij)
            row = np.zeros(nvars)
            row[i] = 1.0
            row[j] = -1.0
            row[2*n + i] = 0.5
            row[2*n + j] = 0.5
            row[a_idx] = -M
            add_ineq(row, 0.0)
            
            # If b_ij=1: x_j + s_j/2 <= x_i - s_i/2
            #   x_j - x_i + s_j/2 + s_i/2 <= M(1 - b_ij)
            row = np.zeros(nvars)
            row[j] = 1.0
            row[i] = -1.0
            row[2*n + j] = 0.5
            row[2*n + i] = 0.5
            row[b_idx] = -M
            add_ineq(row, 0.0)
            
            # If c_ij=1: y_i + s_i/2 <= y_j - s_j/2
            #   y_i - y_j + s_i/2 + s_j/2 <= M(1 - c_ij)
            row = np.zeros(nvars)
            row[n + i] = 1.0
            row[n + j] = -1.0
            row[2*n + i] = 0.5
            row[2*n + j] = 0.5
            row[c_idx] = -M
            add_ineq(row, 0.0)
            
            # If d_ij=1: y_j + s_j/2 <= y_i - s_i/2
            #   y_j - y_i + s_j/2 + s_i/2 <= M(1 - d_ij)
            row = np.zeros(nvars)
            row[n + j] = 1.0
            row[n + i] = -1.0
            row[2*n + j] = 0.5
            row[2*n + i] = 0.5
            row[d_idx] = -M
            add_ineq(row, 0.0)
            
            pair_idx += 1
    
    # Build constraint matrix
    A_ub = np.array(A_ub_list)
    b_ub = np.array(b_ub_list)
    
    constraints = [LinearConstraint(A_ub, -np.inf, b_ub)]
    bounds = Bounds(lb, ub)
    
    result = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
    
    if not result.success:
        # Fallback to grid packing
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
    
    x = result.x
    
    centers = np.zeros((n, 2), dtype=np.float64)
    sides = np.zeros(n, dtype=np.float64)
    for i in range(n):
        centers[i, 0] = x[i]
        centers[i, 1] = x[n + i]
        sides[i] = x[2*n + i]
    
    # Ensure sides are strictly positive and finite
    sides = np.maximum(sides, 1e-9)
    centers = np.where(np.isfinite(centers), centers, 0.5)
    
    return centers, sides


# EVOLVE-BLOCK-END
