import numpy as np
import time
import random
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    n = 129 # 0 to 128
    target_min = 128
    target_max = 256
    
    # Variables:
    # 0..128: x_i (binary)
    # 129: M (continuous, integer in practice)
    # 130..130+K-1: y_{a,b} (binary)
    
    # Identify all pairs (a, b) with a < b and target_min <= a+b <= target_max
    pairs = []
    pair_index = {} # (a, b) -> index in y-variables
    y_count = 0
    for a in range(n):
        for b in range(a + 1, n):
            s = a + b
            if target_min <= s <= target_max:
                pair_index[(a, b)] = 130 + y_count
                y_count += 1
                pairs.append((a, b))
                
    total_vars = 130 + y_count
    
    # Objective: minimize M
    c = np.zeros(total_vars)
    c[129] = 1.0 # M is at index 129
    
    # Constraints
    # We use LinearConstraint(A, lb, ub)
    # A is a matrix, lb and ub are vectors
    
    # List of constraints (row, lb, ub)
    A_rows = []
    lbs = []
    ubs = []
    
    # 1. Linking constraints for y
    # y_ab - x_a <= 0
    # y_ab - x_b <= 0
    # -y_ab + x_a + x_b <= 1
    
    for (a, b) in pairs:
        idx_y = pair_index[(a, b)]
        
        # y_ab - x_a <= 0
        row1 = np.zeros(total_vars)
        row1[idx_y] = 1.0
        row1[a] = -1.0
        A_rows.append(row1)
        lbs.append(-np.inf)
        ubs.append(0.0)
        
        # y_ab - x_b <= 0
        row2 = np.zeros(total_vars)
        row2[idx_y] = 1.0
        row2[b] = -1.0
        A_rows.append(row2)
        lbs.append(-np.inf)
        ubs.append(0.0)
        
        # -y_ab + x_a + x_b <= 1
        row3 = np.zeros(total_vars)
        row3[idx_y] = -1.0
        row3[a] = 1.0
        row3[b] = 1.0
        A_rows.append(row3)
        lbs.append(-np.inf)
        ubs.append(1.0)
        
    # 2. Coverage and Capacity constraints for each target t
    # R_t = sum_{a<b, a+b=t} y_ab + (x_{t/2} if t even else 0)
    # 1 <= R_t <= M
    
    for t in range(target_min, target_max + 1):
        # Build R_t expression
        # Coefficients for x and y
        coeff = np.zeros(total_vars)
        
        # Add y terms
        for a in range(target_min // 2, t // 2 + 1):
            b = t - a
            if a < b and 0 <= a < n and 0 <= b < n:
                if (a, b) in pair_index:
                    coeff[pair_index[(a, b)]] = 1.0
            elif a == b:
                # Self pair, handled by x
                pass
        
        # Add x term for self-pair if t is even
        if t % 2 == 0:
            mid = t // 2
            if 0 <= mid < n:
                coeff[mid] = 1.0
        
        # Constraint: R_t >= 1  =>  -R_t <= -1
        row_lb = -coeff
        A_rows.append(row_lb)
        lbs.append(-np.inf)
        ubs.append(-1.0)
        
        # Constraint: R_t <= M  =>  R_t - M <= 0
        row_ub = coeff.copy()
        row_ub[129] = -1.0 # Subtract M
        A_rows.append(row_ub)
        lbs.append(-np.inf)
        ubs.append(0.0)
        
    # Construct A matrix
    A = np.array(A_rows)
    lb = np.array(lbs)
    ub = np.array(ubs)
    
    # Bounds for variables
    # x_i are binary: 0 or 1
    # M is continuous: 0 or more
    # y_ab are binary: 0 or 1
    
    integrality = np.zeros(total_vars)
    integrality[0:129] = 1.0 # x_i are integers (binary)
    integrality[130:] = 1.0  # y_ab are integers (binary)
    # M (index 129) is continuous
    
    bounds = Bounds(lb=np.zeros(total_vars), ub=np.inf)
    # Set binary bounds for x and y
    for i in range(129):
        bounds.upper[i] = 1.0
    for i in range(130, total_vars):
        bounds.upper[i] = 1.0
    # M can be up to maybe 100?
    bounds.upper[129] = 100.0
    
    # Solve
    try:
        res = milp(c=c,
                   constraints=LinearConstraint(A, lb, ub),
                   integrality=integrality,
                   bounds=bounds)
        
        if res.status == 0:
            x_sol = res.x[0:129]
            # Threshold for binary
            x_bin = (x_sol > 0.5).astype(int)
            result = np.nonzero(x_bin)[0].tolist()
            
            # Verify and return
            if len(result) > 0:
                return result
            else:
                # Fallback if empty
                return [64]
        else:
            # Fallback to a simple heuristic if MILP fails
            # Just return a large set to ensure coverage
            return list(range(64, 129))
            
    except Exception as e:
        # Fallback to a simple heuristic if MILP fails
        return list(range(64, 129))

