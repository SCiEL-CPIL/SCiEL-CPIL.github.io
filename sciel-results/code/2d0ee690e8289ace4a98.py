"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csc_matrix

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 128
    min_sum = 128
    max_sum = 256
    target_sums = np.arange(min_sum, max_sum + 1)
    num_targets = len(target_sums)
    
    # Precompute the incidence matrix for pair sums
    # We want a matrix P where P[t, (i, j)] = 1 if i + j == target_sums[t]
    # Variables are x_k for k in 0..128 (binary).
    # The representation count for sum s is:
    # sum_{i: 2i=s} x_i + sum_{i<j: i+j=s} (x_i x_j)
    # This is quadratic. MILP handles linear constraints.
    # However, we can linearize the product x_i x_j by introducing variables y_{ij} = x_i x_j.
    # Constraints:
    # y_{ij} <= x_i
    # y_{ij} <= x_j
    # y_{ij} >= x_i + x_j - 1
    # y_{ij} >= 0
    # And x_i in {0, 1}, y_{ij} in {0, 1} (implied by above if x are binary)
    
    # This creates a large number of variables.
    # Number of pairs i <= j such that i+j is in [128, 256].
    # Actually, we only need y_{ij} for pairs that sum to a target.
    # Let's identify all unique pairs (i, j) with i <= j such that i+j in [128, 256].
    
    pairs = []
    pair_to_idx = {}
    for i in range(0, 129):
        for j in range(i, 129):
            s = i + j
            if min_sum <= s <= max_sum:
                idx = len(pairs)
                pairs.append((i, j))
                pair_to_idx[(i, j)] = idx
    
    num_pairs = len(pairs)
    
    # Variables:
    # 0..128: x_i (binary)
    # 129..128+num_pairs: y_{ij} (binary/continuous, linearized product)
    
    total_vars = 129 + num_pairs
    
    # Objective: Minimize M (variable at index total_vars)
    # c vector: 0 for all x, 0 for all y, 1 for M
    c = np.zeros(total_vars + 1)
    c[-1] = 1.0
    
    # Constraints:
    # 1. Coverage: For each target sum t, count_t >= 1
    #    count_t = sum_{(i,j) in P_t} y_{ij}  (Note: for i=j, y_ii = x_i, since x_i*x_i=x_i)
    #    Wait, for i=j, y_ii = x_i. So we can just use x_i for diagonal terms.
    #    Let's refine:
    #    If i == j: term is x_i.
    #    If i < j: term is y_{ij}.
    
    # 2. Multiplicity Cap: For each target sum t, count_t <= M
    #    count_t <= M  =>  count_t - M <= 0
    
    # 3. Linearization constraints for i < j:
    #    y_{ij} - x_i <= 0
    #    y_{ij} - x_j <= 0
    #    y_{ij} - x_i - x_j >= -1  =>  -y_{ij} + x_i + x_j <= 1
    
    # Let's build the constraint matrix A_ub and b_ub for A_ub @ v <= b_ub
    
    # Estimate size:
    # num_targets = 129
    # num_pairs ~ 4000-5000?
    #   i ranges 0..128. j ranges i..128.
    #   i+j >= 128.
    #   For i=0, j=128.
    #   For i=64, j=64..128 (65 pairs)
    #   Total pairs is roughly half of 129*129/2? No, restricted range.
    #   Let's just count.
    
    # We will use sparse matrices.
    rows = []
    cols = []
    vals = []
    
    # Helper to add a row
    def add_row(coeffs_dict, bound_val, is_ub=True):
        # coeffs_dict: {var_idx: coeff}
        # We accumulate into global lists
        pass

    # It's easier to build row by row for clarity, then stack.
    # But for efficiency, let's use scipy.sparse.
    
    A_list = []
    b_list = []
    
    # Map pair index to variable index
    def y_idx(pair_idx):
        return 129 + pair_idx
    
    # 1 & 2. Coverage and Cap constraints
    # For each target sum t:
    for t in range(num_targets):
        s_val = target_sums[t]
        
        # Collect coefficients for count_t
        row_coeffs = {}
        
        for (i, j), p_idx in pair_to_idx.items():
            if i + j == s_val:
                if i == j:
                    var_idx = i # x_i
                else:
                    var_idx = y_idx(p_idx)
                row_coeffs[var_idx] = row_coeffs.get(var_idx, 0) + 1
        
        # Constraint: count_t >= 1  =>  -count_t <= -1
        # Create row for -count_t
        row_neg = [0.0] * (total_vars + 1)
        for var_idx, coeff in row_coeffs.items():
            row_neg[var_idx] = -coeff
        A_list.append(row_neg)
        b_list.append(-1.0)
        
        # Constraint: count_t <= M  =>  count_t - M <= 0
        row_cap = [0.0] * (total_vars + 1)
        for var_idx, coeff in row_coeffs.items():
            row_cap[var_idx] = coeff
        row_cap[-1] = -1.0 # -M
        A_list.append(row_cap)
        b_list.append(0.0)
    
    # 3. Linearization constraints for i < j
    for (i, j), p_idx in pair_to_idx.items():
        if i == j:
            continue
        yi = y_idx(p_idx)
        xi = i
        xj = j
        
        # y_{ij} - x_i <= 0
        r1 = [0.0] * (total_vars + 1)
        r1[yi] = 1.0
        r1[xi] = -1.0
        A_list.append(r1)
        b_list.append(0.0)
        
        # y_{ij} - x_j <= 0
        r2 = [0.0] * (total_vars + 1)
        r2[yi] = 1.0
        r2[xj] = -1.0
        A_list.append(r2)
        b_list.append(0.0)
        
        # -y_{ij} + x_i + x_j <= 1
        r3 = [0.0] * (total_vars + 1)
        r3[yi] = -1.0
        r3[xi] = 1.0
        r3[xj] = 1.0
        A_list.append(r3)
        b_list.append(1.0)
    
    # Convert to sparse matrix
    A_ub = np.array(A_list)
    b_ub = np.array(b_list)
    
    # Integality: x_i (0..128) are integer (binary). y_{ij} can be continuous?
    # If x are binary, y determined by linearization is binary.
    # So we only need to enforce integrality on x_i.
    integrality = np.zeros(total_vars + 1)
    for i in range(129):
        integrality[i] = 1.0
    
    # Bounds
    lb = np.zeros(total_vars + 1)
    ub = np.ones(total_vars + 1)
    ub[-1] = np.inf # M can be large
    
    # Solve
    try:
        res = milp(
            c=c,
            constraints=[
                LinearConstraint(A_ub, -np.inf, b_ub),
                Bounds(lb, ub)
            ],
            integrality=integrality
        )
        
        if res.status == 0:
            # Extract x_i
            x_sol = res.x[:129]
            A_set = set(int(i) for i in range(129) if x_sol[i] > 0.5)
            
            # Verify and calculate peak
            # We should verify because MILP might have tiny numerical errors
            # or we might want to double check the peak.
            
            def get_counts(A_set):
                counts = np.zeros(num_targets, dtype=np.int32)
                A_list = list(A_set)
                m = len(A_list)
                for i in range(m):
                    ai = A_list[i]
                    s = 2 * ai
                    if min_sum <= s <= max_sum:
                        idx = s - min_sum
                        counts[idx] += 1
                    for j in range(i + 1, m):
                        aj = A_list[j]
                        s = ai + aj
                        if min_sum <= s <= max_sum:
                            idx = s - min_sum
                            counts[idx] += 1
                return counts

            counts = get_counts(A_set)
            if np.any(counts == 0):
                # Fallback if invalid (should not happen if constraints correct)
                # Try a simple heuristic
                return list(range(0, 129))
            
            peak = np.max(counts)
            return list(A_set)
            
        else:
            # Fallback if MILP fails
            return list(range(0, 129))
            
    except Exception as e:
        # Fallback
        return list(range(0, 129))

# EVOLVE-BLOCK-END
