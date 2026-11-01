import numpy as np
import time
import itertools

def construct():
    """
    Finds a low-rank exact tensor decomposition for 3x3x6 matrix multiplication.
    Strategy:
    1. Define the multiplication tensor T for 3x3x6.
    2. Attempt to find a decomposition of rank R where R is between 45 and 53.
       The decomposition is defined by matrices U (9xR), V (18xR), W (18xR)
       such that T = sum_r u_r v_r^T w_r^T (in appropriate reshaping).
       Specifically, T_{a,b,c} = sum_r U[a,r] V[b,r] W[c,r].
       Here a is in 0..8 (index of A), b in 0..17 (index of B), c in 0..17 (index of C).
       
       We fix U and V to have small integer entries (e.g., -1, 0, 1) and solve for W.
       W must be integer or half-integer.
       
    3. If a valid decomposition is found, return it.
    4. Otherwise, return the schoolbook decomposition (rank 54).
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin

    n, m, p = 3, 3, 6
    dim_A = n * m  # 9
    dim_B = m * p  # 18
    dim_C = p * n  # 18

    # Build the target tensor T as a dictionary of non-zero entries
    # T[(i,j), (j,k), (k,i)] = 1
    # Indices:
    # A index: a = i * m + j
    # B index: b = j * p + k
    # C index: c = k * n + i
    T_nonzero = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_nonzero[(a, b, c)] = 1.0

    # Schoolbook fallback
    def generate_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(dim_A)]
        V = [[0] * rank for _ in range(dim_B)]
        W = [[0] * rank for _ in range(dim_C)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a][col] = 1
                    V[b][col] = 1
                    W[c][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best_result = generate_schoolbook()
    best_rank = n * m * p

    # Search for lower rank decompositions
    # We iterate over possible ranks R from 45 up to 53.
    # For each R, we try random U, V with entries in {-1, 0, 1}.
    # Then we solve for W.
    
    # The equation is:
    # For all (a, b, c), sum_r U[a,r] V[b,r] W[c,r] = T[(a,b,c)]
    
    # This is a linear system in W if U, V are fixed.
    # Let M be a matrix where rows correspond to (a,b,c) triples that are non-zero in T?
    # No, the equation must hold for ALL (a,b,c).
    # However, T is sparse.
    # Let's consider the structure.
    # For a fixed r, the term is u_r v_r w_r.
    # This is a sum of rank-1 tensors.
    
    # We can set up a linear system for W.
    # Let's collect all equations.
    # There are dim_A * dim_B * dim_C = 9 * 18 * 18 = 2916 equations.
    # Unknowns are W[c, r] for c in 0..17, r in 0..R-1. Total 18*R unknowns.
    
    # We can build the matrix A_sys of size (num_equations, 18*R).
    # A_sys[eq_idx, c*R + r] = U[a, r] * V[b, r]
    # b_sys[eq_idx] = T[(a,b,c)]
    
    # To make it faster, we only need to consider equations where T is non-zero?
    # No, if T is zero, the sum must be 0.
    # But the number of equations is manageable (2916).
    
    # However, building a 2916 x (18*R) matrix for R=50 is 2916 x 900.
    # Solving this via lstsq is feasible but might be slow if done many times.
    
    # Optimization:
    # We can use the fact that T is sparse.
    # But the zeros impose constraints too.
    
    # Let's try a different approach:
    # For each c, we have a vector W[c, :] of length R.
    # The equation for a specific (a, b) is:
    # sum_c T[(a,b,c)] W[c, :]^T (element-wise product of U[:, r] and V[:, r]?)
    # No.
    # T[(a,b,c)] = sum_r U[a,r] V[b,r] W[c,r]
    
    # Let's fix a, b.
    # We have a system for W[:, r] across c?
    # Actually, for fixed a, b, we have 18 equations (for c=0..17).
    # sum_r U[a,r] V[b,r] W[c,r] = T[(a,b,c)]
    
    # This is 18 equations in 18*R unknowns? No, W[c,r] are the unknowns.
    # For fixed a, b, the unknowns are W[0..17, 0..R-1].
    # The coefficient of W[c, r] in the equation for output (a,b,c') is:
    # If c == c', coeff is U[a,r] * V[b,r].
    # If c != c', coeff is 0.
    
    # So for fixed a, b, the system is:
    # For each c:
    #   sum_r (U[a,r] * V[b,r]) * W[c,r] = T[(a,b,c)]
    
    # This means that for each c, the vector W[c, :] must satisfy:
    # dot(U[a,:] * V[b,:], W[c,:]) = T[(a,b,c)]
    
    # This decouples the c's for a fixed (a,b)!
    # Wait, W[c, r] appears in the equation for (a,b,c) with coefficient U[a,r]V[b,r].
    # It does NOT appear in the equation for (a,b,c') if c != c'.
    # So yes, for fixed a, b, we have 18 independent linear equations (one for each c)
    # for the 18*R unknowns? No.
    
    # Let's look at the unknowns: W is a matrix of size 18 x R.
    # The unknowns are W[0,0], W[0,1], ..., W[17, R-1].
    # For fixed a, b, the equation for c is:
    # sum_r (U[a,r] * V[b,r]) * W[c,r] = T[(a,b,c)]
    
    # This equation involves only the row W[c, :] of W.
    # So, for fixed a, b, and fixed c, we have one linear equation involving the R unknowns in W[c, :].
    # But we have 18 such equations (for c=0..17) for the same (a,b).
    # And these equations are independent in terms of which row of W they involve.
    # Equation for c involves W[c, :].
    # Equation for c' involves W[c', :].
    # So for fixed a, b, we have 18 independent systems, each of size 1 equation with R unknowns?
    # No, that's not enough to determine W[c, :].
    
    # We need to combine equations from different (a, b) pairs.
    # For a fixed c, W[c, :] is involved in equations for all (a, b).
    # Specifically, for each (a, b), we have:
    # dot(U[a,:] * V[b,:], W[c,:]) = T[(a,b,c)]
    
    # So for a fixed c, we have a system of 9*18 = 162 equations (one for each a, b)
    # for the R unknowns in W[c, :].
    # Let M_c be a 162 x R matrix.
    # M_c[(a,b), r] = U[a,r] * V[b,r]
    # b_c[(a,b)] = T[(a,b,c)]
    
    # We need to solve M_c @ W[c, :] = b_c for each c.
    # This is a least squares problem.
    # If a solution exists, it must be integer or half-integer.
    
    # This is much more efficient!
    # We only need to solve 18 small least squares problems (162 x R) instead of one huge one.
    
    # Let's implement this.
    
    for R in range(45, 54):
        if time.time() > deadline:
            break
        
        # Number of attempts per rank
        # We want to use the time wisely.
        # Solving 18 lstsq problems of size 162 x R is fast.
        # Let's try 1000 attempts per rank.
        attempts = 1000
        
        for attempt in range(attempts):
            if time.time() > deadline:
                break
                
            # Generate random U, V
            U = np.random.randint(-1, 2, size=(dim_A, R))
            V = np.random.randint(-1, 2, size=(dim_B, R))
            
            W = np.zeros((dim_C, R))
            valid = True
            
            for c in range(dim_C):
                # Build M_c: 162 x R
                # Rows correspond to (a, b) pairs
                M_c = np.zeros((dim_A * dim_B, R))
                b_c = np.zeros(dim_A * dim_B)
                
                idx = 0
                for a in range(dim_A):
                    for b in range(dim_B):
                        M_c[idx, :] = U[a, :] * V[b, :]
                        # T[(a,b,c)]
                        if (a, b, c) in T_nonzero:
                            b_c[idx] = 1.0
                        else:
                            b_c[idx] = 0.0
                        idx += 1
                
                # Solve M_c @ w = b_c
                try:
                    w, residuals, rank, s = np.linalg.lstsq(M_c, b_c, rcond=None)
                except:
                    valid = False
                    break
                
                # Check if w is integer or half-integer
                w_doubled = w * 2
                closest_ints = np.round(w_doubled)
                error = np.max(np.abs(w_doubled - closest_ints))
                
                if error < 1e-9:
                    W[c, :] = w
                else:
                    valid = False
                    break
            
            if valid:
                # Verify the solution
                # Check a few random (a,b,c) triples
                # Actually, let's verify all non-zero ones and a few zero ones
                # to be sure.
                is_valid = True
                for (a, b, c), val in T_nonzero.items():
                    s = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(s - val) > 1e-9:
                        is_valid = False
                        break
                
                if is_valid:
                    # Check a few zero entries
                    for _ in range(100):
                        a = np.random.randint(0, dim_A)
                        b = np.random.randint(0, dim_B)
                        c = np.random.randint(0, dim_C)
                        if (a, b, c) not in T_nonzero:
                            s = np.sum(U[a, :] * V[b, :] * W[c, :])
                            if abs(s) > 1e-9:
                                is_valid = False
                                break
                
                if is_valid:
                    # Convert to lists
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    
                    # Ensure entries are integers or half-integers
                    # They should be, but let's round to be clean
                    U_list = [[int(x) for x in row] for row in U_list]
                    V_list = [[int(x) for x in row] for row in V_list]
                    W_list = [[float(x) for x in row] for row in W_list]
                    
                    # Round W to half-integers
                    W_list = [[float(np.round(x * 2) / 2) for x in row] for row in W_list]
                    
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = R
                    return best_result

    return best_result

# EVOLVE-BLOCK-START
def construct():
    return _construct_impl()

def _construct_impl():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = 3, 3, 6
    dim_A = n * m  # 9
    dim_B = m * p  # 18
    dim_C = p * n  # 18

    T_nonzero = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_nonzero[(a, b, c)] = 1.0

    def generate_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(dim_A)]
        V = [[0] * rank for _ in range(dim_B)]
        W = [[0] * rank for _ in range(dim_C)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a][col] = 1
                    V[b][col] = 1
                    W[c][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best_result = generate_schoolbook()
    best_rank = n * m * p

    for R in range(45, 54):
        if time.time() > deadline:
            break
        
        attempts = 1000
        
        for attempt in range(attempts):
            if time.time() > deadline:
                break
                
            U = np.random.randint(-1, 2, size=(dim_A, R))
            V = np.random.randint(-1, 2, size=(dim_B, R))
            
            W = np.zeros((dim_C, R))
            valid = True
            
            for c in range(dim_C):
                M_c = np.zeros((dim_A * dim_B, R))
                b_c = np.zeros(dim_A * dim_B)
                
                idx = 0
                for a in range(dim_A):
                    for b in range(dim_B):
                        M_c[idx, :] = U[a, :] * V[b, :]
                        if (a, b, c) in T_nonzero:
                            b_c[idx] = 1.0
                        else:
                            b_c[idx] = 0.0
                        idx += 1
                
                try:
                    w, residuals, rank, s = np.linalg.lstsq(M_c, b_c, rcond=None)
                except:
                    valid = False
                    break
                
                w_doubled = w * 2
                closest_ints = np.round(w_doubled)
                error = np.max(np.abs(w_doubled - closest_ints))
                
                if error < 1e-9:
                    W[c, :] = w
                else:
                    valid = False
                    break
            
            if valid:
                is_valid = True
                for (a, b, c), val in T_nonzero.items():
                    s = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(s - val) > 1e-9:
                        is_valid = False
                        break
                
                if is_valid:
                    for _ in range(100):
                        a = np.random.randint(0, dim_A)
                        b = np.random.randint(0, dim_B)
                        c = np.random.randint(0, dim_C)
                        if (a, b, c) not in T_nonzero:
                            s = np.sum(U[a, :] * V[b, :] * W[c, :])
                            if abs(s) > 1e-9:
                                is_valid = False
                                break
                
                if is_valid:
                    U_list = [[int(x) for x in row] for row in U.tolist()]
                    V_list = [[int(x) for x in row] for row in V.tolist()]
                    W_list = [[float(np.round(x * 2) / 2) for x in row] for row in W.tolist()]
                    
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = R
                    return best_result

    return best_result
# EVOLVE-BLOCK-END
