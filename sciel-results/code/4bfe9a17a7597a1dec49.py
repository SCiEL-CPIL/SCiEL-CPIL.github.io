import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Build the rank-28 construction as a fallback
    rank_28_result = _build_rank_28()
    
    # Try rank-27 feasibility test
    if time.time() < deadline - 5:
        rank_27_result = _try_rank_27()
        if rank_27_result is not None:
            return rank_27_result
    
    return rank_28_result

def _build_rank_28():
    n, m, p = DIMENSIONS
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: A_L (2x2) * B_T1 (2x2) -> C_1 (cols 0,1)
    # Col 0: M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[4][0] = 1
    V[0][0] = 1; V[6][0] = 1
    W[0][0] = 1; W[3][0] = 1

    # Col 1: M2 = (A10+A11)B00
    U[3][1] = 1; U[4][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1

    # Col 2: M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[6][2] = -1
    W[2][2] = 1; W[3][2] = 1

    # Col 3: M4 = A11(B10-B00)
    U[4][3] = 1
    V[5][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1

    # Col 4: M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[6][4] = 1
    W[0][4] = -1; W[2][4] = 1

    # Col 5: M6 = (A10-A00)(B00+B01)
    U[3][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1

    # Col 6: M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[4][6] = -1
    V[5][6] = 1; V[6][6] = 1
    W[0][6] = 1

    # Part 2: A_L (2x2) * B_T2 (2x2) -> C_2 (cols 2,3)
    # Col 7: M1
    U[0][7] = 1; U[4][7] = 1
    V[2][7] = 1; V[8][7] = 1
    W[4][7] = 1; W[7][7] = 1

    # Col 8: M2
    U[3][8] = 1; U[4][8] = 1
    V[2][8] = 1
    W[5][8] = 1; W[7][8] = -1

    # Col 9: M3
    U[0][9] = 1
    V[3][9] = 1; V[8][9] = -1
    W[6][9] = 1; W[7][9] = 1

    # Col 10: M4
    U[4][10] = 1
    V[7][10] = 1; V[2][10] = -1
    W[4][10] = 1; W[5][10] = 1

    # Col 11: M5
    U[0][11] = 1; U[1][11] = 1
    V[8][11] = 1
    W[4][11] = -1; W[6][11] = 1

    # Col 12: M6
    U[3][12] = 1; U[0][12] = -1
    V[2][12] = 1; V[3][12] = 1
    W[7][12] = 1

    # Col 13: M7
    U[1][13] = 1; U[4][13] = -1
    V[7][13] = 1; V[8][13] = 1
    W[4][13] = 1

    # Part 3: A_L (2x2) * B_T3 (2x1) -> C_3 (col 4)
    # Col 14: A00*B04
    U[0][14] = 1
    V[4][14] = 1
    W[8][14] = 1

    # Col 15: A01*B14
    U[1][15] = 1
    V[9][15] = 1
    W[8][15] = 1

    # Col 16: A10*B04
    U[3][16] = 1
    V[4][16] = 1
    W[9][16] = 1

    # Col 17: A11*B14
    U[4][17] = 1
    V[9][17] = 1
    W[9][17] = 1

    # Part 4: A_R (2x1) * B_B (1x5) -> C_4 (all cols)
    # Col 18: A02*B20
    U[2][18] = 1
    V[10][18] = 1
    W[0][18] = 1

    # Col 19: A02*B21
    U[2][19] = 1
    V[11][19] = 1
    W[2][19] = 1

    # Col 20: A02*B22
    U[2][20] = 1
    V[12][20] = 1
    W[4][20] = 1

    # Col 21: A02*B23
    U[2][21] = 1
    V[13][21] = 1
    W[6][21] = 1

    # Col 22: A02*B24
    U[2][22] = 1
    V[14][22] = 1
    W[8][22] = 1

    # Col 23: A12*B20
    U[5][23] = 1
    V[10][23] = 1
    W[1][23] = 1

    # Col 24: A12*B21
    U[5][24] = 1
    V[11][24] = 1
    W[3][24] = 1

    # Col 25: A12*B22
    U[5][25] = 1
    V[12][25] = 1
    W[5][25] = 1

    # Col 26: A12*B23
    U[5][26] = 1
    V[13][26] = 1
    W[7][26] = 1

    # Col 27: A12*B24
    U[5][27] = 1
    V[14][27] = 1
    W[9][27] = 1

    return {"U": U, "V": V, "W": W}

def _try_rank_27():
    """
    Try to find a rank-27 decomposition by systematically searching for
    valid U, V matrices that allow exact W solution.
    """
    n, m, p = DIMENSIONS
    target_rank = 27
    deadline = time.time() + 125  # Safety margin
    
    # Build the tensor: T[u,v,w] = 1 if C_w depends on A_u * B_v
    # u = i*m + j (i in 0..n-1, j in 0..m-1)
    # v = j*p + k (j in 0..m-1, k in 0..p-1)
    # w = k*n + i (k in 0..p-1, i in 0..n-1)
    tensor = np.zeros((n*m, m*p, p*n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                tensor[u_idx, v_idx, w_idx] = 1
    
    # Start from rank-28
    base = _build_rank_28()
    U_base = np.array(base["U"], dtype=float)
    V_base = np.array(base["V"], dtype=float)
    
    # Strategy: For each pair of columns (a, b), try to modify column b
    # slightly (by small integer changes) and check if a valid W exists.
    # Key insight: if we can find U, V such that the system
    # sum_c U[u,c]*V[v,c]*W[w,c] = tensor[u,v,w] has an exact solution,
    # we have a valid decomposition.
    
    # First, try the simple case: make column b identical to column a
    # and see if W can be solved.
    
    # Build the linear system for W. For each w, we need:
    # sum_c (U[:,c] * V[:,c])_w * W[w,c] = tensor[:, :, w]
    # This is: A_w @ W[:,w] = b_w where A_w[c, (u,v)] = U[u,c]*V[v,c]
    
    def try_solve_W(U, V):
        """Try to solve for W given U and V. Returns W if exact solution exists."""
        W = np.zeros((p*n, target_rank))
        for w in range(p*n):
            # Build the linear system
            # Variables: W[w, 0..target_rank-1]
            # Equations: for each (u,v) pair
            A = np.zeros((n*m*m*p, target_rank))
            b = np.zeros(n*m*m*p)
            
            eq_idx = 0
            for u in range(n*m):
                for v in range(m*p):
                    for c in range(target_rank):
                        A[eq_idx, c] = U[u, c] * V[v, c]
                    b[eq_idx] = tensor[u, v, w]
                    eq_idx += 1
            
            # Solve using least squares and check if exact
            try:
                sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                # Check if the solution is exact
                residual = np.abs(A @ sol - b)
                if np.max(residual) < 1e-10:
                    W[w, :] = sol
                else:
                    return None
            except:
                return None
        
        # Verify the full decomposition
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    computed = 0.0
                    for c in range(target_rank):
                        computed += U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
                    expected = tensor[u_idx, v_idx, w_idx]
                    if abs(computed - expected) > 1e-10:
                        return None
        
        return W
    
    # Try all pairs: set column b = column a
    for col_a in range(target_rank):
        for col_b in range(col_a + 1, target_rank):
            if time.time() > deadline:
                return None
            
            U_new = U_base.copy()
            V_new = V_base.copy()
            
            # Set column b identical to column a
            U_new[:, col_b] = U_new[:, col_a]
            V_new[:, col_b] = V_new[:, col_a]
            
            W_sol = try_solve_W(U_new, V_new)
            if W_sol is not None:
                # Check if entries are integer or half-integer
                U_check = U_new * 2
                V_check = V_new * 2
                W_check = W_sol * 2
                
                if (np.allclose(U_check, np.round(U_check), atol=1e-10) and
                    np.allclose(V_check, np.round(V_check), atol=1e-10) and
                    np.allclose(W_check, np.round(W_check), atol=1e-10)):
                    # Round to exact values
                    U_final = (np.round(U_check) / 2).tolist()
                    V_final = (np.round(V_check) / 2).tolist()
                    W_final = (np.round(W_check) / 2).tolist()
                    return {"U": U_final, "V": V_final, "W": W_final}
    
    # Try small perturbations: add -1, 0, or +1 to some entries of column b
    # Focus on columns that are structurally similar
    for col_a in range(target_rank):
        for col_b in range(col_a + 1, target_rank):
            if time.time() > deadline:
                return None
            
            # Try a few simple perturbations
            for du in [0, 1, -1]:
                for dv in [0, 1, -1]:
                    if du == 0 and dv == 0:
                        continue  # Already tried
                    
                    U_new = U_base.copy()
                    V_new = V_base.copy()
                    
                    # Add perturbation to column b
                    U_new[:, col_b] = U_new[:, col_a] + du
                    V_new[:, col_b] = V_new[:, col_a] + dv
                    
                    # Only consider if entries stay in {-1, 0, 1}
                    if not np.all(np.isin(U_new[:, col_b], [-1, 0, 1])):
                        continue
                    if not np.all(np.isin(V_new[:, col_b], [-1, 0, 1])):
                        continue
                    
                    W_sol = try_solve_W(U_new, V_new)
                    if W_sol is not None:
                        U_check = U_new * 2
                        V_check = V_new * 2
                        W_check = W_sol * 2
                        
                        if (np.allclose(U_check, np.round(U_check), atol=1e-10) and
                            np.allclose(V_check, np.round(V_check), atol=1e-10) and
                            np.allclose(W_check, np.round(W_check), atol=1e-10)):
                            U_final = (np.round(U_check) / 2).tolist()
                            V_final = (np.round(V_check) / 2).tolist()
                            W_final = (np.round(W_check) / 2).tolist()
                            return {"U": U_final, "V": V_final, "W": W_final}
    
    return None

# _solve_linear_system removed - using numpy.linalg.lstsq instead
