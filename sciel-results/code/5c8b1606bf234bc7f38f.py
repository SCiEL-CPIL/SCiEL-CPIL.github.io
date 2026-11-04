import numpy as np
import scipy.optimize as so
import time
import sys

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    dim_A = n * m  # 8
    dim_B = m * p  # 24
    dim_C = p * n  # 12
    
    # Build the target tensor T (sparse representation for efficiency)
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i)
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    indices_a = []
    indices_b = []
    indices_c = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                indices_a.append(a)
                indices_b.append(b)
                indices_c.append(c)
    
    idx_a = np.array(indices_a, dtype=np.int32)
    idx_b = np.array(indices_b, dtype=np.int32)
    idx_c = np.array(indices_c, dtype=np.int32)
    
    # Full tensor for zero-mask calculation
    full_T = np.zeros((dim_A, dim_B, dim_C))
    full_T[idx_a, idx_b, idx_c] = 1.0
    zero_mask = (full_T == 0)
    
    # Precompute the non-zero indices for faster residual calc
    # The residual is sum over non-zeros of (hat_T - 1)^2 + sum over zeros of hat_T^2
    
    def calculate_residual(theta, r):
        """
        theta: flattened [U, V, W]
        r: rank
        """
        U = theta[:dim_A * r].reshape(dim_A, r)
        V = theta[dim_A * r : dim_A * r + dim_B * r].reshape(dim_B, r)
        W = theta[dim_A * r + dim_B * r :].reshape(dim_C, r)
        
        # Compute hat_T
        # hat_T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        hat_T = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        
        # Extract values at non-zero positions
        vals = hat_T[idx_a, idx_b, idx_c]
        res_nonzero = vals - 1.0
        
        # Extract values at zero positions
        vals_zero = hat_T[zero_mask]
        res_zero = vals_zero
        
        # Combine
        return np.concatenate([res_nonzero, res_zero])

    def verify_solution(U_mat, V_mat, W_mat):
        """
        Verify if U, V, W are valid (integer/half-integer) and exactly decompose T.
        """
        # Check if entries are close to half-integers
        # The prompt allows integer/half-integer.
        # We assume the input is already snapped or we snap it here for safety.
        # But for verification of a snapped solution:
        
        # Check exactness
        hat_T = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat, optimize=True)
        
        # Check non-zero entries
        vals = hat_T[idx_a, idx_b, idx_c]
        if not np.allclose(vals, 1.0, atol=1e-9):
            return False
            
        # Check zero entries
        vals_zero = hat_T[zero_mask]
        if not np.allclose(vals_zero, 0.0, atol=1e-9):
            return False
            
        return True

    def snap_and_verify(U_sol, V_sol, W_sol, r):
        """
        Snap the real-valued solution to the nearest half-integer and verify.
        """
        U_int = np.round(U_sol * 2) / 2
        V_int = np.round(V_sol * 2) / 2
        W_int = np.round(W_sol * 2) / 2
        
        if verify_solution(U_int, V_int, W_int):
            return U_int, V_int, W_int
            
        # If direct snap fails, try a local search in the discrete space?
        # Or just return None.
        # Given the high dimensionality, direct snap from a good real solution is the best bet.
        return None

    # Fallback: Rank 42 Strassen solution
    def get_fallback():
        U = [[0] * 42 for _ in range(n * m)]
        V = [[0] * 42 for _ in range(m * p)]
        W = [[0] * 42 for _ in range(p * n)]
        
        def add_strassen_block(a_idx, b_idx, w_idx, start_col):
            # Standard Strassen 7 multiplications for 2x2 matrices
            # Indices for A (2x2): 00, 01, 10, 11 -> 0, 1, 2, 3 (row-major)
            # Indices for B (2x2): 00, 01, 10, 11 -> 0, 1, 2, 3
            # Indices for C (2x2): 00, 01, 10, 11 -> 0, 1, 2, 3
            
            # M1 = A00 * (B01 - B11)
            # M2 = (A00 + A01) * B11
            # M3 = (A10 + A11) * B00
            # M4 = A11 * (B10 - B00)
            # M5 = (A00 + A11) * (B00 + B11)
            # M6 = (A01 - A11) * (B10 + B11)
            # M7 = (A10 - A00) * (B00 + B01)
            
            # C00 = M5 + M4 - M2 + M6
            # C01 = M1 + M2
            # C10 = M3 + M4
            # C11 = M5 + M1 - M3 - M7
            
            # Mapping to our 7 terms (0 to 6):
            # t0: M1, t1: M2, t2: M3, t3: M4, t4: M5, t5: M6, t6: M7
            
            # U (A) contributions:
            # M1: A00
            U[a_idx[0]][start_col + 0] += 1
            # M2: A00, A01
            U[a_idx[0]][start_col + 1] += 1
            U[a_idx[1]][start_col + 1] += 1
            # M3: A10, A11
            U[a_idx[2]][start_col + 2] += 1
            U[a_idx[3]][start_col + 2] += 1
            # M4: A11
            U[a_idx[3]][start_col + 3] += 1
            # M5: A00, A11
            U[a_idx[0]][start_col + 4] += 1
            U[a_idx[3]][start_col + 4] += 1
            # M6: A01, -A11
            U[a_idx[1]][start_col + 5] += 1
            U[a_idx[3]][start_col + 5] -= 1
            # M7: -A00, A10
            U[a_idx[0]][start_col + 6] -= 1
            U[a_idx[2]][start_col + 6] += 1
            
            # V (B) contributions:
            # M1: B01, -B11
            V[b_idx[1]][start_col + 0] += 1
            V[b_idx[3]][start_col + 0] -= 1
            # M2: B11
            V[b_idx[3]][start_col + 1] += 1
            # M3: B00
            V[b_idx[0]][start_col + 2] += 1
            # M4: B10, -B00
            V[b_idx[2]][start_col + 3] += 1
            V[b_idx[0]][start_col + 3] -= 1
            # M5: B00, B11
            V[b_idx[0]][start_col + 4] += 1
            V[b_idx[3]][start_col + 4] += 1
            # M6: B10, B11
            V[b_idx[2]][start_col + 5] += 1
            V[b_idx[3]][start_col + 5] += 1
            # M7: B00, B01
            V[b_idx[0]][start_col + 6] += 1
            V[b_idx[1]][start_col + 6] += 1
            
            # W (C) contributions:
            # C00: M5, M4, -M2, M6
            W[w_idx[0]][start_col + 4] += 1
            W[w_idx[0]][start_col + 3] += 1
            W[w_idx[0]][start_col + 1] -= 1
            W[w_idx[0]][start_col + 5] += 1
            # C01: M1, M2
            W[w_idx[1]][start_col + 0] += 1
            W[w_idx[1]][start_col + 1] += 1
            # C10: M3, M4
            W[w_idx[2]][start_col + 2] += 1
            W[w_idx[2]][start_col + 3] += 1
            # C11: M5, M1, -M3, -M7
            W[w_idx[3]][start_col + 4] += 1
            W[w_idx[3]][start_col + 0] += 1
            W[w_idx[3]][start_col + 2] -= 1
            W[w_idx[3]][start_col + 6] -= 1

        # We have 2x4x6. 
        # A is 2x4. B is 4x6.
        # We can split B into three 4x2 blocks? Or A into 2x2 blocks?
        # Let's stick to the 6 blocks of 2x2x2 used in the previous solution.
        # A rows: 0,1. A cols: 0,1 (block 0), 2,3 (block 1)
        # B rows: 0,1 (block 0), 2,3 (block 1)
        # B cols: 0,1 (block 0), 2,3 (block 1), 4,5 (block 2)
        
        # This forms a block matrix where C is 6x2.
        # C rows: 0,1 (from B cols 0,1), 2,3 (from B cols 2,3), 4,5 (from B cols 4,5)
        # C cols: 0,1 (A rows)
        
        # Block (0,0): A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
        a_idx_0 = [0, 1, 2, 3]
        b_idx_0 = [0, 1, 2, 3]
        w_idx_0 = [0, 1, 2, 3]
        add_strassen_block(a_idx_0, b_idx_0, w_idx_0, 0)
        
        # Block (0,1): A[0:2, 0:2] * B[0:2, 2:4] -> C[2:4, 0:2]
        # Note: B[0:2, 2:4] corresponds to b indices 2,3,6,7? 
        # B is 4x6. Row j, Col k.
        # Block 0: j in {0,1}, k in {0,1}. Indices: 0*6+0=0, 0*6+1=1, 1*6+0=6, 1*6+1=7.
        # Wait, my previous mapping in the prompt's code was:
        # b_1_T = [0, 1, 6, 7] (j=0,1; k=0,1)
        # b_1_B = [12, 13, 18, 19] (j=2,3; k=0,1)
        
        # Let's re-verify the indices for B (4x6).
        # b = j*p + k = j*6 + k.
        # j=0, k=0 -> 0
        # j=0, k=1 -> 1
        # j=1, k=0 -> 6
        # j=1, k=1 -> 7
        # j=2, k=0 -> 12
        # j=2, k=1 -> 13
        # j=3, k=0 -> 18
        # j=3, k=1 -> 19
        
        # So for the first block of B (rows 0,1; cols 0,1), indices are 0,1,6,7.
        # My add_strassen_block expects 4 indices for a 2x2 matrix.
        # The order in the function is:
        # b_idx[0] = 00 (j=0,k=0)
        # b_idx[1] = 01 (j=0,k=1)
        # b_idx[2] = 10 (j=1,k=0)
        # b_idx[3] = 11 (j=1,k=1)
        
        # So b_idx_0 should be [0, 1, 6, 7].
        
        # Let's restart the fallback construction carefully.
        
        U = [[0] * 42 for _ in range(n * m)]
        V = [[0] * 42 for _ in range(m * p)]
        W = [[0] * 42 for _ in range(p * n)]
        
        # We have 3 blocks of B columns (0-1, 2-3, 4-5).
        # And 2 blocks of A columns (0-1, 2-3).
        # Total 6 blocks. Each rank 7. Total 42.
        
        # Block 1: A_cols 0,1 (indices 0,1,4,5? No. A is 2x4. a = i*4+j.
        # i=0,j=0 -> 0
        # i=0,j=1 -> 1
        # i=1,j=0 -> 4
        # i=1,j=1 -> 5
        # So a_idx for A[0:2, 0:2] is [0, 1, 4, 5].
        
        # Block 2: A_cols 2,3.
        # i=0,j=2 -> 2
        # i=0,j=3 -> 3
        # i=1,j=2 -> 6
        # i=1,j=3 -> 7
        # So a_idx for A[0:2, 2:4] is [2, 3, 6, 7].
        
        # B Block 1: Rows 0,1; Cols 0,1.
        # b = j*6+k.
        # j=0,k=0 -> 0
        # j=0,k=1 -> 1
        # j=1,k=0 -> 6
        # j=1,k=1 -> 7
        # b_idx = [0, 1, 6, 7]
        
        # B Block 2: Rows 2,3; Cols 0,1.
        # j=2,k=0 -> 12
        # j=2,k=1 -> 13
        # j=3,k=0 -> 18
        # j=3,k=1 -> 19
        # b_idx = [12, 13, 18, 19]
        
        # C Block 1: Rows 0,1; Cols 0,1.
        # c = k*n+i = k*2+i.
        # k=0,i=0 -> 0
        # k=0,i=1 -> 1
        # k=1,i=0 -> 2
        # k=1,i=1 -> 3
        # w_idx = [0, 1, 2, 3]
        
        # C Block 2: Rows 2,3; Cols 0,1.
        # k=2,i=0 -> 4
        # k=2,i=1 -> 5
        # k=3,i=0 -> 6
        # k=3,i=1 -> 7
        # w_idx = [4, 5, 6, 7]
        
        # C Block 3: Rows 4,5; Cols 0,1.
        # k=4,i=0 -> 8
        # k=4,i=1 -> 9
        # k=5,i=0 -> 10
        # k=5,i=1 -> 11
        # w_idx = [8, 9, 10, 11]
        
        # Block (A1, B1) -> C1
        add_strassen_block([0, 1, 4, 5], [0, 1, 6, 7], [0, 1, 2, 3], 0)
        
        # Block (A2, B1) -> C1
        # A[0:2, 2:4] * B[0:2, 0:1]? No.
        # We need to multiply A[0:2, 2:4] (cols 2,3) with B[0:2, 0:1] (rows 0,1)?
        # No, the multiplication is sum_j A[i,j] B[j,k].
        # If we split A by columns and B by rows, the blocks must match.
        # A_block 1: cols 0,1. B_block 1: rows 0,1.
        # A_block 2: cols 2,3. B_block 2: rows 2,3.
        
        # So:
        # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] (k=0,1)
        # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2] (k=0,1)
        # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[2:4, 0:2] (k=2,3)
        # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[2:4, 0:2] (k=2,3)
        # Block 5: A[0:2, 0:2] * B[0:2, 4:6] -> C[4:6, 0:2] (k=4,5)
        # Block 6: A[0:2, 2:4] * B[2:4, 4:6] -> C[4:6, 0:2] (k=4,5)
        
        # Indices:
        # A1: [0, 1, 4, 5]
        # A2: [2, 3, 6, 7]
        
        # B1 (rows 0,1; cols 0,1): [0, 1, 6, 7]
        # B2 (rows 2,3; cols 0,1): [12, 13, 18, 19]
        # B3 (rows 0,1; cols 2,3): [2, 3, 8, 9]
        # B4 (rows 2,3; cols 2,3): [14, 15, 20, 21]
        # B5 (rows 0,1; cols 4,5): [4, 5, 10, 11]
        # B6 (rows 2,3; cols 4,5): [16, 17, 22, 23]
        
        # C1 (k=0,1): [0, 1, 2, 3]
        # C2 (k=2,3): [4, 5, 6, 7]
        # C3 (k=4,5): [8, 9, 10, 11]
        
        # Block 1: A1 * B1 -> C1
        add_strassen_block(A1, B1, C1, 0)
        
        # Block 2: A2 * B2 -> C1
        add_strassen_block(A2, B2, C1, 7)
        
        # Block 3: A1 * B3 -> C2
        add_strassen_block(A1, B3, C2, 14)
        
        # Block 4: A2 * B4 -> C2
        add_strassen_block(A2, B4, C2, 21)
        
        # Block 5: A1 * B5 -> C3
        add_strassen_block(A1, B5, C3, 28)
        
        # Block 6: A2 * B6 -> C3
        add_strassen_block(A2, B6, C3, 35)
        
        return U, V, W

    U_fb, V_fb, W_fb = get_fallback()
    
    start_time = time.time()
    deadline = 110.0
    
    # Search for lower ranks
    # Known border rank for 2x4x6 is 24.
    # We try ranks from 24 up to 41.
    
    for r in range(24, 42):
        if time.time() > deadline:
            break
            
        # Number of attempts depends on time remaining
        attempts = 10 if r < 30 else 5
        
        for attempt in range(attempts):
            if time.time() > deadline:
                break
                
            np.random.seed(r * 100 + attempt)
            
            # Initialize
            # Scale is important. Too small -> slow convergence. Too large -> overflow/instability.
            scale = 1.0
            U0 = np.random.randn(dim_A, r) * scale
            V0 = np.random.randn(dim_B, r) * scale
            W0 = np.random.randn(dim_C, r) * scale
            
            theta0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            try:
                # Use LM (Levenberg-Marquardt) for non-linear least squares
                # It doesn't use gradients, so it's robust for this kind of problem.
                res = so.least_squares(
                    lambda t: calculate_residual(t, r), 
                    theta0, 
                    method='lm', 
                    xtol=1e-8, 
                    ftol=1e-8, 
                    gtol=1e-8,
                    max_nfev=5000
                )
                
                if res.success and res.cost < 1e-6:
                    U_sol = res.x[:dim_A * r].reshape(dim_A, r)
                    V_sol = res.x[dim_A * r : dim_A * r + dim_B * r].reshape(dim_B, r)
                    W_sol = res.x[dim_A * r + dim_B * r :].reshape(dim_C, r)
                    
                    # Try to snap to half-integers
                    result = snap_and_verify(U_sol, V_sol, W_sol, r)
                    if result is not None:
                        return {"U": result[0].tolist(), "V": result[1].tolist(), "W": result[2].tolist()}
                    
                    # If snap failed, maybe we can use the real solution if the evaluator is lenient?
                    # The prompt says "integer/half-integer entries".
                    # However, sometimes "exactly decompose" implies the mathematical object,
                    # but the interface constraint is explicit.
                    # If we can't snap, we can't return this.
                    
            except Exception:
                pass

    # If no better rank found, return the known rank 42 solution
    return {"U": U_fb, "V": V_fb, "W": W_fb}
