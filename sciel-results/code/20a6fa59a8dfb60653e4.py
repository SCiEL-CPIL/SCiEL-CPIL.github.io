import numpy as np
import time
from scipy.optimize import least_squares

def construct():
    n, m, p = 2, 2, 7
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for return and overhead
    
    # Baseline: Rank 25 solution
    # We will try to find a Rank 24 solution first.
    # Structure:
    # Cols 0,1: Strassen (Rank 7)
    # Cols 2,3: Strassen (Rank 7)
    # Cols 4,5,6: 2x2x3 Block (Target Rank 10)
    # Total Rank: 7 + 7 + 10 = 24
    
    def get_strassen_block(k_start, col_start):
        """
        Returns U, V, W contributions for a 2x2x2 Strassen block.
        k_start: starting column index in B (and C)
        col_start: starting column index in the rank dimension
        """
        U_block = np.zeros((4, 7))
        V_block = np.zeros((14, 7))
        W_block = np.zeros((14, 7))
        
        # Strassen coefficients
        # U (A) coeffs: [A00, A01, A10, A11]
        strassen_U = [
            [1, 0, 0, 1],  # M1: A00 + A11
            [0, 0, 1, 1],  # M2: A10 + A11
            [1, 0, 0, 0],  # M3: A00
            [0, 0, 0, 1],  # M4: A11
            [1, 1, 0, 0],  # M5: A00 + A01
            [-1, 0, 1, 0], # M6: -A00 + A10
            [0, 1, 0, -1]  # M7: A01 - A11
        ]
        
        # V (B) coeffs: [B00, B01, B10, B11]
        # Mapping to global V index: V[j*p + k]
        # For block b, columns are 2b, 2b+1.
        # B00 -> j=0, k=2b -> V[0*p + 2b]
        # B01 -> j=0, k=2b+1 -> V[0*p + 2b+1]
        # B10 -> j=1, k=2b -> V[1*p + 2b]
        # B11 -> j=1, k=2b+1 -> V[1*p + 2b+1]
        strassen_V_local = [
            {0: 1, 3: 1},   # M1: B00 + B11
            {0: 1},         # M2: B00
            {1: 1, 3: -1},  # M3: B01 - B11
            {2: 1, 0: -1},  # M4: B10 - B00
            {3: 1},         # M5: B11
            {0: 1, 1: 1},   # M6: B00 + B01
            {2: 1, 3: 1}    # M7: B10 + B11
        ]
        
        # W (C) coeffs: [C00, C01, C10, C11]
        # Mapping to global W index: W[k*n + i]
        # For block b, columns k are 2b, 2b+1.
        # C00 -> k=2b, i=0 -> W[2b*n + 0]
        # C01 -> k=2b+1, i=0 -> W[(2b+1)*n + 0]
        # C10 -> k=2b, i=1 -> W[2b*n + 1]
        # C11 -> k=2b+1, i=1 -> W[(2b+1)*n + 1]
        strassen_W_local = [
            {0: 1, 3: 1},   # M1: C00 + C11
            {2: 1, 3: -1},  # M2: C10 - C11
            {1: 1, 3: 1},   # M3: C01 + C11
            {0: 1, 2: 1},   # M4: C00 + C10
            {0: -1, 1: 1},  # M5: -C00 + C01
            {3: 1},         # M6: C11
            {0: 1}          # M7: C00
        ]
        
        for r in range(7):
            for i in range(4):
                U_block[i, r] = strassen_U[r][i]
            
            for loc_idx, coeff in strassen_V_local[r].items():
                if loc_idx == 0:
                    v_idx = 0 * p + (2 * k_start)
                elif loc_idx == 1:
                    v_idx = 0 * p + (2 * k_start + 1)
                elif loc_idx == 2:
                    v_idx = 1 * p + (2 * k_start)
                elif loc_idx == 3:
                    v_idx = 1 * p + (2 * k_start + 1)
                V_block[v_idx, r] = coeff
            
            for loc_idx, coeff in strassen_W_local[r].items():
                if loc_idx == 0:
                    w_idx = (2 * k_start) * n + 0
                elif loc_idx == 1:
                    w_idx = (2 * k_start + 1) * n + 0
                elif loc_idx == 2:
                    w_idx = (2 * k_start) * n + 1
                elif loc_idx == 3:
                    w_idx = (2 * k_start + 1) * n + 1
                W_block[w_idx, r] = coeff
        
        return U_block, V_block, W_block

    def find_2x2x3_rank10():
        """
        Attempts to find a Rank 10 decomposition for 2x2x3 matrix multiplication.
        Returns U, V, W (shapes 4x10, 6x10, 6x10) or None if not found.
        """
        if time.time() > deadline - 20:
            return None
            
        R = 10
        # Variables: U (4x10), V (6x10), W (6x10)
        # Total 16*R = 160 variables
        
        def residuals(x):
            U = x[:4*R].reshape(4, R)
            V = x[4*R:4*R+6*R].reshape(6, R)
            W = x[4*R+6*R:].reshape(6, R)
            
            res = np.zeros(12) # 2*2*3 = 12 terms
            idx = 0
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        # Tensor entry T[i, j, k] should be 1 if it's a valid multiplication term?
                        # No, the tensor is defined such that sum U[i,r] V[j,r] W[k,r] = 1 for the "diagonal"
                        # In the context of the full problem, the tensor is 1 for (i,j,k) corresponding to A[i,j] B[j,k] -> C[i,k]
                        # Here i is row of A, j is col of A/row of B, k is col of B/col of C.
                        # So for every i,j,k, the term exists.
                        s = 0.0
                        for r in range(R):
                            s += U[i, r] * V[j*3+k, r] * W[k*2+i, r]
                        res[idx] = s - 1.0
                        idx += 1
            return res

        best_res = float('inf')
        best_x = None
        
        # Try a few random initializations
        for attempt in range(10):
            if time.time() > deadline - 20:
                break
            np.random.seed(attempt * 42)
            x0 = np.random.randn(16*R) * 0.5
            
            try:
                sol = least_squares(residuals, x0, method='lm', max_nfev=2000)
                if sol.cost < best_res:
                    best_res = sol.cost
                    best_x = sol.x
                    # Early stop if cost is very low
                    if sol.cost < 1e-8:
                        break
            except Exception:
                pass
        
        if best_x is not None and best_res < 1e-6:
            U = best_x[:4*R].reshape(4, R)
            V = best_x[4*R:4*R+6*R].reshape(6, R)
            W = best_x[4*R+6*R:].reshape(6, R)
            
            # Try to round to half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Verify rounded solution
            valid = True
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        s = 0.0
                        for r in range(R):
                            s += U_rounded[i, r] * V_rounded[j*3+k, r] * W_rounded[k*2+i, r]
                        if abs(s - 1.0) > 1e-6:
                            valid = False
                            break
                    if not valid: break
                if not valid: break
            
            if valid:
                return U_rounded, V_rounded, W_rounded
        
        return None

    # --- Attempt Rank 24 ---
    U_24 = np.zeros((4, 24))
    V_24 = np.zeros((14, 24))
    W_24 = np.zeros((14, 24))
    
    # Block 1: Cols 0,1 (Strassen)
    U_b1, V_b1, W_b1 = get_strassen_block(0, 0)
    U_24[:, 0:7] += U_b1
    V_24[:, 0:7] += V_b1
    W_24[:, 0:7] += W_b1
    
    # Block 2: Cols 2,3 (Strassen)
    U_b2, V_b2, W_b2 = get_strassen_block(1, 7)
    U_24[:, 7:14] += U_b2
    V_24[:, 7:14] += V_b2
    W_24[:, 7:14] += W_b2
    
    # Block 3: Cols 4,5,6 (2x2x3 Rank 10)
    block3 = find_2x2x3_rank10()
    
    if block3 is not None:
        U_b3, V_b3, W_b3 = block3
        U_24[:, 14:24] += U_b3
        # Map V_b3 (6x10) to V_24 (14x24)
        # V_b3 indices: j*3+k. j in 0,1. k in 0,1,2.
        # Global V indices: j*p + k_global.
        # For this block, k_global is 4, 5, 6.
        # So V_b3[j, k_local] maps to V_24[j*7 + (4+k_local), 14:24]
        for j in range(2):
            for k_loc in range(3):
                k_glob = 4 + k_loc
                V_24[j*7 + k_glob, 14:24] += V_b3[j*3 + k_loc, :]
        
        # Map W_b3 (6x10) to W_24 (14x24)
        # W_b3 indices: k*2+i. k in 0,1,2. i in 0,1.
        # Global W indices: k_global*n + i.
        # For this block, k_global is 4, 5, 6.
        # So W_b3[k_loc, i] maps to W_24[(4+k_loc)*2 + i, 14:24]
        for k_loc in range(3):
            for i in range(2):
                k_glob = 4 + k_loc
                W_24[k_glob*2 + i, 14:24] += W_b3[k_loc*2 + i, :]
        
        # Final verification of the full Rank 24 solution
        if time.time() < deadline - 5:
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a_idx = i * m + j
                        b_idx = j * p + k
                        c_idx = k * n + i
                        val = np.dot(U_24[a_idx, :], V_24[b_idx, :] * W_24[c_idx, :])
                        if abs(val - 1.0) > max_err:
                            max_err = abs(val - 1.0)
            # Check zeros
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for j2 in range(m):
                            if j2 != j:
                                a_idx = i * m + j
                                b_idx = j2 * p + k
                                c_idx = k * n + i
                                val = np.dot(U_24[a_idx, :], V_24[b_idx, :] * W_24[c_idx, :])
                                if abs(val) > max_err:
                                    max_err = abs(val)
            
            if max_err < 1e-6:
                return {"U": U_24.tolist(), "V": V_24.tolist(), "W": W_24.tolist()}
    
    # --- Fallback to Rank 25 ---
    U_25 = np.zeros((4, 25))
    V_25 = np.zeros((14, 25))
    W_25 = np.zeros((14, 25))
    
    # Block 1: Cols 0,1
    U_b1, V_b1, W_b1 = get_strassen_block(0, 0)
    U_25[:, 0:7] += U_b1
    V_25[:, 0:7] += V_b1
    W_25[:, 0:7] += W_b1
    
    # Block 2: Cols 2,3
    U_b2, V_b2, W_b2 = get_strassen_block(1, 7)
    U_25[:, 7:14] += U_b2
    V_25[:, 7:14] += V_b2
    W_25[:, 7:14] += W_b2
    
    # Block 3: Cols 4,5
    U_b3, V_b3, W_b3 = get_strassen_block(2, 14)
    U_25[:, 14:21] += U_b3
    V_25[:, 14:21] += V_b3
    W_25[:, 14:21] += W_b3
    
    # Block 4: Col 6 (Schoolbook 2x2x1)
    # U: A00, A01, A10, A11
    U_25[0, 21] = 1
    U_25[1, 22] = 1
    U_25[2, 23] = 1
    U_25[3, 24] = 1
    
    # V: B[0,6], B[1,6]
    V_25[0*7 + 6, 21] = 1
    V_25[1*7 + 6, 22] = 1
    V_25[0*7 + 6, 23] = 1
    V_25[1*7 + 6, 24] = 1
    
    # W: C[0,6], C[1,6]
    W_25[6*2 + 0, 21] = 1
    W_25[6*2 + 0, 22] = 1
    W_25[6*2 + 1, 23] = 1
    W_25[6*2 + 1, 24] = 1
    
    return {"U": U_25.tolist(), "V": V_25.tolist(), "W": W_25.tolist()}

# EVOLVE-BLOCK-END
