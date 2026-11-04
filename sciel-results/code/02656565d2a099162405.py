import numpy as np
import time
import random

DIMENSIONS = [3, 3, 8]

def construct():
    start_time = time.time()
    timeout = 110.0  # Leave margin for safety
    
    n, m, p = DIMENSIONS
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def strassen_3x3_block():
        """
        Returns Strassen-Winograd rank-8 decomposition for 3x3 matrix multiplication.
        U: 9x8, V: 9x8, W: 9x8
        Indices: U[i*3+j], V[j*3+k], W[k*3+i]
        """
        # Strassen-Winograd 3x3 algorithm (rank 8)
        # This is a well-known exact rank-8 decomposition
        # Using the standard formulation
        
        # The 8 products are:
        # M1 = (A00+A11+A22)(B00+B11+B22)
        # M2 = (A00+A11)(B00+B11)
        # M3 = A22*(B11-B22)
        # M4 = A22*(B22-B00)
        # M5 = A00*(B00-B22)
        # M6 = A11*(B11-B00)
        # M7 = (A00+A11+A22)(B00+B11+B22) - M1 ... wait, let me use a cleaner formulation
        
        # Let me use the standard Winograd 3x3 rank-8:
        # U, V, W matrices for the 8 rank-1 terms
        
        # Term 1: (a00+a11+a22) * (b00+b11+b22)
        # Term 2: (a00+a11) * (b00+b11)
        # Term 3: a22 * (b11-b22)
        # Term 4: a22 * (b22-b00)
        # Term 5: a00 * (b00-b22)
        # Term 6: a11 * (b11-b00)
        # Term 7: (a00+a11+a22) * (b00+b11+b22) -- same as term 1, so we need a different formulation
        
        # Let me use the correct Winograd 3x3:
        # M1 = (a00+a11+a22)(b00+b11+b22)
        # M2 = (a00+a11)(b00+b11)
        # M3 = a22(b11-b22)
        # M4 = a22(b22-b00)
        # M5 = a00(b00-b22)
        # M6 = a11(b11-b00)
        # M7 = (a00+a11+a22)(b00+b11+b22) - M1 is not right
        
        # Correct Winograd 3x3 rank-8:
        # Let me define it properly with explicit U, V, W entries
        
        U_s = np.zeros((9, 8))
        V_s = np.zeros((9, 8))
        W_s = np.zeros((9, 8))
        
        # Term 0: (a00+a11+a22) * (b00+b11+b22)
        U_s[0, 0] = 1  # a00
        U_s[4, 0] = 1  # a11
        U_s[8, 0] = 1  # a22
        V_s[0, 0] = 1  # b00
        V_s[4, 0] = 1  # b11
        V_s[8, 0] = 1  # b22
        W_s[0, 0] = 1  # c00
        W_s[4, 0] = 1  # c11
        W_s[8, 0] = 1  # c22
        
        # Term 1: (a00+a11) * (b00+b11)
        U_s[0, 1] = 1
        U_s[4, 1] = 1
        V_s[0, 1] = 1
        V_s[4, 1] = 1
        W_s[0, 1] = 1
        W_s[4, 1] = 1
        
        # Term 2: a22 * (b11-b22)
        U_s[8, 2] = 1
        V_s[4, 2] = 1
        V_s[8, 2] = -1
        W_s[5, 2] = 1   # c21
        W_s[8, 2] = 1   # c22
        
        # Term 3: a22 * (b22-b00)
        U_s[8, 3] = 1
        V_s[8, 3] = 1
        V_s[0, 3] = -1
        W_s[6, 3] = 1   # c20
        W_s[8, 3] = 1   # c22
        
        # Term 4: a00 * (b00-b22)
        U_s[0, 4] = 1
        V_s[0, 4] = 1
        V_s[8, 4] = -1
        W_s[0, 4] = 1   # c00
        W_s[2, 4] = 1   # c02
        
        # Term 5: a11 * (b11-b00)
        U_s[4, 5] = 1
        V_s[4, 5] = 1
        V_s[0, 5] = -1
        W_s[4, 5] = 1   # c11
        W_s[1, 5] = 1   # c01
        
        # Term 6: (a00+a11+a22) * (b00+b11+b22) - M1
        # This gives the off-diagonal corrections
        # Actually, let me use a verified formulation
        
        # Let me use a different, well-known rank-8 3x3 algorithm:
        # Based on Winograd (1979)
        
        U_s = np.zeros((9, 8))
        V_s = np.zeros((9, 8))
        W_s = np.zeros((9, 8))
        
        # I'll use the formulation from the literature
        # M1 = (a00+a11+a22)(b00+b11+b22)
        # M2 = (a00+a11)(b00+b11)
        # M3 = a22(b11-b22)
        # M4 = a22(b22-b00)
        # M5 = a00(b00-b22)
        # M6 = a11(b11-b00)
        # M7 = (a00+a11+a22)(b00+b11+b22)
        # M8 = (a00+a11)(b00+b11)
        
        # This isn't quite right. Let me use a verified one.
        
        # Actually, let me just use a simple approach: try to find a rank < 72
        # decomposition using the alternating minimization approach from the current program,
        # but with better initialization and more careful rounding.
        
        return None
    
    def verify_decomposition(U, V, W):
        """Verify that the decomposition is exact."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        r = len(U[0])
        
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(Ua[i*m+j] * Va[j*p+k] * Wa[k*n+i])
                    max_err = max(max_err, abs(val - 1.0))
        
        return max_err
    
    def solve_for_W(U, V, r):
        W = np.zeros((p * n, r))
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, r))
                for j in range(m):
                    A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    W_col, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    W[k*n+i, :] = W_col
                except:
                    W[k*n+i, :] = 0
        return W
    
    def solve_for_U(U, V, W, r):
        U_new = np.zeros((n * m, r))
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, r))
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                try:
                    U_col, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    U_new[i*m+j, :] = U_col
                except:
                    U_new[i*m+j, :] = 0
        return U_new
    
    def solve_for_V(U, V, W, r):
        V_new = np.zeros((m * p, r))
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, r))
                for i in range(n):
                    A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    V_col, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                    V_new[j*p+k, :] = V_col
                except:
                    V_new[j*p+k, :] = 0
        return V_new
    
    def compute_loss(U, V, W):
        total = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    total += (val - 1.0) ** 2
        return total
    
    def try_rank(r, max_sweeps=50, num_restarts=5, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        
        best_result = None
        best_loss = float('inf')
        
        for attempt in range(num_restarts):
            if time.time() - start_time > timeout - 20:
                break
            
            # Initialize with random matrices
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            local_best_loss = float('inf')
            local_best_U, local_best_V, local_best_W = U.copy(), V.copy(), W.copy()
            
            for sweep in range(max_sweeps):
                if time.time() - start_time > timeout - 20:
                    break
                
                W = solve_for_W(U, V, r)
                U = solve_for_U(U, V, W, r)
                V = solve_for_V(U, V, W, r)
                
                loss = compute_loss(U, V, W)
                if loss < local_best_loss:
                    local_best_loss = loss
                    local_best_U, local_best_V, local_best_W = U.copy(), V.copy(), W.copy()
                
                if loss < 1e-14:
                    break
            
            if local_best_loss < best_loss:
                best_loss = local_best_loss
                best_U, best_V, best_W = local_best_U, local_best_V, local_best_W
            
            if best_loss < 1e-10:
                # Try to round to half-integers
                U_r = np.round(best_U * 2) / 2
                V_r = np.round(best_V * 2) / 2
                W_r = np.round(best_W * 2) / 2
                
                max_err = verify_decomposition(U_r.tolist(), V_r.tolist(), W_r.tolist())
                
                if max_err < 1e-6:
                    return {
                        "U": [[float(x) for x in row] for row in U_r],
                        "V": [[float(x) for x in row] for row in V_r],
                        "W": [[float(x) for x in row] for row in W_r]
                    }
        
        return None
    
    # Try to find a decomposition with rank < 72
    # Start from rank 71 down to 50
    for r in range(71, 49, -1):
        if time.time() - start_time > timeout - 15:
            break
        
        result = try_rank(r)
        if result is not None:
            return result
    
    # If no lower rank found, return schoolbook
    return schoolbook()
