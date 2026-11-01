"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm = n * m  # 4
    mp = m * p  # 8
    pn = p * n  # 8
    
    def check_decomposition(U, V, W, R):
        """Check if the decomposition is exact."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r] for r in range(R))
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Fallback: naive rank-16 decomposition
    def naive_decomp():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = naive_decomp()
    
    # Strassen-based rank-14 decomposition
    # Decompose 2x2 x 2x4 into two 2x2 x 2x2 Strassen blocks
    # Block 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # Block 2: A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    
    # Strassen for 2x2 x 2x2 has rank 7:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B10)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    def strassen_2x2():
        """Return U, V, W factors for 2x2 x 2x2 Strassen (rank 7)."""
        R = 7
        U = np.zeros((4, R))  # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        V = np.zeros((4, R))  # B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        W = np.zeros((4, R))  # C indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        
        # M1 = (A00+A11)*(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1  # A00, A11
        V[0, 0] = 1; V[3, 0] = 1  # B00, B11
        W[0, 0] = 1; W[1, 0] = 0; W[2, 0] = 0; W[3, 0] = 1  # C00, C11
        
        # M2 = (A10+A11)*B00
        U[2, 1] = 1; U[3, 1] = 1  # A10, A11
        V[0, 1] = 1  # B00
        W[2, 1] = 1; W[3, 1] = -1  # C10, C11
        
        # M3 = A00*(B10-B11)
        U[0, 2] = 1  # A00
        V[2, 2] = 1; V[3, 2] = -1  # B10, B11
        W[1, 2] = 1; W[3, 2] = 1  # C01, C11
        
        # M4 = A11*(B01-B10)
        U[3, 3] = 1  # A11
        V[1, 3] = 1; V[2, 3] = -1  # B01, B10
        W[0, 3] = 1; W[2, 3] = 1  # C00, C10
        
        # M5 = (A00+A01)*B11
        U[0, 4] = 1; U[1, 4] = 1  # A00, A01
        V[3, 4] = 1  # B11
        W[0, 4] = -1; W[1, 4] = 1  # C00, C01
        
        # M6 = (A10-A00)*(B00+B10)
        U[2, 5] = 1; U[0, 5] = -1  # A10, A00
        V[0, 5] = 1; V[2, 5] = 1  # B00, B10
        W[3, 5] = 1  # C11
        
        # M7 = (A01-A11)*(B01+B11)
        U[1, 6] = 1; U[3, 6] = -1  # A01, A11
        V[1, 6] = 1; V[3, 6] = 1  # B01, B11
        W[0, 6] = 1  # C00
        
        return U, V, W
    
    def build_strassen_2x2x4():
        """Build rank-14 decomposition using two Strassen blocks."""
        R = 14
        U = np.zeros((nm, R))  # 4 x 14
        V = np.zeros((mp, R))  # 8 x 14
        W = np.zeros((pn, R))  # 8 x 14
        
        # Get Strassen factors for 2x2 x 2x2
        U_s, V_s, W_s = strassen_2x2()
        
        # Block 1: A[:,0:2] * B[0:2,0:2] -> C[0:2,0:2]
        # A indices: (i,j) with j=0,1 -> rows 0,1,2,3 in U (i*m+j = i*2+j)
        # B indices: (j,k) with j=0,1 and k=0,1 -> rows 0,1,2,3 in V (j*p+k = j*4+k)
        # C indices: (k,i) with k=0,1 and i=0,1 -> rows 0,1,2,3 in W (k*n+i = k*2+i)
        
        # Copy Strassen factors for block 1
        # U: rows 0-3 (A00,A01,A10,A11) -> columns 0-6
        U[0:4, 0:7] = U_s
        # V: rows 0-3 (B00,B01,B10,B11) -> columns 0-6
        V[0:4, 0:7] = V_s
        # W: rows 0-3 (C00,C01,C10,C11) -> columns 0-6
        W[0:4, 0:7] = W_s
        
        # Block 2: A[:,0:2] * B[0:2,2:4] -> C[0:2,2:4]
        # A indices: same as block 1 -> rows 0-3 in U
        # B indices: (j,k) with j=0,1 and k=2,3 -> rows 2,3,6,7 in V (j*4+k)
        #   j=0,k=2 -> row 2; j=0,k=3 -> row 3; j=1,k=2 -> row 6; j=1,k=3 -> row 7
        # C indices: (k,i) with k=2,3 and i=0,1 -> rows 4,5,6,7 in W (k*2+i)
        #   k=2,i=0 -> row 4; k=2,i=1 -> row 5; k=3,i=0 -> row 6; k=3,i=1 -> row 7
        
        # Copy Strassen factors for block 2
        # U: same A indices -> columns 7-13
        U[0:4, 7:14] = U_s
        # V: B indices for k=2,3 -> rows 2,3,6,7
        V[2:4, 7:14] = V_s[0:2]   # j=0, k=2,3
        V[6:8, 7:14] = V_s[2:4]   # j=1, k=2,3
        # W: C indices for k=2,3 -> rows 4,5,6,7
        W[4:6, 7:14] = W_s[0:2]   # k=2, i=0,1
        W[6:8, 7:14] = W_s[2:4]   # k=3, i=0,1
        
        return U, V, W
    
    # Try Strassen rank-14 first
    if time.time() < deadline:
        U_s14, V_s14, W_s14 = build_strassen_2x2x4()
        if check_decomposition(U_s14.tolist(), V_s14.tolist(), W_s14.tolist(), 14):
            best_result = {"U": U_s14.tolist(), "V": V_s14.tolist(), "W": W_s14.tolist()}
            # Try to find rank 13 or lower
            found_better = False
            for R in range(8, 14):
                if time.time() > deadline:
                    break
                
                # Try random search for rank R
                np.random.seed(R * 100 + 42)
                best_loss = float('inf')
                best_state = None
                lr = 0.005
                
                for trial in range(50):
                    if time.time() > deadline:
                        break
                    
                    U = np.random.uniform(-1, 1, (nm, R))
                    V = np.random.uniform(-1, 1, (mp, R))
                    W = np.random.uniform(-1, 1, (pn, R))
                    
                    for it in range(2000):
                        if time.time() > deadline:
                            break
                        
                        # Compute residuals
                        residuals = np.zeros((n, m, p))
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(R))
                                    residuals[i,j,k] = val - 1.0
                        
                        loss = np.sum(residuals**2)
                        if loss < best_loss:
                            best_loss = loss
                            best_state = (U.copy(), V.copy(), W.copy())
                        
                        if loss < 1e-12:
                            break
                        
                        # Gradient
                        dU = np.zeros_like(U)
                        dV = np.zeros_like(V)
                        dW = np.zeros_like(W)
                        
                        for i in range(n):
                            for j in range(m):
                                for r in range(R):
                                    grad_u = 0.0
                                    for k in range(p):
                                        grad_u += residuals[i,j,k] * V[j*p+k, r] * W[k*n+i, r]
                                    dU[i*m+j, r] = 2 * grad_u
                                    
                                    for k in range(p):
                                        idx_v = j*p + k
                                        idx_w = k*n + i
                                        dV[idx_v, r] += 2 * residuals[i,j,k] * U[i*m+j, r] * W[idx_w, r]
                                        dW[idx_w, r] += 2 * residuals[i,j,k] * U[i*m+j, r] * V[idx_v, r]
                        
                        U -= lr * dU
                        V -= lr * dV
                        W -= lr * dW
                        lr *= 0.999  # decay learning rate
                    
                    if best_loss < 1e-10:
                        break
                
                if best_state is not None and best_loss < 1e-10:
                    U_best, V_best, W_best = best_state
                    # Round to nearest half-integer
                    U_rounded = np.round(U_best * 2) / 2
                    V_rounded = np.round(V_best * 2) / 2
                    W_rounded = np.round(W_best * 2) / 2
                    
                    if check_decomposition(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), R):
                        best_result = {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                        found_better = True
                        break
                    else:
                        # Try integer rounding
                        U_int = np.round(U_best)
                        V_int = np.round(V_best)
                        W_int = np.round(W_best)
                        if check_decomposition(U_int.tolist(), V_int.tolist(), W_int.tolist(), R):
                            best_result = {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                            found_better = True
                            break
    
    return best_result
# EVOLVE-BLOCK-END
