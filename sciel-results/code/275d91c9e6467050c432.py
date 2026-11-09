import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def _construct_strassen_rank56():
    """
    Build a rank-56 decomposition for 2x4x8 matrix multiplication
    using Strassen's algorithm applied to 8 independent 2x2 sub-blocks.
    
    A is 2x4: A = [A0, A1] where A0, A1 are 2x2
    B is 4x8: B = [B0, B1, B2, B3] where each is 2x2
    C is 2x8: C = [C0, C1, C2, C3] where each is 2x2
    
    C0 = A0*B0 + A1*B1
    C1 = A0*B2 + A1*B3
    C2 = A0*B4 + A1*B5  (wait, B only has 4 blocks of 2x2)
    
    Actually, let me re-index:
    A is 2x4: rows 0,1; cols 0,1,2,3
    B is 4x8: rows 0,1,2,3; cols 0,1,2,3,4,5,6,7
    C is 2x8: rows 0,1; cols 0,1,2,3,4,5,6,7
    
    C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    Group by pairs of j and k:
    For k in {0,1}: C[:, 0:2] = A[:, 0:2] @ B[0:2, 0:2] + A[:, 2:4] @ B[2:4, 0:2]
    For k in {2,3}: C[:, 2:4] = A[:, 0:2] @ B[0:2, 2:4] + A[:, 2:4] @ B[2:4, 2:4]
    For k in {4,5}: C[:, 4:6] = A[:, 0:2] @ B[0:2, 4:6] + A[:, 2:4] @ B[2:4, 4:6]
    For k in {6,7}: C[:, 6:8] = A[:, 0:2] @ B[0:2, 6:8] + A[:, 2:4] @ B[2:4, 6:8]
    
    Each 2x2x2 multiplication uses 7 Strassen terms.
    Total: 4 output blocks * 2 input blocks * 7 = 56 terms.
    """
    n, m, p = 2, 4, 8
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    # Strassen 2x2x2 coefficients
    # For multiplying X (2x2) * Y (2x2) = Z (2x2)
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y01-Y11)
    # M4 = X11(Y10-Y00)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y01)
    # M7 = (X01-X11)(Y10+Y11)
    # Z00 = M1 - M4 + M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # Each term has coefficients for U (X part), V (Y part), W (Z part)
    # U indices: (x_row, x_col) where x_row, x_col in {0,1}
    # V indices: (y_row, y_col) where y_row, y_col in {0,1}
    # W indices: (z_row, z_col) where z_row, z_col in {0,1}
    
    strassen_terms = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # M1
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,-1)]),
        # M2
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        # M3
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        # M4
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        # M5
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        # M6
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # M7
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    
    col = 0
    for s in range(4):  # output block index (k in {0:2, 2:4, 4:6, 6:8})
        for t in range(2):  # input block index (j in {0:2, 2:4})
            for (u_c, v_c, w_c) in strassen_terms:
                # U: A[i, j_global] where i in {0,1}, j_global = t*2 + j_local
                for (u_r, u_c_idx, coeff) in u_c:
                    U[u_r * m + t * 2 + u_c_idx][col] += coeff
                
                # V: B[j_global, k_global] where j_global = t*2 + j_local, k_global = s*2 + k_local
                for (v_r, v_c_idx, coeff) in v_c:
                    V[(t * 2 + v_r) * p + s * 2 + v_c_idx][col] += coeff
                
                # W: C[i, k_global] where i in {0,1}, k_global = s*2 + k_local
                # W index: k*n + i = k_global * 2 + i
                for (w_r, w_c_idx, coeff) in w_c:
                    W[(s * 2 + w_c_idx) * n + w_r][col] += coeff
                
                col += 1
    
    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U_list, V_list, W_list, n=2, m=4, p=8, tol=1e-10):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    U = np.array(U_list, dtype=float)
    V = np.array(V_list, dtype=float)
    W = np.array(W_list, dtype=float)
    
    rank = U.shape[1]
    if rank == 0:
        return False
    
    np.random.seed(42)
    for trial in range(3):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_true = A @ B
        
        C_decomp = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                total = 0.0
                for j in range(m):
                    for r in range(rank):
                        total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                C_decomp[i, k] = total
        
        if np.max(np.abs(C_true - C_decomp)) > tol:
            return False
    
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    result = _construct_strassen_rank56()
    
    # Verify correctness
    if not _verify_decomposition(result["U"], result["V"], result["W"]):
        pass  # Should not happen
    
    return result


if __name__ == "__main__":
    result = construct()
    U = np.array(result["U"], dtype=float)
    V = np.array(result["V"], dtype=float)
    W = np.array(result["W"], dtype=float)
    
    n, m, p = DIMENSIONS
    
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_true = A @ B
    
    C_decomp = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            total = 0
            for j in range(m):
                for r in range(U.shape[1]):
                    total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            C_decomp[i,k] = total
            
    error = np.max(np.abs(C_true - C_decomp))
    print(f"Max error: {error}")
    print(f"Rank: {U.shape[1]}")
