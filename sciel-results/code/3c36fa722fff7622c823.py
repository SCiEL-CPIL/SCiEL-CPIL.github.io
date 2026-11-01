"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin for 120s limit
    
    size_a = n * m  # 4
    size_b = m * p  # 28
    size_c = p * n  # 28
    
    # Build the true tensor T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T_true = np.zeros((size_a, size_b, size_c), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_true[a, b, c] = 1.0

    def verify(U, V, W):
        """Verify that sum_r U[a,r]*V[b,r]*W[c,r] == T_true[a,b,c] for all a,b,c."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr, optimize=True)
        return np.allclose(T_recon, T_true, atol=1e-9)

    def strassen_decomposition():
        """
        Apply Strassen's algorithm to 2x2 blocks of the 2x14 matrix.
        14 columns = 7 blocks of 2x2.
        Each 2x2 block multiplication uses rank 7 (Strassen).
        Total rank = 7 * 7 = 49.
        """
        rank = 7 * 7  # 49
        U = np.zeros((size_a, rank), dtype=np.float64)
        V = np.zeros((size_b, rank), dtype=np.float64)
        W = np.zeros((size_c, rank), dtype=np.float64)
        
        # Indices for A (U):
        # A11: 0, A12: 1, A21: 2, A22: 3
        
        # Indices for B (V) for block b:
        # B11: 2*b, B12: 2*b+1
        # B21: 14+2*b, B22: 14+2*b+1
        
        # Indices for C (W) for block b:
        # C11: 4*b, C12: 4*b+2
        # C21: 4*b+1, C22: 4*b+3
        
        for b in range(7):
            for t in range(7):
                r = b * 7 + t
                
                if t == 0:  # M0 = (A11 - A22)(B12 + B22)
                    U[0, r] += 1
                    U[3, r] += -1
                    V[2*b+1, r] += 1
                    V[14+2*b+1, r] += 1
                    W[4*b, r] += 1
                    W[4*b+3, r] += 1
                elif t == 1:  # M1 = (A21 + A22)B11
                    U[2, r] += 1
                    U[3, r] += 1
                    V[2*b, r] += 1
                    W[4*b+1, r] += 1
                    W[4*b+3, r] += -1
                elif t == 2:  # M2 = A11(B11 - B21)
                    U[0, r] += 1
                    V[2*b, r] += 1
                    V[14+2*b, r] += -1
                    W[4*b+2, r] += 1
                    W[4*b+3, r] += 1
                elif t == 3:  # M3 = A22(B22 - B12)
                    U[3, r] += 1
                    V[14+2*b+1, r] += 1
                    V[2*b+1, r] += -1
                    W[4*b, r] += 1
                    W[4*b+1, r] += 1
                elif t == 4:  # M4 = (A11 + A12)B22
                    U[0, r] += 1
                    U[1, r] += 1
                    V[14+2*b+1, r] += 1
                    W[4*b, r] += -1
                    W[4*b+2, r] += 1
                elif t == 5:  # M5 = (A21 - A11)(B11 + B12)
                    U[2, r] += 1
                    U[0, r] += -1
                    V[2*b, r] += 1
                    V[2*b+1, r] += 1
                    W[4*b+3, r] += 1
                elif t == 6:  # M6 = (A12 - A22)(B21 + B22)
                    U[1, r] += 1
                    U[3, r] += -1
                    V[14+2*b, r] += 1
                    V[14+2*b+1, r] += 1
                    W[4*b, r] += 1
        
        # Verify the decomposition
        if verify(U, V, W):
            return U.tolist(), V.tolist(), W.tolist()
        
        return None

    # Try Strassen-based decomposition (rank 49)
    result = strassen_decomposition()
    if result:
        return {"U": result[0], "V": result[1], "W": result[2]}
    
    # Fallback to Schoolbook Rank 56
    rank = n * m * p
    U = np.zeros((size_a, rank), dtype=int)
    V = np.zeros((size_b, rank), dtype=int)
    W = np.zeros((size_c, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
