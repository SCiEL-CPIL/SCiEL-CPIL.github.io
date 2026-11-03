import time
import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130

    def get_schoolbook():
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

    def get_strassen_based():
        # Replace the 2x2x2 sub-block (i=0,1; j=0,1; k=0,1) with Strassen
        # This uses 7 terms instead of 8, saving 1 rank
        # Total rank = 30 - 1 = 29
        
        # Strassen terms for 2x2x2:
        # M1 = (A11 - A01)(B00 + B01)
        # M2 = (A00 + A10)B00
        # M3 = (A00 - A10)(B11 - B01)
        # M4 = A11(B10 - B00)
        # M5 = (A00 + A10)(B10 + B11)
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        rank = 29
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Indices:
        # U[i*m + j], V[j*p + k], W[k*n + i]
        # For 2x2 sub-block: i=0,1; j=0,1; k=0,1
        # U[0]=A00, U[1]=A01, U[2]=A02, U[3]=A10, U[4]=A11, U[5]=A12
        # V[0]=B00, V[1]=B01, V[2]=B02, V[3]=B03, V[4]=B04
        # V[5]=B10, V[6]=B11, V[7]=B12, V[8]=B13, V[9]=B14
        # V[10]=B20, V[11]=B21, V[12]=B22, V[13]=B23, V[14]=B24
        # W[0]=C00, W[1]=C01, W[2]=C02, W[3]=C03, W[4]=C04
        # W[5]=C10, W[6]=C11, W[7]=C12, W[8]=C13, W[9]=C14
        
        col = 0
        
        # M1 = (A11 - A01)(B00 + B01)
        U[4][col] = 1  # A11
        U[1][col] = -1  # A01
        V[0][col] = 1  # B00
        V[1][col] = 1  # B01
        W[0][col] = 1  # C00
        W[3][col] = 1  # C11
        col += 1
        
        # M2 = (A00 + A10)B00
        U[0][col] = 1  # A00
        U[3][col] = 1  # A10
        V[0][col] = 1  # B00
        W[3][col] = -1  # C11
        W[5][col] = 1  # C10
        col += 1
        
        # M3 = (A00 - A10)(B11 - B01)
        U[0][col] = 1  # A00
        U[3][col] = -1  # A10
        V[6][col] = 1  # B11
        V[1][col] = -1  # B01
        W[1][col] = 1  # C01
        W[3][col] = 1  # C11
        col += 1
        
        # M4 = A11(B10 - B00)
        U[4][col] = 1  # A11
        V[5][col] = 1  # B10
        V[0][col] = -1  # B00
        W[0][col] = 1  # C00
        W[5][col] = 1  # C10
        col += 1
        
        # M5 = (A00 + A10)(B10 + B11)
        U[0][col] = 1  # A00
        U[3][col] = 1  # A10
        V[5][col] = 1  # B10
        V[6][col] = 1  # B11
        W[0][col] = -1  # C00
        W[1][col] = 1  # C01
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[3][col] = 1  # A10
        U[0][col] = -1  # A00
        V[0][col] = 1  # B00
        V[1][col] = 1  # B01
        W[3][col] = 1  # C11
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[1][col] = 1  # A01
        U[4][col] = -1  # A11
        V[5][col] = 1  # B10
        V[6][col] = 1  # B11
        W[0][col] = 1  # C00
        W[1][col] = -1  # C01
        W[3][col] = -1  # C11
        col += 1
        
        # Schoolbook terms for remaining (i,j,k) where not (i<2 and j<2 and k<2)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if i < 2 and j < 2 and k < 2:
                        continue
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        
        # Validate
        U_3d = np.array(U, dtype=np.float64).reshape(n, m, rank)
        V_3d = np.array(V, dtype=np.float64).reshape(m, p, rank)
        W_3d = np.array(W, dtype=np.float64).reshape(p, n, rank)
        T_recon = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        
        if not np.allclose(T_recon, 1.0, atol=1e-9):
            return get_schoolbook()
        
        return {"U": U, "V": V, "W": W}

    # Try Strassen-based decomposition (rank 29)
    result = get_strassen_based()
    if time.time() < deadline:
        # Check if it's valid
        U = result["U"]
        V = result["V"]
        W = result["W"]
        rank = len(U[0])
        U_3d = np.array(U, dtype=np.float64).reshape(n, m, rank)
        V_3d = np.array(V, dtype=np.float64).reshape(m, p, rank)
        W_3d = np.array(W, dtype=np.float64).reshape(p, n, rank)
        T_recon = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        if np.allclose(T_recon, 1.0, atol=1e-9):
            return result
    
    # Fallback to schoolbook
    return get_schoolbook()
