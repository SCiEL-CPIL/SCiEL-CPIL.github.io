import numpy as np
import time

n, m, p = 2, 5, 6
nm, mp, pn = n * m, m * p, p * n

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def check_decomposition(U, V, W, tol=1e-9):
        R = U.shape[1]
        U_3d = U.reshape(n, m, R)
        V_3d = V.reshape(m, p, R)
        W_3d = W.reshape(p, n, R)
        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        if not np.allclose(T, 1.0, atol=tol):
            return False
        return True

    def get_strassen_222():
        """Returns U, V, W for 2x2x2 Strassen decomposition. Shapes: 4x7, 4x7, 4x7."""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        # M0: (A00+A11)(B00+B11)
        U[0, 0] += 1; U[3, 0] += 1
        V[0, 0] += 1; V[3, 0] += 1
        W[0, 0] += 1; W[3, 0] += 1
        # M1: (A10+A11)B00
        U[2, 1] += 1; U[3, 1] += 1
        V[0, 1] += 1
        W[1, 1] += 1; W[3, 1] -= 1
        # M2: A00(B01-B11)
        U[0, 2] += 1
        V[1, 2] += 1; V[3, 2] -= 1
        W[2, 2] += 1; W[3, 2] += 1
        # M3: A11(B10-B00)
        U[3, 3] += 1
        V[2, 3] += 1; V[0, 3] -= 1
        W[0, 3] += 1; W[1, 3] += 1
        # M4: (A00+A01)B11
        U[0, 4] += 1; U[1, 4] += 1
        V[3, 4] += 1
        W[0, 4] -= 1; W[2, 4] += 1
        # M5: (A10-A00)(B00+B01)
        U[2, 5] += 1; U[0, 5] -= 1
        V[0, 5] += 1; V[1, 5] += 1
        W[3, 5] += 1
        # M6: (A01-A11)(B10+B11)
        U[1, 6] += 1; U[3, 6] -= 1
        V[2, 6] += 1; V[3, 6] += 1
        W[0, 6] += 1
        return U, V, W

    def get_schoolbook_221():
        """Returns U, V, W for 2x2x1 schoolbook decomposition. Shapes: 4x2, 2x2, 2x2."""
        # A is 2x2, B is 2x1, C is 1x2 (p=1, n=2)
        # Indices: i in 0..1, j in 0..1, k in 0..0
        U = np.zeros((4, 2))
        V = np.zeros((2, 2)) # m*p = 2*1 = 2
        W = np.zeros((2, 2)) # p*n = 1*2 = 2
        # Term 1: A00 B00 -> C00
        U[0, 0] = 1
        V[0, 0] = 1
        W[0, 0] = 1
        # Term 2: A10 B00 -> C10
        U[2, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1
        return U, V, W

    def build_223_block():
        """Builds 2x2x3 block with Rank 9.
        A is 2x2, B is 2x3, C is 3x2.
        U shape: 4x9, V shape: 6x9, W shape: 6x9.
        """
        U_out = np.zeros((4, 9))
        V_out = np.zeros((6, 9))
        W_out = np.zeros((6, 9))
        
        # Part 1: 2x2x2 (Strassen) for B columns 0,1
        U_s, V_s, W_s = get_strassen_222()
        U_out[:, :7] = U_s
        # V_s is 4x7 (j in 0..1, k in 0..1). V_out is 6x9 (j in 0..1, k in 0..2).
        # Map V_s[j, k] to V_out[j*3+k, r]
        for j in range(2):
            for k in range(2):
                V_out[j*3+k, :7] = V_s[j*2+k, :7]
        # W_s is 4x7 (k in 0..1, i in 0..1). W_out is 6x9 (k in 0..2, i in 0..1).
        # Map W_s[k, i] to W_out[k*2+i, r]
        for k in range(2):
            for i in range(2):
                W_out[k*2+i, :7] = W_s[k*2+i, :7]
        
        # Part 2: 2x2x1 (Schoolbook) for B column 2
        U_sb, V_sb, W_sb = get_schoolbook_221()
        U_out[:, 7:9] = U_sb
        # V_sb is 2x2 (j in 0..1, k in 0..0). We want k=2.
        # Map V_sb[j, 0] to V_out[j*3+2, 7:9]
        for j in range(2):
            V_out[j*3+2, 7:9] = V_sb[j, :]
        # W_sb is 2x2 (k in 0..0, i in 0..1). We want k=2.
        # Map W_sb[0, i] to W_out[2*2+i, 7:9]
        for i in range(2):
            W_out[2*2+i, 7:9] = W_sb[0, i]
            
        return U_out, V_out, W_out

    def build_233_block_rank15():
        """Builds 2x3x3 block with Rank 15 (Sum of three 2x1x3).
        A is 2x3, B is 3x3, C is 3x2.
        U shape: 6x15, V shape: 9x15, W shape: 9x15.
        """
        U_out = np.zeros((6, 15))
        V_out = np.zeros((9, 15))
        W_out = np.zeros((9, 15))
        
        # 2x1x3 block for each j in 0,1,2
        # Rank 6 for each.
        # Total 18? No, 2x1x3 rank is 6?
        # Let's re-verify 2x1x3 rank.
        # C_{i,k} = A_i B_k.
        # T_{i, j, k} = 1 if j is the only index.
        # We need sum_r U_{i,r} V_{j,r} W_{k,r} = 1.
        # If we use 6 terms, one for each (i,k) pair.
        # r=0..5.
        # U_{i, r} = 1 if r corresponds to i.
        # V_{j, r} = 1.
        # W_{k, r} = 1 if r corresponds to k.
        # Then sum is 1.
        # So Rank 6 is correct.
        
        for j in range(3):
            base_r = j * 6
            for i in range(2):
                for k in range(3):
                    r = base_r + i*3 + k
                    U_out[i*3+j, r] = 1
                    V_out[j*3+k, r] = 1
                    W_out[k*2+i, r] = 1
                    
        return U_out, V_out, W_out

    def get_decomposition():
        # Partition m=5 into 2, 3.
        # Partition p=6 into 3, 3.
        # Blocks:
        # 1. A[0:2, 0:2] * B[0:2, 0:3] -> 2x2x3 (Rank 9)
        # 2. A[0:2, 0:2] * B[0:2, 3:6] -> 2x2x3 (Rank 9)
        # 3. A[0:2, 2:5] * B[2:5, 0:3] -> 2x3x3 (Rank 15)
        # 4. A[0:2, 2:5] * B[2:5, 3:6] -> 2x3x3 (Rank 15)
        
        R_total = 9 + 9 + 15 + 15
        U = np.zeros((nm, R_total))
        V = np.zeros((mp, R_total))
        W = np.zeros((pn, R_total))
        
        col = 0
        
        # Block 1: i=0:2, j=0:2, k=0:3
        U1, V1, W1 = build_223_block()
        U[0:4, col:col+9] = U1
        # V1 is 6x9 (j=0:2, k=0:3). Global V is (j=0:5, k=0:6).
        # j_global = 0:2, k_global = 0:3.
        for j in range(2):
            for k in range(3):
                V[j*6+k, col:col+9] = V1[j*3+k, :]
        # W1 is 6x9 (k=0:3, i=0:2). Global W is (k=0:6, i=0:2).
        for k in range(3):
            for i in range(2):
                W[k*2+i, col:col+9] = W1[k*2+i, :]
        col += 9
        
        # Block 2: i=0:2, j=0:2, k=3:6
        U2, V2, W2 = build_223_block()
        U[0:4, col:col+9] = U2
        for j in range(2):
            for k in range(3):
                V[j*6+(k+3), col:col+9] = V2[j*3+k, :]
        for k in range(3):
            for i in range(2):
                W[(k+3)*2+i, col:col+9] = W2[k*2+i, :]
        col += 9
        
        # Block 3: i=0:2, j=2:5, k=0:3
        U3, V3, W3 = build_233_block_rank15()
        # U3 is 6x15 (i=0:2, j=0:3). Global j=2:5.
        for i in range(2):
            for j in range(3):
                U[i*5+(j+2), col:col+15] = U3[i*3+j, :]
        # V3 is 9x15 (j=0:3, k=0:3). Global j=2:5, k=0:3.
        for j in range(3):
            for k in range(3):
                V[(j+2)*6+k, col:col+15] = V3[j*3+k, :]
        # W3 is 9x15 (k=0:3, i=0:2). Global k=0:3, i=0:2.
        for k in range(3):
            for i in range(2):
                W[k*2+i, col:col+15] = W3[k*2+i, :]
        col += 15
        
        # Block 4: i=0:2, j=2:5, k=3:6
        U4, V4, W4 = build_233_block_rank15()
        for i in range(2):
            for j in range(3):
                U[i*5+(j+2), col:col+15] = U4[i*3+j, :]
        for j in range(3):
            for k in range(3):
                V[(j+2)*6+(k+3), col:col+15] = V4[j*3+k, :]
        for k in range(3):
            for i in range(2):
                W[(k+3)*2+i, col:col+15] = W4[k*2+i, :]
        col += 15
        
        if col != R_total:
            return None
            
        return U, V, W

    U, V, W = get_decomposition()
    if U is None:
        # Fallback to schoolbook
        R = nm * mp // n # No, rank = n*m*p
        R = n * m * p
        U = np.zeros((nm, R))
        V = np.zeros((mp, R))
        W = np.zeros((pn, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    if check_decomposition(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        R = n * m * p
        U = np.zeros((nm, R))
        V = np.zeros((mp, R))
        W = np.zeros((pn, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
