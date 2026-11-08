"""Matrix multiplication 6,6,6 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110  # safety margin before 120s limit
    n, m, p = DIMENSIONS
    
    # Start with schoolbook (rank 216) as fallback
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
    
    # Try to construct lower-rank decomposition using known small matrix multiplication algorithms
    # Strategy: Use tensor product of smaller decompositions
    # R(6) <= R(2)*R(3) via tensor product
    # R(2) = 7 (Strassen), R(3) <= 23 (best known)
    
    # Define Strassen decomposition for 2x2 (rank 7)
    # Indices: i,j in {0,1} for A (2x2), k in {0,1} for B (2x2)
    # Output C[i,k] where C = A*B
    # W index: k*n2 + i where n2=2
    
    # Strassen 7 products:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    def strassen_2x2():
        """Return (U, V, W) for 2x2 matrix multiplication, rank 7.
        U has shape (2*2, 7) = (4, 7)
        V has shape (2*2, 7) = (4, 7)  
        W has shape (2*2, 7) = (4, 7)
        W index convention: W[k*2+i, r] for output C[i,k]
        """
        rank2 = 7
        U2 = [[0]*rank2 for _ in range(4)]
        V2 = [[0]*rank2 for _ in range(4)]
        W2 = [[0]*rank2 for _ in range(4)]
        
        # A indices: row 0 = A00 (i=0,j=0), row 1 = A01 (i=0,j=1), row 2 = A10 (i=1,j=0), row 3 = A11 (i=1,j=1)
        # B indices: row 0 = B00 (j=0,k=0), row 1 = B01 (j=0,k=1), row 2 = B10 (j=1,k=0), row 3 = B11 (j=1,k=1)
        # W indices: C[i,k] -> index k*2+i
        #   C00 -> W[0*2+0]=W[0]
        #   C01 -> W[1*2+0]=W[2]
        #   C10 -> W[0*2+1]=W[1]
        #   C11 -> W[1*2+1]=W[3]
        
        # M1 = (A00+A11)*(B00+B11)
        U2[0][0] = 1  # A00
        U2[3][0] = 1  # A11
        V2[0][0] = 1  # B00
        V2[3][0] = 1  # B11
        W2[0][0] = 1  # C00
        W2[3][0] = 1  # C11
        
        # M2 = (A10+A11)*B00
        U2[2][1] = 1  # A10
        U2[3][1] = 1  # A11
        V2[0][1] = 1  # B00
        W2[1][1] = 1  # C10
        W2[3][1] = -1 # C11
        
        # M3 = A00*(B01-B11)
        U2[0][2] = 1  # A00
        V2[1][2] = 1  # B01
        V2[3][2] = -1 # B11
        W2[2][2] = 1  # C01
        W2[3][2] = 1  # C11
        
        # M4 = A11*(B10-B00)
        U2[3][3] = 1  # A11
        V2[2][3] = 1  # B10
        V2[0][3] = -1 # B00
        W2[0][3] = 1  # C00
        W2[1][3] = 1  # C10
        
        # M5 = (A00+A01)*B11
        U2[0][4] = 1  # A00
        U2[1][4] = 1  # A01
        V2[3][4] = 1  # B11
        W2[0][4] = -1 # C00
        W2[2][4] = 1  # C01
        
        # M6 = (A10-A00)*(B00+B01)
        U2[2][5] = 1  # A10
        U2[0][5] = -1 # A00
        V2[0][5] = 1  # B00
        V2[1][5] = 1  # B01
        W2[3][5] = 1  # C11
        
        # M7 = (A01-A11)*(B10+B11)
        U2[1][6] = 1  # A01
        U2[3][6] = -1 # A11
        V2[2][6] = 1  # B10
        V2[3][6] = 1  # B11
        W2[0][6] = 1  # C00
        W2[2][6] = 1  # C01
        
        return U2, V2, W2
    
    U2, V2, W2 = strassen_2x2()
    
    # For 3x3, use schoolbook (rank 27) to ensure correctness
    # This gives R(6) <= 7 * 27 = 189
    def schoolbook_3x3():
        """Return (U, V, W) for 3x3 matrix multiplication, rank 27 (schoolbook).
        U has shape (3*3, 27) = (9, 27)
        V has shape (3*3, 27) = (9, 27)
        W has shape (3*3, 27) = (9, 27)
        W index convention: W[k*3+i, r] for output C[i,k]
        """
        rank3 = 27
        U3 = [[0]*rank3 for _ in range(9)]
        V3 = [[0]*rank3 for _ in range(9)]
        W3 = [[0]*rank3 for _ in range(9)]
        
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    c = i * 3 + j  # U index: i*m+j
                    v = j * 3 + k  # V index: j*p+k
                    w = k * 3 + i  # W index: k*n+i
                    col = i * 9 + j * 3 + k
                    U3[c][col] = 1
                    V3[v][col] = 1
                    W3[w][col] = 1
        
        return U3, V3, W3
    
    U3, V3, W3 = schoolbook_3x3()
    
    # Tensor product: 6x6 = 2x3 blocks
    # Split 6x6 into 2x2 block of 3x3 submatrices
    # A (6x6) -> 2x2 blocks, each 3x3
    # B (6x6) -> 2x2 blocks, each 3x3
    # C (6x6) -> 2x2 blocks, each 3x3
    
    # U for 6x6: shape (36, 189)
    # For each (i6, j6) where i6 in 0..5, j6 in 0..5:
    #   i2 = i6 // 3, j2 = j6 // 3, i3 = i6 % 3, j3 = j6 % 3
    #   For each Strassen component c2 in 0..6 and schoolbook component c3 in 0..26:
    #     col = c2 * 27 + c3
    #     U6[i6*6+j6][col] = U2[i2*2+j2][c2] * U3[i3*3+j3][c3]
    
    rank_combined = 7 * 27  # 189
    U6 = [[0]*rank_combined for _ in range(36)]
    V6 = [[0]*rank_combined for _ in range(36)]
    W6 = [[0]*rank_combined for _ in range(36)]
    
    for i6 in range(6):
        for j6 in range(6):
            i2 = i6 // 3
            j2 = j6 // 3
            i3 = i6 % 3
            j3 = j6 % 3
            for c2 in range(7):
                u2_val = U2[i2*2+j2][c2]
                if u2_val == 0:
                    continue
                for c3 in range(27):
                    u3_val = U3[i3*3+j3][c3]
                    if u3_val == 0:
                        continue
                    col = c2 * 27 + c3
                    U6[i6*6+j6][col] = u2_val * u3_val
    
    for j6 in range(6):
        for k6 in range(6):
            j2 = j6 // 3
            k2 = k6 // 3
            j3 = j6 % 3
            k3 = k6 % 3
            for c2 in range(7):
                v2_val = V2[j2*2+k2][c2]
                if v2_val == 0:
                    continue
                for c3 in range(27):
                    v3_val = V3[j3*3+k3][c3]
                    if v3_val == 0:
                        continue
                    col = c2 * 27 + c3
                    V6[j6*6+k6][col] = v2_val * v3_val
    
    for k6 in range(6):
        for i6 in range(6):
            k2 = k6 // 3
            i2 = i6 // 3
            k3 = k6 % 3
            i3 = i6 % 3
            for c2 in range(7):
                w2_val = W2[k2*2+i2][c2]
                if w2_val == 0:
                    continue
                for c3 in range(27):
                    w3_val = W3[k3*3+i3][c3]
                    if w3_val == 0:
                        continue
                    col = c2 * 27 + c3
                    W6[k6*6+i6][col] = w2_val * w3_val
    
    # Verify correctness
    def verify(U, V, W, n, m, p):
        """Verify that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if i==j==k... 
        Actually for matrix mult tensor: the decomposition satisfies
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{condition}
        For standard matrix multiplication C = AB, the tensor T[i,j,k] = 1 for all i,j,k
        and C[i,k] = sum_j A[i,j] B[j,k]
        The decomposition: C[i,k] = sum_r (sum_j U[i*m+j, r] A[i,j]) * (sum_j V[j*p+k, r] B[j,k]) * W[k*n+i, r]
        This should equal sum_j A[i,j] B[j,k] for all A, B
        """
        # Convert to numpy
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        rank = len(U[0])
        
        # Check: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should equal 1
        # Wait, that's not right. The tensor for matrix multiplication has T[i,j,k] = 1 for all valid indices
        # The decomposition means: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # For matrix multiplication, T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        
        # Actually let me reconsider. The standard formulation:
        # We want C = AB where C is n x p, A is n x m, B is m x p
        # C[i,k] = sum_j A[i,j] B[j,k]
        # Tensor form: sum_{i,j,k} A[i,j] B[j,k] C[i,k] = sum_{i,j,k} A[i,j] B[j,k] C[i,k]
        # Decomposition: sum_r (sum_j U[i*m+j, r] A[i,j]) (sum_j V[j*p+k, r] B[j,k]) W[k*n+i, r] C[i,k]
        # For this to equal the original, we need:
        # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i, j, k
        
        # Wait, that's only if we're decomposing the identity-like tensor. Let me think again.
        # The multiplication tensor T has entries T[i,j,k] for the coefficient of A[i,j]*B[j,k] in C[i,k]
        # Actually the standard way: the tensor is defined such that
        # sum_{r} U_{i,j}^{(r)} V_{j,k}^{(r)} W_{i,k}^{(r)} = delta_{i,k}... no
        
        # Let me use the direct check: generate random matrices and verify
        np.random.seed(42)
        A = np.random.randint(-2, 3, size=(n, m))
        B = np.random.randint(-2, 3, size=(m, p))
        C_expected = A @ B
        
        # Compute via decomposition
        # U_a[i*m+j, r], V_b[j*p+k, r], W_c[k*n+i, r]
        # C[i,k] = sum_r (sum_j U[i*m+j,r]*A[i,j]) * (sum_j V[j*p+k,r]*B[j,k]) * W[k*n+i,r]
        # Wait, that's not standard either. Let me re-read the task.
        
        # The task says: "exactly decompose the matrix multiplication tensor"
        # with output factor W[k*n+i, r]
        # Standard: C[i,k] = sum_r U[i*m+j, r] A[i,j] * V[j*p+k, r] B[j,k] * W[k*n+i, r] summed appropriately
        
        # Actually the cleanest way: the rank-r decomposition of the multiplication tensor means:
        # For all A, B: C = AB can be computed as
        # C[i,k] = sum_{r=0}^{rank-1} (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
        
        # Let's verify this
        C_computed = np.zeros((n, p))
        for r in range(rank):
            U_col = U_np[:, r]  # shape (n*m,)
            V_col = V_np[:, r]  # shape (m*p,)
            W_val = W_np[:, r]  # shape (p*n,)
            
            # sum_j U[i*m+j, r] * A[i,j] for each i
            UA = np.zeros(n)
            for i in range(n):
                for j in range(m):
                    UA[i] += U_np[i*m+j, r] * A[i, j]
            
            # sum_j V[j*p+k, r] * B[j,k] for each k
            VB = np.zeros(p)
            for k in range(p):
                for j in range(m):
                    VB[k] += V_np[j*p+k, r] * B[j, k]
            
            # C[i,k] += UA[i] * VB[k] * W[k*n+i, r]
            for i in range(n):
                for k in range(p):
                    C_computed[i, k] += UA[i] * VB[k] * W_np[k*n+i, r]
        
        if np.allclose(C_computed, C_expected, atol=1e-9):
            return True
        else:
            return False
    
    if verify(U6, V6, W6, n, m, p):
        return {"U": U6, "V": V6, "W": W6}
    else:
        # Fallback to schoolbook
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
