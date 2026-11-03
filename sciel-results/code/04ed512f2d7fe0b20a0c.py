import time
import numpy as np
from scipy.optimize import nnls

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """Construct a rank-minimal (or lower-rank) tensor decomposition for 4x4 * 4x7 matrix multiplication."""
    n, m, p = 4, 4, 7
    deadline = time.time() + 105  # Leave 15s safety margin
    
    def verify(U, V, W, n, m, p, tol=1e-7):
        """Verify that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k."""
        U = np.asarray(U, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        W = np.asarray(W, dtype=np.float64)
        rank = U.shape[1]
        # Efficient verification using einsum
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # Reshape to 3D
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        # sum over c
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        # T[i,j,k] should be 1
        return np.allclose(T, 1.0, atol=tol)
    
    def schoolbook_full(n, m, p):
        """Schoolbook decomposition: rank = n*m*p."""
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    def strassen_2x2x2():
        """
        Return U, V, W for 2x2x2 matrix multiplication with rank 7 (Strassen).
        Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (a+d)(e+h): U[0]=1,U[3]=1; V[0]=1,V[3]=1; W[0]=1,W[3]=1
        U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
        # M2 = (c+d)e: U[2]=1,U[3]=1; V[0]=1; W[2]=-1,W[3]=1
        U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=-1; W[3,1]=1
        # M3 = a(f-h): U[0]=1; V[1]=1,V[3]=-1; W[1]=1,W[3]=1
        U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=1
        # M4 = d(g-e): U[3]=1; V[2]=1,V[0]=-1; W[0]=-1,W[2]=1
        U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=-1; W[2,3]=1
        # M5 = (a+b)h: U[0]=1,U[1]=1; V[3]=1; W[0]=-1,W[1]=1
        U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[1,4]=1
        # M6 = (c-a)(e+f): U[2]=1,U[0]=-1; V[0]=1,V[1]=1; W[3]=-1
        U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=-1
        # M7 = (b-d)(g+h): U[1]=1,U[3]=-1; V[2]=1,V[3]=1; W[0]=1
        U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
        
        # Verify
        if not verify(U, V, W, 2, 2, 2):
            raise ValueError("Strassen 2x2x2 verification failed")
        return U, V, W
    
    def kronecker_strassen_4x4x4():
        """
        Build rank-49 decomposition for 4x4 * 4x4 using Kronecker product of Strassen 2x2x2.
        
        The 4x4 matrix is split into 2x2 blocks of 2x2 submatrices.
        A = [[A00, A01], [A10, A11]] where each Ai is 2x2
        B = [[B00, B01], [B10, B11]]
        C = A * B computed via Strassen at block level, with each block product
        computed via Strassen at scalar level.
        
        Total rank = 7 (block-level) * 7 (scalar-level) = 49
        """
        U2, V2, W2 = strassen_2x2x2()
        
        # Block-level Strassen coefficients
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 - M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 - M6
        
        # A_block_coeffs[r][block] for r=0..6, block=0..3 (0=A00,1=A01,2=A10,3=A11)
        A_bc = [
            [1, 0, 0, 1],   # M1
            [0, 0, 1, 1],   # M2
            [1, 0, 0, 0],   # M3
            [0, 0, 0, 1],   # M4
            [1, 1, 0, 0],   # M5
            [-1, 0, 1, 0],  # M6
            [0, 1, 0, -1],  # M7
        ]
        B_bc = [
            [1, 0, 0, 1],   # M1
            [1, 0, 0, 0],   # M2
            [0, 1, 0, -1],  # M3
            [0, 0, 1, -1],  # M4
            [0, 0, 0, 1],   # M5
            [1, 1, 0, 0],   # M6
            [0, 0, 1, 1],   # M7
        ]
        # C_out_coeffs[c_block][r] for c_block=0..3 (0=C00,1=C01,2=C10,3=C11), r=0..6
        C_oc = [
            [1, 0, 0, 0, -1, 0, 1],   # C00: M1 - M4 - M5 + M7
            [0, 0, 1, 0, 1, 0, 0],    # C01: M3 + M5
            [0, 1, 0, 1, 0, 0, 0],    # C10: M2 + M4
            [1, -1, 1, 0, 0, -1, 0],  # C11: M1 - M2 + M3 - M6
        ]
        
        rank = 49
        U4 = np.zeros((16, rank))
        V4 = np.zeros((16, rank))
        W4 = np.zeros((16, rank))
        
        # For 4x4 matrices, block (br, bc) of size 2x2 has global indices:
        # local (lr, lc) -> global row = br*2+lr, global col = bc*2+lc
        # U index for A: row*4 + col
        # V index for B: row*4 + col  
        # W index for C: col*4 + row (since W[k*n+i] where n=4)
        
        def block_indices_A(br, bc):
            """Indices in U (size 16) for A block (br,bc)."""
            idx = []
            for lr in range(2):
                for lc in range(2):
                    idx.append((br*2+lr)*4 + (bc*2+lc))
            return idx
        
        def block_indices_B(br, bc):
            """Indices in V (size 16) for B block (br,bc)."""
            idx = []
            for lr in range(2):
                for lc in range(2):
                    idx.append((br*2+lr)*4 + (bc*2+lc))
            return idx
        
        def block_indices_C(br, bc):
            """Indices in W (size 16) for C block (br,bc). W[k*4+i] where k=col, i=row."""
            idx = []
            for lr in range(2):
                for lc in range(2):
                    idx.append((bc*2+lc)*4 + (br*2+lr))
            return idx
        
        # For each block-level product r (0..6), and each scalar-level product s (0..6):
        # The scalar-level product s within Strassen 2x2x2 has:
        #   U2[:, s] gives which A-entries are involved
        #   V2[:, s] gives which B-entries are involved
        #   W2[:, s] gives which C-entries are involved (with coefficients)
        
        for r in range(7):
            for s in range(7):
                col = r * 7 + s
                # For scalar product s, U2[i,s] for i=0..3 corresponds to
                # local A entries: 0=A[0,0], 1=A[0,1], 2=A[1,0], 3=A[1,1]
                # Similarly for V2 and W2.
                
                # For each A-block ab with A_bc[r][ab] != 0:
                for ab in range(4):
                    if A_bc[r][ab] == 0:
                        continue
                    ab_r, ab_c = divmod(ab, 2)
                    a_indices = block_indices_A(ab_r, ab_c)
                    for local_i in range(4):
                        if U2[local_i, s] != 0:
                            U4[a_indices[local_i], col] += A_bc[r][ab] * U2[local_i, s]
                
                # For each B-block bb with B_bc[r][bb] != 0:
                for bb in range(4):
                    if B_bc[r][bb] == 0:
                        continue
                    bb_r, bb_c = divmod(bb, 2)
                    b_indices = block_indices_B(bb_r, bb_c)
                    for local_j in range(4):
                        if V2[local_j, s] != 0:
                            V4[b_indices[local_j], col] += B_bc[r][bb] * V2[local_j, s]
                
                # For C output: for each C-block cb with C_oc[cb][r] != 0:
                for cb in range(4):
                    if C_oc[cb][r] == 0:
                        continue
                    cb_r, cb_c = divmod(cb, 2)
                    c_indices = block_indices_C(cb_r, cb_c)
                    for local_k in range(4):
                        if W2[local_k, s] != 0:
                            W4[c_indices[local_k], col] += C_oc[cb][r] * W2[local_k, s]
        
        # Verify
        if not verify(U4, V4, W4, 4, 4, 4):
            raise ValueError("Kronecker Strassen 4x4x4 verification failed")
        return U4, V4, W4
    
    def schoolbook_4x4x3():
        """Schoolbook decomposition for 4x4 * 4x3, rank 48."""
        rank = 4 * 4 * 3
        U = np.zeros((16, rank))
        V = np.zeros((12, rank))
        W = np.zeros((12, rank))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(3):
                    U[i * 4 + j, col] = 1
                    V[j * 3 + k, col] = 1
                    W[k * 4 + i, col] = 1
                    col += 1
        return U, V, W
    
    def combine_decompositions(U1, V1, W1, U2, V2, W2, n, m, p):
        """
        Combine two decompositions:
        - First: for 4x4 * 4x4 (columns 0-3 of B)
        - Second: for 4x4 * 4x3 (columns 4-6 of B)
        
        U1: (16, r1), V1: (16, r1), W1: (16, r1)
        U2: (16, r2), V2: (12, r2), W2: (12, r2)
        
        Full: U: (16, r1+r2), V: (28, r1+r2), W: (28, r1+r2)
        """
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        total_rank = r1 + r2
        
        U_full = np.zeros((16, total_rank))
        V_full = np.zeros((28, total_rank))
        W_full = np.zeros((28, total_rank))
        
        # First part: 4x4 * 4x4
        U_full[:, :r1] = U1
        V_full[:16, :r1] = V1
        W_full[:16, :r1] = W1
        
        # Second part: 4x4 * 4x3
        U_full[:, r1:] = U2
        V_full[16:, r1:] = V2
        W_full[16:, r1:] = W2
        
        return U_full, V_full, W_full
    
    # Step 1: Get the rank-97 decomposition (49 + 48)
    try:
        U44, V44, W44 = kronecker_strassen_4x4x4()
        U43, V43, W43 = schoolbook_4x4x3()
        U97, V97, W97 = combine_decompositions(U44, V44, W44, U43, V43, W43, 4, 4, 7)
        
        if not verify(U97, V97, W97, 4, 4, 7):
            raise ValueError("Combined 97-rank verification failed")
        
        # Step 2: Try tensor completion to reduce rank
        # Strategy: Try to find a rank-R decomposition (R < 97) by:
        # 1. Start with the rank-97 decomposition
        # 2. Try to zero out some entries via optimization
        # 3. Use the structure: the 4x4x4 part (49 terms) and 4x4x3 part (48 terms)
        #    might share some rank-1 terms
        
        # For now, return the verified rank-97 decomposition
        return {"U": U97.tolist(), "V": V97.tolist(), "W": W97.tolist()}
        
    except Exception as e:
        # Fallback to schoolbook
        U, V, W = schoolbook_full(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
