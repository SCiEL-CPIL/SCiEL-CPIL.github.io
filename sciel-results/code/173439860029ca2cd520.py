"""Rewritten program for matrix multiplication tensor decomposition M_{3,5,7}."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140
    n, m, p = 3, 5, 7
    
    # Schoolbook decomposition (rank = n*m*p = 105)
    def schoolbook():
        r = n * m * p
        U = np.zeros((n * m, r), dtype=np.int64)
        V = np.zeros((m * p, r), dtype=np.int64)
        W = np.zeros((p * n, r), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    # Strassen decomposition for M_{2,2,2}
    # Returns U (4x7), V (4x7), W (4x7)
    # U indexed by (i,j) with i,j in {0,1}, row = i*2+j
    # V indexed by (j,k) with j,k in {0,1}, row = j*2+k
    # W indexed by (k,i) with k,i in {0,1}, row = k*2+i
    def strassen_222():
        U = np.zeros((4, 7), dtype=np.int64)
        V = np.zeros((4, 7), dtype=np.int64)
        W = np.zeros((4, 7), dtype=np.int64)
        # M1 = (a00+a11)(b00+b11)
        U[0,0]=1; U[3,0]=1
        V[0,0]=1; V[1,0]=1; V[2,0]=1; V[3,0]=1
        W[0,0]=1; W[3,0]=-1
        # M2 = (a10+a11)b00
        U[2,1]=1; U[3,1]=1
        V[0,1]=1
        W[2,1]=1
        # M3 = a00(b10-b11)
        U[0,2]=1
        V[2,2]=1; V[3,2]=-1
        W[1,2]=1
        # M4 = a11(b01-b00)
        U[3,3]=1
        V[1,3]=1; V[0,3]=-1
        W[2,3]=1
        # M5 = (a00+a01)b11
        U[0,4]=1; U[1,4]=1
        V[3,4]=1
        W[1,4]=1
        # M6 = (a10-a00)(b00+b10)
        U[2,5]=1; U[0,5]=-1
        V[0,5]=1; V[2,5]=1
        W[0,5]=1
        # M7 = (a01-a11)(b11+b01)
        U[1,6]=1; U[3,6]=-1
        V[3,6]=1; V[1,6]=1
        W[3,6]=1
        return U, V, W

    def verify(U, V, W):
        """Verify the decomposition against random matrices."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        
        # Random test matrices
        A = np.random.randint(-5, 6, (n, m))
        B = np.random.randint(-5, 6, (m, p))
        C_expected = A @ B
        
        # Compute using decomposition
        # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        
        # Flatten A and B to match U and V indices
        # U index: i*m + j -> A[i, j]
        # V index: j*p + k -> B[j, k]
        # W index: k*n + i -> C[k, i] (Note: W[k*n+i, r] corresponds to C[i, k] in standard C = A@B? 
        # Let's check the convention: W[k*n+i, r]. 
        # Standard C[i, k] = sum_j A[i, j] B[j, k].
        # Tensor T_{i, j, k} = 1 if i,j,k are valid indices.
        # Decomposition: T = sum_r U_{i, j, r} V_{j, k, r} W_{k, i, r}?
        # The prompt says: "output factor follows the official trace convention W[k*n+i,r]".
        # Usually, the tensor is T_{i, j, k}.
        # If W is indexed by k*n+i, then the third index of the tensor product is mapped to the row of W.
        # Let's assume the standard bilinear form:
        # C[i, k] = sum_r (sum_j U[i, j, r] A[i, j]) * V[j, k, r] ... no, that's not right.
        
        # Correct interpretation:
        # The tensor M_{n,m,p} is defined by T_{i,j,k} = 1.
        # A decomposition is T = sum_r U_r \otimes V_r \otimes W_r.
        # Where U_r is a vector in R^{n*m}, V_r in R^{m*p}, W_r in R^{p*n}.
        # U_r corresponds to the coefficient for A_{i,j}.
        # V_r corresponds to the coefficient for B_{j,k}.
        # W_r corresponds to the coefficient for C_{i,k} (but stored as W[k*n+i]).
        
        # So, C_{i,k} = sum_r W_r[k*n+i] * (sum_j U_r[i*m+j] A_{i,j}) * (sum_j V_r[j*p+k] B_{j,k})?
        # No, the multiplication is:
        # Intermediate_r = (U_r^T A_flat) * (V_r^T B_flat)
        # C_flat = sum_r W_r * Intermediate_r
        
        A_flat = A.flatten() # index i*m+j
        B_flat = B.flatten() # index j*p+k
        
        U_proj = U @ A_flat # shape (rank,)
        V_proj = V @ B_flat # shape (rank,)
        
        C_flat_computed = W @ (U_proj * V_proj) # shape (p*n,)
        
        # Map C_flat_computed back to C matrix
        # Index in C_flat_computed is k*n + i.
        # We want C[i, k].
        C_computed = np.zeros((n, p))
        for k in range(p):
            for i in range(n):
                C_computed[i, k] = C_flat_computed[k*n + i]
        
        return np.allclose(C_expected, C_computed)

    def try_strassen_block():
        """
        Construct a rank-103 decomposition using Strassen on sub-blocks.
        Split A (3x5) into A1 (2x5) and A2 (1x5).
        Split B (5x7) into B1 (5x5) and B2 (5x2).
        
        Sub-problems:
        1. C11 = A1 @ B1 (2x5 @ 5x5 -> 2x5)
        2. C12 = A1 @ B2 (2x5 @ 5x2 -> 2x2)
        3. C21 = A2 @ B1 (1x5 @ 5x5 -> 1x5)
        4. C22 = A2 @ B2 (1x5 @ 5x2 -> 1x2)
        
        Ranks:
        1. M_{2,5,5}: Rank 49 (using Strassen on 2x2x2 blocks and schoolbook for rest)
        2. M_{2,5,2}: Rank 19 (using Strassen on 2x2x2 and schoolbook for 2x3x2)
        3. M_{1,5,5}: Rank 25 (Schoolbook)
        4. M_{1,5,2}: Rank 10 (Schoolbook)
        Total: 49 + 19 + 25 + 10 = 103.
        """
        # We will build the full U, V, W matrices of size (15, R), (35, R), (21, R)
        # R = 103
        
        U = np.zeros((15, 103), dtype=np.int64)
        V = np.zeros((35, 103), dtype=np.int64)
        W = np.zeros((21, 103), dtype=np.int64)
        
        col = 0
        
        # Helper to add a sub-decomposition
        def add_sub(U_sub, V_sub, W_sub, u_offset, v_offset, w_offset):
            """
            U_sub: (n_sub*m_sub, r_sub)
            V_sub: (m_sub*p_sub, r_sub)
            W_sub: (p_sub*n_sub, r_sub)
            Offsets map local indices to global indices.
            """
            r_sub = U_sub.shape[1]
            U[u_offset:u_offset+n_sub*m_sub, col:col+r_sub] = U_sub
            V[v_offset:v_offset+m_sub*p_sub, col:col+r_sub] = V_sub
            W[w_offset:w_offset+p_sub*n_sub, col:col+r_sub] = W_sub
            return col + r_sub

        # --- Sub-problem 3: M_{1,5,5} (A2 is 1x5, B1 is 5x5, C21 is 1x5) ---
        # A2 is row 2 of A. B1 is cols 0-4 of B. C21 is row 2 of C, cols 0-4.
        # n=1, m=5, p=5.
        # Schoolbook rank 25.
        U_sub3 = np.zeros((5, 25), dtype=np.int64)
        V_sub3 = np.zeros((25, 25), dtype=np.int64)
        W_sub3 = np.zeros((5, 25), dtype=np.int64)
        c = 0
        for j in range(5):
            for k in range(5):
                U_sub3[j, c] = 1
                V_sub3[j*5+k, c] = 1
                W_sub3[k*1+0, c] = 1 # W index k*n+i = k*1+0 = k
                c += 1
        # Offsets:
        # A2 is row 2 of A. Global U index for A2[j] is 2*5 + j = 10+j.
        # B1 is cols 0-4 of B. Global V index for B1[j,k] is j*7 + k.
        # C21 is row 2 of C, cols 0-4. Global W index for C21[k] is k*3 + 2.
        col = add_sub(U_sub3, V_sub3, W_sub3, 10, 0, 2) # W offset: k*3+2. Min index is 2. Max is 4*3+2=14.
        # Wait, W_sub3 has 5 rows (k=0..4). The global W indices are k*3+2.
        # This mapping is not a contiguous block in W!
        # My add_sub function assumes contiguous blocks.
        # I need to handle non-contiguous offsets or build the matrices directly.
        
        # Let's restart the construction by directly filling the global matrices.
        U = np.zeros((15, 103), dtype=np.int64)
        V = np.zeros((35, 103), dtype=np.int64)
        W = np.zeros((21, 103), dtype=np.int64)
        col = 0
        
        # --- Sub-problem 3: M_{1,5,5} ---
        # A2 (row 2 of A), B1 (cols 0-4 of B), C21 (row 2 of C, cols 0-4)
        for j in range(5):
            for k in range(5):
                U[2*5 + j, col] = 1
                V[j*7 + k, col] = 1
                W[k*3 + 2, col] = 1
                col += 1
        # col is now 25
        
        # --- Sub-problem 4: M_{1,5,2} ---
        # A2 (row 2 of A), B2 (cols 5-6 of B), C22 (row 2 of C, cols 5-6)
        for j in range(5):
            for k in range(2):
                # k_local 0->5, 1->6
                U[2*5 + j, col] = 1
                V[j*7 + (5+k), col] = 1
                W[(5+k)*3 + 2, col] = 1
                col += 1
        # col is now 25 + 10 = 35
        
        # --- Sub-problem 2: M_{2,5,2} ---
        # A1 (rows 0-1 of A), B2 (cols 5-6 of B), C12 (rows 0-1 of C, cols 5-6)
        # Split inner dim 5 into 2+3.
        # Part 2a: M_{2,2,2} (j=0,1)
        # Part 2b: M_{2,3,2} (j=2,3,4)
        
        # Part 2a: Strassen on j=0,1
        U_s, V_s, W_s = strassen_222()
        # Map U_s (i,j) to global U (i, j) where i in 0,1 and j in 0,1.
        # Global U index: i*5 + j.
        # Map V_s (j,k) to global V (j, k) where j in 0,1 and k in 5,6.
        # Global V index: j*7 + (5+k_local).
        # Map W_s (k,i) to global W (k, i) where k in 5,6 and i in 0,1.
        # Global W index: (5+k_local)*3 + i.
        
        for r in range(7):
            # U
            for i in range(2):
                for j in range(2):
                    U[i*5 + j, col + r] = U_s[i*2 + j, r]
            # V
            for j in range(2):
                for k in range(2):
                    V[j*7 + (5+k), col + r] = V_s[j*2 + k, r]
            # W
            for k in range(2):
                for i in range(2):
                    W[(5+k)*3 + i, col + r] = W_s[k*2 + i, r]
        col += 7
        # col is now 35 + 7 = 42
        
        # Part 2b: Schoolbook on j=2,3,4
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    U[i*5 + (2+j), col] = 1
                    V[(2+j)*7 + (5+k), col] = 1
                    W[(5+k)*3 + i, col] = 1
                    col += 1
        # col is now 42 + 2*3*2 = 42 + 12 = 54
        
        # --- Sub-problem 1: M_{2,5,5} ---
        # A1 (rows 0-1 of A), B1 (cols 0-4 of B), C11 (rows 0-1 of C, cols 0-4)
        # Split inner dim 5 into 2+3.
        # Part 1a: M_{2,2,5} (j=0,1)
        # Part 1b: M_{2,3,5} (j=2,3,4)
        
        # Part 1a: M_{2,2,5}
        # Split outer dim 5 into 2+3.
        # Part 1a-i: M_{2,2,2} (j=0,1, k=0,1)
        U_s, V_s, W_s = strassen_222()
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    U[i*5 + j, col + r] = U_s[i*2 + j, r]
            for j in range(2):
                for k in range(2):
                    V[j*7 + k, col + r] = V_s[j*2 + k, r]
            for k in range(2):
                for i in range(2):
                    W[k*3 + i, col + r] = W_s[k*2 + i, r]
        col += 7
        # col is now 54 + 7 = 61
        
        # Part 1a-ii: M_{2,2,3} (j=0,1, k=2,3,4)
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    U[i*5 + j, col] = 1
                    V[j*7 + (2+k), col] = 1
                    W[(2+k)*3 + i, col] = 1
                    col += 1
        # col is now 61 + 2*2*3 = 61 + 12 = 73
        
        # Part 1b: M_{2,3,5} (j=2,3,4)
        # Split outer dim 5 into 2+3.
        # Part 1b-i: M_{2,3,2} (j=2,3,4, k=0,1)
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    U[i*5 + (2+j), col] = 1
                    V[(2+j)*7 + k, col] = 1
                    W[k*3 + i, col] = 1
                    col += 1
        # col is now 73 + 2*3*2 = 73 + 12 = 85
        
        # Part 1b-ii: M_{2,3,3} (j=2,3,4, k=2,3,4)
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    U[i*5 + (2+j), col] = 1
                    V[(2+j)*7 + (2+k), col] = 1
                    W[(2+k)*3 + i, col] = 1
                    col += 1
        # col is now 85 + 2*3*3 = 85 + 18 = 103
        
        # Verify
        if verify(U.tolist(), V.tolist(), W.tolist()):
            return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
        else:
            # Fallback to schoolbook if verification fails
            return schoolbook()

    return try_strassen_block()
# EVOLVE-BLOCK-END
