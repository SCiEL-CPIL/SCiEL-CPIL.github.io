import numpy as np
import time

DIMENSIONS = [2, 2, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # The tensor for matrix multiplication M_{n,m,p} is defined by T_{i,j,k} = 1.
    # We seek a rank-R decomposition:
    # C_{i,k} = sum_r W_{k,i,r} * (sum_j U_{i,j,r} A_{i,j}) * (sum_j V_{j,k,r} B_{j,k})
    #
    # The current best known rank is 35, obtained by splitting the p=10 dimension
    # into 5 blocks of size 2 and applying the rank-7 Strassen algorithm to each block.
    # R = 5 * 7 = 35.
    #
    # We attempt to verify this and return it.
    # Note: The seed program uses rank 40 (schoolbook).
    # Score = Seed / Current = 40 / 35 = 1.1428...
    
    def build_strassen_rank_35():
        """
        Constructs a rank-35 decomposition using Strassen's algorithm on 5 blocks of 2x2 matrices.
        """
        R = 35
        U = [[0] * R for _ in range(n * m)]  # 4 x 35
        V = [[0] * R for _ in range(m * p)]  # 20 x 35
        W = [[0] * R for _ in range(p * n)]  # 20 x 35
        
        # Strassen coefficients for 2x2 * 2x2
        # A = [[A00, A01], [A10, A11]]
        # B = [[B00, B01], [B10, B11]]
        # C = A * B
        #
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B10 - B11)
        # M4 = A11 * (B01 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        #
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # U indices: i*m + j. For 2x2, indices are 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # V indices: j*p + k. For a block q (columns 2q, 2q+1):
        #   j=0, k=0 -> 0*10 + 2q = 2q
        #   j=0, k=1 -> 0*10 + 2q+1 = 2q+1
        #   j=1, k=0 -> 1*10 + 2q = 10+2q
        #   j=1, k=1 -> 1*10 + 2q+1 = 10+2q+1
        # W indices: k*n + i. For a block q:
        #   k=0, i=0 -> 0*2 + 0 = 0 (global 4q? No, W index is k*n+i. 
        #      Wait, W is (p*n, R). Index is k*n + i.
        #      For block q, k is 2q or 2q+1.
        #      k=2q, i=0 -> 2q*2 + 0 = 4q
        #      k=2q, i=1 -> 2q*2 + 1 = 4q+1
        #      k=2q+1, i=0 -> (2q+1)*2 + 0 = 4q+2
        #      k=2q+1, i=1 -> (2q+1)*2 + 1 = 4q+3
        
        strassen_terms = [
            # (U_vec, V_vec, W_vec)
            # M1
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            # M2
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
            # M3
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
            # M4
            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
            # M5
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
            # M6
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            # M7
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
        ]
        
        col = 0
        for q in range(5):
            for u_v, v_v, w_v in strassen_terms:
                # Fill U
                for i in range(4):
                    U[i][col] = u_v[i]
                
                # Fill V
                # V indices: 2q, 2q+1, 10+2q, 10+2q+1
                V[2*q][col] = v_v[0]
                V[2*q+1][col] = v_v[1]
                V[10+2*q][col] = v_v[2]
                V[10+2*q+1][col] = v_v[3]
                
                # Fill W
                # W indices: 4q, 4q+1, 4q+2, 4q+3
                W[4*q][col] = w_v[0]
                W[4*q+1][col] = w_v[1]
                W[4*q+2][col] = w_v[2]
                W[4*q+3][col] = w_v[3]
                
                col += 1
                
        return U, V, W

    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that the decomposition exactly computes matrix multiplication."""
        n, m, p = 2, 2, 10
        R = len(U[0])
        np.random.seed(42)
        for trial in range(10):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            
            C_computed = np.zeros((n, p))
            for r in range(R):
                # u_A[i] = sum_j U[i*m+j, r] * A[i,j]
                u_A = np.zeros(n)
                for i in range(n):
                    for j in range(m):
                        u_A[i] += U[i*m + j][r] * A[i, j]
                
                # v_B[k] = sum_j V[j*p+k, r] * B[j,k]
                v_B = np.zeros(p)
                for k in range(p):
                    for j in range(m):
                        v_B[k] += V[j*p + k][r] * B[j, k]
                
                # C[i,k] += W[k*n+i, r] * u_A[i] * v_B[k]
                for i in range(n):
                    for k in range(p):
                        C_computed[i, k] += W[k*n + i][r] * u_A[i] * v_B[k]
            
            if not np.allclose(C_computed, C_expected, atol=tol, rtol=tol):
                return False
        return True

    U, V, W = build_strassen_rank_35()
    
    # Verification is expensive, but we have time.
    if not verify_decomposition(U, V, W):
        # Fallback to schoolbook rank 40 if Strassen fails (should not happen)
        R = 40
        U = [[0] * R for _ in range(n * m)]
        V = [[0] * R for _ in range(m * p)]
        W = [[0] * R for _ in range(p * n)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        
        if not verify_decomposition(U, V, W):
            raise ValueError("Decomposition failed")
        
    return {"U": U, "V": V, "W": W}
