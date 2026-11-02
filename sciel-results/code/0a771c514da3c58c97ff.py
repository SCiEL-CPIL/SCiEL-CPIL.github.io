"""Matrix multiplication 2x2x15: rank-minimal exact tensor decomposition."""
import time
import numpy as np
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=15
    
    # Rank 53 decomposition: 7 Strassen blocks (14 cols) + 1 schoolbook (1 col)
    rank = 53
    U = [[0] * rank for _ in range(n * m)]  # 4 x 53
    V = [[0] * rank for _ in range(m * p)]  # 30 x 53
    W = [[0] * rank for _ in range(p * n)]  # 30 x 53
    
    # A indices in U: 0:A00, 1:A01, 2:A10, 3:A11
    # B indices in V: j*p + k
    # C indices in W: k*n + i
    
    def add_strassen_block(base_r, k0, k1):
        r1, r2, r3, r4, r5, r6, r7 = base_r, base_r+1, base_r+2, base_r+3, base_r+4, base_r+5, base_r+6
        
        # P1: (A00+A11)(B0k0+B1k1)
        U[0][r1] = 1; U[3][r1] = 1
        V[0*p+k0][r1] = 1; V[1*p+k1][r1] = 1
        # P2: (A10+A11)B0k0
        U[2][r2] = 1; U[3][r2] = 1
        V[0*p+k0][r2] = 1
        # P3: A00(B0k1-B1k1)
        U[0][r3] = 1
        V[0*p+k1][r3] = 1; V[1*p+k1][r3] = -1
        # P4: A11(B1k0-B0k0)
        U[3][r4] = 1
        V[1*p+k0][r4] = 1; V[0*p+k0][r4] = -1
        # P5: (A00+A01)B1k1
        U[0][r5] = 1; U[1][r5] = 1
        V[1*p+k1][r5] = 1
        # P6: (A10-A00)(B0k0+B0k1)
        U[2][r6] = 1; U[0][r6] = -1
        V[0*p+k0][r6] = 1; V[0*p+k1][r6] = 1
        # P7: (A01-A11)(B1k0+B1k1)
        U[1][r7] = 1; U[3][r7] = -1
        V[1*p+k0][r7] = 1; V[1*p+k1][r7] = 1
        
        # Reconstruction:
        # C0k0 = P1 + P4 - P5 + P7
        W[2*k0][r1] = 1; W[2*k0][r4] = 1; W[2*k0][r5] = -1; W[2*k0][r7] = 1
        # C1k0 = P2 + P4
        W[2*k0+1][r2] = 1; W[2*k0+1][r4] = 1
        # C0k1 = P3 + P5
        W[2*k1][r3] = 1; W[2*k1][r5] = 1
        # C1k1 = P1 - P2 + P3 + P6
        W[2*k1+1][r1] = 1; W[2*k1+1][r2] = -1; W[2*k1+1][r3] = 1; W[2*k1+1][r6] = 1
    
    # 7 Strassen blocks for column pairs (0,1), (2,3), ..., (12,13)
    for i in range(7):
        add_strassen_block(7*i, 2*i, 2*i+1)
    
    # Trivial block for column 14 (rank 4)
    base_r = 49
    U[0][base_r] = 1; V[0*p+14][base_r] = 1; W[28][base_r] = 1
    U[1][base_r+1] = 1; V[1*p+14][base_r+1] = 1; W[28][base_r+1] = 1
    U[2][base_r+2] = 1; V[0*p+14][base_r+2] = 1; W[29][base_r+2] = 1
    U[3][base_r+3] = 1; V[1*p+14][base_r+3] = 1; W[29][base_r+3] = 1
    
    # Verify correctness with a random test
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Random test matrices
    A_test = np.random.rand(n, m)
    B_test = np.random.rand(m, p)
    
    # Direct multiplication
    C_direct = A_test @ B_test
    
    # Tensor decomposition multiplication
    # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # But actually the tensor decomposition works as:
    # C[i,k] = sum_j sum_r (sum over j) ... 
    # The correct interpretation: C = sum_r U[:,r] * V[:,r]^T * W[:,r] (as outer products)
    # More precisely: C[i,k] = sum_r (sum_j U[i*m+j,r] * V[j*p+k,r]) * W[k*n+i,r]
    
    C_tensor = np.zeros((n, p))
    for r in range(rank):
        # Compute the intermediate: for each (i,k), sum over j of U[i*m+j,r]*V[j*p+k,r]
        # This is a 2x15 matrix
        M_r = np.zeros((n, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    M_r[i,k] += U_np[i*m+j, r] * V_np[j*p+k, r]
        C_tensor += M_r * W_np[k*n+i, r] if False else 0  # placeholder
    
    # Actually, let me do this more carefully:
    # The tensor T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # And C[i,k] = sum_j T[i,j,k]
    
    C_tensor = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            s = 0.0
            for j in range(m):
                for r in range(rank):
                    s += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
            C_tensor[i,k] = s
    
    # Check if they match
    if not np.allclose(C_direct, C_tensor, atol=1e-10):
        # If verification fails, return schoolbook as fallback
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(n * m)]
        V_s = [[0] * rank_school for _ in range(m * p)]
        W_s = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j][col] = 1
                    V_s[j*p+k][col] = 1
                    W_s[k*n+i][col] = 1
                    col += 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
