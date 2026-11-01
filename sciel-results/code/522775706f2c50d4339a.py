"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-reduced exact tensor decomposition for multiplying a 2x2 matrix by a 2x12 matrix.
    Attempts rank-32 decomposition using 4 blocks of 3 columns with rank-8 per block.
    Falls back to rank-42 Strassen if verification fails.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    # Try rank-32 first
    num_blocks = p // 3  # 4
    rank_per_block = 8
    rank = num_blocks * rank_per_block  # 32
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t in range(num_blocks):
        base = t * rank_per_block
        c0 = 3 * t
        c1 = 3 * t + 1
        c2 = 3 * t + 2
        
        # Term 0: (A00+A11)*(B00+B10)
        s = 0
        c = base + s
        U[0][c] = 1; U[3][c] = 1
        V[0*p+c0][c] = 1; V[1*p+c1][c] = 1
        W[c0*n+0][c] = 1; W[c1*n+1][c] = 1
        
        # Term 1: (A10+A11)*B00
        s = 1
        c = base + s
        U[2][c] = 1; U[3][c] = 1
        V[0*p+c0][c] = 1
        W[c0*n+1][c] = 1; W[c1*n+1][c] = -1
        
        # Term 2: A00*(B10-B00)
        s = 2
        c = base + s
        U[0][c] = 1
        V[1*p+c1][c] = 1; V[0*p+c0][c] = -1
        W[c0*n+0][c] = 1; W[c0*n+1][c] = 1
        
        # Term 3: A11*(B01-B10)
        s = 3
        c = base + s
        U[3][c] = 1
        V[0*p+c1][c] = 1; V[1*p+c1][c] = -1
        W[c1*n+0][c] = 1; W[c1*n+1][c] = 1
        
        # Term 4: (A00+A01)*B10
        s = 4
        c = base + s
        U[0][c] = 1; U[1][c] = 1
        V[1*p+c1][c] = 1
        W[c0*n+0][c] = -1; W[c1*n+0][c] = 1
        
        # Term 5: (A10-A00)*(B00+B01)
        s = 5
        c = base + s
        U[2][c] = 1; U[0][c] = -1
        V[0*p+c0][c] = 1; V[0*p+c1][c] = 1
        W[c1*n+1][c] = 1
        
        # Term 6: (A01-A11)*(B10+B00)
        s = 6
        c = base + s
        U[1][c] = 1; U[3][c] = -1
        V[1*p+c1][c] = 1; V[0*p+c0][c] = 1
        W[c0*n+0][c] = 1
        
        # Term 7: For column c2, need C02=A00*B02+A01*B12, C12=A10*B02+A11*B12
        # This term: A01*B02 -> contributes to C02 and C12
        s = 7
        c = base + s
        U[1][c] = 1
        V[0*p+c2][c] = 1
        W[c2*n+0][c] = 1; W[c2*n+1][c] = 1

    # Verify with random test
    import numpy as np
    np.random.seed(42)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_expected = A @ B
    
    # Compute C from decomposition
    C_computed = np.zeros((n, p))
    for r in range(rank):
        u_vec = np.array(U)[:, r]
        v_vec = np.array(V)[:, r]
        w_vec = np.array(W)[:, r]
        
        # u_vec is 4 elements: [A00, A01, A10, A11]
        # v_vec is 24 elements: [B00,B01,...,B0,11, B10,B11,...,B1,11]
        # w_vec is 24 elements: [W[0,0],W[0,1],W[1,0],W[1,1],...]
        
        a_val = (u_vec[0] * A[0,0] + u_vec[1] * A[0,1] + 
                 u_vec[2] * A[1,0] + u_vec[3] * A[1,1])
        b_val = sum(v_vec[i*m + j] * B[i, j] for i in range(m) for j in range(p))
        
        for k in range(p):
            for i in range(n):
                C_computed[i, k] += w_vec[k*n + i] * a_val * b_val
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Fall back to rank-42 Strassen
        rank = 42
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        num_pairs = p // 2  # 6
        
        for t in range(num_pairs):
            base = t * 7
            for s in range(7):
                c = base + s
                if s == 0:
                    U[0][c] = 1; U[3][c] = 1
                    V[0*p+2*t][c] = 1; V[1*p+2*t+1][c] = 1
                    W[2*t*n+0][c] = 1; W[(2*t+1)*n+1][c] = 1
                elif s == 1:
                    U[2][c] = 1; U[3][c] = 1
                    V[0*p+2*t][c] = 1
                    W[2*t*n+1][c] = 1; W[(2*t+1)*n+1][c] = -1
                elif s == 2:
                    U[0][c] = 1
                    V[0*p+2*t+1][c] = 1; V[1*p+2*t+1][c] = -1
                    W[(2*t+1)*n+0][c] = 1; W[(2*t+1)*n+1][c] = 1
                elif s == 3:
                    U[3][c] = 1
                    V[1*p+2*t][c] = 1; V[0*p+2*t][c] = -1
                    W[2*t*n+0][c] = 1; W[2*t*n+1][c] = 1
                elif s == 4:
                    U[0][c] = 1; U[1][c] = 1
                    V[1*p+2*t+1][c] = 1
                    W[2*t*n+0][c] = -1; W[(2*t+1)*n+0][c] = 1
                elif s == 5:
                    U[2][c] = 1; U[0][c] = -1
                    V[0*p+2*t][c] = 1; V[0*p+2*t+1][c] = 1
                    W[(2*t+1)*n+1][c] = 1
                elif s == 6:
                    U[1][c] = 1; U[3][c] = -1
                    V[1*p+2*t][c] = 1; V[1*p+2*t+1][c] = 1
                    W[2*t*n+0][c] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
