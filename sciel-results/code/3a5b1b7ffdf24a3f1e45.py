import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Try rank 22 construction (Strassen for 2x2*2x4 + outer product for 2x1*1x4)
    result = _try_rank_22()
    if result is not None:
        if _verify(result, n, m, p):
            return result
    
    # Try rank 23
    if time.time() < deadline:
        result = _try_rank_23()
        if result is not None:
            if _verify(result, n, m, p):
                return result
    
    # Fall back to schoolbook rank 24
    return _schoolbook()

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(decomp, n, m, p, trials=100):
    """Verify that the decomposition exactly decomposes matrix multiplication."""
    U = np.array(decomp["U"], dtype=np.float64)
    V = np.array(decomp["V"], dtype=np.float64)
    W = np.array(decomp["W"], dtype=np.float64)
    rank = U.shape[1]
    
    for _ in range(trials):
        # Random integer matrices
        A = np.random.randint(-5, 6, size=(n, m))
        B = np.random.randint(-5, 6, size=(m, p))
        C_expected = A @ B  # shape (n, p)
        
        # Compute using tensor decomposition
        # u = U^T A_flat, v = V^T B_flat, w = W^T C_flat
        A_flat = A.flatten()  # shape (n*m,), index i*m+j
        B_flat = B.flatten()  # shape (m*p,), index j*p+k
        C_flat = np.zeros(n * p)
        
        for r in range(rank):
            u_r = U[:, r] @ A_flat  # scalar
            v_r = V[:, r] @ B_flat  # scalar
            w_r = W[:, r]
            C_flat += u_r * v_r * w_r
        
        # C_flat has index k*n+i for C[i,k]
        # So C_flat[k*n + i] = C[i, k]
        C_reconstructed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                C_reconstructed[i, k] = C_flat[k * n + i]
        
        if not np.allclose(C_reconstructed, C_expected, atol=1e-10):
            return False
    
    return True

def _try_rank_22():
    """
    Construct rank-22 decomposition:
    - Split A (2x3) into A1 (2x2, cols 0,1) and A2 (2x1, col 2)
    - Split B (3x4) into B1 (2x4, rows 0,1) and B2 (1x4, row 2)
    - C = A1*B1 + A2*B2
    - A1*B1: 2x2 * 2x4, use 2 Strassen blocks (7+7=14 mults)
    - A2*B2: 2x1 * 1x4, outer product (8 mults)
    - Total: 22
    """
    n, m, p = DIMENSIONS
    rank = 22
    U = [[0] * rank for _ in range(n * m)]  # 6 rows
    V = [[0] * rank for _ in range(m * p)]  # 12 rows
    W = [[0] * rank for _ in range(p * n)]  # 8 rows
    
    # A1 entries (2x2):
    # A[0,0] -> U index 0*3+0 = 0
    # A[0,1] -> U index 0*3+1 = 1
    # A[1,0] -> U index 1*3+0 = 3
    # A[1,1] -> U index 1*3+1 = 4
    # A2 entry:
    # A[0,2] -> U index 0*3+2 = 2
    # A[1,2] -> U index 1*3+2 = 5
    
    # B1 entries (2x4):
    # B[0,0] -> V index 0*4+0 = 0
    # B[0,1] -> V index 0*4+1 = 1
    # B[0,2] -> V index 0*4+2 = 2
    # B[0,3] -> V index 0*4+3 = 3
    # B[1,0] -> V index 1*4+0 = 4
    # B[1,1] -> V index 1*4+1 = 5
    # B[1,2] -> V index 1*4+2 = 6
    # B[1,3] -> V index 1*4+3 = 7
    # B2 entries (1x4):
    # B[2,0] -> V index 2*4+0 = 8
    # B[2,1] -> V index 2*4+1 = 9
    # B[2,2] -> V index 2*4+2 = 10
    # B[2,3] -> V index 2*4+3 = 11
    
    # C entries (2x4), W index k*n+i = k*2+i:
    # C[0,0] -> W[0*2+0] = W[0]
    # C[0,1] -> W[1*2+0] = W[2]
    # C[0,2] -> W[2*2+0] = W[4]
    # C[0,3] -> W[3*2+0] = W[6]
    # C[1,0] -> W[0*2+1] = W[1]
    # C[1,1] -> W[1*2+1] = W[3]
    # C[1,2] -> W[2*2+1] = W[5]
    # C[1,3] -> W[3*2+1] = W[7]
    
    # Strassen for 2x2 * 2x2:
    # Given matrices [[a,b],[c,d]] and [[e,f],[g,h]]
    # P1 = (a+b)(e+h)
    # P2 = (c+d)(e)
    # P3 = (a-c)(g+h)
    # P4 = (d-b)(f+h)
    # P5 = (a+d)(e+f)
    # P6 = (b+d)(g-e)
    # P7 = (c+a)(h-f)
    # Result:
    # [P1+P4-P5+P7, P3+P5]
    # [P2+P6,        P1-P2+P3+P6]
    
    def add_strassen(u_a, u_b, u_c, u_d, v_e, v_f, v_g, v_h, w_00, w_01, w_10, w_11, col):
        """Add one Strassen 2x2 block at column col."""
        # P1 = (a+b)(e+h)
        U[u_a][col] += 1; U[u_b][col] += 1
        V[v_e][col] += 1; V[v_h][col] += 1
        W[w_00][col] += 1; W[w_01][col] += 0; W[w_10][col] += 0; W[w_11][col] += 1
        
        # P2 = (c+d)(e)
        U[u_c][col + 1] += 1; U[u_d][col + 1] += 1
        V[v_e][col + 1] += 1
        W[w_10][col + 1] += 1; W[w_11][col + 1] += -1
        
        # P3 = (a-c)(g+h)
        U[u_a][col + 2] += 1; U[u_c][col + 2] += -1
        V[v_g][col + 2] += 1; V[v_h][col + 2] += 1
        W[w_01][col + 2] += 1; W[w_11][col + 2] += 1
        
        # P4 = (d-b)(f+h)
        U[u_d][col + 3] += 1; U[u_b][col + 3] += -1
        V[v_f][col + 3] += 1; V[v_h][col + 3] += 1
        W[w_00][col + 3] += 1; W[w_11][col + 3] += 1
        
        # P5 = (a+d)(e+f)
        U[u_a][col + 4] += 1; U[u_d][col + 4] += 1
        V[v_e][col + 4] += 1; V[v_f][col + 4] += 1
        W[w_00][col + 4] += -1; W[w_01][col + 4] += 1; W[w_10][col + 4] += 0; W[w_11][col + 4] += 0
        
        # P6 = (b+d)(g-e)
        U[u_b][col + 5] += 1; U[u_d][col + 5] += 1
        V[v_g][col + 5] += 1; V[v_e][col + 5] += -1
        W[w_10][col + 5] += 1; W[w_11][col + 5] += 1
        
        # P7 = (c+a)(h-f)
        U[u_c][col + 6] += 1; U[u_a][col + 6] += 1
        V[v_h][col + 6] += 1; V[v_f][col + 6] += -1
        W[w_00][col + 6] += 1; W[w_01][col + 6] += 0; W[w_10][col + 6] += 0; W[w_11][col + 6] += 1
    
    # Strassen block 1: A1 * B1[:, 0:2]
    # A1: a=A[0,0]=U[0], b=A[0,1]=U[1], c=A[1,0]=U[3], d=A[1,1]=U[4]
    # B1[:,0:2]: e=B[0,0]=V[0], f=B[0,1]=V[1], g=B[1,0]=V[4], h=B[1,1]=V[5]
    # W: C[0,0]=W[0], C[0,1]=W[2], C[1,0]=W[1], C[1,1]=W[3]
    add_strassen(0, 1, 3, 4, 0, 1, 4, 5, 0, 2, 1, 3, 0)
    
    # Strassen block 2: A1 * B1[:, 2:4]
    # B1[:,2:4]: e=B[0,2]=V[2], f=B[0,3]=V[3], g=B[1,2]=V[6], h=B[1,3]=V[7]
    # W: C[0,2]=W[4], C[0,3]=W[6], C[1,2]=W[5], C[1,3]=W[7]
    add_strassen(0, 1, 3, 4, 2, 3, 6, 7, 4, 6, 5, 7, 7)
    
    # Outer product: A2 * B2
    # A2: A[0,2]=U[2], A[1,2]=U[5]
    # B2: B[2,0]=V[8], B[2,1]=V[9], B[2,2]=V[10], B[2,3]=V[11]
    # C[0,k] += A[0,2]*B[2,k], C[1,k] += A[1,2]*B[2,k]
    # W indices: C[0,0]=W[0], C[0,1]=W[2], C[0,2]=W[4], C[0,3]=W[6]
    #            C[1,0]=W[1], C[1,1]=W[3], C[1,2]=W[5], C[1,3]=W[7]
    
    for k in range(4):
        col = 14 + k
        U[2][col] = 1  # A[0,2]
        U[5][col] = 1  # A[1,2]
        V[8 + k][col] = 1  # B[2,k]
        W[k * 2 + 0][col] = 1  # C[0,k]
        W[k * 2 + 1][col] = 1  # C[1,k]
    
    return {"U": U, "V": V, "W": W}

def _try_rank_23():
    """Try a rank-23 construction if rank 22 fails."""
    # For now, just return None to fall back to schoolbook
    return None

# EVOLVE-BLOCK-END
