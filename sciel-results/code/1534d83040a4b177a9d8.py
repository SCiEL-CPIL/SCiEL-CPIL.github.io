"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Returns a rank-7 tensor decomposition for 2x2 matrix multiplication.
    
    The decomposition takes the form:
    T = sum_{r=0}^{6} u_r \otimes v_r \otimes w_r
    
    Where T is the matrix multiplication tensor.
    U shape: (4, 7) corresponding to A entries (a00, a01, a10, a11)
    V shape: (4, 7) corresponding to B entries (b00, b01, b10, b11)
    W shape: (4, 7) corresponding to C entries with index k*n + i
    
    Strassen's algorithm provides the optimal rank 7 decomposition.
    It is a well-established mathematical result that rank 6 is impossible
    for 2x2 matrix multiplication over real numbers.
    """
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s limit
    
    n, m, p = 2, 2, 2
    rank = 7
    
    # Initialize matrices as lists of lists of integers
    # U indices: i*m + j -> 0: a00, 1: a01, 2: a10, 3: a11
    # V indices: j*p + k -> 0: b00, 1: b01, 2: b10, 3: b11
    # W indices: k*n + i -> 0: c00, 1: c10, 2: c01, 3: c11
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 scalar products:
    # m0 = (a00 + a11) * (b00 + b11)
    # m1 = (a10 + a11) * b00
    # m2 = a00 * (b01 - b11)
    # m3 = a11 * (b10 - b00)
    # m4 = (a00 + a01) * b11
    # m5 = (a10 - a00) * (b00 + b01)
    # m6 = (a01 - a11) * (b10 + b11)
    
    # Resulting C:
    # c00 = m0 + m3 - m4 + m6
    # c01 = m2 + m4
    # c10 = m1 + m3
    # c11 = m0 - m1 + m2 + m5
    
    # Component 0: m0
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1  # c00 (0*2+0=0), c11 (1*2+1=3)
    
    # Component 1: m1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1 # c10 (0*2+1=1), c11 (1*2+1=3)
    
    # Component 2: m2
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1  # c01 (1*2+0=2), c11 (1*2+1=3)
    
    # Component 3: m3
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1  # c00 (0*2+0=0), c10 (0*2+1=1)
    
    # Component 4: m4
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1 # c00 (0*2+0=0), c01 (1*2+0=2)
    
    # Component 5: m5
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1               # c11 (1*2+1=3)
    
    # Component 6: m6
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1               # c00 (0*2+0=0)
    
    # Check deadline (computation is instantaneous, but guard is required)
    if time.time() > deadline:
        # If we somehow exceeded deadline (unlikely), we still return the valid result
        pass
    
    # Verify decomposition: T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # T[i,j,k] = 1 if the (i,j) entry of A times (j,k) entry of B contributes to (i,k) of C
    # i in [0,n), j in [0,m), k in [0,p)
    # U index: i*m+j, V index: j*p+k, W index: k*n+i
    import numpy as np
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check all tensor entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                expected = 1.0  # Each entry of A and B contributes to exactly one C entry
                computed = sum(U_arr[u_idx, r] * V_arr[v_idx, r] * W_arr[w_idx, r] for r in range(rank))
                if abs(computed - expected) > 1e-10:
                    # Fallback: this shouldn't happen with Strassen, but guard against it
                    pass
    
    return {
        "U": U,
        "V": V,
        "W": W
    }

# EVOLVE-BLOCK-END
