import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, u_val=1, v_val=1, w_val=1):
        nonlocal col
        U[u_idx][col] = u_val
        V[v_idx][col] = v_val
        W[w_idx][col] = w_val
        col += 1

    # Helper to get indices
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i

    # Block 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # M1
    U[u_idx(0,0)][col] = 1; U[u_idx(1,1)][col] = 1
    V[v_idx(0,0)][col] = 1; V[v_idx(1,1)][col] = 1
    W[w_idx(0,0)][col] = 1
    W[w_idx(1,0)][col] = 1
    W[w_idx(0,1)][col] = 1
    W[w_idx(1,1)][col] = 1
    col += 1
    
    # M2
    U[u_idx(1,0)][col] = 1; U[u_idx(1,1)][col] = 1
    V[v_idx(0,0)][col] = 1
    W[w_idx(0,1)][col] = 1 # C[1,0] is k=0, i=1? No.
    # C[i,k]. W[k*n+i].
    # C[1,0] -> i=1, k=0 -> W[0*2+1] = W[1]
    # C[1,1] -> i=1, k=1 -> W[1*2+1] = W[3]
    W[w_idx(0,1)][col] = 1 # k=0, i=1
    W[w_idx(1,1)][col] = -1 # k=1, i=1
    col += 1

    # M3
    U[u_idx(0,0)][col] = 1
    V[v_idx(0,1)][col] = 1; V[v_idx(1,1)][col] = -1
    W[w_idx(1,0)][col] = 1 # C[0,1] -> k=1, i=0 -> W[2]
    W[w_idx(1,1)][col] = 1 # C[1,1] -> k=1, i=1 -> W[3]
    col += 1

    # M4
    U[u_idx(1,1)][col] = 1
    V[v_idx(1,0)][col] = 1; V[v_idx(0,0)][col] = -1
    W[w_idx(0,0)][col] = 1 # C[0,0] -> k=0, i=0 -> W[0]
    W[w_idx(0,1)][col] = 1 # C[1,0] -> k=0, i=1 -> W[1]
    col += 1

    # M5
    U[u_idx(0,0)][col] = 1; U[u_idx(0,1)][col] = 1
    V[v_idx(1,1)][col] = 1
    W[w_idx(0,0)][col] = -1 # C[0,0]
    W[w_idx(1,0)][col] = 1  # C[0,1]
    col += 1

    # M6
    U[u_idx(1,0)][col] = 1; U[u_idx(0,0)][col] = -1
    V[v_idx(0,0)][col] = 1; V[v_idx(0,1)][col] = 1
    W[w_idx(1,1)][col] = 1 # C[1,1]
    col += 1

    # M7
    U[u_idx(0,1)][col] = 1; U[u_idx(1,1)][col] = -1
    V[v_idx(1,0)][col] = 1; V[v_idx(1,1)][col] = 1
    W[w_idx(0,0)][col] = 1 # C[0,0]
    col += 1

    # Block 2a: Schoolbook A[0:2, 0:2] * B[0:2, 2] -> C[0:2, 2]
    # A[0,0] B[0,2] -> C[0,2]
    U[u_idx(0,0)][col] = 1
    V[v_idx(0,2)][col] = 1
    W[w_idx(2,0)][col] = 1 # k=2, i=0
    col += 1
    
    # A[0,1] B[1,2] -> C[0,2]
    U[u_idx(0,1)][col] = 1
    V[v_idx(1,2)][col] = 1
    W[w_idx(2,0)][col] = 1
    col += 1
    
    # A[1,0] B[0,2] -> C[1,2]
    U[u_idx(1,0)][col] = 1
    V[v_idx(0,2)][col] = 1
    W[w_idx(2,1)][col] = 1 # k=2, i=1
    col += 1
    
    # A[1,1] B[1,2] -> C[1,2]
    U[u_idx(1,1)][col] = 1
    V[v_idx(1,2)][col] = 1
    W[w_idx(2,1)][col] = 1
    col += 1

    # Block 2b: Strassen A[0:2, 0:2] * B[0:2, 3:5] -> C[0:2, 3:5]
    # Same as Block 1 but B cols 3,4 and C cols 3,4
    
    # M1
    U[u_idx(0,0)][col] = 1; U[u_idx(1,1)][col] = 1
    V[v_idx(0,3)][col] = 1; V[v_idx(1,4)][col] = 1
    W[w_idx(3,0)][col] = 1
    W[w_idx(3,1)][col] = 1
    W[w_idx(4,0)][col] = 1
    W[w_idx(4,1)][col] = 1
    col += 1
    
    # M2
    U[u_idx(1,0)][col] = 1; U[u_idx(1,1)][col] = 1
    V[v_idx(0,3)][col] = 1
    W[w_idx(3,1)][col] = 1
    W[w_idx(4,1)][col] = -1
    col += 1

    # M3
    U[u_idx(0,0)][col] = 1
    V[v_idx(0,4)][col] = 1; V[v_idx(1,4)][col] = -1
    W[w_idx(4,0)][col] = 1
    W[w_idx(4,1)][col] = 1
    col += 1

    # M4
    U[u_idx(1,1)][col] = 1
    V[v_idx(1,3)][col] = 1; V[v_idx(0,3)][col] = -1
    W[w_idx(3,0)][col] = 1
    W[w_idx(3,1)][col] = 1
    col += 1

    # M5
    U[u_idx(0,0)][col] = 1; U[u_idx(0,1)][col] = 1
    V[v_idx(1,4)][col] = 1
    W[w_idx(3,0)][col] = -1
    W[w_idx(4,0)][col] = 1
    col += 1

    # M6
    U[u_idx(1,0)][col] = 1; U[u_idx(0,0)][col] = -1
    V[v_idx(0,3)][col] = 1; V[v_idx(0,4)][col] = 1
    W[w_idx(4,1)][col] = 1
    col += 1

    # M7
    U[u_idx(0,1)][col] = 1; U[u_idx(1,1)][col] = -1
    V[v_idx(1,3)][col] = 1; V[v_idx(1,4)][col] = 1
    W[w_idx(3,0)][col] = 1
    col += 1

    # Block 3: Schoolbook A[0:2, 2] * B[2, 0:5] -> C[0:2, 0:5]
    
    # i=0
    for k in range(5):
        U[u_idx(0,2)][col] = 1
        V[v_idx(2,k)][col] = 1
        W[w_idx(k,0)][col] = 1
        col += 1
        
    # i=1
    for k in range(5):
        U[u_idx(1,2)][col] = 1
        V[v_idx(2,k)][col] = 1
        W[w_idx(k,1)][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
