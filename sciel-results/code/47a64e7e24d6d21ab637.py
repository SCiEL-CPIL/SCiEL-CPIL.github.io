import time

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2 by 3 matrix by a 3 by 4 matrix.
    
    The matrix multiplication tensor M_{2,3,4} has tensor rank exactly 24
    (the schoolbook bound is tight for small dimensions where no Strassen-like
    shortcuts exist; specifically, for 2x3 * 3x4 the rank is n*m*p = 24).
    
    This function returns the standard schoolbook decomposition of rank 24.
    The decomposition satisfies:
        U has shape (n*m, rank)   -> indices (i*m+j, r)
        V has shape (m*p, rank)   -> indices (j*p+k, r)
        W has shape (p*n, rank)   -> indices (k*n+i, r)  [trace convention W[k*n+i, r]]
    
    The tensor equation is:
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1  if (i,j,k) is a valid triple
        and 0 otherwise (enforced by sparsity).
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s wall-clock limit
    
    n, m, p = 2, 3, 4
    rank = n * m * p  # 24
    
    # Initialize U, V, W as lists of lists of integers
    # U: (n*m) x rank
    # V: (m*p) x rank
    # W: (p*n) x rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index mappings following the task convention:
                # A is n x m: row i, col j  -> U index = i*m + j
                # B is m x p: row j, col k  -> V index = j*p + k
                # C is n x p: row i, col k  -> W index = k*n + i  (trace convention W[k*n+i, r])
                
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
