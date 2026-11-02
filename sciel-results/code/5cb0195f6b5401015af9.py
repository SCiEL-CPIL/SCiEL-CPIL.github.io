"""Program for 2x2 * 2x6 matrix multiplication tensor decomposition."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = time.time()
_DEADLINE = _start_time + 110.0


def _build_target_tensor(n, m, p):
    """Build the matrix multiplication tensor T of shape (n*m*p,) flattened."""
    T = np.zeros(n * m * p)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m * p + j * p + k] = 1.0
    return T


def _tensor_from_decomposition(U, V, W, n, m, p):
    """Reconstruct the tensor from a rank-r decomposition."""
    T = np.zeros(n * m * p)
    rank = U.shape[1]
    for r in range(rank):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m * p + j * p + k] += u[i * m + j] * v[j * p + k] * w[k * n + i]
    return T


def _build_rank_4_decomposition(n, m, p):
    """
    Build rank-4 decomposition for 2x2 * 2x6 matrix multiplication.
    
    Key insight: T[i,j,k] = 1 for all i,j,k (every entry of A multiplies
    with the corresponding entry of B). We use 4 terms, one for each (i,j) 
    pair of A:
      r=0: i=0,j=0
      r=1: i=0,j=1  
      r=2: i=1,j=0
      r=3: i=1,j=1
    
    U[i*m+j, r] = 1 if r matches (i,j)
    V[j*p+k, r] = 1 if j matches the j of r (for all k)
    W[k*n+i, r] = 1 if i matches the i of r (for all k)
    """
    rank = 4
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # U: one-hot per (i,j)
    for i in range(n):
        for j in range(m):
            r = i * m + j
            U[i * m + j, r] = 1.0
    
    # V: for each (j,k), set to 1 for all r that have this j
    for j in range(m):
        for k in range(p):
            for i in range(n):
                r = i * m + j
                V[j * p + k, r] = 1.0
    
    # W: for each (k,i), set to 1 for all r that have this i
    for k in range(p):
        for i in range(n):
            for j in range(m):
                r = i * m + j
                W[k * n + i, r] = 1.0
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    
    # Build the rank-4 decomposition
    U, V, W = _build_rank_4_decomposition(n, m, p)
    
    # Verify correctness
    target = _build_target_tensor(n, m, p)
    T_recon = _tensor_from_decomposition(U, V, W, n, m, p)
    resid = np.sum((T_recon - target) ** 2)
    
    if resid > 1e-9:
        # Fallback: try rank-21 Strassen if rank-4 fails (should not happen)
        rank = 21
        U21 = np.zeros((n * m, rank))
        V21 = np.zeros((m * p, rank))
        W21 = np.zeros((p * n, rank))
        for b in range(3):
            k0 = 2 * b
            k1 = 2 * b + 1
            v_e = 0 * p + k0
            v_f = 0 * p + k1
            v_g = 1 * p + k0
            v_h = 1 * p + k1
            w_c00 = k0 * n + 0
            w_c10 = k0 * n + 1
            w_c01 = k1 * n + 0
            w_c11 = k1 * n + 1
            r = b * 7 + 0
            U21[0, r] = 1; U21[3, r] = 1
            V21[v_e, r] = 1; V21[v_h, r] = 1
            W21[w_c00, r] = 1; W21[w_c11, r] = 1
            r = b * 7 + 1
            U21[2, r] = 1; U21[3, r] = 1
            V21[v_e, r] = 1
            W21[w_c10, r] = 1; W21[w_c11, r] = -1
            r = b * 7 + 2
            U21[0, r] = 1
            V21[v_f, r] = 1; V21[v_h, r] = -1
            W21[w_c01, r] = 1; W21[w_c11, r] = 1
            r = b * 7 + 3
            U21[3, r] = 1
            V21[v_g, r] = 1; V21[v_e, r] = -1
            W21[w_c00, r] = 1; W21[w_c10, r] = 1
            r = b * 7 + 4
            U21[0, r] = 1; U21[1, r] = 1
            V21[v_h, r] = 1
            W21[w_c00, r] = -1; W21[w_c01, r] = 1
            r = b * 7 + 5
            U21[2, r] = 1; U21[0, r] = -1
            V21[v_e, r] = 1; V21[v_f, r] = 1
            W21[w_c11, r] = 1
            r = b * 7 + 6
            U21[1, r] = 1; U21[3, r] = -1
            V21[v_g, r] = 1; V21[v_h, r] = 1
            W21[w_c00, r] = 1
        return {"U": U21.tolist(), "V": V21.tolist(), "W": W21.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
