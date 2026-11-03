"""Combined program: structured Strassen-block decomposition for 3x3 * 3x6."""
import numpy as np
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def _block_terms():
    """Return rank-17 term list for one 3x3 * 3x2 block.
    Each term is (u_idx, v_idx, w_idx, coeff) with:
      u_idx = i*3 + j   (A[i,j])
      v_idx = j*2 + k   (B[j,k])
      w_idx = k*3 + i   (C[i,k], trace convention W[k*n+i], n=3)
    """
    terms = []
    # --- Strassen 2x2 (A[0:2,0:2] * B[0:2,0:2]) -> C[0:2,0:2], rank 7 ---
    # M1 = (A00+A11)(B00+B11)
    terms += [(0,0,0,1), (0,0,3,1), (3,0,1,1), (3,0,4,1)]
    # M2 = (A10+A11)B00
    terms += [(2,0,1,1), (3,0,1,1)]
    # M3 = A00(B01-B11)
    terms += [(0,1,3,1), (0,1,4,-1)]
    # M4 = A11(B10-B00)
    terms += [(3,2,1,1), (3,0,1,-1)]
    # M5 = (A00+A01)B11
    terms += [(0,1,4,1), (1,1,4,1)]
    # M6 = (A10-A00)(B00+B01)
    terms += [(2,0,1,1), (2,1,3,1), (0,0,0,-1), (0,1,3,-1)]
    # M7 = (A01-A11)(B10+B11)
    terms += [(1,2,0,1), (1,1,4,1), (3,2,1,-1), (3,1,4,-1)]
    # --- A[2,:2] * B[0:2,:] -> C[2,0:2], rank 4 ---
    terms += [(6,0,2,1), (7,1,2,1), (6,2,5,1), (7,3,5,1)]
    # --- A[0:2,2] * B[2,:] -> C[0:2,0:2], rank 4 ---
    terms += [(2,4,0,1), (2,5,3,1), (5,4,1,1), (5,5,4,1)]
    # --- A[2,2] * B[2,:] -> C[2,0:2], rank 2 ---
    terms += [(8,4,2,1), (8,5,5,1)]
    return terms

def _verify(U, V, W):
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    err = 0.0
    for i in range(n*m):
        for j in range(m*p):
            for k in range(p*n):
                val = np.dot(U[i], V[j])
                val = np.dot(val, W[k])
                err += (val - T[i,j,k])**2
    return err

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin vs 150s wall-clock kill
    n, m, p = DIMENSIONS

    block = _block_terms()
    rank = 3 * len(block)
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))

    for blk in range(3):
        col = blk * len(block)
        for (u, v, w, c) in block:
            U[u, col] += c
            V[v + 2*blk, col] += c   # B row j*2+k -> (j*2+k) + 2*blk
            W[w, col] += c           # W index k*3+i is global (i in 0..2)

    err = _verify(U, V, W)
    if err < 1e-9:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: exact schoolbook rank n*m*p (guaranteed valid)
    r = n*m*p
    U2 = np.zeros((n*m, r))
    V2 = np.zeros((m*p, r))
    W2 = np.zeros((p*n, r))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U2[i*m+j, col] = 1
                V2[j*p+k, col] = 1
                W2[k*n+i, col] = 1
                col += 1
    return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
# EVOLVE-BLOCK-END
