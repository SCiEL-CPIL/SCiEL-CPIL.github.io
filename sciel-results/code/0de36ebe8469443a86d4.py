"""
Rank-49 exact tensor decomposition for 4x4 matrix multiplication.

This program constructs a rank-49 decomposition of the 4x4x4 matrix multiplication tensor
using the Kronecker product of the rank-7 Strassen decomposition for 2x2 matrix multiplication.

The tensor identity to satisfy is:
sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] = 1  for all i, j, k in {0, 1, 2, 3}
where n=m=p=4.
"""

import time
import numpy as np

def _get_strassen_matrices():
    """
    Returns the U, V, W matrices for the Strassen 2x2 decomposition.
    """
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)
    
    return U2, V2, W2

def _compute_permutation_UV():
    """
    Computes the permutation to map Kronecker product rows to standard row-major indices for U and V.
    Standard index S = i*4 + j.
    Kron index K = P1*4 + P2, where P1 = i1*2 + j1, P2 = i0*2 + j0.
    i = i1*2 + i0, j = j1*2 + j0.
    """
    perm = np.zeros(16, dtype=int)
    for S in range(16):
        i = S // 4
        j = S % 4
        i1, i0 = divmod(i, 2)
        j1, j0 = divmod(j, 2)
        P1 = i1 * 2 + j1
        P2 = i0 * 2 + j0
        K = P1 * 4 + P2
        perm[S] = K
    return perm

def _compute_permutation_W():
    """
    Computes the permutation to map Kronecker product rows to standard indices for W.
    Standard index S = k*4 + i (as per task description W[k*n+i]).
    Kron index K = Q1*4 + Q2, where Q1 = k1*2 + i1, Q2 = k0*2 + i0.
    k = k1*2 + k0, i = i1*2 + i0.
    """
    perm = np.zeros(16, dtype=int)
    for S in range(16):
        k = S // 4
        i = S % 4
        k1, k0 = divmod(k, 2)
        i1, i0 = divmod(i, 2)
        Q1 = k1 * 2 + i1
        Q2 = k0 * 2 + i0
        K = Q1 * 4 + Q2
        perm[S] = K
    return perm

def construct():
    """
    Constructs a rank-49 decomposition for 4x4 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 100  # Safety margin within 120s

    # Get the 2x2 Strassen matrices
    U2, V2, W2 = _get_strassen_matrices()
    
    # Kronecker product for 4x4
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    if time.time() > deadline:
        return _schoolbook_fallback()

    # Permutations
    perm_uv = _compute_permutation_UV()
    perm_w = _compute_permutation_W()
    
    # Apply permutations
    U4_fixed = U4[perm_uv, :]
    V4_fixed = V4[perm_uv, :]
    W4_fixed = W4[perm_w, :]
    
    if time.time() > deadline:
        return _schoolbook_fallback()

    # Verification step (optional but good for correctness)
    # We verify a few random points or all points if fast enough.
    # The tensor identity is sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r] = delta(i,k)
    # This is equivalent to checking if the reconstructed matrix product matches the identity
    # for basis inputs.
    
    # Let's do a quick check on a few points to catch gross errors.
    # Full check is 4*4*4*49 = 3136 operations, which is very fast.
    for i in range(4):
        for j in range(4):
            for k in range(4):
                u_val = U4_fixed[i * 4 + j]
                v_val = V4_fixed[j * 4 + k]
                w_val = W4_fixed[k * 4 + i]
                total = np.dot(np.dot(u_val, v_val), w_val)
                expected = 1 if i == k else 0
                if abs(total - expected) > 1e-9:
                    # If verification fails, fallback to schoolbook
                    return _schoolbook_fallback()
                if time.time() > deadline:
                    return _schoolbook_fallback()

    # Convert to lists of lists with integer entries
    U_out = U4_fixed.astype(int).tolist()
    V_out = V4_fixed.astype(int).tolist()
    W_out = W4_fixed.astype(int).tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}

def _schoolbook_fallback():
    """Fallback to rank-64 schoolbook decomposition."""
    n = 4
    rank = 64
    U = [[0] * rank for _ in range(n * n)]
    V = [[0] * rank for _ in range(n * n)]
    W = [[0] * rank for _ in range(n * n)]
    col = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                U[i * n + j][col] = 1
                V[j * n + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
