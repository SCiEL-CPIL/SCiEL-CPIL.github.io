"""Matrix multiplication 2,4,6 — rank-minimal exact tensor decomposition."""

import numpy as np
import time

DIMENSIONS = [2, 4, 6]


# EVOLVE-BLOCK-START
def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _strassen_2x2x2():
    """Return U, V, W for Strassen's 2x2x2 matrix multiplication, rank 7."""
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # p1 = (a+d)(e+h)
    U[0, 0] = 1.0; U[3, 0] = 1.0
    V[0, 0] = 1.0; V[3, 0] = 1.0
    W[0, 0] = 1.0; W[3, 0] = 1.0
    
    # p2 = (c+d)e
    U[2, 1] = 1.0; U[3, 1] = 1.0
    V[0, 1] = 1.0
    W[2, 1] = 1.0; W[3, 1] = -1.0
    
    # p3 = a(f-h)
    U[0, 2] = 1.0
    V[1, 2] = 1.0; V[3, 2] = -1.0
    W[1, 2] = 1.0; W[3, 2] = 1.0
    
    # p4 = d(g-e)
    U[3, 3] = 1.0
    V[2, 3] = 1.0; V[0, 3] = -1.0
    W[0, 3] = 1.0; W[2, 3] = 1.0
    
    # p5 = (a+b)h
    U[0, 4] = 1.0; U[1, 4] = 1.0
    V[3, 4] = 1.0
    W[0, 4] = -1.0; W[1, 4] = 1.0
    
    # p6 = (c-a)(e+f)
    U[2, 5] = 1.0; U[0, 5] = -1.0
    V[0, 5] = 1.0; V[1, 5] = 1.0
    W[3, 5] = 1.0
    
    # p7 = (b-d)(g+h)
    U[1, 6] = 1.0; U[3, 6] = -1.0
    V[2, 6] = 1.0; V[3, 6] = 1.0
    W[0, 6] = 1.0; W[1, 6] = 1.0
    
    return U, V, W


def _schoolbook_2x2x1():
    """Return U, V, W for 2x2 by 2x1 matrix multiplication, rank 2.
    A is 2x2, B is 2x1. C is 2x1.
    C0 = a*e + b*h
    C1 = c*e + d*h
    """
    U = np.zeros((4, 2), dtype=np.float64)
    V = np.zeros((2, 2), dtype=np.float64)
    W = np.zeros((2, 2), dtype=np.float64)
    
    # p1 = a*e
    U[0, 0] = 1.0
    V[0, 0] = 1.0
    W[0, 0] = 1.0
    
    # p2 = b*h
    U[1, 0] = 1.0
    V[1, 0] = 1.0
    W[0, 0] += 1.0 # Wait, C0 = ae + bh.
    # Let's do it properly.
    # p1 = a*e -> C0
    # p2 = b*h -> C0
    # p3 = c*e -> C1
    # p4 = d*h -> C1
    # Rank 4?
    # C0 = ae + bh
    # C1 = ce + dh
    # Can we do it in 2?
    # No, 4 scalar products are needed if we don't share.
    # But wait, the tensor is 2x2x1.
    # T[i, j, k] where i in 0,1 (row of C), j in 0,1 (col of A), k in 0 (col of B).
    # T[0,0,0]=1, T[0,1,0]=1, T[1,0,0]=1, T[1,1,0]=1.
    # This is rank 2?
    # T = e0 (e0 + e1) + e1 (e0 + e1) ? No.
    # T = (e0+e1) (e0+e1)? No.
    # T[0,0]=1, T[0,1]=1, T[1,0]=1, T[1,1]=1.
    # This is the outer product of [1,1] and [1,1].
    # Rank 1?
    # C0 = a*e + b*h.
    # C1 = c*e + d*h.
    # If we use 1 product: p = (a+b)(e+h) = ae + ah + be + bh.
    # We have extra terms ah, be.
    # So rank 1 is not enough.
    # Rank 2?
    # p1 = a*e, p2 = b*h, p3 = c*e, p4 = d*h.
    # This is rank 4.
    # Can we do better?
    # C0 = ae + bh
    # C1 = ce + dh
    # This is just 4 independent products.
    # Rank 4.
    
    # So 2x2x1 has rank 4?
    # Let's check.
    # T[i,j,k] = delta_{i,j} delta_{k,0}? No.
    # C_i = sum_j A_ij B_j.
    # T[i, j, k] where k is output index.
    # Output is 2x1. So k=0,1.
    # T[i, j, k] = 1 if i=k and j is the inner index?
    # No, the tensor is defined by T[i*m+j, j*p+k, k*n+i].
    # For n=2, m=2, p=1.
    # i in 0,1. j in 0,1. k in 0.
    # T[i*2+j, j*1+0, 0*2+i] = 1.
    # T[i*2+j, j, i] = 1.
    # i=0, j=0: T[0, 0, 0] = 1.
    # i=0, j=1: T[1, 1, 0] = 1.
    # i=1, j=0: T[2, 0, 1] = 1.
    # i=1, j=1: T[3, 1, 1] = 1.
    # This is 4 disjoint terms.
    # Rank 4.
    
    # So 2x2x1 rank is 4.
    
    U = np.zeros((4, 4), dtype=np.float64)
    V = np.zeros((2, 4), dtype=np.float64)
    W = np.zeros((2, 4), dtype=np.float64)
    
    U[0, 0] = 1.0; V[0, 0] = 1.0; W[0, 0] = 1.0
    U[1, 1] = 1.0; V[1, 1] = 1.0; W[0, 1] = 1.0
    U[2, 2] = 1.0; V[0, 2] = 1.0; W[1, 2] = 1.0
    U[3, 3] = 1.0; V[1, 3] = 1.0; W[1, 3] = 1.0
    
    return U, V, W


def _construct_rank36():
    """Construct a rank-36 decomposition.
    2x4x6 = 2 * (2x2x6).
    2x2x6 = 2x2x2 (Strassen, rank 7) + 2x2x1 (Schoolbook, rank 4).
    Total rank per 2x2x6 is 11.
    Total rank 22.
    Wait, 2x2x1 rank is 4.
    7 + 4 = 11.
    2 * 11 = 22.
    """
    n, m, p = 2, 4, 6
    dim_a = n * m  # 8
    dim_b = m * p  # 24
    dim_c = p * n  # 12
    
    rank = 22
    U = np.zeros((dim_a, rank), dtype=np.float64)
    V = np.zeros((dim_b, rank), dtype=np.float64)
    W = np.zeros((dim_c, rank), dtype=np.float64)
    
    U_str, V_str, W_str = _strassen_2x2x2()
    U_sb, V_sb, W_sb = _schoolbook_2x2x1()
    
    col = 0
    for a_block in range(2):
        # Strassen part for cols 0,1 of B (local)
        for s in range(7):
            r = col
            # A indices
            for idx in range(4):
                val = U_str[idx, s]
                if val != 0:
                    i = idx // 2
                    j_local = idx % 2
                    j_global = a_block * 2 + j_local
                    U[i * m + j_global, r] += val
            # B indices (cols 0,1)
            for idx in range(4):
                val = V_str[idx, s]
                if val != 0:
                    i_local = idx // 2
                    k_local = idx % 2
                    i_global = a_block * 2 + i_local
                    k_global = k_local # 0 or 1
                    V[i_global * p + k_global, r] += val
            # C indices (cols 0,1)
            for idx in range(4):
                val = W_str[idx, s]
                if val != 0:
                    i_out = idx // 2
                    k_out_local = idx % 2
                    k_out_global = k_out_local
                    W[k_out_global * n + i_out, r] += val
            col += 1
        
        # Schoolbook part for col 2 of B (local)
        for s in range(4):
            r = col
            # A indices
            for idx in range(4):
                val = U_sb[idx, s]
                if val != 0:
                    i = idx // 2
                    j_local = idx % 2
                    j_global = a_block * 2 + j_local
                    U[i * m + j_global, r] += val
            # B indices (col 2)
            for idx in range(2):
                val = V_sb[idx, s]
                if val != 0:
                    i_local = idx # 0 or 1
                    k_global = 2
                    i_global = a_block * 2 + i_local
                    V[i_global * p + k_global, r] += val
            # C indices (col 2)
            for idx in range(2):
                val = W_sb[idx, s]
                if val != 0:
                    i_out = idx
                    k_out_global = 2
                    W[k_out_global * n + i_out, r] += val
            col += 1
    
    return U, V, W


def _validate(U, V, W, T, tol=1e-6):
    """Check if U, V, W (each shape (dim, rank)) exactly reconstruct T."""
    T_hat = np.einsum('ir,jr,kr->ijk', U, V, W)
    return np.allclose(T_hat, T, atol=tol)


def construct():
    n, m, p = 2, 4, 6
    start_time = time.time()
    deadline = start_time + 110
    
    U, V, W = _construct_rank36()
    T = _build_tensor(n, m, p)
    
    if _validate(U, V, W, T, tol=1e-6):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to rank 42
    U, V, W = _construct_rank42()
    if _validate(U, V, W, T, tol=1e-6):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Fallback to schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
