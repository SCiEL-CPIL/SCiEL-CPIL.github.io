import numpy as np
import time as _time
import random

_DEADLINE = 130.0

def _try_find_2x3x3_rank(target_rank):
    """
    Attempts to find a rank decomposition of the 2x3x3 matrix multiplication tensor.
    Returns (U, V, W) lists if found, else None.
    """
    # Tensor T[i,j,k] = 1 if j is the connecting index, else 0?
    # No, the tensor is defined by the multiplication.
    # T[i,j,k] is the coefficient of A[i,j] B[j,k] in C[i,k].
    # So T[i,j,k] = 1 if the indices match the multiplication structure.
    # Actually, the tensor is T = sum_{j} e_i \otimes e_j \otimes e_k * delta_{jk}? No.
    # The multiplication tensor is T = sum_{i,j,k} e_i \otimes e_j \otimes e_k such that C_ik = sum_j A_ij B_jk.
    # So T[i,j,k] = 1 if j is the index summed over.
    # In the standard tensor notation for matmul M_{n,m,p}:
    # T[i,j,k] = 1 if the entry A[i,j] and B[j,k] contribute to C[i,k].
    # So for 2x3x3:
    # i in 0..1, j in 0..2, k in 0..2.
    # T[i,j,k] = 1 for all i,j,k? No.
    # The tensor is T = sum_{i=0}^1 sum_{j=0}^2 sum_{k=0}^2 e_i \otimes e_j \otimes e_k * (1 if j is the middle index).
    # Wait, the tensor is fixed. T[i,j,k] = 1 if the indices are consistent.
    # For matmul, T[i,j,k] = 1 if j is the index connecting A and B.
    # So T[i,j,k] = 1 for all i,j,k in the ranges?
    # No, T is a 3-tensor. T[i,j,k] = 1 if the term A[i,j]B[j,k] exists.
    # So T[i,j,k] = 1 for all i in [0,1], j in [0,2], k in [0,2].
    # Yes, because every A[i,j] multiplies with every B[j,k] to contribute to C[i,k].
    # So the tensor is all 1s?
    # No, the tensor is T[i,j,k] = 1.
    # We want to decompose T = sum_r u_r \otimes v_r \otimes w_r.
    # Such that sum_r u_r[i] v_r[j] w_r[k] = 1 for all i,j,k?
    # No, that's not right.
    # The condition is: for all i,k, sum_j T[i,j,k] A[i,j] B[j,k] = C[i,k].
    # This is equivalent to: sum_j A[i,j] B[j,k] = sum_r (sum_j u_r[i] A[i,j] v_r[j]) (sum_k w_r[k] B[j,k]?? No).
    # The decomposition is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # We approximate this with sum_r P_r(A) Q_r(B).
    # P_r(A) = sum_j u_r[i] v_r[j] A[i,j]? No.
    # P_r is a linear form in A. P_r = sum_{i,j} u_r[i] v_r[j] A[i,j].
    # Q_r is a linear form in B. Q_r = sum_{j,k} w_r[j] z_r[k] B[j,k].
    # Then C[i,k] = sum_r P_r(A) Q_r(B) ?
    # No, C[i,k] is a matrix. The decomposition gives C[i,k] = sum_r (u_r \otimes v_r \otimes w_r) contracted with A and B.
    # Specifically: C[i,k] = sum_r (sum_{j} u_r[i] v_r[j] A[i,j]) * (sum_{j'} w_r[j'] z_r[k] B[j',k])?
    # The standard form is:
    # C[i,k] = sum_r (sum_{j} U[r, i, j] A[i,j]) (sum_{j'} V[r, j', k] B[j',k])?
    # No, the factors are:
    # U: shape (n, rank) ? No, U is (n*m, rank).
    # Let's use the interface definition.
    # U: (n*m, rank). V: (m*p, rank). W: (p*n, rank).
    # C[i,k] = sum_r W[r, k*n+i] * (sum_{j} U[r, i*m+j] A[i,j]) * (sum_{j'} V[r, j'*p+k] B[j',k])?
    # No, the standard tensor decomposition for matmul is:
    # T[i,j,k] = sum_r U[r,i,j] V[r,j,k] W[r,i,k]?
    # The task says:
    # U: (n*m, rank). Index (i*m + j, r).
    # V: (m*p, rank). Index (j*p + k, r).
    # W: (p*n, rank). Index (k*n + i, r).
    # Then C[i,k] = sum_j A[i,j] B[j,k] = sum_r W[r, k*n+i] * (sum_j U[r, i*m+j] A[i,j]) * (sum_j V[r, j*p+k] B[j,k])?
    # No, that would be a product of three sums.
    # The correct identity is:
    # C[i,k] = sum_r W[r, k*n+i] * (sum_j U[r, i*m+j] A[i,j]) * (sum_j V[r, j*p+k] B[j,k]) is WRONG.
    # The correct identity for tensor decomposition of multiplication is:
    # C[i,k] = sum_r (sum_j U[r, i*m+j] A[i,j]) * (sum_j V[r, j*p+k] B[j,k]) * W[r, k*n+i] ?
    # No, usually it is:
    # C[i,k] = sum_r P_r(A) Q_r(B)
    # where P_r(A) = sum_{i,j} U[r, i*m+j] A[i,j]
    # and Q_r(B) = sum_{j,k} V[r, j*p+k] B[j,k]
    # and the result is added to C[i,k] with coefficient W[r, k*n+i].
    # So C[i,k] = sum_r W[r, k*n+i] P_r(A) Q_r(B).
    # We need this to equal sum_j A[i,j] B[j,k].
    # This implies that the tensor T[i,j,k] (where T[i,j,k] is the coeff of A[i,j]B[j,k] in C[i,k]) must be decomposed.
    # T[i,j,k] = 1 if j is the middle index.
    # So we need:
    # sum_r W[r, k*n+i] U[r, i*m+j] V[r, j*p+k] = 1 for all i,j,k.
    # Note that U depends on i,j. V depends on j,k. W depends on i,k.
    # This is a very constrained decomposition.

    # For 2x3x3:
    # n=2, m=3, p=3.
    # i in 0..1, j in 0..2, k in 0..2.
    # We need sum_r W[r, k*2+i] U[r, i*3+j] V[r, j*3+k] = 1.

    pass

def construct():
    _t0 = _time.monotonic()
    n, m, p = 2, 4, 6
    
    # Fallback to the known rank 42 solution
    rank = 42
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    
    U_tbl = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (0, 0, -1)],
        [(0, 1, 1), (1, 1, -1)]
    ]
    V_tbl = [
        [(0, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(0, 1, 1), (1, 1, -1)],
        [(1, 0, 1), (0, 0, -1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (1, 1, 1)]
    ]
    W_tbl = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]
    
    for a in range(2):
        j_off = a * 2
        for b in range(3):
            k_off = b * 2
            for s in range(7):
                col = a * 21 + b * 7 + s
                if col >= rank:
                    continue
                if _time.monotonic() - _t0 > _DEADLINE:
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
                for i_loc, j_loc, c in U_tbl[s]:
                    U[i_loc * m + (j_off + j_loc), col] += c
                for j_loc, k_loc, c in V_tbl[s]:
                    V[(j_off + j_loc) * p + (k_off + k_loc), col] += c
                for i_loc, k_loc, c in W_tbl[s]:
                    W[(k_off + k_loc) * n + i_loc, col] += c
                    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
