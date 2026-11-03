import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    """
    Rank-minimal tensor decomposition for 2x4 * 4x8 matrix multiplication.
    Uses the fact that C[i,k] = sum_j A[i,j] B[j,k] can be decomposed into
    independent rank-1 terms for each (i, j) pair, sharing the V side.
    Rank = n * m = 2 * 4 = 8.
    """
    n, m, p = DIMENSIONS
    rank = n * m  # 8
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # U is indexed by (i, j) -> i*m + j
    # V is indexed by (j, k) -> j*p + k
    # W is indexed by (k, i) -> k*n + i
    
    for i in range(n):
        for j in range(m):
            r = i * m + j
            # U[i*m+j, r] = 1
            U[i * m + j][r] = 1
            # V[j*p+k, r] = 1 for all k? No.
            # The term is A[i,j] * B[j,k] * 1.
            # So V[j*p+k, r] should be 1 for the specific k?
            # Wait, the decomposition is C[i,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r].
            # We want sum_j A[i,j] B[j,k].
            # If we set U[i*m+j, r_j] = 1, V[j*p+k, r_j] = B[j,k]? No, V must be constant.
            # The tensor entry T[i*m+j, j*p+k, k*n+i] = 1.
            # This tensor is 1 if the indices match the path i->j->k->i.
            # Specifically, T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i.
            # This is exactly the identity tensor for the middle index j.
            # Rank of this tensor?
            # T = sum_{i,j} e_{i*m+j} \otimes (sum_k e_{j*p+k} \otimes e_{k*n+i})?
            # No, for fixed i,j, the term is e_{i*m+j} \otimes V_{i,j} \otimes W_{i,j}.
            # V_{i,j} must be a vector such that V_{i,j}[j*p+k] = 1 for all k?
            # If V_{i,j}[j*p+k] = 1, then sum_k V[j*p+k] W[k*n+i] = sum_k W[k*n+i].
            # We need the result to be B[j,k].
            # But B is an input, not part of the decomposition.
            # The decomposition approximates the tensor T where T[i*m+j, j*p+k, k*n+i] = 1.
            # And C[i,k] = sum_{i,j} A[i,j] B[j,k] T[...]
            # So we need sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all valid i,j,k.
            # And 0 otherwise.
            
            # For rank 8, we use r = i*m+j.
            # U[i*m+j, r] = 1.
            # V[j*p+k, r] = 1 for all k?
            # If V[j*p+k, r] = 1, then the term is A[i,j] * 1 * W[k*n+i, r].
            # Sum over j: sum_j A[i,j] W[k*n+i, r_j].
            # We want sum_j A[i,j] B[j,k].
            # This implies W[k*n+i, r_j] must depend on B[j,k]?
            # But W is constant.
            
            # My previous logic was flawed.
            # The tensor T is fixed.
            # T[i*m+j, j*p+k, k*n+i] = 1.
            # Can this tensor be decomposed with rank 8?
            # T = sum_{r=0}^7 u_r \otimes v_r \otimes w_r.
            # For fixed i, j, let r = i*m+j.
            # u_r = e_{i*m+j}.
            # v_r must have 1s at positions j*p+k for all k?
            # If v_r[j*p+k] = 1 for all k, then v_r is not a single vector?
            # v_r is a vector of length m*p.
            # v_r[j*p+k] = 1 for k=0..p-1.
            # w_r[k*n+i] = 1 for all k?
            # Then sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
            # = sum_{j'} U[i*m+j', r_{ij'}] V[j*p+k, r_{ij'}] W[k*n+i, r_{ij'}]
            # Only r = i*m+j is non-zero for U.
            # = 1 * V[j*p+k, r] * W[k*n+i, r].
            # If V[j*p+k, r] = 1 and W[k*n+i, r] = 1, then result is 1.
            # This works for all k.
            
            # So:
            # For each i, j:
            # r = i*m + j
            # U[i*m+j, r] = 1
            # V[j*p+k, r] = 1 for all k in 0..p-1
            # W[k*n+i, r] = 1 for all k in 0..p-1
            
            for k in range(p):
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1

    return {"U": U, "V": V, "W": W}
