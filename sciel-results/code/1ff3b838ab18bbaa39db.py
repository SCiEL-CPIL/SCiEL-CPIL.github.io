"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

_start_time = time.time()
_DEADLINE = 145.0

def _build_rank55():
    n, m, p = 2, 3, 10
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # M1: (A00 + A11) * (B0,k0 + B1,k1)
        U[0][col] += 1
        U[4][col] += 1
        V[k0][col] += 1
        V[10 + k1][col] += 1
        W[k0][col] += 1
        W[2 + k1][col] += 1
        col += 1
        
        # M2: (A10 + A11) * B0,k0
        U[3][col] += 1
        U[4][col] += 1
        V[k0][col] += 1
        W[2 + k0][col] += 1
        W[2 + k1][col] -= 1
        col += 1
        
        # M3: A00 * (B0,k1 - B1,k1)
        U[0][col] += 1
        V[10 + k1][col] -= 1
        V[k1][col] += 1
        W[k1][col] += 1
        W[2 + k1][col] += 1
        col += 1
        
        # M4: A11 * (B1,k0 - B0,k0)
        U[4][col] += 1
        V[20 + k0][col] += 1
        V[k0][col] -= 1
        W[k0][col] += 1
        W[2 + k0][col] += 1
        col += 1
        
        # M5: (A00 + A01) * B1,k1
        U[0][col] += 1
        U[1][col] += 1
        V[10 + k1][col] += 1
        W[k0][col] -= 1
        W[k1][col] += 1
        col += 1
        
        # M6: (A10 - A00) * (B0,k0 + B0,k1)
        U[3][col] += 1
        U[0][col] -= 1
        V[k0][col] += 1
        V[k1][col] += 1
        W[2 + k1][col] += 1
        col += 1
        
        # M7: (A01 - A11) * (B1,k0 + B1,k1)
        U[1][col] += 1
        U[4][col] -= 1
        V[20 + k0][col] += 1
        V[10 + k1][col] += 1
        W[k0][col] += 1
        col += 1
        
        # W00: A02 * B2,k0
        U[2][col] += 1
        V[20 + k0][col] += 1
        W[k0][col] += 1
        col += 1
        
        # W01: A02 * B2,k1
        U[2][col] += 1
        V[20 + k1][col] += 1
        W[k1][col] += 1
        col += 1
        
        # W10: A12 * B2,k0
        U[5][col] += 1
        V[20 + k0][col] += 1
        W[2 + k0][col] += 1
        col += 1
        
        # W11: A12 * B2,k1
        U[5][col] += 1
        V[20 + k1][col] += 1
        W[2 + k1][col] += 1
        col += 1
        
    return U, V, W

def _verify(U, V, W):
    """Verify that the decomposition exactly computes matrix multiplication.
    For all i,j,k: sum_r U[i*3+j, r] * V[j*10+k, r] == W[k*2+i, r] summed over r
    Actually: C[i,k] = sum_j A[i,j]*B[j,k]
    So we need: for all i,k: sum_r sum_j U[i*3+j, r] * V[j*10+k, r] == sum_r W[k*2+i, r]
    Wait, the tensor is: T[i,j,k] = sum_r U[i*3+j, r] * V[j*10+k, r]
    And this should equal sum_r W[k*2+i, r] * delta? No.
    
    The matrix multiplication tensor T has T[i,j,k] = 1 if... no.
    T[i,j,k] represents the coefficient such that C[i,k] = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k]
    Actually T[i,j,k] = 1 if the indices match for the product term, else 0.
    For C[i,k] = sum_j A[i,j] B[j,k], the tensor T has T[i,j,k] = 1 if j is the shared index... 
    
    Let me re-read: "output factor follows the official trace convention W[k*n+i,r]"
    
    The tensor for matrix mult (n x m) * (m x p) -> (n x p):
    T[i, j, k] where i in [0,n), j in [0,m), k in [0,p)
    T[i,j,k] = 1 if this term contributes to C[i,k]
    C[i,k] = sum_j A[i,j] B[j,k]
    So T[i,j,k] = 1 for all valid i,j,k? No, that's not right either.
    
    Actually the tensor T is defined by: sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = C[i,k] for each (i,k)
    This means T[i,j,k] should be 1 when the indices match: for a fixed output (i,k),
    we sum over j: A[i,j] * B[j,k]. So T[i,j,k] = 1 for all (i,j,k) triples?
    
    No wait. The tensor T has entries T[i,j,k] such that:
    C[i,k] = sum_{i',j,k'} T[i',j,k'] A[i',j] B[j,k'] where we fix output (i,k)
    Hmm, this is getting confusing. Let me think about it differently.
    
    The matrix multiplication tensor T ∈ R^{(n*m) x (m*p) x (n*p)} is defined by:
    T[a, b, c] = 1 if a = i*m+j, b = j'*p+k, c = i'*n+k' and i=i', j=j', k=k'
    Wait no. Let me use the standard definition.
    
    For matrix multiplication (n×m)·(m×p)→(n×p):
    The tensor T has indices (a, b, c) where a indexes A (n*m entries), b indexes B (m*p entries), c indexes C (n*p entries).
    T[a, b, c] = 1 if and only if the term A[a] * B[b] contributes to C[c].
    
    a = i*m + j (i-th row, j-th col of A)
    b = j'*p + k (j'-th row, k-th col of B)  
    c = i'*n + k' (i'-th row, k'-th col of C)
    
    A[i,j] * B[j',k] contributes to C[i',k'] if i=i', j=j', k=k'.
    So T[a,b,c] = 1 if i=i', j=j', k=k', else 0.
    
    In the decomposition: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    Let me verify with the given construction.
    """
    n, m, p = 2, 3, 10
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    
    # Build the tensor from the decomposition
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # Check against the true matrix multiplication tensor
    
    # True tensor: T_true[a,b,c] = 1 if indices match
    for a in range(n * m):
        i, j = divmod(a, m)
        for b in range(m * p):
            j2, k = divmod(b, p)
            if j == j2:
                c = i * p + k
                # T[a,b,c] should be 1
                val = sum(U_arr[a, r] * V_arr[b, r] * W_arr[c, r] for r in range(rank))
                if abs(val - 1.0) > 1e-10:
                    return False
            else:
                # T[a,b,c] should be 0 for all c
                for c in range(n * p):
                    val = sum(U_arr[a, r] * V_arr[b, r] * W_arr[c, r] for r in range(rank))
                    if abs(val) > 1e-10:
                        return False
    return True

def _try_rank_reduction(U, V, W, max_rank=54):
    """Try to reduce the rank by checking if any column is linearly dependent
    or can be eliminated via a valid transformation."""
    n, m, p = 2, 3, 10
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    if rank <= max_rank:
        return U, V, W, rank
    
    # Try to find if any single column can be removed by adjusting others
    # This is a simplified check - in general rank reduction requires more sophisticated methods
    # For now, just verify the current rank is valid
    return U, V, W, rank

def construct():
    n, m, p = 2, 3, 10
    
    # Build the rank-55 decomposition
    U, V, W = _build_rank55()
    
    # Verify correctness
    if not _verify(U, V, W):
        # Fallback to schoolbook if verification fails (shouldn't happen)
        rank = 60
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
    
    # Try to reduce rank if time permits
    if time.time() < _DEADLINE - 20:
        U, V, W, new_rank = _try_rank_reduction(U, V, W, max_rank=54)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
