"""Matrix multiplication 2,2,9 rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [2, 2, 9]

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
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

def _check_decomp(U, V, W, tol=1e-9):
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    if U.shape[0] != n * m:
        return False
    if V.shape[0] != m * p:
        return False
    if W.shape[0] != p * n:
        return False
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    # Build tensor T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # T should equal the standard multiplication tensor: T[i,j,k] = 1 for all i,j,k
    # Actually, the multiplication tensor for A(n x m) * B(m x p) = C(n x p)
    # has T[i, j, k] such that C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: C[i,k] = sum_{j,c} A[i,j] B[j,k] * (delta structure)
    # The decomposition means: for each c, U[:,c], V[:,c], W[:,c] contribute
    # The condition is that the tensor T_{ijk} = sum_c U_{im+j,c} V_{jp+k,c} W_{kn+i,c} 
    # must equal the standard multiplication tensor.
    # For matrix multiplication, the standard tensor has T_{i,j,k} = 1 for the specific structure.
    # More precisely: C[i,k] = sum_j A[i,j] B[j,k]
    # This corresponds to tensor M with M[i,j,k] = 1 (since we're multiplying, each term A[i,j]*B[j,k] contributes to C[i,k])
    # Wait, let me reconsider. The tensor for matrix mult M_{n,m,p} has entries M[i,j,k] = 1 for all i,j,k? No.
    # The multiplication is: C[i,k] = sum_{j} A[i,j] B[j,k]
    # So the tensor T has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # And the decomposition gives: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = T[i,j,k] = 1
    # So we need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k.
    
    # Compute for all (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _reduce_single(U, V, W, deadline):
    """Try to remove one term at a time."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    best = (rank, U, V, W)
    
    for attempt in range(rank):
        if time.time() > deadline - 5:
            break
        # Try removing term c
        c = attempt
        new_rank = rank - 1
        if new_rank < 1:
            break
        # Create copies without column c
        U_new = [row[:c] + row[c+1:] for row in U]
        V_new = [row[:c] + row[c+1:] for row in V]
        W_new = [row[:c] + row[c+1:] for row in W]
        
        if _check_decomp(U_new, V_new, W_new):
            return (new_rank, U_new, V_new, W_new)
    
    return best

def _reduce_pair(U, V, W, deadline):
    """Try to remove two terms at a time."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    best = (rank, U, V, W)
    
    tried = 0
    for c1 in range(rank):
        if time.time() > deadline - 5:
            break
        for c2 in range(c1 + 1, rank):
            if time.time() > deadline - 5:
                break
            tried += 1
            new_rank = rank - 2
            if new_rank < 1:
                break
            # Remove columns c1 and c2
            cols_to_remove = {c1, c2}
            U_new = [ [row[c] for c in range(rank) if c not in cols_to_remove] for row in U ]
            V_new = [ [row[c] for c in range(rank) if c not in cols_to_remove] for row in V ]
            W_new = [ [row[c] for c in range(rank) if c not in cols_to_remove] for row in W ]
            
            if _check_decomp(U_new, V_new, W_new):
                return (new_rank, U_new, V_new, W_new)
    
    return best

def construct():
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Start with schoolbook
    result = _schoolbook()
    U, V, W = result["U"], result["V"], result["W"]
    rank = len(U[0])
    
    # Try iterative single-term removal
    for _ in range(20):
        if time.time() > deadline - 10:
            break
        new_rank, U2, V2, W2 = _reduce_single(U, V, W, deadline)
        if new_rank < rank:
            rank = new_rank
            U, V, W = U2, V2, W2
        else:
            break
    
    # Try iterative pair-term removal
    for _ in range(10):
        if time.time() > deadline - 10:
            break
        new_rank, U2, V2, W2 = _reduce_pair(U, V, W, deadline)
        if new_rank < rank:
            rank = new_rank
            U, V, W = U2, V2, W2
        else:
            break
    
    return {"U": U, "V": V, "W": W}
