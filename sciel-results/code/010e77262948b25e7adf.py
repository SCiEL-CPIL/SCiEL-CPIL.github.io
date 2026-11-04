"""Rewritten program: search for low-rank exact tensor decomposition of 4x4 * 4x5."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 5]
DEADLINE = time.time() + 110  # safety margin before 150s kill

def _verify(U, V, W, n=4, m=4, p=5):
    """Check exactness: for all i,j,k, sum_r U[i*m+j,r]*V[j*p+k,r] == W[k*n+i,r]... 
    Actually the tensor decomposition means: T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]? 
    No, the standard form: the multiplication tensor T_{i,j,k}^{(n,m,p)} where C_{ik} = sum_j A_{ij} B_{jk}.
    The tensor T has entries T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k].
    Rank-r decomposition: T = sum_{r=1}^{rank} u_r * v_r * w_r where
    u_r[i,j] = U[i*m+j, r], v_r[j,k] = V[j*p+k, r], w_r[k,i] = W[k*n+i, r].
    So we need: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 (since T[i,j,k]=1 for the multiplication).
    Wait, actually the multiplication tensor for A (n x m) * B (m x p) -> C (n x p) is:
    T[i,j,k] = A[i,j] * B[j,k] contributes to C[i,k].
    As a 3-tensor: T[i,j,k] = 1 for the "identity" mapping? No.
    
    Let me re-read the task. The tensor for matrix multiplication: we have A (n*m), B (m*p), C (p*n) [note: W is indexed [k*n+i, r], so output dimension is p*n with index k*n+i].
    
    The standard formulation: C_{i,k} = sum_j A_{i,j} * B_{j,k}.
    The tensor T has indices (i, j, k) and T[i,j,k] = 1 (it's the "structure tensor").
    A rank-r decomposition writes: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    
    So we need: for all i in [0,n), j in [0,m), k in [0,p):
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # Start with schoolbook as fallback
    U_fb, V_fb, W_fb = _schoolbook()
    
    # Try to find a lower-rank decomposition
    # Strategy: initialize with random or structured tensors, then use ALS or similar
    # For exact decomposition, we need rational/half-integer entries ideally, but float is ok for scoring.
    
    # Let's try a gradient-based approach on the factor matrices
    # Objective: minimize sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
    # with some regularization to encourage sparsity/low-rank structure
    
    best_rank = n * m * p
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    
    # Try various ranks
    for rank in range(80, n*m*p + 1):  # 4*4*5 = 80, so just 80 first
        if time.time() > DEADLINE - 10:
            break
        # Initialize
        rng = np.random.RandomState(rank)
        U = rng.randn(n*m, rank) * 0.1
        V = rng.randn(m*p, rank) * 0.1
        W = rng.randn(p*n, rank) * 0.1
        
        # Use simple iterative updates
        # For each factor, fix the other two and solve
        lr = 0.1
        for it in range(200):
            if time.time() > DEADLINE - 5:
                break
            # Update U
            # For each (i,j), U[i*m+j, :] should satisfy:
            # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
            # This is a linear system: for fixed (i,j), let v_k[r] = V[j*p+k,r]*W[k*n+i,r] for k in range(p)
            # Then we need U[i*m+j,:] * v_k = 1 for all k
            # So solve: U[i*m+j,:] * VjW = ones(p)
            # where VjW[k,r] = V[j*p+k,r] * W[k*n+i,r]
            for i in range(n):
                for j in range(m):
                    # Build p x rank matrix
                    VjW = np.zeros((p, rank))
                    for k in range(p):
                        VjW[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    # Solve U[i*m+j, :] * VjW = ones(p)
                    # This is underdetermined if rank > p, use least squares
                    try:
                        U[i*m+j, :], residuals, rank_v, sv = np.linalg.lstsq(VjW.T, np.ones(p), rcond=None)
                    except:
                        pass
            # Update V
            for j in range(m):
                for k in range(p):
                    UkW = np.zeros((n, rank))
                    for i in range(n):
                        UkW[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        V[j*p+k, :], residuals, rank_v, sv = np.linalg.lstsq(UkW.T, np.ones(n), rcond=None)
                    except:
                        pass
            # Update W
            for k in range(p):
                for i in range(n):
                    UkV = np.zeros((m, rank))
                    for j in range(m):
                        UkV[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        W[k*n+i, :], residuals, rank_v, sv = np.linalg.lstsq(UkV.T, np.ones(m), rcond=None)
                    except:
                        pass
        
        # Check if this gives a valid decomposition
        if _verify(U, V, W, n, m, p):
            if rank < best_rank:
                best_rank = rank
                best_U = U.tolist()
                best_V = V.tolist()
                best_W = W.tolist()
            break  # Found valid at this rank, try lower
    
    # Return the best found
    return {"U": best_U, "V": best_V, "W": best_W}
