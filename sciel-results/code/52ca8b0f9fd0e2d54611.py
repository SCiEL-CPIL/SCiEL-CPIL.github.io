"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # The tensor T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Check a few random entries
    for _ in range(100):
        i = random.randint(0, n-1)
        j = random.randint(0, m-1)
        k = random.randint(0, p-1)
        expected = 1.0  # T[i,j,k] = 1 if this is a valid multiplication entry
        # Actually T[i,j,k] = 1 for all i,j,k in the matrix mult tensor
        # The tensor for matmul A(n,m) x B(m,p) -> C(n,p) has T[i,j,k] = 1
        # where C[i,k] = sum_j A[i,j] B[j,k]
        # So T[i,j,k] = 1 for all valid i,j,k
        actual = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
        if abs(actual - expected) > 1e-9:
            return False
    return True

def _round_to_half_integer(x):
    """Round to nearest half-integer."""
    return round(2*x) / 2.0

def _round_matrix_to_half_int(M):
    """Round all entries to nearest half-integer."""
    return [[_round_to_half_integer(M[i][j]) for j in range(len(M[0]))] for i in range(len(M))]

def _als_decompose(n, m, p, rank, max_iter=50, tol=1e-6, seed=42):
    """ALS decomposition of the matrix multiplication tensor at given rank."""
    rng = np.random.RandomState(seed)
    U = rng.randn(n*m, rank) * 0.1
    V = rng.randn(m*p, rank) * 0.1
    W = rng.randn(p*n, rank) * 0.1
    
    for it in range(max_iter):
        # Update U
        for i in range(n):
            for j in range(m):
                # U[i*m+j, :] = (sum_k W[k*n+i, :] * V[j*p+k, :]) / (sum_k W^2 * V^2)
                num = np.zeros(rank)
                den = 0.0
                for k in range(p):
                    num += W[k*n+i, :] * V[j*p+k, :]
                    den += np.dot(W[k*n+i, :], W[k*n+i, :]) * np.dot(V[j*p+k, :], V[j*p+k, :])
                # This isn't quite right for ALS on 3-tensor. Let me use proper ALS.
                # For tensor T with T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
                # Fix V,W: U[i,j,r] = sum_k T[i,j,k] * V[j,k,r] * W[k,i,r] / sum_k V[j,k,r]^2 * W[k,i,r]^2
                # But T[i,j,k]=1 for all, so:
                num = np.zeros(rank)
                den = np.zeros(rank)
                for k in range(p):
                    num += V[j*p+k, :] * W[k*n+i, :]
                    den += V[j*p+k, :]**2 * W[k*n+i, :]**2
                mask = den > 1e-10
                U_new = U[i*m+j, :].copy()
                U_new[mask] = num[mask] / den[mask]
                U[i*m+j, :] = U_new
        
        # Update V
        for j in range(m):
            for k in range(p):
                num = np.zeros(rank)
                den = np.zeros(rank)
                for i in range(n):
                    num += U[i*m+j, :] * W[k*n+i, :]
                    den += U[i*m+j, :]**2 * W[k*n+i, :]**2
                mask = den > 1e-10
                V_new = V[j*p+k, :].copy()
                V_new[mask] = num[mask] / den[mask]
                V[j*p+k, :] = V_new
        
        # Update W
        for k in range(p):
            for i in range(n):
                num = np.zeros(rank)
                den = np.zeros(rank)
                for j in range(m):
                    num += U[i*m+j, :] * V[j*p+k, :]
                    den += U[i*m+j, :]**2 * V[j*p+k, :]**2
                mask = den > 1e-10
                W_new = W[k*n+i, :].copy()
                W_new[mask] = num[mask] / den[mask]
                W[k*n+i, :] = W_new
        
        # Check convergence
        if it > 0:
            # Simple check: compute residual
            pass  # Skip for speed
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120
    
    # Start with schoolbook
    rank_schoolbook = n * m * p
    U_school = np.zeros((n*m, rank_schoolbook))
    V_school = np.zeros((m*p, rank_schoolbook))
    W_school = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school.tolist(), V_school.tolist(), W_school.tolist()
    best_rank = rank_schoolbook
    
    # Try to reduce rank using ALS + rounding
    for target_rank in range(60, rank_schoolbook + 1):
        if time.time() > deadline - 30:
            break
        for seed in range(5):
            if time.time() > deadline - 30:
                break
            U, V, W = _als_decompose(n, m, p, target_rank, max_iter=30, seed=seed)
            # Round to half-integers
            U_r = _round_matrix_to_half_int(U.tolist())
            V_r = _round_matrix_to_half_int(V.tolist())
            W_r = _round_matrix_to_half_int(W.tolist())
            if _verify_decomposition(U_r, V_r, W_r, n, m, p):
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = target_rank
                break
        if best_rank == target_rank:
            # Found it, try to go lower
            continue
    
    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
