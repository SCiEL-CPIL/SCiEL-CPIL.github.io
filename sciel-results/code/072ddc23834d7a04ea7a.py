"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from scipy.optimize import linprog

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START

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


def _verify(U, V, W, tol=1e-6):
    """Check that U @ diag? No, the tensor equation is:
    For each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    and for all other combinations it should be 0.
    """
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # The tensor T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # We need T[i,j,k] = 1 for all (i,j,k) in the multiplication tensor.
    # Actually for matrix multiplication C = A*B where A is n x m, B is m x p, C is n x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] = 1 for all valid (i,j,k).
    
    # Check: for each r, the outer product U[:,r] ⊗ V[:,r] ⊗ W[:,r] should sum to the identity tensor.
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Build the full tensor check
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > max_err:
                    max_err = abs(val - 1.0)
    
    # Also check that no spurious terms appear (though the structure guarantees only (i,j,k) terms)
    # Actually the way the indices work, U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] only produces
    # the (i,j,k) component, so we just need T[i,j,k]=1 for all.
    
    return max_err < tol


def _try_rank(rank, time_budget=30.0, seed=42):
    """Try to find a rank-`rank` decomposition via random initialization + local search."""
    n, m, p = DIMENSIONS
    rng = np.random.RandomState(seed)
    start_time = time.time()
    
    # Random initialization with small values
    best_err = float('inf')
    best_U = best_V = best_W = None
    
    for trial in range(10):
        if time.time() - start_time > time_budget:
            break
        
        # Initialize with random values
        U = rng.randn(n * m, rank) * 0.5
        V = rng.randn(m * p, rank) * 0.5
        W = rng.randn(p * n, rank) * 0.5
        
        # Gradient descent on the objective: sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
        lr = 0.01
        for step in range(500):
            if time.time() - start_time > time_budget:
                break
            
            # Compute T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
            T = np.zeros((n, m, p))
            for r in range(rank):
                # U[:,r] has shape (n*m,), reshape to (n,m)
                Ur = U[:, r].reshape(n, m)
                Vr = V[:, r].reshape(m, p)
                Wr = W[:, r].reshape(p, n)
                # T[i,j,k] += Ur[i,j] * Vr[j,k] * Wr[k,i]
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T[i, j, k] += Ur[i, j] * Vr[j, k] * Wr[k, i]
            
            err = np.sum((T - 1.0)**2)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-8:
                break
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for r in range(rank):
                Ur = U[:, r].reshape(n, m)
                Vr = V[:, r].reshape(m, p)
                Wr = W[:, r].reshape(p, n)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            dT = 2.0 * (T[i, j, k] - 1.0)
                            dU[i*m+j, r] += dT * Vr[j, k] * Wr[k, i]
                            dV[j*p+k, r] += dT * Ur[i, j] * Wr[k, i]
                            dW[k*n+i, r] += dT * Ur[i, j] * Vr[j, k]
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        # Try to snap to half-integers
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        # Verify snapped version
        if _verify(U_snap, V_snap, W_snap):
            return U_snap.tolist(), V_snap.tolist(), W_snap.tolist()
        
        # Also try original
        if best_err < 1e-8:
            U_r = np.round(best_U * 2) / 2
            V_r = np.round(best_V * 2) / 2
            W_r = np.round(best_W * 2) / 2
            if _verify(U_r, V_r, W_r):
                return U_r.tolist(), V_r.tolist(), W_r.tolist()
    
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 48
    
    # Try progressively smaller ranks
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(8, 24, 12) = 24
    # For 2x4x6, the theoretical minimum is likely around 24-30
    
    # First, return schoolbook as fallback
    fallback = _schoolbook()
    
    # Try ranks from lower bound up to schoolbook
    for rank in range(24, schoolbook_rank + 1):
        if time.time() > deadline:
            break
        
        time_budget = min(20.0, deadline - time.time() - 5.0)
        if time_budget < 5.0:
            break
        
        result = _try_rank(rank, time_budget=time_budget, seed=rank)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    return fallback

# EVOLVE-BLOCK-END
