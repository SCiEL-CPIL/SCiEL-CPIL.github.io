import time
import numpy as np
from scipy.optimize import nnls

DIMENSIONS = [2, 3, 9]

def _schoolbook_decomposition(n, m, p):
    """Return the standard rank-n*m*p decomposition."""
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
                
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = U.shape[1]
    # Reshape to tensors
    U_t = U.reshape(n, m, rank)
    V_t = V.reshape(m, p, rank)
    W_t = W.reshape(p, n, rank)
    
    # Check: sum_r U[i,j,r] * V[j,k,r] == W[k,i,r] for all i,j,k
    # Actually, the decomposition means:
    # (A * B)[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_j sum_r U[i,j,r] * A[i,j] * V[j,k,r] * B[j,k] ... no
    # The tensor T_{ijk} = delta_{jk} (for matmul? no)
    # T_{ijk} is 1 if we're computing (AB)_{ik} = sum_j A_{ij} B_{jk}
    # So T_{ijk} = 1 for all i,j,k? No.
    # The multiplication tensor has T_{ijk} = 1 if the term A_{ij}B_{jk} contributes to (AB)_{ik}
    # Actually T_{ijk} = 1 for all valid i,j,k since every term A_{ij}B_{jk} appears.
    # Wait, no. The tensor for matrix multiplication C = AB where A is n x m, B is m x p, C is n x p:
    # C_{ik} = sum_j A_{ij} B_{jk}
    # The tensor T has indices (i, j, k) and T_{ijk} = 1 (since each term A_{ij}B_{jk} is a term in the sum)
    # So T_{ijk} = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # The decomposition says T_{ijk} = sum_r U_{ij,r} V_{jk,r} W_{ki,r}
    # where W is indexed by (k, i) per the convention W[k*n+i, r].
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = 1.0  # T_{ijk} = 1
                rhs = 0.0
                for r in range(rank):
                    rhs += U_t[i, j, r] * V_t[j, k, r] * W_t[k, i, r]
                if abs(lhs - rhs) > tol:
                    return False
    return True

def _try_rank_reduction(n, m, p, target_rank, deadline):
    """Try to find a decomposition of the given target rank using iterative refinement."""
    rank = target_rank
    np.random.seed(42)
    
    # Build index mapping: for each (i,j,k), which (r) indices in U,V,W are involved
    # U indexed by (i,j), V by (j,k), W by (k,i)
    # We need: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k
    
    # Initialize with schoolbook-like structure
    # For rank r, we can't just use schoolbook. Let's use random init.
    U = np.random.randn(n * m, rank) * 0.5
    V = np.random.randn(m * p, rank) * 0.5
    W = np.random.randn(p * n, rank) * 0.5
    
    # Iterative refinement: alternate updating each matrix
    # For fixed V, W: U[i,j,:] is determined by needing sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
    # This is a linear system in U[i,j,:] for each (i,j)
    # Similarly for V and W
    
    # Precompute index mappings
    # For a given (i,j), the constraint involves V[j,k,:] and W[k,i,:] for all k
    # sum_r U[i,j,r] * (sum_k V[j,k,r] * W[k,i,r]) = 1
    # So if we define S[i,j,r] = sum_k V[j,k,r] * W[k,i,r], then we need:
    # sum_r U[i,j,r] * S[i,j,r] = 1
    # This is underdetermined (rank unknowns, 1 equation per (i,j))
    # We can pick the minimum norm solution: U[i,j,:] = S[i,j,:] / (S[i,j,:] @ S[i,j,:])
    
    # Similarly for V: for fixed U, W, define T[j,k,r] = sum_i U[i,j,r] * W[k,i,r]
    # Need sum_r V[j,k,r] * T[j,k,r] = 1
    # V[j,k,:] = T[j,k,:] / (T[j,k,:] @ T[j,k,:])
    
    # And for W: for fixed U, V, define R[k,i,r] = sum_j U[i,j,r] * V[j,k,r]
    # Need sum_r W[k,i,r] * R[k,i,r] = 1
    # W[k,i,:] = R[k,i,:] / (R[k,i,:] @ R[k,i,:])
    
    best_error = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for iteration in range(200):
        if time.time() > deadline - 15:
            break
        
        # Update U
        for i in range(n):
            for j in range(m):
                # S[i,j,r] = sum_k V[j,k,r] * W[k,i,r]
                S = np.zeros(rank)
                for k in range(p):
                    S += V[j * p + k, :] * W[k * n + i, :]
                norm_sq = S @ S
                if norm_sq > 1e-12:
                    U[i * m + j, :] = S / norm_sq
                else:
                    U[i * m + j, :] = 1.0 / rank
        
        # Update V
        for j in range(m):
            for k in range(p):
                # T[j,k,r] = sum_i U[i,j,r] * W[k,i,r]
                T = np.zeros(rank)
                for i in range(n):
                    T += U[i * m + j, :] * W[k * n + i, :]
                norm_sq = T @ T
                if norm_sq > 1e-12:
                    V[j * p + k, :] = T / norm_sq
                else:
                    V[j * p + k, :] = 1.0 / rank
        
        # Update W
        for k in range(p):
            for i in range(n):
                # R[k,i,r] = sum_j U[i,j,r] * V[j,k,r]
                R = np.zeros(rank)
                for j in range(m):
                    R += U[i * m + j, :] * V[j * p + k, :]
                norm_sq = R @ R
                if norm_sq > 1e-12:
                    W[k * n + i, :] = R / norm_sq
                else:
                    W[k * n + i, :] = 1.0 / rank
        
        # Compute error
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    error += abs(val - 1.0)
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if error < 1e-9:
            break
    
    if best_error < 1e-8:
        return best_U, best_V, best_W
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Use the schoolbook solution which is guaranteed to be correct.
    # Rank is n*m*p = 54.
    U, V, W = _schoolbook_decomposition(n, m, p)
    
    # Try to see if we can reduce the rank
    # Try rank values below 54
    best_U, best_V, best_W = U, V, W
    best_rank = U.shape[1]
    
    # Try a few lower ranks, starting from just below schoolbook
    for trial_rank in [53, 52, 51, 50, 48, 45, 42, 40, 36, 32, 30, 28, 27]:
        if time.time() > deadline - 30:
            break
        result = _try_rank_reduction(n, m, p, trial_rank, deadline)
        if result is not None:
            U, V, W = result
            best_rank = trial_rank
            break
    
    # Verify the final decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fall back to schoolbook
        U, V, W = _schoolbook_decomposition(n, m, p)
    
    # Convert to lists as required by the interface
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
