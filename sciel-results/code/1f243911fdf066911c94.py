"""Matrix multiplication 2x3x10 rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 for all valid indices."""
    T = np.ones((n, m, p), dtype=np.float64)
    return T


def _check_exact(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    # Reshape to 3D tensors
    U3 = np.array(U).reshape(n, m, rank)
    V3 = np.array(V).reshape(m, p, rank)
    W3 = np.array(W).reshape(p, n, rank)
    
    # Compute the reconstructed tensor
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    T_recon = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    # Check against ones
    if not np.allclose(T_recon, 1.0, atol=tol):
        return False
    return True


def _round_to_halfint(arr):
    """Round array to nearest half-integer."""
    return np.round(2 * arr) / 2.0


def _try_rank_als(n, m, p, rank, time_budget=30.0, n_restarts=5):
    """
    Try to find a rank decomposition using ALS.
    Returns (U, V, W) as lists of lists if found, else None.
    """
    start = time.time()
    T = _build_tensor(n, m, p)
    
    for restart in range(n_restarts):
        if time.time() - start > time_budget - 5:
            break
        
        # Initialize
        np.random.seed(42 + restart)
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        W = np.random.randn(p * n, rank) * 0.5
        
        for iteration in range(200):
            if time.time() - start > time_budget - 5:
                break
            
            # Update U: for each (i,j), solve for U[i*m+j, :]
            U3 = U.reshape(n, m, rank)
            V3 = V.reshape(m, p, rank)
            W3 = W.reshape(p, n, rank)
            
            for i in range(n):
                for j in range(m):
                    # We need: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all k
                    # This is: U[i,j,:] @ (V[j,:,:] * W[:,i,:])^T = ones(p)
                    # where (V[j,:,:] * W[:,i,:])[k,r] = V[j,k,r] * W[k,i,r]
                    mat = np.zeros((rank, p))
                    for k in range(p):
                        mat[:, k] = V3[j, k, :] * W3[k, i, :]
                    # Solve U[i,j,:] @ mat = ones(p)
                    # i.e., mat^T @ U[i,j,:] = ones(p)
                    try:
                        U3[i, j, :] = np.linalg.lstsq(mat.T, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V: for each (j,k), solve for V[j*p+k, :]
            U3 = U.reshape(n, m, rank)
            W3 = W.reshape(p, n, rank)
            
            for j in range(m):
                for k in range(p):
                    # We need: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i
                    # This is: V[j,k,:] @ (U[:,j,:] * W[k,:])^T = ones(n)
                    mat = np.zeros((rank, n))
                    for i in range(n):
                        mat[:, i] = U3[i, j, :] * W3[k, i, :]
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(mat.T, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Update W: for each (k,i), solve for W[k*n+i, :]
            U3 = U.reshape(n, m, rank)
            V3 = V.reshape(m, p, rank)
            
            for k in range(p):
                for i in range(n):
                    # We need: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all j
                    # This is: W[k,i,:] @ (U[i,:,:] * V[:,k,:])^T = ones(m)
                    mat = np.zeros((rank, m))
                    for j in range(m):
                        mat[:, j] = U3[i, j, :] * V3[j, k, :]
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(mat.T, np.ones(m), rcond=None)[0]
                    except:
                        pass
        
        # Check if we found a valid solution
        if _check_exact(U, V, W, n, m, p, tol=1e-6):
            # Try rounding to half-integers
            U_rounded = _round_to_halfint(U)
            V_rounded = _round_to_halfint(V)
            W_rounded = _round_to_halfint(W)
            
            if _check_exact(U_rounded, V_rounded, W_rounded, n, m, p, tol=1e-9):
                return (U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist())
            
            # Also try rounding to integers
            U_int = np.round(U)
            V_int = np.round(V)
            W_int = np.round(W)
            
            if _check_exact(U_int, V_int, W_int, n, m, p, tol=1e-9):
                return (U_int.tolist(), V_int.tolist(), W_int.tolist())
            
            # If not exactly rounded, still return the float solution if it's very close
            if _check_exact(U, V, W, n, m, p, tol=1e-8):
                # Convert to half-integer representation if possible
                return (U.tolist(), V.tolist(), W.tolist())
    
    return None


def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
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
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = n * m * p
    
    # Try to find lower rank decompositions using ALS
    # Try ranks from a reasonable lower bound up to schoolbook
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(6, 30, 20) = 30
    # But matrix multiplication rank is typically higher
    
    # Try a few ranks
    for rank in range(30, best_rank):
        if time.time() > deadline - 15:
            break
        
        result = _try_rank_als(n, m, p, rank, time_budget=10.0, n_restarts=3)
        if result is not None:
            U_r, V_r, W_r = result
            if len(U_r[0]) < best_rank:
                best_rank = len(U_r[0])
                best_result = {"U": U_r, "V": V_r, "W": W_r}
                # Found a lower rank, stop searching
                break
    
    return best_result

# EVOLVE-BLOCK-END
