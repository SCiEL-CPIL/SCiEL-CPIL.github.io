"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START

def _schoolbook_decomp(n, m, p):
    """Return schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _tensor_entries(U, V, W, n, m, p):
    """
    Given U (n*m x rank), V (m*p x rank), W (p*n x rank),
    compute the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    as a 3D array.
    """
    rank = U.shape[1]
    # Reshape U to (n, m, rank)
    U3 = U.reshape(n, m, rank)
    # Reshape V to (m, p, rank)
    V3 = V.reshape(m, p, rank)
    # Reshape W to (p, n, rank)
    W3 = W.reshape(p, n, rank)
    
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Use einsum: U[i,j,c] * V[j,k,c] * W[k,i,c] summed over c
    # First: UV[i,j,k,c] = U[i,j,c] * V[j,k,c] -- but this is large
    # Better: for each c, outer product structure
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    T = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        Uc = U3[:, :, c]  # (n, m)
        Vc = V3[:, :, c]  # (m, p)
        Wc = W3[:, :, c]  # (p, n)
        # For each (i,j,k): Uc[i,j] * Vc[j,k] * Wc[k,i]
        # Build via: for each j, Uc[:,j] (n,) * Vc[j,:] (p,) gives (n,p), then multiply elementwise by Wc.T[i,k]? 
        # Wc is (p,n), so Wc[k,i] = Wc.T[i,k]
        # T[i,j,k] += Uc[i,j] * Vc[j,k] * Wc[k,i]
        # Let's do: M[i,k] = sum_j Uc[i,j] * Vc[j,k] = Uc @ Vc  (n x p)
        # Then T[i,j,k] is not just M[i,k] * Wc[k,i] because of the j index
        # Actually: T[i,j,k] = Uc[i,j] * Vc[j,k] * Wc[k,i]
        # For fixed j: T[:,j,:] = Uc[:,j] (n,) outer Vc[j,:] (p,) elementwise * Wc.T (n,p)
        for j in range(m):
            T[:, j, :] += (Uc[:, j, np.newaxis] * Vc[j, np.newaxis, :] * Wc.T)
    return T


def _target_tensor(n, m, p):
    """
    The target multiplication tensor: T[i,j,k] = 1 if we're computing (A@B)[i,k] where A is n x m, B is m x p.
    (A@B)[i,k] = sum_j A[i,j] * B[j,k]
    So the tensor T[i,j,k] = 1 for all i,j,k (it's the identity mapping in the bilinear form).
    Actually the tensor is: T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    because (A@B)[i,k] = sum_j 1 * A[i,j] * B[j,k]
    """
    return np.ones((n, m, p), dtype=np.float64)


def _validate_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if U,V,W exactly decompose the target tensor."""
    T = _tensor_entries(U, V, W, n, m, p)
    T_target = _target_tensor(n, m, p)
    err = np.max(np.abs(T - T_target))
    return err < tol, err


def _is_half_integer(arr, tol=1e-9):
    """Check if all entries are integer or half-integer (multiples of 0.5)."""
    doubled = 2 * arr
    rounded = np.round(doubled)
    return np.max(np.abs(doubled - rounded)) < tol


def _try_rank(r, n, m, p, max_iters=200, deadline=None, n_restarts=5, seed=42):
    """
    Try to find a rank-r decomposition using iterative refinement.
    Returns (U, V, W) if found valid, else None.
    """
    rng = np.random.default_rng(seed)
    
    for restart in range(n_restarts):
        if deadline is not None and time.time() > deadline:
            break
        
        # Random initialization
        U = rng.standard_normal((n * m, r))
        V = rng.standard_normal((m * p, r))
        W = rng.standard_normal((p * n, r))
        
        best_err = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for it in range(max_iters):
            if deadline is not None and time.time() > deadline:
                break
            
            # Alternating least squares: fix two, solve for third
            # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            # For fixed V,W: T[i,j,k] = sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c])
            # Let B[i,j,k,c] = V[j*p+k,c] * W[k*n+i,c], then T[i,j,k] = sum_c U[i*m+j,c] * B[i,j,k,c]
            # This is a linear system in U for each (i,j): sum_k T[i,j,k] * ... no, it's coupled.
            # Better: use gradient descent or coordinate descent.
            
            # Compute current error
            T_cur = _tensor_entries(U, V, W, n, m, p)
            T_tgt = _target_tensor(n, m, p)
            err = np.max(np.abs(T_cur - T_tgt))
            
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if err < 1e-9:
                return best_U, best_V, best_W
            
            # Gradient descent with small step
            lr = 0.001
            T_diff = T_cur - T_tgt  # (n,m,p)
            
            # dU[i*m+j,c] = sum_k T_diff[i,j,k] * V[j*p+k,c] * W[k*n+i,c]
            # dV[j*p+k,c] = sum_i T_diff[i,j,k] * U[i*m+j,c] * W[k*n+i,c]
            # dW[k*n+i,c] = sum_j T_diff[i,j,k] * U[i*m+j,c] * V[j*p+k,c]
            
            U3 = U.reshape(n, m, r)
            V3 = V.reshape(m, p, r)
            W3 = W.reshape(p, n, r)
            
            dU = np.zeros((n, m, r))
            dV = np.zeros((m, p, r))
            dW = np.zeros((p, n, r))
            
            for c in range(r):
                Uc = U3[:, :, c]
                Vc = V3[:, :, c]
                Wc = W3[:, :, c]
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            dU[i, j, c] += T_diff[i, j, k] * Vc[j, k] * Wc[k, i]
                            dV[j, k, c] += T_diff[i, j, k] * Uc[i, j] * Wc[k, i]
                            dW[k, i, c] += T_diff[i, j, k] * Uc[i, j] * Vc[j, k]
            
            U -= lr * dU.reshape(n*m, r)
            V -= lr * dV.reshape(m*p, r)
            W -= lr * dW.reshape(p*n, r)
        
        if best_err < 1e-9:
            return best_U, best_V, best_W
    
    return None


def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 140  # 140s deadline with 10s margin
    
    # Schoolbook as fallback
    U_sb, V_sb, W_sb = _schoolbook_decomp(n, m, p)
    
    # Try to find lower rank decompositions
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(6, 18, 12) = 18
    # Also rank >= n*p if m=1, etc. For 2x3x6, lower bound is at least 18.
    # Try ranks from 18 to 35
    
    for r in range(18, 36):
        if time.time() > deadline:
            break
        
        result = _try_rank(r, n, m, p, max_iters=50, deadline=deadline, n_restarts=3, seed=r)
        if result is not None:
            U, V, W = result
            # Validate
            valid, err = _validate_decomposition(U, V, W, n, m, p)
            if valid and _is_half_integer(U) and _is_half_integer(V) and _is_half_integer(W):
                # Convert to lists
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
            elif valid:
                # Valid but not half-integer; still return (score is rank)
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
    
    # Fallback to schoolbook
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }

# EVOLVE-BLOCK-END
