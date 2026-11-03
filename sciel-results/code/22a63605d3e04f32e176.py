"""Matrix multiplication 3,4,5 rank-minimal tensor decomposition search."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 5]

# Deadline guard: kill after 150s, keep safety margin
_DEADLINE = 145.0
_START_TIME = time.time()

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return max(0.0, _DEADLINE - _elapsed())

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 if i,j,k correspond to a term in C=A*B."""
    # T is a 3D array of shape (n*m, m*p, p*n)
    # U index: i*m + j (i in [0,n), j in [0,m))
    # V index: j*p + k (j in [0,m), k in [0,p))
    # W index: k*n + i (k in [0,p), i in [0,n))
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
    return T

def _validate(U, V, W, n, m, p, tol=1e-6):
    """Check if U, V, W exactly decompose the tensor."""
    T = _build_tensor(n, m, p)
    # Compute sum over c of U[:,c] * V[:,c] * W[:,c]
    # Use einsum: T_recon[i,j,k] = sum_c U[i,c]*V[j,c]*W[k,c]
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    T_recon = np.einsum('ic,jc,kc->ijk', Ua, Va, Wa)
    return np.max(np.abs(T_recon - T)) < tol

def _rank_of(U, V, W):
    """Return the rank (number of columns)."""
    return len(U[0]) if U else 0

def _try_rank(n, m, p, rank, max_iter=2000, lr=0.01, seed=0):
    """Try to find a rank-`rank` decomposition using gradient descent on the tensor."""
    T = _build_tensor(n, m, p)
    rng = np.random.RandomState(seed)
    
    # Initialize
    U = rng.randn(n*m, rank) * 0.1
    V = rng.randn(m*p, rank) * 0.1
    W = rng.randn(p*n, rank) * 0.1
    
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if _remaining() < 5.0:
            break
        # Compute gradient: d/dU[i,c] = sum_{j,k} T[i,j,k] * V[j,c] * W[k,c] - U[i,c] * (sum over j,k of T_recon... )
        # Actually, the objective is ||T - sum_c U_c ⊗ V_c ⊗ W_c||^2
        # Gradient w.r.t. U[i,c] = -2 * sum_{j,k} (T[i,j,k] - sum_d U[i,d]V[j,d]W[k,d]) * V[j,c] * W[k,c]
        
        # Compute T_recon
        T_recon = np.einsum('ic,jc,kc->ijk', U, V, W)
        diff = T - T_recon  # shape (n*m, m*p, p*n)
        
        # Gradient for U: dU[i,c] = -2 * sum_{j,k} diff[i,j,k] * V[j,c] * W[k,c]
        # = -2 * einsum('ijk,jc,kc->ic', diff, V, W)
        dU = -2.0 * np.einsum('ijk,jc,kc->ic', diff, V, W)
        dV = -2.0 * np.einsum('ijk,ic,kc->jc', diff, U, W)
        dW = -2.0 * np.einsum('ijk,ic,jc->kc', diff, U, V)
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        err = np.max(np.abs(T_recon - T))
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_err < 1e-8:
            break
    
    if best_err < 1e-6:
        return best_U, best_V, best_W, best_err
    return None

def _try_rank_als(n, m, p, rank, max_iter=500, seed=0):
    """Try ALS (alternating least squares) for tensor decomposition."""
    T = _build_tensor(n, m, p)
    rng = np.random.RandomState(seed)
    
    U = rng.randn(n*m, rank) * 0.1
    V = rng.randn(m*p, rank) * 0.1
    W = rng.randn(p*n, rank) * 0.1
    
    for it in range(max_iter):
        if _remaining() < 5.0:
            break
        
        # Update U: for each c, U[:,c] = (V_c^T W_c)^{-1} * (contract T with V_c, W_c)
        # T[i,j,k] = sum_c U[i,c] V[j,c] W[k,c]
        # Fix V,W: for each c, minimize ||T - sum_d U_d V_d W_d||^2 w.r.t. U[:,c]
        # This is a linear system. For each c:
        # sum_i U[i,c] V[j,c] W[k,c] = T[i,j,k] (approx, if rank=1 it's exact)
        # Actually for general rank, we solve:
        # For fixed V,W, the best U[:,c] satisfies:
        # (sum_{j,k} V[j,c]W[k,c] V[j,d]W[k,d]) U[:,c] ... no this is getting complex.
        
        # Simpler: use the fact that T[i,j,k] = sum_c U[i,c] V[j,c] W[k,c]
        # For fixed V,W, U[i,c] can be found by:
        # U[i,c] = sum_{j,k} T[i,j,k] V[j,c] W[k,c] / sum_{j,k} V[j,c]^2 W[k,c]^2
        # This is the optimal solution in the least-squares sense for each c independently
        # if we treat each c as a separate rank-1 component.
        
        denom_U = np.sum(V**2 * W**2, axis=(0,1))  # shape (rank,) - wait, V is (m*p, rank), W is (p*n, rank)
        # V[j,c] * W[k,c] summed over j,k: sum_j sum_k V[j,c]^2 * W[k,c]^2 = (sum_j V[j,c]^2) * (sum_k W[k,c]^2)
        sum_V2 = np.sum(V**2, axis=0)  # shape (rank,)
        sum_W2 = np.sum(W**2, axis=0)  # shape (rank,)
        denom = sum_V2 * sum_W2  # shape (rank,)
        
        # numerator for U[i,c]: sum_{j,k} T[i,j,k] V[j,c] W[k,c]
        # = einsum('ijk,jc,kc->ic', T, V, W)
        num_U = np.einsum('ijk,jc,kc->ic', T, V, W)
        U_new = num_U / np.maximum(denom, 1e-12)
        
        # Similarly for V[j,c]:
        sum_U2 = np.sum(U_new**2, axis=0)
        denom_V = sum_U2 * sum_W2
        num_V = np.einsum('ijk,ic,kc->jc', T, U_new, W)
        V_new = num_V / np.maximum(denom_V, 1e-12)
        
        # For W[k,c]:
        sum_U2 = np.sum(U_new**2, axis=0)
        sum_V2 = np.sum(V_new**2, axis=0)
        denom_W = sum_U2 * sum_V2
        num_W = np.einsum('ijk,ic,jc->kc', T, U_new, V_new)
        W_new = num_W / np.maximum(denom_W, 1e-12)
        
        U, V, W = U_new, V_new, W_new
        
        T_recon = np.einsum('ic,jc,kc->ijk', U, V, W)
        err = np.max(np.abs(T_recon - T))
        if err < 1e-8:
            return U, V, W, err
    
    T_recon = np.einsum('ic,jc,kc->ijk', U, V, W)
    err = np.max(np.abs(T_recon - T))
    if err < 1e-6:
        return U, V, W, err
    return None

def _to_list(arr):
    """Convert numpy array to list of lists with rounded values."""
    return [[float(x) for x in row] for row in arr]

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    # Known lower bound for matrix multiplication rank: at least n*m*p / something
    # For 3x4x5, the schoolbook rank is 60.
    # Let's try ranks from 1 upward, but realistically the minimum is likely > 20.
    # Start from a reasonable lower bound and go up.
    
    # Theoretical lower bound: rank >= max(n*m, m*p, p*n) / min(n,m,p) ... not quite.
    # Actually, for matrix multiplication, rank >= n*m*p / max(n,m,p) is a known bound? No.
    # Let's just try from rank 10 upward, since we have time.
    
    best_result = None
    best_rank = n * m * p  # schoolbook
    
    # First, provide the schoolbook as fallback
    U_school = np.zeros((n*m, n*m*p))
    V_school = np.zeros((m*p, n*m*p))
    W_school = np.zeros((p*n, n*m*p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1
    
    # Try decreasing rank from a high estimate down to lower bounds
    # Actually, try increasing rank from low to find the minimum.
    # But we want the MINIMUM rank. So try rank 1, 2, 3, ... until we find one.
    # This is expensive. Let's try a range.
    
    # Try ranks: 20, 25, 30, 35, 40, 45, 50, 55, 60
    # Start from lower to find minimum.
    
    for rank in range(15, 61):
        if _remaining() < 10.0:
            break
        
        found = False
        for seed in range(5):
            if _remaining() < 5.0:
                break
            
            # Try ALS
            result = _try_rank_als(n, m, p, rank, max_iter=200, seed=seed)
            if result is not None:
                U, V, W, err = result
                # Round to nearest 0.5 or integer if close
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Validate rounded version
                T_recon = np.einsum('ic,jc,kc->ijk', U_rounded, V_rounded, W_rounded)
                if np.max(np.abs(T_recon - T)) < 1e-6:
                    return {
                        "U": _to_list(U_rounded),
                        "V": _to_list(V_rounded),
                        "W": _to_list(W_rounded)
                    }
                else:
                    # Use unrounded if it works
                    T_recon = np.einsum('ic,jc,kc->ijk', U, V, W)
                    if np.max(np.abs(T_recon - T)) < 1e-6:
                        return {
                            "U": _to_list(U),
                            "V": _to_list(V),
                            "W": _to_list(W)
                        }
        
        # Also try gradient descent for this rank
        for seed in range(3):
            if _remaining() < 5.0:
                break
            result = _try_rank(n, m, p, rank, max_iter=1000, lr=0.005, seed=seed)
            if result is not None:
                U, V, W, err = result
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                T_recon = np.einsum('ic,jc,kc->ijk', U_rounded, V_rounded, W_rounded)
                if np.max(np.abs(T_recon - T)) < 1e-6:
                    return {
                        "U": _to_list(U_rounded),
                        "V": _to_list(V_rounded),
                        "W": _to_list(W_rounded)
                    }
                else:
                    T_recon = np.einsum('ic,jc,kc->ijk', U, V, W)
                    if np.max(np.abs(T_recon - T)) < 1e-6:
                        return {
                            "U": _to_list(U),
                            "V": _to_list(V),
                            "W": _to_list(W)
                        }
    
    # Fallback: schoolbook
    return {
        "U": _to_list(U_school),
        "V": _to_list(V_school),
        "W": _to_list(W_school)
    }
