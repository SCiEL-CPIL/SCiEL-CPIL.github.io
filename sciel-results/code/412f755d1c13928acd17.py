"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 3, 3]
N, M, P = DIMENSIONS
RANK_MAX = N * M * P  # 18

# Deadline guard: evaluator kills at 150s wall-clock. We reserve margin.
_DEADLINE = time.time() + 140.0


def _remaining():
    return _DEADLINE - time.time()


def _build_tensor():
    """Build the matrix multiplication tensor T[i,j,k] for (n,m,p) = (2,3,3).
    The tensor is the canonical multiplication tensor: T[i,j,k] = 1 for all valid indices."""
    return np.ones((N, M, P), dtype=float)


def _compute_residuals(U, V, W):
    """Vectorized computation of residuals for given U, V, W matrices."""
    nm, mp, pn = N*M, M*P, P*N
    rank = U.shape[1]
    
    # Precompute index mappings
    # U[i*M+j, r], V[j*P+k, r], W[k*N+i, r]
    # For each (i,j,k), we need sum_r U[i*M+j, r] * V[j*P+k, r] * W[k*N+i, r]
    
    # Use einsum for efficient computation
    # Reshape U, V, W to 3D: U[i, j, r], V[j, k, r], W[k, i, r]
    U_3d = U.reshape(N, M, rank)
    V_3d = V.reshape(M, P, rank)
    W_3d = W.reshape(P, N, rank)
    
    # For each (i,j,k): sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # We can compute this using einsum
    # First, contract U and V over j and r: UV[i,k,r] = sum_j U[i,j,r] * V[j,k,r]
    # But this isn't quite right because we need the same r for all three.
    
    # Better approach: use broadcasting
    # U_3d[i,j,r] * V_3d[j,k,r] * W_3d[k,i,r]
    # Sum over r for each (i,j,k)
    
    # Create full tensor and sum over r
    # This might be memory intensive, but for small dimensions it's fine
    full = U_3d[:, :, None, :] * V_3d[None, :, :, :] * W_3d[None, None, :, :]
    # full has shape (N, M, P, rank)
    # Sum over last axis
    result = np.sum(full, axis=3)  # shape (N, M, P)
    
    return (result - 1.0).flatten()


def _try_rank_decomposition(rank, timeout=None):
    """
    Attempt to find a rank-`rank` decomposition of the multiplication tensor.
    Uses scipy.optimize.least_squares with multiple random restarts.
    """
    from scipy.optimize import least_squares
    
    def _residuals(params, rank):
        """Unpack parameters into U, V, W and compute residuals."""
        nm, mp, pn = N*M, M*P, P*N
        u_size = nm * rank
        v_size = mp * rank
        w_size = pn * rank
        
        U = params[:u_size].reshape(nm, rank)
        V = params[u_size:u_size+v_size].reshape(mp, rank)
        W = params[u_size+v_size:u_size+v_size+w_size].reshape(pn, rank)
        
        return _compute_residuals(U, V, W)
    
    best_result = None
    best_cost = float('inf')
    max_attempts = 20  # More restarts to find a solution
    
    for attempt in range(max_attempts):
        if _remaining() < 5:
            break
        # Use different seeds for diversity
        seed = attempt * 12345 + 67890
        rng = np.random.RandomState(seed)
        
        # Initialize with random values, scaled appropriately
        scale = 0.5
        u0 = rng.randn(N*M, rank) * scale
        v0 = rng.randn(M*P, rank) * scale
        w0 = rng.randn(P*N, rank) * scale
        params0 = np.concatenate([u0.flatten(), v0.flatten(), w0.flatten()])
        
        try:
            result = least_squares(
                _residuals, 
                params0, 
                args=(rank,), 
                method='lm', 
                max_nfev=2000,
                ftol=1e-15,
                xtol=1e-15,
                gtol=1e-15
            )
            if result.cost < best_cost:
                best_cost = result.cost
                if result.cost < 1e-12:
                    # Found a solution
                    u_size = N*M*rank
                    v_size = M*P*rank
                    w_size = P*N*rank
                    U = result.x[:u_size].reshape(N*M, rank)
                    V = result.x[u_size:u_size+v_size].reshape(M*P, rank)
                    W = result.x[u_size+v_size:u_size+v_size+w_size].reshape(P*N, rank)
                    
                    # Try to round to nearest 0.5 or 0.25 if close
                    for denom in [2, 4, 8]:
                        U_r = np.round(U * denom) / denom
                        V_r = np.round(V * denom) / denom
                        W_r = np.round(W * denom) / denom
                        
                        # Verify the rounded version
                        res = _compute_residuals(U_r, V_r, W_r)
                        if np.max(np.abs(res)) < 1e-8:
                            return U_r, V_r, W_r
                    
                    # If rounding doesn't work, return the float solution
                    return U, V, W
        except Exception as e:
            pass
    
    return None


def construct():
    """Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x3 matrix."""
    global _DEADLINE
    _DEADLINE = time.time() + 140.0
    
    # Try ranks from lower values up to the schoolbook rank
    # Start from a reasonable lower bound
    start_rank = 8
    
    for rank in range(start_rank, N*M*P + 1):
        if _remaining() < 15:
            break
        result = _try_rank_decomposition(rank)
        if result is not None:
            U, V, W = result
            # Convert to list of lists
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook decomposition
    rank = N*M*P
    U = [[0]*rank for _ in range(N*M)]
    V = [[0]*rank for _ in range(M*P)]
    W = [[0]*rank for _ in range(P*N)]
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i*M+j][col] = 1
                V[j*P+k][col] = 1
                W[k*N+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
