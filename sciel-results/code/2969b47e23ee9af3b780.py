"""Program that constructs a rank-reduced decomposition for 3x4 * 4x8 matrix multiplication.

Uses a randomized local-search / coordinate-descent approach to find a valid
decomposition at rank < 96 (schoolbook).  If no valid decomposition is found
within the time budget, falls back to the schoolbook decomposition.
"""
import time
import random
import numpy as np

_DIMS = (3, 4, 8)


def _validate(U, V, W):
    """Check whether the factor matrices correctly decompose the matmul tensor."""
    n, m, p = _DIMS
    # Reconstruct the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # and compare with the standard matmul tensor T_std[i,j,k] = 1 (since the
    # matrix multiplication tensor has all-ones entries).
    #
    # The tensor for multiplying an n x m matrix by an m x p matrix is:
    #   T[i, j, k] = sum over the rank dimension
    #   where the output is C[i,k] = sum_j A[i,j] B[j,k]
    #   In the standard convention:
    #     U has shape (n*m, rank), indexed by (i,j)
    #     V has shape (m*p, rank), indexed by (j,k)
    #     W has shape (p*n, rank), indexed by (k,i)
    #
    # The tensor entry T[i,j,k] should equal 1 for all i,j,k.
    
    rank = U.shape[1]
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    
    # Build the tensor via sum over rank: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We can do this efficiently:
    # For each c, the outer product contribution is:
    #   U_col[i*m+j] * V_col[j*p+k] * W_col[k*n+i]
    
    # Efficient check: for each (i, j, k), compute the sum
    # Use broadcasting: reshape to 3D
    # U_r: shape (rank, n*m) -> (rank, n, m)
    # V_r: shape (rank, m*p) -> (rank, m, p)
    # W_r: shape (rank, p*n) -> (rank, p, n)
    
    U3 = U.T.reshape(rank, n, m)  # (rank, n, m)
    V3 = V.T.reshape(rank, m, p)  # (rank, m, p)
    W3 = W.T.reshape(rank, p, n)  # (rank, p, n)
    
    # T[i,j,k] = sum_c U3[c,i,j] * V3[c,j,k] * W3[c,k,i]
    # We need to sum over c with matching indices.
    # This is a tensor contraction. Let's do it via einsum or loops.
    
    # For small sizes, loop over c:
    T = np.zeros((n, m, p))
    for c in range(rank):
        # U3[c, i, j], V3[c, j, k], W3[c, k, i]
        # Build contribution: for each (i,j,k): U3[c,i,j]*V3[c,j,k]*W3[c,k,i]
        # This is a 3D outer product with index matching.
        # Use einsum: 'cij,cjk,cki->ijk'
        contrib = np.einsum('cij,cjk,cki->ijk', U3[c], V3[c], W3[c])
        T += contrib
    
    # Check if T is all ones
    return np.allclose(T, 1.0, atol=1e-9)


def _schoolbook():
    """Return the schoolbook decomposition of rank 96."""
    n, m, p = _DIMS
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}


def _random_init(rank, n, m, p):
    """Initialize random factor matrices."""
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    return U, V, W


def _compute_residual(U, V, W, n, m, p):
    """Compute the residual tensor (deviation from all-ones)."""
    rank = U.shape[1]
    U3 = U.T.reshape(rank, n, m)
    V3 = V.T.reshape(rank, m, p)
    W3 = W.T.reshape(rank, p, n)
    T = np.zeros((n, m, p))
    for c in range(rank):
        contrib = np.einsum('cij,cjk,cki->ijk', U3[c], V3[c], W3[c])
        T += contrib
    return T - 1.0  # residual


def _local_search(U, V, W, rank, n, m, p, max_iters=500, lr=0.05, deadline=None):
    """Coordinate descent / random perturbation local search."""
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_residual = np.max(np.abs(_compute_residual(best_U, best_V, best_W, n, m, p)))
    
    U3 = best_U.T.reshape(rank, n, m)
    V3 = best_V.T.reshape(rank, m, p)
    W3 = best_W.T.reshape(rank, p, n)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Pick a random factor and a random entry to perturb
        which = random.choice(['U', 'V', 'W'])
        if which == 'U':
            row = random.randint(0, n * m - 1)
            col = random.randint(0, rank - 1)
            # Current contribution to residual from this entry
            # d T[i,j,k] / d U[row, col] = V[j*p+k, col] * W[k*n+i, col]
            # where row = i*m+j
            i, j = divmod(row, m)
            # For each k, the derivative is V[j*p+k, col] * W[k*n+i, col]
            derivs = np.array([V[j * p + k, col] * W[k * n + i, col] for k in range(p)])
            # Residual for this (i, j, :)
            res = np.array([_compute_residual(best_U, best_V, best_W, n, m, p)[i, j, k] for k in range(p)])
            # Gradient step
            grad = np.sum(res * derivs)
            step = -lr * grad
            # But we need the full residual contribution... let's just do a simple random perturbation
            # Actually, let's use a proper gradient step for the entry
            # The residual squared loss: L = sum_{i,j,k} (T[i,j,k] - 1)^2
            # dL/dU[row,col] = 2 * sum_{k} res[i,j,k] * V[j*p+k,col] * W[k*n+i,col]
            best_U[row, col] += step
        elif which == 'V':
            row = random.randint(0, m * p - 1)
            col = random.randint(0, rank - 1)
            j, k = divmod(row, p)
            i_vals = list(range(n))
            derivs = np.array([U[i * m + j, col] * W[k * n + i, col] for i in i_vals])
            res = np.array([_compute_residual(best_U, best_V, best_W, n, m, p)[i, j, k] for i in i_vals])
            grad = np.sum(res * derivs)
            step = -lr * grad
            best_V[row, col] += step
        else:
            row = random.randint(0, p * n - 1)
            col = random.randint(0, rank - 1)
            k, i = divmod(row, n)
            j_vals = list(range(m))
            derivs = np.array([U[i * m + j, col] * V[j * p + k, col] for j in j_vals])
            res = np.array([_compute_residual(best_U, best_V, best_W, n, m, p)[i, j, k] for j in j_vals])
            grad = np.sum(res * derivs)
            step = -lr * grad
            best_W[row, col] += step
        
        # Check improvement
        res_tensor = _compute_residual(best_U, best_V, best_W, n, m, p)
        cur_res = np.max(np.abs(res_tensor))
        if cur_res < best_residual:
            best_residual = cur_res
            if best_residual < 1e-9:
                break
        else:
            # Revert
            if which == 'U':
                best_U[row, col] -= step
            elif which == 'V':
                best_V[row, col] -= step
            else:
                best_W[row, col] -= step
            # Decrease learning rate occasionally
            if it % 100 == 99:
                lr *= 0.5
    
    return best_U, best_V, best_W, best_residual


def construct():
    """Construct the factor matrices for the 3x4 * 4x8 matrix multiplication tensor.
    
    Returns a dict with keys 'U', 'V', 'W' containing the factor matrices.
    """
    deadline = time.time() + 110  # Safety margin before 120s limit
    n, m, p = _DIMS
    
    # Try to find a rank-reduced decomposition
    # Start from schoolbook and try to reduce rank
    schoolbook = _schoolbook()
    
    # Try ranks from 95 downward, but limit the search
    # For each rank, try a few random restarts
    best_result = schoolbook
    best_rank = 96
    
    # First, try a few random initializations at lower ranks
    # The theoretical minimum rank for 3x4*4x8 is unknown but <= 96
    # Known results: rank(2,2,2)=7, rank(3,3,3)>=23 (best known 23)
    # For 3x4x8, let's try ranks in a range
    
    # Try a systematic approach: start with a random init at various ranks
    ranks_to_try = list(range(90, 96)) + list(range(80, 90, 2)) + list(range(70, 80, 5)) + [60, 50, 40, 30]
    
    random.seed(42)
    
    for rank in ranks_to_try:
        if time.time() > deadline:
            break
        if rank >= best_rank:
            continue
        
        # Try a few restarts at this rank
        for restart in range(5):
            if time.time() > deadline:
                break
            
            U, V, W = _random_init(rank, n, m, p)
            
            # Run local search
            U, V, W, residual = _local_search(
                U, V, W, rank, n, m, p,
                max_iters=300, lr=0.1,
                deadline=deadline
            )
            
            if residual < 1e-9:
                # Found a valid decomposition at this rank
                best_rank = rank
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                # Try to go lower
                break
        
        if time.time() > deadline:
            break
    
    # If we found a valid rank-reduced decomposition, return it
    if best_rank < 96:
        return best_result
    
    # Fallback to schoolbook
    return schoolbook
