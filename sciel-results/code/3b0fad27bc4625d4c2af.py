"""Tensor decomposition search for 2x3 * 3x9 matrix multiplication."""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def _schoolbook():
    """Return the standard rank 54 decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}


def _matmul_tensor(n, m, p):
    """Return the matrix multiplication tensor T[i,j,k] = 1 if indices valid."""
    # T is a 3D array of shape (n*m, m*p, p*n)
    # Actually, the tensor for matmul is T[i, j, k] where:
    #   i indexes row of A (0..n-1)
    #   j indexes col of A / row of B (0..m-1)
    #   k indexes col of B (0..p-1)
    # The tensor T has entries T[(i,j), (j,k), (k,i)] = 1
    # In the factorization: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    # So the tensor is T[i,j,k] = 1 for all valid i,j,k
    T = np.ones((n * m, m * p, p * n), dtype=float)
    # But wait - the tensor is only non-zero at specific indices.
    # Actually for matrix multiplication, the tensor is:
    # T[a, b, c] where a = (i, j), b = (j, k), c = (k, i)
    # T is 1 when the middle indices match: a[1] == b[0] and b[1] == c[0]
    # Let me re-construct properly.
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T


def _check_decomposition(U, V, W, T, tol=1e-9):
    """Check if U, V, W exactly decompose tensor T."""
    # Compute the reconstructed tensor
    # T_recon[a,b,c] = sum_c' U[a,c'] * V[b,c'] * W[c,c']
    # Use einsum
    recon = np.einsum('ac,bc,dc->abd', U, V, W)
    return np.allclose(recon, T, atol=tol, rtol=tol)


def _try_rank_with_greedy(rank, n, m, p, max_iter=5000, deadline=None):
    """Try to find a rank decomposition using greedy/randomized approach."""
    T = _matmul_tensor(n, m, p)
    
    best_score = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    np.random.seed(42)
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Random initialization
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # Normalize columns
        for c in range(rank):
            norm = np.sqrt(np.sum(U[:, c]**2) + np.sum(V[:, c]**2) + np.sum(W[:, c]**2))
            if norm > 1e-15:
                U[:, c] /= norm
                V[:, c] /= norm
                W[:, c] /= norm
        
        # Iterate: alternate updates
        for inner in range(20):
            if deadline is not None and time.time() > deadline:
                break
            
            # Update U
            for a in range(n * m):
                # U[a, c] = sum_{b,d} T[a,b,d] * V[b,c] * W[d,c] / (sum_c' V[:,c']^2 * W[:,c']^2)
                # Simplified: U[a,c] = (T[a,:,:] * (V*W)) / norm
                VW = V * W  # (m*p, p*n) element-wise? No, need to be careful
                # Actually: sum_{b,d} T[a,b,d] V[b,c] W[d,c]
                # = sum_b sum_d T[a,b,d] V[b,c] W[d,c]
                new_U = np.zeros(rank)
                for b in range(m * p):
                    for d in range(p * n):
                        if T[a, b, d] != 0:
                            new_U += V[b, :] * W[d, :] * T[a, b, d]
                # Normalize
                denom = np.sum((V * V) * (W * W), axis=0)
                denom[denom < 1e-15] = 1e-15
                new_U /= denom
                U[a, :] = new_U
            
            # Update V
            for b in range(m * p):
                new_V = np.zeros(rank)
                for a in range(n * m):
                    for d in range(p * n):
                        if T[a, b, d] != 0:
                            new_V += U[a, :] * W[d, :] * T[a, b, d]
                denom = np.sum((U * U) * (W * W), axis=0)
                denom[denom < 1e-15] = 1e-15
                new_V /= denom
                V[b, :] = new_V
            
            # Update W
            for d in range(p * n):
                new_W = np.zeros(rank)
                for a in range(n * m):
                    for b in range(m * p):
                        if T[a, b, d] != 0:
                            new_W += U[a, :] * V[b, :] * T[a, b, d]
                denom = np.sum((U * U) * (V * V), axis=0)
                denom[denom < 1e-15] = 1e-15
                new_W /= denom
                W[d, :] = new_W
        
        # Check convergence
        recon = np.einsum('ac,bc,dc->abd', U, V, W)
        err = np.max(np.abs(recon - T))
        if err < best_score:
            best_score = err
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
        
        if best_score < 1e-9:
            break
    
    if best_score < 1e-9:
        return {"U": best_U, "V": best_V, "W": best_W}
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s safety margin
    
    n, m, p = DIMENSIONS
    T = _matmul_tensor(n, m, p)
    
    # Start with schoolbook as fallback
    result = _schoolbook()
    result["U"] = result["U"].tolist()
    result["V"] = result["V"].tolist()
    result["W"] = result["W"].tolist()
    
    # Try to find lower rank decompositions
    # Lower bound: max(n*m, m*p, p*n) = max(6, 27, 18) = 27
    # Known results suggest rank might be around 30-40 for this size
    
    for rank in range(27, 54):
        if time.time() > deadline - 10.0:
            break
        
        sol = _try_rank_with_greedy(rank, n, m, p, max_iter=200, deadline=deadline - 5.0)
        if sol is not None:
            # Verify
            if _check_decomposition(sol["U"], sol["V"], sol["W"], T, tol=1e-8):
                result = {
                    "U": sol["U"].tolist(),
                    "V": sol["V"].tolist(),
                    "W": sol["W"].tolist()
                }
                break
    
    return result
# EVOLVE-BLOCK-END
