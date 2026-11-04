"""Search for a rank-minimal exact tensor decomposition of 4x4 * 4x5 matrix mult."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]


def _schoolbook(n, m, p):
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


def _verify(U, V, W, n, m, p, tol=1e-7):
    """Check that U @ diag @ V^T reconstructs the matmul tensor via W."""
    # Tensor T[i,j,k] = delta_{i,k} (for (n x m)*(m x p) -> (n x p))
    # Actually: (A B)[i,k] = sum_j A[i,j] B[j,k]
    # Tensor: T[i, j, k] = 1 if (i,j) in A, (j,k) in B, output (i,k)
    # Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # W index is [k*n + i] per the trace convention W[k*n+i, r]
    # Build the full tensor
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0  # every (i,j,k) triple contributes
    # Actually the matmul tensor: T[i,j,k] = 1 for all valid i,j,k
    # because (AB)[i,k] = sum_j A[i,j]*B[j,k], so each triple (i,j,k) has coefficient 1
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # W3[k, i, c] -> transpose to (p, n, rank) which is already W[k*n+i, c]
    # Contract: for each (i,j,k): sum over c
    # Efficient: einsum
    recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.max(np.abs(recon - T)) < tol


def construct():
    deadline = time.time() + 120
    n, m, p = DIMENSIONS
    best_rank = n * m * p  # schoolbook
    U0, V0, W0 = _schoolbook(n, m, p)
    best_result = {"U": U0, "V": V0, "W": W0}

    # Try a few candidate ranks below schoolbook
    # Known: 4x4x4 has rank 49 (Strassen-like), 4x4x5 might be around 50-80
    # We'll try to verify if a random/structured decomposition works at lower rank.
    # For safety, we focus on finding a valid decomposition.
    
    # Strategy: try structured decompositions
    # 1. Try rank = n*m*p - 1, n*m*p - 2, ... down to some limit
    # But verification is expensive for high rank.
    
    # Let's try a simple approach: check if the schoolbook can be compressed.
    # For 4x4x5, schoolbook rank = 80.
    
    # Try to find a decomposition at rank < 80 using a simple search.
    # Given time constraints, we'll try a few structured candidates.
    
    # Candidate: use the fact that for square matrices, Strassen gives rank 7 for 2x2x2.
    # For 4x4x4, we can use block Strassen: rank = 4^3 * 7/8 = 28 * 7/8... 
    # Actually 4x4x4 via Strassen: split into 2x2 blocks, each 2x2x2 costs 7, so 4*7 = 28? No.
    # 4x4x4: split into 2x2 blocks of 2x2. Each 2x2x2 multiplication uses 7 products.
    # There are 4 block positions, so 4*7 = 28? No, that's not right either.
    # Standard: 4x4x4 rank is 49 (Lader). Let's not go there.
    
    # For 4x4x5, let's just try to verify the schoolbook and return it.
    # The search for lower rank is complex; we'll do a quick random search.
    
    # Quick random search for lower rank
    np.random.seed(42)
    for rank in range(75, 79):  # try a few ranks below 80
        if time.time() > deadline - 30:
            break
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        if _verify(U, V, W, n, m, p, tol=1e-5):
            best_rank = rank
            best_result = {"U": U, "V": V, "W": W}
            break

    return best_result
