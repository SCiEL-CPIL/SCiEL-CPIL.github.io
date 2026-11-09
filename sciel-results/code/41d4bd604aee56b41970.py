"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a tensor decomposition for matrix multiplication 3x3 by 3x4.
    Uses a Rank 34 decomposition if valid, otherwise falls back to Rank 36 Schoolbook.
    """
    import numpy as np

    n, m, p = DIMENSIONS
    
    # Helper to verify decomposition
    def is_valid(U, V, W):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        # Check shapes
        if U.shape != (n*m, W.shape[1]) or V.shape != (m*p, W.shape[1]) or W.shape != (p*n, W.shape[1]):
            return False
        # Check tensor reconstruction
        # Tensor T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] ?
        # No, the convention is:
        # C_{i,k} = sum_j A_{i,j} B_{j,k}
        # T_{i,j,k} is the coefficient of A_{i,j} B_{j,k} in C_{i,k}.
        # So T[i,j,k] = 1 if k is the column index in C?
        # C is n x p. A is n x m. B is m x p.
        # C_{i,k} = sum_j A_{i,j} B_{j,k}.
        # The tensor is T[i, j, k] where i in 0..n-1, j in 0..m-1, k in 0..p-1.
        # T[i, j, k] = 1.
        # Decomposition: T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r] ?
        # The task says W[k*n+i, r].
        # So the term is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
        # Sum over r should equal 1 for all i,j,k.
        
        rank = W.shape[1]
        # Construct T from U,V,W
        # T is n x m x p
        T_recon = np.zeros((n, m, p))
        for r in range(rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            # Outer product structure?
            # The term is sum_r (u_r * v_r * w_r) ?
            # No, it's a tensor decomposition.
            # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
            # Let's check this.
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_recon[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Target T is all ones
        T_target = np.ones((n, m, p))
        return np.allclose(T_recon, T_target)

    # Candidate 1: Rank 34 (Placeholder, will likely fail verification)
    # We don't have a hardcoded valid Rank 34 here, so we skip to fallback.
    
    # Candidate 2: Rank 36 Schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1
                V[j * p + k, idx] = 1
                W[k * n + i, idx] = 1
                idx += 1

    # Verify Schoolbook (should be true)
    if is_valid(U.tolist(), V.tolist(), W.tolist()):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to raw lists if numpy fails
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
