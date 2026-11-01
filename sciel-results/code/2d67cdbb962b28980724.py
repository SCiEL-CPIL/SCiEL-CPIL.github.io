import time
import numpy as np

def _strassen_decomposition():
    """Return the known rank-7 Strassen decomposition."""
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))

    terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(1, 1), (3, -1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(2, 1), (3, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (1, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (2, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]

    for r, (u_contribs, v_contribs, w_contribs) in enumerate(terms):
        for idx, coef in u_contribs:
            U[idx, r] += coef
        for idx, coef in v_contribs:
            V[idx, r] += coef
        for idx, coef in w_contribs:
            W[idx, r] += coef

    return U, V, W


def _verify_decomposition(U, V, W, tol=1e-10):
    """
    Verify that U @ diag(W) @ V^T equals the matrix multiplication tensor.
    Tensor T[i,j,k] = delta_{i,k} (so T[i,j,k] = 1 if i==k else 0)
    where i indexes rows of A (0..n-1), j indexes cols of A / rows of B,
    k indexes cols of B.
    
    With our flattening:
    - A index: i*m + j  (row-major, n*m entries)
    - B index: j*p + k  (row-major, m*p entries)
    - C index: k*n + i  (trace convention, p*n entries)
    
    T[A_idx, B_idx, C_idx] = 1 if (i == k) else 0
    """
    n, m, p = 2, 2, 2
    # Build the target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i*m + j
                b_idx = j*p + k
                c_idx = k*n + i
                T[a_idx, b_idx, c_idx] = 1.0
    
    # Compute decomposition tensor: D[a,b,c] = sum_r U[a,r] * W[c,r] * V[b,r]
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    rank = U.shape[1]
    D = np.einsum('ar,cr,br->abc', U, W, V)
    
    max_err = np.max(np.abs(D - T))
    return max_err < tol, max_err


def _attempt_rank6_search(deadline):
    """
    Attempt to find a rank-6 decomposition using parameterized search.
    Uses the structure of matrix multiplication tensor and tries to find
    a valid rank-6 decomposition via numerical optimization.
    
    Returns (U, V, W) if found, None otherwise.
    """
    n, m, p = 2, 2, 2
    rank = 6
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i*m + j
                b_idx = j*p + k
                c_idx = k*n + i
                T[a_idx, b_idx, c_idx] = 1.0
    
    # Random initialization with various seeds
    rng = np.random.RandomState(42)
    
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        # Initialize U, V, W randomly
        U = rng.randn(n*m, rank)
        V = rng.randn(m*p, rank)
        W = rng.randn(p*n, rank)
        
        # Normalize columns to avoid scale issues
        for r in range(rank):
            norm_u = np.linalg.norm(U[:, r])
            norm_v = np.linalg.norm(V[:, r])
            norm_w = np.linalg.norm(W[:, r])
            if norm_u > 1e-10:
                U[:, r] /= norm_u
            if norm_v > 1e-10:
                V[:, r] /= norm_v
            if norm_w > 1e-10:
                W[:, r] /= norm_w
        
        # Gradient descent on the tensor reconstruction error
        lr = 0.01
        for step in range(200):
            if time.time() > deadline:
                return None
            
            # Compute D = sum_r U[:,r] * W[:,r] * V[:,r]^T
            D = np.einsum('ar,cr,br->abc', U, W, V)
            err = D - T
            
            # Gradients
            # dL/dU[a,r] = sum_{b,c} err[a,b,c] * W[c,r] * V[b,r]
            dU = np.einsum('abc,cr,br->ar', err, W, V)
            dV = np.einsum('abc,ar,cr->br', err, U, W)
            dW = np.einsum('abc,ar,br->cr', err, U, V)
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Periodic check
            if step % 50 == 0:
                max_err = np.max(np.abs(D - T))
                if max_err < 1e-10:
                    valid, _ = _verify_decomposition(U, V, W)
                    if valid:
                        return U, V, W
        
        # Final check
        valid, max_err = _verify_decomposition(U, V, W)
        if valid:
            return U, V, W
    
    return None


def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 matrix multiplication.
    
    First attempts to find a rank-6 decomposition (which would be an improvement
    over the known Strassen rank of 7). If no valid rank-6 decomposition is found
    within the time budget, falls back to the proven rank-7 Strassen algorithm.
    
    The tensor rank of 2x2 matrix multiplication is known to be 7 over the
    rationals (Strassen's algorithm). A rank-6 decomposition would be a
    significant mathematical discovery.
    """
    
    start_time = time.time()
    deadline = start_time + 130.0  # Leave 20s safety margin before 150s kill
    
    # Attempt rank-6 search
    result = _attempt_rank6_search(deadline)
    
    if result is not None:
        U, V, W = result
        # Verify before returning
        valid, max_err = _verify_decomposition(U, V, W, tol=1e-10)
        if valid:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: Strassen rank-7 decomposition
    U, V, W = _strassen_decomposition()
    
    # Verify Strassen (should always pass)
    valid, max_err = _verify_decomposition(U, V, W)
    if not valid:
        # This should never happen, but handle gracefully
        raise RuntimeError(f"Strassen verification failed with max_err={max_err}")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
