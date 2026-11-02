"""Structural change: ALS-based tensor decomposition with rational rounding.

Instead of random perturbation or block algebra, we use alternating least squares
to iteratively minimize the tensor approximation error, then round to rationals
and verify. We also check the rank-1 case (single Kronecker product) first.
"""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 7]


def _build_target_tensor(n, m, p):
    """Build the matrix multiplication tensor as a 3D array.
    T[u, v, w] = 1 if (u,v,w) corresponds to a valid multiplication term, else 0.
    u = i*m + j, v = j*p + k, w = k*n + i
    """
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    return T


def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that sum_r U[:,r] * V[:,r] * W[:,r] == T element-wise."""
    rank = U.shape[1]
    # Compute the reconstructed tensor
    recon = np.zeros(T.shape, dtype=np.float64)
    for r in range(rank):
        recon += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
    return np.allclose(recon, T, atol=tol)


def _round_to_rational(arr, max_den=8):
    """Round each element to the nearest multiple of 1/max_den (half-integer grid)."""
    return np.round(arr * max_den) / max_den


def _als_decompose(T, rank, max_iter=200, tol=1e-12):
    """Alternating least squares for tensor decomposition.
    Minimize ||T - sum_r u_r v_r w_r||_F^2
    """
    n1, n2, n3 = T.shape
    # Initialize randomly
    np.random.seed(42)
    U = np.random.randn(n1, rank)
    V = np.random.randn(n2, rank)
    W = np.random.randn(n3, rank)
    
    prev_err = np.inf
    for it in range(max_iter):
        # Update U: for each row i, U[i,:] solves least squares
        # T[i,:,:] = sum_r U[i,r] * V[:,r] * W[:,r]
        # Reshape: T[i,:,:] flattened = V_flat * diag(W_flat) ... 
        # Actually: T[i] (n2 x n3) = sum_r U[i,r] * (V[:,r] * W[:,r]^T)
        # So T[i].flatten() = sum_r U[i,r] * (V[:,r] * W[:,r]).flatten()
        # Build matrix B where B[:,r] = (V[:,r] * W[:,r]).flatten()
        B = np.zeros((n2 * n3, rank))
        for r in range(rank):
            B[:, r] = (V[:, r] * W[:, r]).flatten()
        # U[i] = (B^T B)^{-1} B^T T[i].flatten()
        BtB = B.T @ B
        BtB += 1e-10 * np.eye(rank)  # regularization
        for i in range(n1):
            Ti = T[i].flatten()
            U[i, :] = np.linalg.solve(BtB, B.T @ Ti)
        
        # Update V
        A = np.zeros((n1 * n3, rank))
        for r in range(rank):
            A[:, r] = (U[:, r] * W[:, r]).flatten()
        AtA = A.T @ A
        AtA += 1e-10 * np.eye(rank)
        for j in range(n2):
            # T[:,j,:] -> flatten to n1*n3
            Tj = T[:, j, :].reshape(-1)
            # T[:,j,:] = sum_r V[j,r] * (U[:,r] * W[:,r])
            # But T[:,j,:] is n1 x n3, and (U[:,r] * W[:,r]) is n1 x n3
            # So T[:,j,:].flatten() = sum_r V[j,r] * (U[:,r] * W[:,r]).flatten()
            V[j, :] = np.linalg.solve(AtA, A.T @ Tj)
        
        # Update W
        C = np.zeros((n1 * n2, rank))
        for r in range(rank):
            C[:, r] = (U[:, r] * V[:, r]).flatten()
        CtC = C.T @ C
        CtC += 1e-10 * np.eye(rank)
        for k in range(n3):
            Tk = T[:, :, k].reshape(-1)
            W[k, :] = np.linalg.solve(CtC, C.T @ Tk)
        
        # Compute error
        recon = np.zeros(T.shape, dtype=np.float64)
        for r in range(rank):
            recon += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        err = np.linalg.norm(T - recon)
        
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    
    return U, V, W, prev_err


def _try_rank1(T, n, m, p):
    """Try to find a rank-1 decomposition: T = u ⊗ v ⊗ w.
    This means T[i,j,k] = u[i]*v[j]*w[k] for all i,j,k.
    Only works if T has the special structure of a single outer product.
    """
    # For matrix multiplication tensor, T is NOT rank-1 in general.
    # Check: pick T[0,0,0], T[1,0,0], etc.
    # If T is rank-1, then T[i,j,k] = a_i * b_j * c_k
    # T[0,0,0] must be nonzero for this to work
    if T[0, 0, 0] == 0:
        return None
    
    n1, n2, n3 = T.shape
    a = np.zeros(n1)
    b = np.zeros(n2)
    c = np.zeros(n3)
    
    # Set a[0] = 1, then b[j] = T[0,j,0], c[k] = T[0,0,k]
    a[0] = 1.0
    for j in range(n2):
        b[j] = T[0, j, 0]
    for k in range(n3):
        c[k] = T[0, 0, k]
    
    # Now determine a[i] from T[i,0,0] = a[i]*b[0]*c[0]
    if b[0] * c[0] == 0:
        return None
    for i in range(n1):
        a[i] = T[i, 0, 0] / (b[0] * c[0])
    
    # Verify
    recon = np.einsum('i,j,k->ijk', a, b, c)
    if np.allclose(recon, T, atol=1e-9):
        return a, b, c
    return None


def _to_fraction_list(arr, max_den=16):
    """Convert numpy array to list of Fraction values (for output)."""
    result = []
    for val in arr.flatten():
        # Round to nearest 1/max_den
        rounded = round(val * max_den) / max_den
        # Convert to fraction
        frac = Fraction(rounded).limit_denominator(max_den)
        result.append(frac)
    return result


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    T = _build_target_tensor(n, m, p)
    
    # Try rank 1 first (very unlikely to work, but fast to check)
    r1 = _try_rank1(T, n, m, p)
    if r1 is not None:
        a, b, c = r1
        U = a.reshape(-1, 1)
        V = b.reshape(-1, 1)
        W = c.reshape(-1, 1)
        if _verify_decomposition(U, V, W, T):
            return {
                'U': U.tolist(),
                'V': V.tolist(),
                'W': W.tolist()
            }
    
    # Try ALS for various ranks, starting from low
    # The schoolbook rank is n*m*p = 112
    # Try a range of ranks
    best_result = None
    best_rank = n * m * p  # schoolbook
    
    # Schoolbook as fallback
    U_sb = np.zeros((n * m, n * m * p))
    V_sb = np.zeros((m * p, n * m * p))
    W_sb = np.zeros((p * n, n * m * p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    if time.time() > deadline - 20:
        return {'U': U_sb.tolist(), 'V': V_sb.tolist(), 'W': W_sb.tolist()}
    
    # Try ALS with various ranks
    # Start with smaller ranks and work up
    ranks_to_try = []
    # Try a few ranks below schoolbook
    for r in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]:
        if r < n * m * p:
            ranks_to_try.append(r)
    
    for rank in ranks_to_try:
        if time.time() > deadline - 15:
            break
        
        # Try a few random initializations
        found = False
        for seed in range(3):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(seed * 1000 + rank)
            U, V, W, err = _als_decompose(T, rank, max_iter=100)
            
            # Round to rational grid
            U_r = _round_to_rational(U, max_den=4)
            V_r = _round_to_rational(V, max_den=4)
            W_r = _round_to_rational(W, max_den=4)
            
            if _verify_decomposition(U_r, V_r, W_r, T, tol=1e-9):
                best_result = {'U': U_r, 'V': V_r, 'W': W_r}
                best_rank = rank
                found = True
                break
        
        if found:
            # Found a valid decomposition at this rank; try lower
            continue
        
        # If we haven't found anything yet and error is small, try harder rounding
        if err < 1e-6:
            # Try finer rounding
            for max_den in [2, 4, 8, 16]:
                U_r = _round_to_rational(U, max_den=max_den)
                V_r = _round_to_rational(V, max_den=max_den)
                W_r = _round_to_rational(W, max_den=max_den)
                if _verify_decomposition(U_r, V_r, W_r, T, tol=1e-9):
                    best_result = {'U': U_r, 'V': V_r, 'W': W_r}
                    best_rank = rank
                    break
    
    if best_result is not None:
        return {
            'U': best_result['U'].tolist(),
            'V': best_result['V'].tolist(),
            'W': best_result['W'].tolist()
        }
    
    # Fallback: schoolbook
    return {'U': U_sb.tolist(), 'V': V_sb.tolist(), 'W': W_sb.tolist()}


if __name__ == '__main__':
    result = construct()
    n, m, p = DIMENSIONS
    rank = len(result['U'][0]) if result['U'] else 0
    print(f"Rank: {rank}")
    T = _build_target_tensor(n, m, p)
    U = np.array(result['U'])
    V = np.array(result['V'])
    W = np.array(result['W'])
    valid = _verify_decomposition(U, V, W, T)
    print(f"Valid: {valid}")
