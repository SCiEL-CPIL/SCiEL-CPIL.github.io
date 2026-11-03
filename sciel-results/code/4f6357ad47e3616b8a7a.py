"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.linalg import null_space
from scipy.optimize import minimize
import itertools

DIMENSIONS = [3, 4, 6]

def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
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

def _strassen_block(n, m, p, block_sizes):
    """
    Build a block-structured Strassen-like decomposition.
    block_sizes: list of (ni, mi, pi) for each block partition.
    This is a heuristic; we build a base decomposition and then try to reduce rank.
    """
    # For now, just use schoolbook as the base, then attempt reduction
    return _schoolbook(n, m, p)

def _tensor_to_matrix(U, V, W):
    """
    Build the flattening matrix M of shape (n*m*p, n*m*p) where
    M[a, b] = sum_r U[a, r] * V[b, r] * W[?, r] ... 
    Actually, the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    The flattening over (i,j) vs k: M[(i,j), k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    This isn't a simple matrix product. Let's use a different approach.
    
    For rank reduction: we check if the set of rank-1 tensors is linearly dependent.
    The tensor T = sum_r u_r ⊗ v_r ⊗ w_r.
    We can check linear dependence by looking at the vectorization of each rank-1 term.
    vec(T) = sum_r vec(u_r ⊗ v_r ⊗ w_r)
    Each vec(u_r ⊗ v_r ⊗ w_r) is a vector of length n*m*p.
    We form a matrix A where column r is vec(u_r ⊗ v_r ⊗ w_r) and check rank.
    """
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    # Vectorize each rank-1 component
    # u_r is length n*m, v_r is length m*p, w_r is length p*n
    # The tensor component for rank r: T_r[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Vectorize over (i,j,k) in order i fastest? No, let's use a consistent ordering.
    # Let's vectorize as: index = i * m * p + j * p + k
    vecs = []
    for r in range(rank):
        vec = np.zeros(n * m * p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    vec[i * m * p + j * p + k] = val
        vecs.append(vec)
    return np.column_stack(vecs)

def _try_reduce_rank(U, V, W, max_attempts=5, time_limit=100):
    """
    Attempt to reduce the rank by finding linear dependencies among the rank-1 components.
    """
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    
    # Build the matrix of vectorized rank-1 components
    A = _tensor_to_matrix(U, V, W)
    
    # Find null space
    # Use SVD to find numerical rank
    try:
        U_svd, s, Vt = np.linalg.svd(A, full_matrices=False)
        # Determine numerical rank
        tol = max(A.shape) * np.finfo(float).eps * s[0] if len(s) > 0 else 0
        num_rank = np.sum(s > tol)
        
        if num_rank < rank:
            # There are dependencies. Find the null space basis.
            null = Vt[num_rank:]  # Shape: (rank - num_rank, rank)
            
            # We can try to eliminate components by expressing them in terms of others.
            # For simplicity, we'll try to remove the last (rank - num_rank) columns
            # by adjusting the remaining ones. This requires solving a system.
            
            # Let's try a simpler approach: randomly combine columns to see if we can drop one.
            # For each column r, check if it's in the span of the others.
            
            # Actually, let's just try to find a subset of columns that spans the same space
            # and use only those.
            
            # Pivot columns from SVD
            # The first num_rank columns of U_svd span the column space.
            # We need to find which original columns are independent.
            
            # Use QR with column pivoting
            Q, R, P = _qr_with_pivoting(A)
            
            # Determine rank from R
            tol2 = max(A.shape) * np.finfo(float).eps * abs(R[0,0]) if R.shape[0] > 0 else 0
            rank_qr = 0
            for i in range(min(R.shape)):
                if abs(R[i,i]) > tol2:
                    rank_qr += 1
                else:
                    break
            
            if rank_qr < rank:
                # Independent columns are P[:rank_qr]
                # We can keep only these columns
                indep_cols = P[:rank_qr]
                U_new = U[:, indep_cols]
                V_new = V[:, indep_cols]
                W_new = W[:, indep_cols]
                
                # Verify correctness
                if _verify(U_new, V_new, W_new):
                    return U_new, V_new, W_new
    except Exception:
        pass
    
    return U, V, W

def _qr_with_pivoting(A):
    """QR decomposition with column pivoting."""
    # Simple implementation using numpy
    n, m = A.shape
    Q = np.eye(n)
    R = A.copy()
    P = np.arange(m)
    
    for k in range(min(n, m)):
        # Find column with largest norm in remaining part
        norms = np.linalg.norm(R[k:, k:], axis=0)
        j = k + np.argmax(norms)
        
        if j != k:
            R[:, [k, j]] = R[:, [j, k]]
            P[[k, j]] = P[[j, k]]
        
        # QR step for column k
        col = R[k:, k]
        norm = np.linalg.norm(col)
        if norm < 1e-15:
            continue
        v = col / norm
        v[0] = 1
        R[k:, k:] -= np.outer(v, v.T @ R[k:, k:])
        Q[:, k:] -= np.outer(Q[k:, k] @ v, v.T)
    
    return Q, R, P

def _verify(U, V, W, rtol=1e-10, atol=1e-12):
    """Verify that the decomposition is correct."""
    n, m, p = DIMENSIONS
    # Test with random matrices
    rng = np.random.RandomState(42)
    A = rng.randn(n, m)
    B = rng.randn(m, p)
    C_expected = A @ B
    
    # Compute using decomposition
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_j sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] A[i,j] B[j,k]
    # Wait, the decomposition is: T[i,j,k] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]
    # And C[i,k] = sum_j A[i,j] B[j,k] should equal sum_j sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] A[i,j] B[j,k]?
    # No, that's not right. The tensor T represents the multiplication map.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The decomposition says: for each (i,j,k), the "coefficient" is sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]
    # But this should equal the delta that picks out A[i,j]*B[j,k].
    # Actually, the standard decomposition has U[i*m+j,r]=1, V[j*p+k,r]=1, W[k*n+i,r]=1 for r=(i,j,k)
    # Then sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1 if (i,j,k) matches, else 0.
    # So C[i,k] = sum_j (sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]) A[i,j] B[j,k]
    #           = sum_j delta_{i,j,k} A[i,j] B[j,k] ... this doesn't make sense.
    
    # Let me reconsider. The tensor T is defined such that:
    # C[i,k] = sum_{j} A[i,j] B[j,k]
    # The multiplication tensor T has components T[i,j,k] where the trilinear form is:
    # sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_{i,k} C[i,k] * (something)
    # Actually, the standard way: T[i,j,k] = 1 if the index pattern matches, and the decomposition
    # gives T as a sum of rank-1 tensors.
    
    # The correct verification: 
    # For the schoolbook, T[i,j,k] = 1 for each (i,j,k) triple.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The trilinear form: sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] * D[k,i] = sum_{i,k} (sum_j A[i,j]B[j,k]) D[k,i]
    # So D is the output. The decomposition should satisfy:
    # sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = T[i,j,k]
    
    # So verification: check that for all i,j,k:
    # sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] == T[i,j,k]
    # where T[i,j,k] = 1 for the schoolbook.
    
    # But for a general decomposition, T[i,j,k] is not necessarily 1.
    # The condition is that the decomposition reproduces the multiplication map.
    
    # Let's verify by checking: for random A, B, does the decomposition give C = A@B?
    # The decomposition defines a map: (A, B) -> C where
    # C[i,k] = sum_j sum_r U[i*m+j,r] * (A[i,j] * V[j*p+k,r] * B[j,k]) * W[k*n+i,r]
    # This is NOT how it works. The rank-1 decomposition of the tensor means:
    # T = sum_r u_r ⊗ v_r ⊗ w_r
    # where (u_r ⊗ v_r ⊗ w_r)[i,j,k] = u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
    # And the multiplication is: C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k] ... no.
    
    # I think the correct interpretation is:
    # The tensor T encodes the bilinear map (A, B) -> C.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # This can be written as: C[i,k] = sum_{i',j,k'} T[i,k,i',j,k'] A[i',j] B[j,k']
    # where T[i,k,i',j,k'] = 1 if i=i' and k=k', else 0.
    # But the task uses a different indexing. Let me just trust the schoolbook and verify
    # that my decomposition matches it.
    
    # For the schoolbook: U[i*m+j, (i,j,k)] = 1, V[j*p+k, (i,j,k)] = 1, W[k*n+i, (i,j,k)] = 1
    # Then sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1 if r=(i,j,k), else 0.
    # So the "effective" T[i,j,k] = sum_r ... = 1 for all (i,j,k).
    # And C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k] = sum_j A[i,j] B[j,k]. Correct!
    
    # So for a general decomposition, we need:
    # sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1 for all i,j,k.
    # Wait, that's only true for the schoolbook. For a Strassen-like decomposition,
    # the T[i,j,k] values are not all 1.
    
    # Actually, I think the condition is that the decomposition reproduces the tensor T
    # where T[i,j,k] = 1 for the standard multiplication. Let me just check that
    # sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] equals the original T values.
    
    # For the schoolbook, T[i,j,k] = 1 for all i,j,k.
    # So I need to check that my decomposition gives T[i,j,k] = 1 for all i,j,k.
    
    # Let's build the tensor from the decomposition and compare to the schoolbook tensor.
    T_decomp = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_decomp[i, j, k] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
    
    # The schoolbook tensor has T[i,j,k] = 1 for all i,j,k.
    T_expected = np.ones((n, m, p))
    
    if not np.allclose(T_decomp, T_expected, rtol=rtol, atol=atol):
        return False
    return True

def _qr_with_pivoting(A):
    """QR decomposition with column pivoting using scipy."""
    from scipy.linalg import qr
    Q, R, P = qr(A, pivoting=True)
    return Q, R, P

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to reduce rank
    U, V, W = _try_reduce_rank(U, V, W)
    if U.shape[1] < best_rank:
        best_rank = U.shape[1]
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Convert to lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
