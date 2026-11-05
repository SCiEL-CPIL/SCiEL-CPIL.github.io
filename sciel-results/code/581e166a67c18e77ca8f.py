import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START

def _tensor_index(n, m, p):
    """Return function to flatten (i,j,k) to linear index for the multiplication tensor."""
    # T[i*m+j, j*p+k, k*n+i] = 1
    pass

def _build_tensor(n=3, m=5, p=7):
    """Build the full multiplication tensor as a 3D array T[i,j,k]."""
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _tensor_to_matrices(T, n, m, p):
    """Convert tensor T[i,j,k] to the three factor matrices at rank 1.
    Returns U (n*m,), V (m*p,), W (p*n,) for a single rank-1 term."""
    U = T.reshape(n * m)  # T[i,j,k] for fixed k... no, this is 3D
    # Actually for rank-1: T[i,j,k] = u[i,j] * v[j,k] * w[k,i]
    # We need to extract u, v, w from T
    pass

def _extract_rank1(T, n, m, p):
    """Extract a single rank-1 component from tensor T.
    Returns (u, v, w) where T ≈ u[i,j]*v[j,k]*w[k,i] + residual."""
    # Use iterative power method on the tensor
    # Initialize w randomly or from max slice
    n, m, p = n, m, p
    # Initialize
    u = np.ones((n, m))
    v = np.ones((m, p))
    w = np.ones((p, n))
    
    for _ in range(100):
        # Update u: u[i,j] = sum_{k} T[i,j,k] * v[j,k] * w[k,i]
        u_new = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                s = 0.0
                for k in range(p):
                    s += T[i, j, k] * v[j, k] * w[k, i]
                u_new[i, j] = s
        # Normalize
        norm_u = np.linalg.norm(u_new)
        if norm_u < 1e-15:
            break
        u = u_new / norm_u
        
        # Update v: v[j,k] = sum_{i} T[i,j,k] * u[i,j] * w[k,i]
        v_new = np.zeros((m, p))
        for j in range(m):
            for k in range(p):
                s = 0.0
                for i in range(n):
                    s += T[i, j, k] * u[i, j] * w[k, i]
                v_new[j, k] = s
        norm_v = np.linalg.norm(v_new)
        if norm_v < 1e-15:
            break
        v = v_new / norm_v
        
        # Update w: w[k,i] = sum_{j} T[i,j,k] * u[i,j] * v[j,k]
        w_new = np.zeros((p, n))
        for k in range(p):
            for i in range(n):
                s = 0.0
                for j in range(m):
                    s += T[i, j, k] * u[i, j] * v[j, k]
                w_new[k, i] = s
        norm_w = np.linalg.norm(w_new)
        if norm_w < 1e-15:
            break
        w = w_new / norm_w
    
    # Scale: make the product u*v*w match T as closely as possible
    # Compute the actual rank-1 approximation
    approx = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                approx[i, j, k] = u[i, j] * v[j, k] * w[k, i]
    
    # Scale factor: least squares
    # T ≈ alpha * approx
    if np.linalg.norm(approx) > 1e-15:
        alpha = np.dot(T.flatten(), approx.flatten()) / np.dot(approx.flatten(), approx.flatten())
    else:
        alpha = 0.0
    
    u = alpha * u
    # Keep v and w as is (absorbed into u)
    
    return u, v, w

def _verify(U, V, W, n=3, m=5, p=7):
    """Verify that the decomposition is exact."""
    r = len(U[0])
    for c in range(r):
        uc = np.array([U[i][c] for i in range(n * m)])
        vc = np.array([V[j][c] for j in range(m * p)])
        wc = np.array([W[k][c] for k in range(p * n)])
        u_mat = uc.reshape(n, m)
        v_mat = vc.reshape(m, p)
        prod = u_mat @ v_mat  # (n, p)
        w_mat = wc.reshape(p, n)
        if not np.allclose(prod, w_mat.T, atol=1e-10):
            return False
    return True

def _construct_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _greedy_rank1_extraction(deadline):
    """Greedy rank-1 extraction to find a low-rank decomposition."""
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    T_residual = T.copy()
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    max_iter = 105  # schoolbook rank
    start = time.time()
    
    for iteration in range(max_iter):
        if time.time() > deadline - 10:
            break
        
        # Extract rank-1 component
        u, v, w = _extract_rank1(T_residual, n, m, p)
        
        # Compute the actual rank-1 term to subtract
        approx = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    approx[i, j, k] = u[i, j] * v[j, k] * w[k, i]
        
        # Check if this is a meaningful component
        norm_approx = np.linalg.norm(approx)
        if norm_approx < 1e-10:
            break
        
        # Subtract from residual
        T_residual = T_residual - approx
        
        # Check if residual is zero
        if np.linalg.norm(T_residual) < 1e-10:
            break
        
        # Store the rank-1 factors
        # U: (n*m, 1), V: (m*p, 1), W: (p*n, 1)
        u_flat = (u * 1.0).flatten().tolist()  # u[i,j] -> U[i*m+j]
        v_flat = (v * 1.0).flatten().tolist()  # v[j,k] -> V[j*p+k]
        w_flat = (w * 1.0).flatten().tolist()  # w[k,i] -> W[k*n+i]
        
        U_cols.append(u_flat)
        V_cols.append(v_flat)
        W_cols.append(w_flat)
    
    if not U_cols:
        return None
    
    # Transpose to (n*m, r), (m*p, r), (p*n, r)
    r = len(U_cols)
    U = [[U_cols[c][i] for c in range(r)] for i in range(n * m)]
    V = [[V_cols[c][j] for c in range(r)] for j in range(m * p)]
    W = [[W_cols[c][k] for c in range(r)] for k in range(p * n)]
    
    return {"U": U, "V": V, "W": W}

def _try_fixed_rank1_patterns(deadline):
    """Try structured rank-1 patterns: each rank-1 term covers a subset of (i,j,k)."""
    n, m, p = DIMENSIONS
    # Try: for each k, use Strassen-like 2x2 blocks where possible
    # Or: try all combinations of "support" patterns
    
    best = None
    best_rank = 105
    
    # Strategy: try rank-r where r < 105 by using the fact that
    # T[i,j,k] = 1 for all i,j,k means the tensor is all-ones.
    # The all-ones tensor has rank 1! T[i,j,k] = 1*1*1
    # Wait, that's not right for matrix multiplication tensor.
    # The multiplication tensor has T[i,j,k] = 1 iff the (i,j) entry of A, 
    # (j,k) entry of B, and (k,i) entry of C are coupled.
    # Actually T[i,j,k] = 1 for ALL i,j,k in the standard multiplication tensor.
    # So it IS the all-ones tensor, which has rank 1!
    
    # But wait - that can't be right. The standard matrix multiplication tensor
    # T_{ijk} = 1 for all i,j,k means C_{ik} = sum_j A_{ij} B_{jk}.
    # The tensor is indeed all-ones. Its rank as a 3-way tensor is 1.
    # But for matrix multiplication, we need a specific structure.
    
    # Let me reconsider. The multiplication tensor for C = A*B where
    # A is n×m, B is m×p, C is n×p:
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # This is the all-ones tensor of size n×m×p, which has tensor rank 1.
    
    # So the answer should be rank 1!
    # U[i*m+j] = 1, V[j*p+k] = 1, W[k*n+i] = 1
    # Then sum_c U[i*m+j,c]*V[j*p+k,c] = 1*1 = 1 = W[k*n+i,c]
    
    # Let's verify this!
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    return None

# EVOLVE-BLOCK-END

def construct():
    deadline = time.time() + 130  # safety margin
    
    # First, try the trivial rank-1 solution
    rank1 = _try_fixed_rank1_patterns(deadline)
    if rank1 is not None:
        return rank1
    
    # If that fails (shouldn't, but just in case), try greedy extraction
    greedy = _greedy_rank1_extraction(deadline)
    if greedy is not None and _verify(greedy["U"], greedy["V"], greedy["W"]):
        r = len(greedy["U"][0])
        if r <= 105:
            return greedy
    
    # Fallback to schoolbook
    return _construct_schoolbook()
