"""Matrix multiplication 2x4x4 with optimized tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]


def _strassen_2x2():
    """
    Return the 7 Strassen multiplication patterns for 2x2 matrix multiplication.
    """
    patterns = [
        # M1 = (a11+a22)(b11+b22)
        {'a': {0: 1, 3: 1}, 'b': {0: 1, 3: 1}, 'c': {0: 1, 3: 1}},
        # M2 = (a21+a22)(b11)
        {'a': {2: 1, 3: 1}, 'b': {0: 1}, 'c': {2: 1, 3: -1}},
        # M3 = (a11)(b12-b22)
        {'a': {0: 1}, 'b': {1: 1, 3: -1}, 'c': {1: 1, 3: 1}},
        # M4 = (a22)(b21-b11)
        {'a': {3: 1}, 'b': {2: 1, 0: -1}, 'c': {0: 1, 2: 1}},
        # M5 = (a11+a12)(b22)
        {'a': {0: 1, 1: 1}, 'b': {3: 1}, 'c': {0: -1, 1: 1}},
        # M6 = (a21-a11)(b11+b12)
        {'a': {2: 1, 0: -1}, 'b': {0: 1, 1: 1}, 'c': {3: 1}},
        # M7 = (a12-a22)(b21+b22)
        {'a': {1: 1, 3: -1}, 'b': {2: 1, 3: 1}, 'c': {0: 1}},
    ]
    return patterns


def _construct_strassen_28():
    """
    Construct the rank-28 Strassen decomposition.
    Returns U, V, W as numpy arrays.
    """
    n, m, p = DIMENSIONS  # 2, 4, 4
    
    U_strassen = np.zeros((n * m, 28))
    V_strassen = np.zeros((m * p, 28))
    W_strassen = np.zeros((p * n, 28))
    
    strassen_patterns = _strassen_2x2()
    blocks = [
        (0, 0, 0, 0), # A_L * B_11 -> C_L
        (2, 2, 0, 0), # A_R * B_21 -> C_L
        (0, 0, 2, 2), # A_L * B_12 -> C_R
        (2, 2, 2, 2), # A_R * B_22 -> C_R
    ]
    
    col = 0
    for (a_bj, b_bj, b_bk, c_bk) in blocks:
        for pattern in strassen_patterns:
            for local_a_idx, coeff in pattern['a'].items():
                local_r = local_a_idx // 2
                local_c = local_a_idx % 2
                global_i = local_r
                global_j = a_bj + local_c
                U_strassen[global_i * m + global_j, col] = coeff
            
            for local_b_idx, coeff in pattern['b'].items():
                local_r = local_b_idx // 2
                local_c = local_b_idx % 2
                global_j = b_bj + local_r
                global_k = b_bk + local_c
                V_strassen[global_j * p + global_k, col] = coeff
            
            for local_c_idx, coeff in pattern['c'].items():
                local_r = local_c_idx // 2
                local_c = local_c_idx % 2
                global_i = local_r
                global_k = c_bk + local_c
                W_strassen[global_k * n + global_i, col] = coeff
            
            col += 1
            
    return U_strassen, V_strassen, W_strassen


def _get_tensor(n, m, p):
    """
    Construct the matrix multiplication tensor T[i, j, k] = 1 if k == i*m + j? 
    No, the convention is W[k*n+i, r].
    The tensor T is defined such that C = A * B.
    C[i, k] = sum_j A[i, j] B[j, k].
    In the decomposition:
    C[i, k] = sum_r U[i, r] W[k, r] ... wait.
    The standard decomposition is:
    A[i, j] B[j, k] -> C[i, k]
    Tensor T has indices i, j, k.
    T[i, j, k] = 1 if the term A[i, j] B[j, k] contributes to C[i, k].
    Actually, the matrix multiplication tensor is:
    T[i, j, k] = delta_{i, k}? No.
    
    Let's check the interface:
    U: (n*m, rank) -> indices (i, j)
    V: (m*p, rank) -> indices (j, k)
    W: (p*n, rank) -> indices (k, i) ?
    "W[k*n+i,r]"
    
    The decomposition identity is:
    sum_{j} A[i, j] B[j, k] = sum_{r} U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] ?
    No, U depends on i, j. V depends on j, k. W depends on k, i.
    The sum over r gives a term for each (i, j, k) triple?
    Actually, the tensor T is defined by:
    T[i, j, k] = 1 if the product A[i, j] B[j, k] is part of the sum for C[i, k].
    In standard matrix mult, for a fixed i, k, C[i, k] = sum_j A[i, j] B[j, k].
    So for each (i, j, k), there is a contribution if we are computing C[i, k] using A[i, j] and B[j, k].
    This is always true for all i, j, k?
    No, A[i, j] is only defined for valid i, j.
    The tensor T has shape (n*m, m*p, p*n)?
    Indices:
    U index: i, j (0..n-1, 0..m-1)
    V index: j, k (0..m-1, 0..p-1)
    W index: k, i (0..p-1, 0..n-1)
    
    The tensor T is 1 at ( (i, j), (j, k), (k, i) ) ?
    Let's check:
    Sum_r U[i, j, r] V[j, k, r] W[k, i, r]
    If we set U[i, j, r] = A[i, j], V[j, k, r] = B[j, k], W[k, i, r] = 1?
    No, the decomposition is:
    C[i, k] = sum_j A[i, j] B[j, k]
    This can be written as:
    C[i, k] = sum_r ( sum_{j} U[i, j, r] A[i, j] ) ... no.
    
    The standard tensor decomposition for matrix multiplication $M_{n,m,p}$ is:
    $M_{n,m,p} = \sum_{r=1}^R u^{(r)} \otimes v^{(r)} \otimes w^{(r)}$
    where $u^{(r)} \in \mathbb{R}^{n \times m}$, $v^{(r)} \in \mathbb{R}^{m \times p}$, $w^{(r)} \in \mathbb{R}^{p \times n}$.
    And the condition is:
    $\sum_{r} u^{(r)}_{i, j} v^{(r)}_{j, k} w^{(r)}_{k, i} = \delta_{i, k}$? No.
    
    The identity is:
    $\sum_{j} A_{i, j} B_{j, k} = \sum_{r} (\sum_{j} U_{i, j}^{(r)} A_{i, j}) (\sum_{k} V_{j, k}^{(r)} B_{j, k}) \dots$
    
    Actually, the tensor $T$ is defined such that:
    $T_{i, j, k} = 1$ if the term $A_{i, j} B_{j, k}$ appears in the sum for $C_{i, k}$.
    In standard matrix multiplication, $C_{i, k} = \sum_{j} A_{i, j} B_{j, k}$.
    So for every valid i, j, k, the term $A_{i, j} B_{j, k}$ contributes to $C_{i, k}$.
    Thus, $T_{i, j, k} = 1$ for all i, j, k?
    No, the indices of the factors in the decomposition are:
    U: index (i, j)
    V: index (j, k)
    W: index (k, i)
    
    The contraction is:
    $\sum_{r} U_{i, j}^{(r)} V_{j, k}^{(r)} W_{k, i}^{(r)}$
    This sum must equal 1 if $A_{i, j} B_{j, k}$ is a term in the product, and 0 otherwise?
    Actually, the matrix multiplication tensor is:
    $T = \sum_{i, j, k} E_{i, j} \otimes E_{j, k} \otimes E_{k, i}$
    where $E_{i, j}$ is the matrix with 1 at (i, j).
    So $T_{(i, j), (j, k), (k, i)} = 1$.
    All other entries are 0.
    
    So the tensor T has shape (n*m, m*p, p*n).
    T[idx_u, idx_v, idx_w] = 1 if idx_u corresponds to (i, j), idx_v to (j, k), and idx_w to (k, i).
    Note that the indices must be consistent:
    idx_u = i*m + j
    idx_v = j*p + k
    idx_w = k*n + i
    
    So T is a sparse tensor with 1s at these specific coordinates.
    """
    n, m, p = n, m, p
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i*m + j
                idx_v = j*p + k
                idx_w = k*n + i
                T[idx_u, idx_v, idx_w] = 1.0
    return T


def _als_decompose(T, rank, U_init, V_init, W_init, max_iters=100, tol=1e-10):
    """
    Perform ALS to find a decomposition of T with the given rank.
    """
    n_u, n_v, n_w = T.shape
    U = U_init[:, :rank].copy()
    V = V_init[:, :rank].copy()
    W = W_init[:, :rank].copy()
    
    # Normalize to avoid scaling issues
    for r in range(rank):
        norm = np.linalg.norm(U[:, r])
        if norm > 1e-10:
            U[:, r] /= norm
            W[:, r] *= norm
    
    for _ in range(max_iters):
        # Update W
        # W[:, r] = sum_{u, v} T[u, v, :] * U[u, r] * V[v, r] / (U[:, r].T @ V[:, r] * U[:, r].T @ V[:, r]?)
        # Actually, we fix U, V and solve for W.
        # Minimize || T - sum_r u_r v_r w_r^T ||^2
        # For each r, w_r is a vector of length n_w.
        # The term involving w_r is:
        # sum_{u, v} (T[u, v, :] - ... ) u_r[u] v_r[v] w_r
        # Gradient w.r.t w_r[k] is:
        # sum_{u, v} T[u, v, k] u_r[u] v_r[v] - sum_{u, v} (sum_{r'} u_{r'}[u] v_{r'}[v] w_{r'}[k]) u_r[u] v_r[v]
        
        # Efficiently:
        # Let M_r be the outer product u_r v_r^T (shape n_u x n_v).
        # The contribution of r to the tensor is M_r \otimes w_r? No.
        # The tensor is sum_r u_r (v_r^T w_r^T)?
        
        # Let's use the matricization.
        # T is (n_u, n_v, n_w).
        # Fix U, V.
        # For each r, we want to find w_r.
        # The term in the sum for a specific k is:
        # sum_r u_r^T (T[:, :, k] * v_r) w_r[k] ?
        
        # Simpler:
        # Reshape T to (n_u*n_v, n_w).
        # Let B_r = u_r \otimes v_r (length n_u*n_v).
        # Then T_mat = sum_r B_r w_r^T.
        # We want to find W (rank x n_w) such that T_mat = (U \otimes V) W^T.
        # Where U \otimes V is (n_u*n_v, rank).
        
        UV = np.kron(U, V) # Shape (n_u*n_v, rank)
        T_mat = T.reshape(n_u*n_v, n_w)
        
        # Solve UV W^T = T_mat
        # W^T = UV^+ T_mat
        # W = (UV^+ T_mat)^T
        
        # Use least squares:
        # T_mat = UV X, where X = W^T (rank x n_w)
        X, residuals, rank_uv, s = np.linalg.lstsq(UV, T_mat, rcond=None)
        W = X.T
        
        # Update U
        # Fix V, W.
        # T_mat_v = T.transpose(1, 0, 2).reshape(n_v*n_u, n_w)?
        # T is (n_u, n_v, n_w).
        # We want to solve for U.
        # Term: sum_r u_r (v_r \otimes w_r)^T ?
        # Matricize along n_v, n_w.
        # T_mat_v = T.transpose(0, 1, 2) -> (n_u, n_v, n_w)
        # Reshape to (n_u, n_v*n_w).
        T_mat_v = T.reshape(n_u, n_v*n_w)
        # V \otimes W is (n_v*n_w, rank)
        VW = np.kron(V, W)
        # T_mat_v = U (VW)^T
        # U = T_mat_v (VW)^+
        U, residuals, rank_vw, s = np.linalg.lstsq(T_mat_v, VW, rcond=None) # This solves U VW^T = T_mat_v? No.
        # We want U such that U @ VW.T = T_mat_v.T?
        # T_mat_v is (n_u, n_v*n_w).
        # U is (n_u, rank).
        # VW is (n_v*n_w, rank).
        # U @ VW.T has shape (n_u, rank) @ (rank, n_v*n_w) = (n_u, n_v*n_w).
        # So we solve: U X = T_mat_v, where X = VW.T.
        # U = T_mat_v X^+
        U, residuals, rank_x, s = np.linalg.lstsq(T_mat_v, VW.T, rcond=None)
        
        # Update V
        # Fix U, W.
        # T_mat_w = T.transpose(1, 0, 2).reshape(n_v, n_u*n_w)
        T_mat_w = T.transpose(1, 0, 2).reshape(n_v, n_u*n_w)
        UW = np.kron(U, W) # (n_u*n_w, rank)
        # V UW.T = T_mat_w
        V, residuals, rank_uw, s = np.linalg.lstsq(T_mat_w, UW.T, rcond=None)
        
        # Check convergence
        # Calculate error
        # Reconstruct T
        T_recon = np.zeros_like(T)
        for r in range(rank):
            T_recon += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        error = np.linalg.norm(T - T_recon)
        
        if error < tol:
            break
            
    return U, V, W, error


def _is_integer_like(arr, tol=1e-6):
    """Check if array entries are close to integers or half-integers."""
    # Multiply by 2 to check for integers
    arr_2 = arr * 2
    rounded = np.round(arr_2)
    diff = np.abs(arr_2 - rounded)
    return np.max(diff) < tol


def construct():
    """
    Construct a low-rank tensor decomposition for 2x4 * 4x4 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    
    # Get base Strassen decomposition
    U_base, V_base, W_base = _construct_strassen_28()
    
    # Get target tensor
    T = _get_tensor(n, m, p)
    
    # Try to reduce rank
    best_rank = 28
    best_U = U_base
    best_V = V_base
    best_W = W_base
    
    # Try ranks 27 down to 24
    for target_rank in range(27, 23, -1):
        # Initialize with base decomposition (truncate)
        # This is a crude initialization, but might work if the Strassen solution is close to the optimal one
        U_init = U_base
        V_init = V_base
        W_init = W_base
        
        # Run ALS
        U, V, W, error = _als_decompose(T, target_rank, U_init, V_init, W_init, max_iters=200)
        
        if error < 1e-6:
            # Check if entries are integer-like
            if _is_integer_like(U) and _is_integer_like(V) and _is_integer_like(W):
                # Round to nearest half-integer
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify exact decomposition
                T_recon = np.zeros_like(T)
                for r in range(target_rank):
                    T_recon += np.einsum('i,j,k->ijk', U_rounded[:, r], V_rounded[:, r], W_rounded[:, r])
                
                if np.max(np.abs(T - T_recon)) < 1e-6:
                    best_rank = target_rank
                    best_U = U_rounded
                    best_V = V_rounded
                    best_W = W_rounded
                    break # Found the best possible in this loop
    
    # Convert to lists for return
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
