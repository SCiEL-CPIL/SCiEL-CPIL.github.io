"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 4, 6]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 115  # 5s safety margin before 120s kill

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Build the matrix-multiplication tensor T[i,j,k] where T[i,m*j+k, p*n+i] = 1
    Actually: T[a, b, c] = 1 iff a = i, b = m*j+k, c = p*n+i... 
    Let's use the standard: T[i, j, k] for the tensor where
    (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}
    Tensor T[i, j, k] = delta (it's 1 for all valid i,j,k in the product sense)
    
    Better formulation: The multiplication tensor has indices (i, j, k) where
    i in [0,n), j in [0,m), k in [0,p)
    T[i, j, k] = 1 for all i,j,k (it's the identity tensor for this multiplication)
    
    Wait, that's not right either. Let me think carefully.
    
    The tensor for matrix multiplication C = A*B where A is n×m, B is m×p, C is n×p:
    C_{i,k} = sum_j A_{i,j} B_{j,k}
    
    The tensor T has three modes:
    - Mode 1: (i, j) for A, size n*m
    - Mode 2: (j, k) for B, size m*p  
    - Mode 3: (i, k) for C, size n*p
    
    T[(i,j), (j',k), (i',k')] = 1 if i=i' and j=j' and k=k', else 0
    
    So T is essentially a "diagonal" tensor of size (n*m) × (m*p) × (n*p) with
    exactly n*m*p ones on the "diagonal" where all three indices match appropriately.
    """
    n, m, p = DIMENSIONS
    # T[a, b, c] = 1 iff:
    #   a = i*m + j, b = j'*p + k, c = i'*n + k'
    #   and i = i', j = j', k = k'
    # So: a = i*m+j, b = j*p+k, c = i*n+k where i,j,k are the same
    T = np.zeros((n*m, m*p, n*p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = i * n + k
                T[a, b, c] = 1.0
    return T


def _check_decomposition(U, V, W, T, tol=1e-9):
    """Check if sum_r U[:,r] * V[:,r] * W[:,r] == T"""
    rank = U.shape[1]
    # Compute the tensor as outer sum
    # T_reconstructed[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # Use einsum for efficiency
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_recon, T, atol=tol)


def _try_remove_term(U, V, W, T, col, tol=1e-9):
    """Try removing column 'col' by adjusting other columns.
    Returns (success, new_U, new_V, new_W) or (False, None, None, None)
    
    Strategy: If we remove column r, we need to absorb its contribution into
    other columns. This is possible if the outer product U[:,r]⊗V[:,r]⊗W[:,r]
    can be expressed as a linear combination of the remaining outer products.
    
    A simpler approach: try setting the column to zero and check if the
    tensor can still be recovered by adjusting a few other columns.
    
    Actually, the cleanest approach: for each pair of columns (r, s), check if
    we can merge them. Or: for each column r, check if removing it and solving
    for adjustments to one other column s works.
    """
    rank = U.shape[1]
    if rank <= 1:
        return False, None, None, None
    
    n_rows_U, n_rows_V, n_rows_W = U.shape[0], V.shape[0], W.shape[0]
    
    # Try removing column 'col' and see if the residual can be absorbed
    # by modifying one other column.
    # Residual after removing col: R = U[:,col]*V[:,col]*W[:,col]
    # We need: R = sum_{s != col} delta_U_s * V[:,s] * W[:,s] + ... 
    # This is complex. Let's use a simpler strategy:
    
    # Strategy: For each other column s, check if we can find scalars such that
    # U[:,col]*V[:,col]*W[:,col] = alpha * U[:,s]*V[:,s]*W[:,s]
    # This only works for very specific cases.
    
    # Better strategy: Try to find if column 'col' is a linear combination
    # in the tensor sense. We'll use a rank-reduction approach:
    # 
    # The tensor T = sum_r u_r ⊗ v_r ⊗ w_r
    # If we can find coefficients such that for some r:
    # u_r ⊗ v_r ⊗ w_r = sum_{s≠r} c_s * (u_s' ⊗ v_s' ⊗ w_s')
    # where u_s', v_s', w_s' are modified versions...
    #
    # Simplest effective approach: try removing each column and checking
    # if the remaining rank-1 tensors span the original tensor.
    # We can do this by checking if the tensor without column r has the
    # same "support" structure.
    
    # Most practical: try to eliminate a column by solving a small linear system
    # For column r to be removable, we need:
    # u_r ⊗ v_r ⊗ w_r ∈ span{u_s ⊗ v_s ⊗ w_s : s ≠ r}
    #
    # This is a subspace membership test. We can check by seeing if
    # u_r ⊗ v_r ⊗ w_r is in the span of the other rank-1 tensors.
    
    # Build the matrix of other terms flattened
    other_cols = [c for c in range(rank) if c != col]
    if not other_cols:
        return False, None, None, None
    
    # Flatten all rank-1 tensors
    target = np.kron(np.kron(U[:, col], V[:, col]), W[:, col])
    basis = np.array([np.kron(np.kron(U[:, c], V[:, c]), W[:, c]) for c in other_cols])
    
    # Check if target is in span of basis
    try:
        # Use least squares
        coeffs, residuals, rank_basis, sv = np.linalg.lstsq(basis.T, target, rcond=None)
        recon = basis.T @ coeffs
        if np.allclose(recon, target, atol=tol):
            # Column col is redundant! Remove it.
            new_U = np.delete(U, col, axis=1)
            new_V = np.delete(V, col, axis=1)
            new_W = np.delete(W, col, axis=1)
            return True, new_U, new_V, new_W
    except:
        pass
    
    return False, None, None, None


def _try_merge_columns(U, V, W, T, col1, col2, tol=1e-9):
    """Try to merge two columns into one by finding a rank-1 tensor
    that replaces both. This requires:
    u1⊗v1⊗w1 + u2⊗v2⊗w2 = u⊗v⊗w for some u,v,w
    
    This is possible if the sum of two rank-1 tensors has rank 1,
    which happens when they are proportional.
    """
    # Check if the two rank-1 tensors are proportional
    t1 = np.kron(np.kron(U[:, col1], V[:, col1]), W[:, col1])
    t2 = np.kron(np.kron(U[:, col2], V[:, col2]), W[:, col2])
    
    # Check proportionality
    if np.allclose(t1, 0, atol=tol) or np.allclose(t2, 0, atol=tol):
        return False, None, None, None
    
    # Check if t1 = alpha * t2 for some alpha
    # Use first non-zero element
    mask = np.abs(t1) > tol
    if not np.any(mask):
        return False, None, None, None
    
    idx = np.argmax(mask)
    alpha = t1[idx] / t2[idx]
    
    if np.allclose(t1, alpha * t2, atol=tol):
        # They are proportional, so their sum is also rank-1
        # New term: (1+alpha) * t2 (or t1)
        # We can just keep one and scale it
        new_U = np.delete(U, col2, axis=1)
        new_V = np.delete(V, col2, axis=1)
        new_W = np.delete(W, col2, axis=1)
        # Scale the remaining column by (1 + alpha)
        new_U[:, col1] *= (1 + alpha)
        return True, new_U, new_V, new_W
    
    return False, None, None, None


def _get_strassen_2x2_terms():
    """
    Returns U, V, W matrices for the rank-7 Strassen decomposition of 2x2 matrix multiplication.
    U shape: (4, 7)
    V shape: (4, 7)
    W shape: (4, 7)
    Indices:
    A = [[a, b], [c, d]] -> U index 0:a, 1:b, 2:c, 3:d
    B = [[e, f], [g, h]] -> V index 0:e, 1:f, 2:g, 3:h
    C = [[y, z], [w, x]] -> W index 0:y, 1:z, 2:w, 3:x
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # M1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1 # C11
    W[3, 0] = 1 # C22
    
    # M2 = (c+d)h
    U[2, 1] = 1; U[3, 1] = 1
    V[3, 1] = 1
    W[2, 1] = 1 # C21
    W[3, 1] = 1 # C22
    
    # M3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1 # C12
    W[3, 2] = -1 # C22
    
    # M4 = d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1 # C11
    W[2, 3] = 1 # C21
    
    # M5 = (a+b)e
    U[0, 4] = 1; U[1, 4] = 1
    V[0, 4] = 1
    W[0, 4] = -1 # C11
    W[1, 4] = 1 # C12
    
    # M6 = (c-a)(e+f)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = -1 # C22
    
    # M7 = (b-d)(g+h)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1 # C11
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # Use Strassen for 2x2 blocks
    # A is 2x4, B is 4x6, C is 2x6
    # Split A into A1 (2x2), A2 (2x2)
    # Split B into B1 (2x6), B2 (2x6)
    # C = A1 B1 + A2 B2
    # B1 is 2x6, split into B1_1, B1_2, B1_3 (each 2x2)
    # B2 is 2x6, split into B2_1, B2_2, B2_3 (each 2x2)
    # C1 = A1 B1_1 (2x2)
    # C2 = A1 B1_2 (2x2)
    # C3 = A1 B1_3 (2x2)
    # C4 = A2 B2_1 (2x2)
    # C5 = A2 B2_2 (2x2)
    # C6 = A2 B2_3 (2x2)
    # C = [C1, C2, C3, C4, C5, C6]
    
    U_str, V_str, W_str = _get_strassen_2x2_terms()
    
    # Total rank = 6 * 7 = 42
    total_rank = 42
    U = np.zeros((n * m, total_rank), dtype=np.float64)
    V = np.zeros((m * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)
    
    # Map indices
    # A: 2x4. Indices: i in 0..1, j in 0..3.
    # A1: j in 0..1. A2: j in 2..3.
    # B: 4x6. Indices: j in 0..3, k in 0..5.
    # B1: j in 0..1. B2: j in 2..3.
    # B1_1: k in 0..1. B1_2: k in 2..3. B1_3: k in 4..5.
    # B2_1: k in 0..1. B2_2: k in 2..3. B2_3: k in 4..5.
    # C: 2x6. Indices: i in 0..1, k in 0..5.
    # C1: k in 0..1. C2: k in 2..3. C3: k in 4..5.
    # C4: k in 0..1. C5: k in 2..3. C6: k in 4..5.
    # Note: C = A1 B1 + A2 B2.
    # C1 = A1 B1_1 contributes to C[:, 0:2]
    # C2 = A1 B1_2 contributes to C[:, 2:4]
    # C3 = A1 B1_3 contributes to C[:, 4:6]
    # C4 = A2 B2_1 contributes to C[:, 0:2]
    # C5 = A2 B2_2 contributes to C[:, 2:4]
    # C6 = A2 B2_3 contributes to C[:, 4:6]
    # So C[:, 0:2] = C1 + C4
    # C[:, 2:4] = C2 + C5
    # C[:, 4:6] = C3 + C6
    
    # We need to sum the contributions from A1 and A2 for the same C block.
    # The Strassen terms for A1 B1_1 and A2 B2_1 are independent in U and V,
    # but they add up in W.
    # So for each pair of terms (r1 from A1, r2 from A2), we can't just use one term.
    # We have 7 terms for A1 B1_1 and 7 terms for A2 B2_1.
    # Total 14 terms for C[:, 0:2].
    # Total 14 terms for C[:, 2:4].
    # Total 14 terms for C[:, 4:6].
    # Total 42 terms.
    
    # Term index 0-6: A1 * B1_1
    # Term index 7-13: A2 * B2_1
    # Term index 14-20: A1 * B1_2
    # Term index 21-27: A2 * B2_2
    # Term index 28-34: A1 * B1_3
    # Term index 35-41: A2 * B2_3
    
    for r in range(7):
        # A1 * B1_1
        # U: A1 indices (i, j) where j in 0..1.
        # U_str index 0: a (i=0, j=0) -> U[0, 0]
        # U_str index 1: b (i=0, j=1) -> U[1, 0]
        # U_str index 2: c (i=1, j=0) -> U[4, 0]
        # U_str index 3: d (i=1, j=1) -> U[5, 0]
        U[0, r] = U_str[0, r]
        U[1, r] = U_str[1, r]
        U[4, r] = U_str[2, r]
        U[5, r] = U_str[3, r]
        
        # V: B1_1 indices (j, k) where j in 0..1, k in 0..1.
        # V_str index 0: e (j=0, k=0) -> V[0, r]
        # V_str index 1: f (j=0, k=1) -> V[1, r]
        # V_str index 2: g (j=1, k=0) -> V[2, r]
        # V_str index 3: h (j=1, k=1) -> V[3, r]
        V[0, r] = V_str[0, r]
        V[1, r] = V_str[1, r]
        V[2, r] = V_str[2, r]
        V[3, r] = V_str[3, r]
        
        # W: C1 indices (i, k) where k in 0..1.
        # W_str index 0: y (i=0, k=0) -> W[0, r]
        # W_str index 1: z (i=0, k=1) -> W[1, r]
        # W_str index 2: w (i=1, k=0) -> W[2, r]
        # W_str index 3: x (i=1, k=1) -> W[3, r]
        W[0, r] = W_str[0, r]
        W[1, r] = W_str[1, r]
        W[2, r] = W_str[2, r]
        W[3, r] = W_str[3, r]
        
        # A2 * B2_1
        # U: A2 indices (i, j) where j in 2..3.
        # U_str index 0: a (i=0, j=2) -> U[2, r+7]
        # U_str index 1: b (i=0, j=3) -> U[3, r+7]
        # U_str index 2: c (i=1, j=2) -> U[6, r+7]
        # U_str index 3: d (i=1, j=3) -> U[7, r+7]
        U[2, r+7] = U_str[0, r]
        U[3, r+7] = U_str[1, r]
        U[6, r+7] = U_str[2, r]
        U[7, r+7] = U_str[3, r]
        
        # V: B2_1 indices (j, k) where j in 2..3, k in 0..1.
        # V_str index 0: e (j=2, k=0) -> V[4, r+7]
        # V_str index 1: f (j=2, k=1) -> V[5, r+7]
        # V_str index 2: g (j=3, k=0) -> V[6, r+7]
        # V_str index 3: h (j=3, k=1) -> V[7, r+7]
        V[4, r+7] = V_str[0, r]
        V[5, r+7] = V_str[1, r]
        V[6, r+7] = V_str[2, r]
        V[7, r+7] = V_str[3, r]
        
        # W: C4 indices (i, k) where k in 0..1. Adds to C[:, 0:2].
        # W_str index 0: y (i=0, k=0) -> W[0, r+7]
        # W_str index 1: z (i=0, k=1) -> W[1, r+7]
        # W_str index 2: w (i=1, k=0) -> W[2, r+7]
        # W_str index 3: x (i=1, k=1) -> W[3, r+7]
        W[0, r+7] = W_str[0, r]
        W[1, r+7] = W_str[1, r]
        W[2, r+7] = W_str[2, r]
        W[3, r+7] = W_str[3, r]
        
        # A1 * B1_2
        # U: A1 indices (i, j) where j in 0..1.
        U[0, r+14] = U_str[0, r]
        U[1, r+14] = U_str[1, r]
        U[4, r+14] = U_str[2, r]
        U[5, r+14] = U_str[3, r]
        
        # V: B1_2 indices (j, k) where j in 0..1, k in 2..3.
        # V_str index 0: e (j=0, k=2) -> V[2, r+14]
        # V_str index 1: f (j=0, k=3) -> V[3, r+14]
        # V_str index 2: g (j=1, k=2) -> V[4, r+14]
        # V_str index 3: h (j=1, k=3) -> V[5, r+14]
        V[2, r+14] = V_str[0, r]
        V[3, r+14] = V_str[1, r]
        V[4, r+14] = V_str[2, r]
        V[5, r+14] = V_str[3, r]
        
        # W: C2 indices (i, k) where k in 2..3.
        # W_str index 0: y (i=0, k=2) -> W[2, r+14]
        # W_str index 1: z (i=0, k=3) -> W[3, r+14]
        # W_str index 2: w (i=1, k=2) -> W[4, r+14]
        # W_str index 3: x (i=1, k=3) -> W[5, r+14]
        W[2, r+14] = W_str[0, r]
        W[3, r+14] = W_str[1, r]
        W[4, r+14] = W_str[2, r]
        W[5, r+14] = W_str[3, r]
        
        # A2 * B2_2
        # U: A2 indices (i, j) where j in 2..3.
        U[2, r+21] = U_str[0, r]
        U[3, r+21] = U_str[1, r]
        U[6, r+21] = U_str[2, r]
        U[7, r+21] = U_str[3, r]
        
        # V: B2_2 indices (j, k) where j in 2..3, k in 2..3.
        # V_str index 0: e (j=2, k=2) -> V[6, r+21]
        # V_str index 1: f (j=2, k=3) -> V[7, r+21]
        # V_str index 2: g (j=3, k=2) -> V[8, r+21]
        # V_str index 3: h (j=3, k=3) -> V[9, r+21]
        V[6, r+21] = V_str[0, r]
        V[7, r+21] = V_str[1, r]
        V[8, r+21] = V_str[2, r]
        V[9, r+21] = V_str[3, r]
        
        # W: C5 indices (i, k) where k in 2..3. Adds to C[:, 2:4].
        # W_str index 0: y (i=0, k=2) -> W[2, r+21]
        # W_str index 1: z (i=0, k=3) -> W[3, r+21]
        # W_str index 2: w (i=1, k=2) -> W[4, r+21]
        # W_str index 3: x (i=1, k=3) -> W[5, r+21]
        W[2, r+21] = W_str[0, r]
        W[3, r+21] = W_str[1, r]
        W[4, r+21] = W_str[2, r]
        W[5, r+21] = W_str[3, r]
        
        # A1 * B1_3
        # U: A1 indices (i, j) where j in 0..1.
        U[0, r+28] = U_str[0, r]
        U[1, r+28] = U_str[1, r]
        U[4, r+28] = U_str[2, r]
        U[5, r+28] = U_str[3, r]
        
        # V: B1_3 indices (j, k) where j in 0..1, k in 4..5.
        # V_str index 0: e (j=0, k=4) -> V[8, r+28]
        # V_str index 1: f (j=0, k=5) -> V[9, r+28]
        # V_str index 2: g (j=1, k=4) -> V[10, r+28]
        # V_str index 3: h (j=1, k=5) -> V[11, r+28]
        V[8, r+28] = V_str[0, r]
        V[9, r+28] = V_str[1, r]
        V[10, r+28] = V_str[2, r]
        V[11, r+28] = V_str[3, r]
        
        # W: C3 indices (i, k) where k in 4..5.
        # W_str index 0: y (i=0, k=4) -> W[4, r+28]
        # W_str index 1: z (i=0, k=5) -> W[5, r+28]
        # W_str index 2: w (i=1, k=4) -> W[6, r+28]
        # W_str index 3: x (i=1, k=5) -> W[7, r+28]
        W[4, r+28] = W_str[0, r]
        W[5, r+28] = W_str[1, r]
        W[6, r+28] = W_str[2, r]
        W[7, r+28] = W_str[3, r]
        
        # A2 * B2_3
        # U: A2 indices (i, j) where j in 2..3.
        U[2, r+35] = U_str[0, r]
        U[3, r+35] = U_str[1, r]
        U[6, r+35] = U_str[2, r]
        U[7, r+35] = U_str[3, r]
        
        # V: B2_3 indices (j, k) where j in 2..3, k in 4..5.
        # V_str index 0: e (j=2, k=4) -> V[12, r+35]
        # V_str index 1: f (j=2, k=5) -> V[13, r+35]
        # V_str index 2: g (j=3, k=4) -> V[14, r+35]
        # V_str index 3: h (j=3, k=5) -> V[15, r+35]
        V[12, r+35] = V_str[0, r]
        V[13, r+35] = V_str[1, r]
        V[14, r+35] = V_str[2, r]
        V[15, r+35] = V_str[3, r]
        
        # W: C6 indices (i, k) where k in 4..5. Adds to C[:, 4:6].
        # W_str index 0: y (i=0, k=4) -> W[4, r+35]
        # W_str index 1: z (i=0, k=5) -> W[5, r+35]
        # W_str index 2: w (i=1, k=4) -> W[6, r+35]
        # W_str index 3: x (i=1, k=5) -> W[7, r+35]
        W[4, r+35] = W_str[0, r]
        W[5, r+35] = W_str[1, r]
        W[6, r+35] = W_str[2, r]
        W[7, r+35] = W_str[3, r]
        
    # Verify
    T = _build_tensor(n, m, p)
    if _check_decomposition(U, V, W, T):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        rank0 = n * m * p
        U_fb = np.zeros((n * m, rank0), dtype=np.float64)
        V_fb = np.zeros((m * p, rank0), dtype=np.float64)
        W_fb = np.zeros((p * n, rank0), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1.0
                    V_fb[j * p + k, col] = 1.0
                    W_fb[k * n + i, col] = 1.0
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

# EVOLVE-BLOCK-END
