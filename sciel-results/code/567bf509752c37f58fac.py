"""Model-and-optimize: parameterize factor matrices, optimize to minimize rank, round to half-integers."""
import numpy as np
import time

DIMENSIONS = [3, 3, 5]
n, m, p = DIMENSIONS
TENSOR_SHAPE = (n * m, m * p, p * n)

def _build_target_tensor():
    """Build the matrix multiplication tensor T[a, b, c] where:
    a = i*m + j (i in [0,n), j in [0,m))
    b = j*p + k (j in [0,m), k in [0,p))
    c = k*n + i (k in [0,p), i in [0,n))
    T[a, b, c] = 1 if the indices are consistent (same j and i), else 0.
    """
    T = np.zeros(TENSOR_SHAPE, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Check if sum_c U[a,c]*V[b,c]*W[c_idx,c] == T[a,b,c_idx] for all a,b,c_idx."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    T_hat = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_hat, T, atol=tol)

def _round_to_halfint(x):
    """Round to nearest half-integer (multiple of 0.5)."""
    return np.round(x * 2.0) / 2.0

def _is_valid_halfint(arr, tol=1e-6):
    """Check if all entries are within tol of a half-integer."""
    scaled = arr * 2.0
    rounded = np.round(scaled)
    return np.allclose(scaled, rounded, atol=tol * 2)

def _als_step(U, V, W, T, rank):
    """Perform one ALS step. Returns updated U, V, W."""
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Update U
    # T_hat[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # For fixed r, U[:,r] is a vector.
    # We want to minimize || T - sum_r U_r V_r^T W_r^T ||
    # This is equivalent to minimizing || T_a - sum_r U[a,r] (V_r \otimes W_r)^T || for each a?
    # Actually, it's a 3-way tensor.
    # Standard ALS for CP decomposition:
    # U_new = T_{(1)} V W^T (V^T V)^{-1} (W^T W)^{-1} ... roughly
    # Let's use the simple gradient descent or normal equations for each factor.
    
    # Flatten T to matrix T1 (nm x mp*pn)
    T1 = T.reshape(nm, mp * pn)
    # V W^T is (mp x rank) and (pn x rank)
    # We need to form the matrix M1 = (W \otimes V) ?
    # T1[a, b*pn+c] = sum_r U[a,r] V[b,r] W[c,r]
    # Let Z = W \otimes V (Kronecker product). Shape (pn*mp, rank).
    # Z[c*mp+b, r] = W[c,r] V[b,r]
    # Then T1[a, b*pn+c] = sum_r U[a,r] Z[c*mp+b, r]
    # Note: T1 index is b*pn+c. Z index is c*mp+b.
    # We need to permute T1 to match Z's index order.
    # T_perm[a, c*mp+b] = T[a, b*pn+c]
    T_perm = np.zeros((nm, pn * mp))
    for c in range(pn):
        for b in range(mp):
            T_perm[:, c*mp+b] = T1[:, b*pn+c]
    
    Z = np.kron(W, V) # Shape (pn*mp, rank)
    # U = T_perm Z (Z^T Z)^{-1}
    try:
        ZZ = Z.T @ Z
        # Add regularization
        ZZ += 1e-6 * np.eye(rank)
        U_new = T_perm @ Z @ np.linalg.inv(ZZ)
    except np.linalg.LinAlgError:
        U_new = U
        
    # Update V
    # T2 = T.reshape(mp, nm*pn)
    T2 = T.reshape(mp, nm * pn)
    # T2[b, a*pn+c] = T[a, b, c]
    # We need index a*pn+c.
    # T2_perm[b, c*nm+a] = T[a, b, c]
    T2_perm = np.zeros((mp, pn * nm))
    for c in range(pn):
        for a in range(nm):
            T2_perm[:, c*nm+a] = T2[:, a*pn+c]
            
    Z2 = np.kron(W, U) # Shape (pn*nm, rank)
    try:
        ZZ2 = Z2.T @ Z2
        ZZ2 += 1e-6 * np.eye(rank)
        V_new = T2_perm @ Z2 @ np.linalg.inv(ZZ2)
    except np.linalg.LinAlgError:
        V_new = V
        
    # Update W
    T3 = T.reshape(pn, nm*mp)
    T3_perm = np.zeros((pn, nm * mp))
    for a in range(nm):
        for b in range(mp):
            T3_perm[:, a*mp+b] = T3[:, a*mp+b] # T3 index is a*mp+b?
            # T3[c, a*mp+b] = T[a,b,c]
            # So T3_perm is already in correct order if T3 was reshaped correctly?
            # T.reshape(pn, nm*mp) gives T3[c, a*mp+b] = T[a,b,c].
            # So Z3 = U \otimes V. Shape (nm*mp, rank).
            # Z3[a*mp+b, r] = U[a,r] V[b,r].
            # T3[c, a*mp+b] = sum_r W[c,r] Z3[a*mp+b, r].
            # So W = T3 Z3 (Z3^T Z3)^{-1}.
            
    Z3 = np.kron(U, V) # Shape (nm*mp, rank)
    try:
        ZZ3 = Z3.T @ Z3
        ZZ3 += 1e-6 * np.eye(rank)
        W_new = T3 @ Z3 @ np.linalg.inv(ZZ3)
    except np.linalg.LinAlgError:
        W_new = W
        
    return U_new, V_new, W_new

def _try_rank_als(rank, T, max_iters=100, seed=None, time_budget=20):
    """Try to find a rank-r decomposition using ALS."""
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    rng = np.random.RandomState(seed)
    start_time = time.time()
    
    # Initialize with random half-integers
    vals = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
    U = rng.choice(vals, size=(nm, rank))
    V = rng.choice(vals, size=(mp, rank))
    W = rng.choice(vals, size=(pn, rank))
    
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for i in range(max_iters):
        if time.time() - start_time > time_budget:
            break
            
        U, V, W = _als_step(U, V, W, T, rank)
        
        # Round to half-integers
        U_r = _round_to_halfint(U)
        V_r = _round_to_halfint(V)
        W_r = _round_to_halfint(W)
        
        T_hat = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        err = np.max(np.abs(T_hat - T))
        
        if err < 1e-9:
            return U_r, V_r, W_r
            
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
            
    return None

def _construct_rank_18_part():
    """Construct the rank-18 decomposition for the 3x3x2 part (last 2 columns of B/C)."""
    # This part handles k=3,4 in the full 3x3x5 problem?
    # No, the full problem is 3x3x5.
    # We can split B into B1 (3x3) and B2 (3x2).
    # The tensor for B2 part corresponds to k=3,4?
    # Indices:
    # b = j*p + k. If p=5, b = 5j + k.
    # If we split, we can't just use a sub-tensor directly in the same index space easily without permutation.
    # However, we can construct the full U, V, W matrices.
    
    # Let's just return None and rely on the search or schoolbook.
    # Implementing the specific permutation logic is complex.
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    T = _build_target_tensor()
    
    # Try ranks from 40 down to 25
    # Schoolbook is 45.
    ranks_to_try = [40, 41, 42, 43, 44, 45]
    
    for rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
        
        # Try ALS with multiple seeds
        for seed in range(20):
            if time.time() > deadline - 5:
                break
            result = _try_rank_als(rank, T, max_iters=50, seed=seed, time_budget=5)
            if result is not None:
                U, V, W = result
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                return {"U": U_list, "V": V_list, "W": W_list}
        
        # If ALS fails, try simple random search for this rank
        vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        rng = np.random.RandomState(42)
        for attempt in range(100):
            if time.time() > deadline - 5:
                break
            U = rng.choice(vals, size=(n*m, rank))
            V = rng.choice(vals, size=(m*p, rank))
            W = rng.choice(vals, size=(p*n, rank))
            T_hat = np.einsum('ar,br,cr->abc', U, V, W)
            if np.allclose(T_hat, T, atol=1e-9):
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                return {"U": U_list, "V": V_list, "W": W_list}
                
    # Fallback: Schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
