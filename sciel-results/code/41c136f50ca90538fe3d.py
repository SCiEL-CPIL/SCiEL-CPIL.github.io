"""Matrix multiplication 2x3x10 tensor decomposition search."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 10]
n, m, p = DIMENSIONS
N = n * m  # 6
M = m * p  # 30
P = p * n  # 20

def _build_tensor():
    """Build the matrix multiplication tensor T[i, j, k] for C = A*B.
    T[i, j, k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k].
    We need: C[i,k] = sum_j A[i,j] * B[j,k]
    Tensor form: C[i,k] = sum_{j} A[i,j] B[j,k]
    The tensor T has indices (i, j, k) where T[i,j,k] = 1 for all i,j,k.
    Actually the tensor is: T[i, k, j] = 1 (the mode-2 index is the summation index)
    
    For rank-r decomposition:
    C[i,k] = sum_r U[i, r] * W[r, k] * V[j, r]  ... no
    C[i,k] = sum_j sum_r U[i,j,r]... 
    
    Let me re-read the contract:
    U: (n*m, rank) -> U[i*m+j, r]
    V: (m*p, rank) -> V[j*p+k, r]  
    W: (p*n, rank) -> W[k*n+i, r]
    
    The decomposition is: for each r, we have a rank-1 tensor
    U[:,r] * V[:,r] * W[:,r]
    
    The full tensor T[i*m+j, j'*p+k, k'*n+i'] ... hmm this is getting confusing.
    
    Let me think about it differently. The matrix multiplication tensor T is a 3-tensor
    with dimensions (n*m) x (m*p) x (p*n) ... no that's not right either.
    
    Actually, for matrix multiplication C = A * B where A is n x m, B is m x p:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T has three modes:
    - Mode 1: (i, j) pairs, size n*m = 6
    - Mode 2: (j, k) pairs, size m*p = 30
    - Mode 3: (k, i) pairs, size p*n = 20
    
    T[(i,j), (j',k), (k',i')] = 1 if j == j' and k == k' and i == i'
    
    Wait, that's the standard formulation. T[a, b, c] where:
    a = i*m + j
    b = j'*p + k  
    c = k'*n + i'
    
    T[a,b,c] = 1 if j == j' and k == k' and i == i'
    
    For a rank-r decomposition:
    T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    So we need: sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r] = 1 if j==j', k==k', i==i' else 0
    """
    T = np.zeros((N, M, P), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify(U, V, W):
    """Verify that U, V, W form a valid decomposition."""
    rank = len(U[0]) if isinstance(U, list) else U.shape[1]
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Reconstruct tensor
    T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    # Compare with true tensor
    T_true = _build_tensor()
    
    max_err = np.max(np.abs(T_recon - T_true))
    return max_err < 1e-8

def _to_rational_list(arr, tol=1e-10):
    """Convert float array to list of Fractions (integers or halves)."""
    result = []
    for row in arr:
        r = []
        for val in row:
            # Try to find a rational with small denominator
            frac = Fraction(val).limit_denominator(1000)
            # Check if it's close to a half-integer or integer
            if abs(frac - round(2*frac)) < tol:
                # It's a half-integer
                half_int = round(2*frac)
                if half_int % 2 == 0:
                    r.append(half_int // 2)
                else:
                    r.append(Fraction(half_int, 2))
            else:
                r.append(frac)
        result.append(r)
    return result

def _als_step(U, V, W, T, max_iter=1000, tol=1e-12):
    """ALS optimization for tensor decomposition."""
    rank = U.shape[1]
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    for it in range(max_iter):
        # Update U: for each column a, U[a,:] = argmin sum_{b,c} (T[a,b,c] - V[b,:]^T W[c,:])^2
        # This is a least squares problem
        # T_a = reshape T[a,:,:] to (M*P,)
        # The residual for U[a,r] is: T[a,b,c] - sum_r' V[b,r'] W[c,r']
        # We want to minimize sum_{b,c} (T[a,b,c] - sum_r' V[b,r'] W[c,r'])^2
        # = sum_{b,c} T[a,b,c]^2 - 2 sum_r U[a,r] sum_{b,c} T[a,b,c] V[b,r'] W[c,r'] + ...
        # The normal equations: U[a,:] = (V^T W)^{-1} * (T[a] * (V^T W))
        
        # Build the Gram matrix G = V^T * W (r x r) where G[r1,r2] = sum_{b,c} V[b,r1]*W[c,r2]
        # Wait, that's not quite right. Let me think again.
        
        # For fixed a, we want to find u = U[a,:] such that
        # min sum_{b,c} (T[a,b,c] - sum_r u[r] * V[b,r] * W[c,r])^2
        # 
        # Let's define for each r: M_r[b,c] = V[b,r] * W[c,r]
        # Then the model is sum_r u[r] * M_r[b,c]
        # 
        # This is a linear least squares problem:
        # min || T[a] - A u ||^2 where A is (M*P) x r with A[b,c,r] = V[b,r]*W[c,r]
        
        A = V[:, :, None] * W[None, :, :]  # (M, P, r)
        A_flat = A.reshape(M * P, rank)
        
        # Solve for each a
        T_a = T[a].reshape(M * P)
        
        # Normal equations: (A^T A) u = A^T T_a
        ATA = A_flat.T @ A_flat
        ATb = A_flat.T @ T_a
        
        try:
            u = np.linalg.solve(ATA, ATb)
        except np.linalg.LinAlgError:
            u = np.linalg.lstsq(ATA, ATb, rcond=None)[0]
        
        U[a] = u
        
        # Similarly update V
        B = U[:, :, None] * W[None, :, :]  # (N, P, r)
        B_flat = B.reshape(N * P, rank)
        
        for b in range(M):
            T_b = T[:, b, :].reshape(N * P)
            BTB = B_flat.T @ B_flat
            BTc = B_flat.T @ T_b
            try:
                v = np.linalg.solve(BTB, BTc)
            except np.linalg.LinAlgError:
                v = np.linalg.lstsq(BTB, BTc, rcond=None)[0]
            V[b] = v
        
        # Update W
        C = U[:, :, None] * V[None, :, :]  # (N, M, r)
        C_flat = C.reshape(N * M, rank)
        
        for c in range(P):
            T_c = T[:, :, c].reshape(N * M)
            CTC = C_flat.T @ C_flat
            CTe = C_flat.T @ T_c
            try:
                w = np.linalg.solve(CTC, CTe)
            except np.linalg.LinAlgError:
                w = np.linalg.lstsq(CTC, CTe, rcond=None)[0]
            W[c] = w
        
        # Check convergence
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.max(np.abs(T_recon - T))
        if err < tol:
            break
    
    return U, V, W, err

def _try_rank(rank, T, max_iters=500, n_restarts=3):
    """Try to find a valid decomposition of given rank."""
    best_err = float('inf')
    best_result = None
    
    for restart in range(n_restarts):
        # Random initialization
        np.random.seed(42 + restart * 1000)
        U = np.random.randn(N, rank)
        V = np.random.randn(M, rank)
        W = np.random.randn(P, rank)
        
        U, V, W, err = _als_step(U, V, W, T, max_iter=max_iters, tol=1e-12)
        
        if err < best_err:
            best_err = err
            best_result = (U.copy(), V.copy(), W.copy())
        
        if best_err < 1e-8:
            break
    
    if best_result is not None and best_err < 1e-8:
        U, V, W = best_result
        # Convert to rational form
        U_r = _to_rational_list(U)
        V_r = _to_rational_list(V)
        W_r = _to_rational_list(W)
        return U_r, V_r, W_r, True
    return None, None, None, False

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    T = _build_tensor()
    
    # Start from a known valid decomposition (schoolbook) and try to reduce rank
    # The schoolbook rank is n*m*p = 60
    # Known lower bounds suggest rank could be significantly lower
    
    # Try ranks from low to high
    # For 2x3x10, let's try a range of ranks
    # The trivial upper bound is 60
    
    best_rank = None
    best_decomp = None
    
    # First, let's try some reasonable ranks
    # Based on the structure, rank might be achievable around 20-40
    ranks_to_try = [20, 22, 24, 25, 26, 28, 30, 32, 35, 40, 45, 50, 55, 58, 60]
    
    for rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
        
        if time.time() > deadline:
            break
        
        # Try to find decomposition at this rank
        U_r, V_r, W_r, success = _try_rank(rank, T, max_iters=300, n_restarts=2)
        
        if success:
            best_rank = rank
            best_decomp = (U_r, V_r, W_r)
            break
    
    # If no rank worked, fall back to schoolbook
    if best_decomp is None:
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
    
    U_r, V_r, W_r = best_decomp
    return {"U": U_r, "V": V_r, "W": W_r}
