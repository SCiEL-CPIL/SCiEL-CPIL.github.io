"""Program for 3x3 by 3x6 matrix multiplication - rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T of shape (n*m, m*p, p*n).
    T[i*m+k, k*p+j, j*n+i] = 1 for all valid i,k,j."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for k in range(m):
            for j in range(p):
                T[i * m + k, k * p + j, j * n + i] = 1
    return T

def _verify(U, V, W, T, tol=1e-9):
    """Verify that sum_r U[:,r] * V[:,r] * W[:,r] == T."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    n, m, p = T.shape[0], 3, 6  # hardcoded for this problem
    reconstructed = np.zeros_like(T)
    for r in range(U.shape[1]):
        reconstructed += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return np.allclose(reconstructed, T, atol=tol)

def _greedy_decompose(T, max_rank=45, time_limit=100):
    """Greedy rank decomposition: iteratively pick the best rank-1 slice."""
    n, m, p = T.shape
    start_time = time.time()
    
    # Try to find a decomposition by iterative deflation
    # Start with schoolbook rank and try to reduce
    T_remaining = T.copy()
    U_cols = []
    V_cols = []
    W_cols = []
    
    # We'll use a randomized approach: try many random rank-1 approximations
    # and pick the one that best reduces the residual
    
    best_result = None
    best_rank = n * m * p  # schoolbook
    
    # First, get schoolbook as baseline
    U_sch = []
    V_sch = []
    W_sch = []
    for i in range(n):
        for k in range(m):
            for j in range(p):
                u = np.zeros(n * m)
                v = np.zeros(m * p)
                w = np.zeros(p * n)
                u[i * m + k] = 1
                v[k * p + j] = 1
                w[j * n + i] = 1
                U_sch.append(u)
                V_sch.append(v)
                W_sch.append(w)
    
    U_sch = np.array(U_sch).T
    V_sch = np.array(V_sch).T
    W_sch = np.array(W_sch).T
    
    # Verify schoolbook
    recon = np.zeros_like(T)
    for r in range(U_sch.shape[1]):
        recon += np.einsum('a,b,c->abc', U_sch[:, r], V_sch[:, r], W_sch[:, r])
    if not np.allclose(recon, T, atol=1e-9):
        return U_sch.tolist(), V_sch.tolist(), W_sch.tolist()
    
    # Now try to find a better decomposition using iterative methods
    # Use a simple approach: try random initializations and do gradient-like descent
    
    rng = np.random.RandomState(42)
    
    for target_rank in range(30, 54):
        if time.time() - start_time > time_limit:
            break
        
        # Try multiple random initializations
        for trial in range(20):
            if time.time() - start_time > time_limit:
                break
            
            U = rng.randn(n * m, target_rank) * 0.1
            V = rng.randn(m * p, target_rank) * 0.1
            W = np.zeros((p * n, target_rank))
            
            # Initialize W to help convergence
            for r in range(target_rank):
                # Compute best W[:,r] given U[:,r] and V[:,r]
                # T = sum U[:,r] * V[:,r] * W[:,r]
                # For fixed U,V, optimal W[:,r] satisfies:
                # W[:,r] = argmin ||T - sum_{r'} U[:,r']*V[:,r']*W[:,r']||
                # This is a linear system if we fix other W's
                pass
            
            # Alternative: use a simpler iterative scheme
            # Initialize all three randomly
            U = rng.randn(n * m, target_rank) * 0.5
            V = rng.randn(m * p, target_rank) * 0.5
            W = rng.randn(p * n, target_rank) * 0.5
            
            # Iterate: update each factor in turn
            converged = False
            for it in range(200):
                if time.time() - start_time > time_limit:
                    break
                
                # Update U: for each r, U[:,r] = argmin ||T - sum_{r'} U[:,r']*V[:,r']*W[:,r']||
                # This is: U[:,r] should be such that the residual is minimized
                # Residual = T - sum_{r' != r} U[:,r']*V[:,r']*W[:,r'] - U[:,r]*V[:,r]*W[:,r]
                # Optimal U[:,r] = (residual_without_r) : (V[:,r]*W[:,r]) / ||V[:,r]*W[:,r]||^2
                # where : denotes tensor contraction
                
                # Build the "outer product" tensor for each r
                # OV[r] = V[:,r] * W[:,r] (shape m*p, p*n)
                # Then U[:,r] = <T - sum_{r'!=r} U[:,r']*OV[r'], OV[r]> / <OV[r], OV[r]>
                
                # Compute all OV
                OV = np.zeros((target_rank, m * p, p * n))
                for r in range(target_rank):
                    OV[r] = np.outer(V[:, r], W[:, r])
                
                # Update U
                for r in range(target_rank):
                    # Residual without r
                    res = T.copy()
                    for r2 in range(target_rank):
                        if r2 == r:
                            continue
                        res -= np.einsum('a,bc->abc', U[:, r2], OV[r2])
                    # Project
                    numerator = np.einsum('abc,bc->a', res, OV[r])
                    denominator = np.sum(OV[r] ** 2)
                    if denominator > 1e-15:
                        U[:, r] = numerator / denominator
                
                # Update V
                for r in range(target_rank):
                    res = T.copy()
                    for r2 in range(target_rank):
                        if r2 == r:
                            continue
                        res -= np.einsum('a,bc->abc', U[:, r2], OV[r2])
                    # Need OV2[r] = U[:,r] * W[:,r] (shape n*m, p*n)
                    OV2_r = np.outer(U[:, r], W[:, r])
                    numerator = np.einsum('abc,ac->bc', res, OV2_r)
                    denominator = np.sum(OV2_r ** 2)
                    if denominator > 1e-15:
                        V[:, r] = numerator / denominator
                    # Update OV
                    OV[r] = np.outer(V[:, r], W[:, r])
                
                # Update W
                for r in range(target_rank):
                    res = T.copy()
                    for r2 in range(target_rank):
                        if r2 == r:
                            continue
                        res -= np.einsum('a,bc->abc', U[:, r2], OV[r2])
                    # Need OV3[r] = U[:,r] * V[:,r] (shape n*m, m*p)
                    OV3_r = np.outer(U[:, r], V[:, r])
                    numerator = np.einsum('abc,ab->bc', res, OV3_r)
                    denominator = np.sum(OV3_r ** 2)
                    if denominator > 1e-15:
                        W[:, r] = numerator / denominator
                    # Update OV
                    OV[r] = np.outer(V[:, r], W[:, r])
                
                # Check convergence
                recon = np.zeros_like(T)
                for r in range(target_rank):
                    recon += np.einsum('a,bc->abc', U[:, r], OV[r])
                if np.allclose(recon, T, atol=1e-8):
                    converged = True
                    break
            
            if converged:
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_result = (U.copy(), V.copy(), W.copy())
                break  # Found a valid decomposition for this rank, try lower
    
    if best_result is not None:
        U, V, W = best_result
        # Round to nearest half-integer if possible
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        # Verify rounded version
        recon = np.zeros_like(T)
        for r in range(U_rounded.shape[1]):
            recon += np.einsum('a,bc->abc', U_rounded[:, r], V_rounded[:, r], W_rounded[:, r])
        
        if np.allclose(recon, T, atol=1e-6):
            return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
        else:
            # Return unrounded
            return U.tolist(), V.tolist(), W.tolist()
    else:
        # Fall back to schoolbook
        return U_sch.tolist(), V_sch.tolist(), W_sch.tolist()

def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    start_time = time.time()
    
    # Try to find a better decomposition
    try:
        U, V, W = _greedy_decompose(T, max_rank=45, time_limit=110)
    except Exception:
        # Fall back to schoolbook
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
    
    # Safety check: verify the result
    if not _verify(U, V, W, T):
        # Fall back to schoolbook
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
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
