"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 8]
START_TIME = time.time()
TIME_LIMIT = 145.0  # seconds, leave margin before 150s kill

# EVOLVE-BLOCK-START
def _check_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W form a valid decomposition of the matrix multiplication tensor."""
    rank = len(U[0])
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # For each rank factor c, the product U[:,c] * V[:,c]^T gives a rank-1 slice
    # We need sum_c U[:,c] * V[:,c]^T to reconstruct the multiplication map
    # The multiplication tensor T[i,j,k] = 1 if the multiplication uses A[i,j]*B[j,k]
    # In the decomposition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should give
    # the coefficient of A[i,j]*B[j,k] in C[k,i]... 
    
    # Actually, the standard decomposition is:
    # C[k,i] = sum_{j} A[i,j] * B[j,k]
    # = sum_{j} sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] * A[i,j] * B[j,k]... no
    
    # The tensor decomposition means:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta (appropriate Kronecker deltas)
    # Specifically, the multiplication tensor has T_{i,j,k} = 1 for each (i,j,k) triple
    # and the decomposition is: T_{i,j,k} = sum_c U_{i*m+j, c} * V_{j*p+k, c} * W_{k*n+i, c}
    
    # Build the full multiplication tensor as a 3D array
    # T[i,j,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # Efficient check using matrix operations:
    # For each c, form outer product of U[:,c] and V[:,c], then check against W[:,c]
    # Better: check that for all (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # Vectorized check
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = np.array(U).reshape(n, m, rank)
    V3 = np.array(V).reshape(m, p, rank)
    W3 = np.array(W).reshape(p, n, rank)
    
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # We can compute this as: for each j, sum over c and k...
    # Actually let's just do it directly for small dimensions
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = np.dot(U3[i, j, :], V3[j, k, :]) * 0  # placeholder
                # No, it's sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
                val = 0.0
                for c in range(rank):
                    val += U3[i, j, c] * V3[j, k, c] * W3[k, i, c]
                T[i, j, k] = val
    
    # The multiplication tensor should be all ones (each (i,j,k) contributes A[i,j]*B[j,k] to C[k,i])
    # Wait, actually C[k,i] = sum_j A[i,j] * B[j,k], so T[i,j,k] = 1 for all valid (i,j,k)
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True

def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
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
    return {"U": U, "V": V, "W": W, "rank": rank}

def _try_random_search(n, m, p, max_rank, num_trials=100, deadline=None):
    """Try to find a decomposition with rank <= max_rank using random initialization + verification."""
    best = None
    for trial in range(num_trials):
        if deadline is not None and time.time() > deadline:
            break
        rank = max_rank
        # Random initialization with small integers
        U = np.random.randint(-2, 3, size=(n * m, rank)).astype(float)
        V = np.random.randint(-2, 3, size=(m * p, rank)).astype(float)
        W = np.random.randint(-2, 3, size=(p * n, rank)).astype(float)
        
        # Quick check using tensor contraction
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U3[i, j, :] * V3[j, k, :] * W3[k, i, :])
                    if abs(val - 1.0) > 1e-6:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            # Convert to lists
            best = {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist(),
                "rank": rank
            }
            return best
    return best

def _greedy_construct(n, m, p, target_rank, deadline=None):
    """
    Greedy approach: build the decomposition factor by factor.
    Start with the full multiplication tensor and greedily extract rank-1 components.
    """
    # The target tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We want T = sum_{c=1}^{rank} U_c * V_c * W_c where U_c is a vector of size n*m,
    # V_c is a vector of size m*p, W_c is a vector of size p*n
    
    # Initialize residual tensor
    T = np.ones((n, m, p))
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    for c in range(target_rank):
        if deadline is not None and time.time() > deadline:
            break
        
        # Find best rank-1 approximation of residual T
        # T[i,j,k] approx U[i*m+j] * V[j*p+k] * W[k*n+i]
        # This is a 3D tensor, find best rank-1 via ALS or power method
        
        # Use power iteration on the tensor
        # Initialize randomly
        Uv = np.random.randn(n * m)
        Vv = np.random.randn(m * p)
        Wv = np.random.randn(p * n)
        
        # Power iteration for tensor
        for _ in range(50):
            # Update Uv: Uv[i*m+j] = sum_{k} T[i,j,k] * Vv[j*p+k] * Wv[k*n+i]
            Uv_new = np.zeros(n * m)
            for i in range(n):
                for j in range(m):
                    s = 0.0
                    for k in range(p):
                        s += T[i, j, k] * Vv[j * p + k] * Wv[k * n + i]
                    Uv_new[i * m + j] = s
            Uv = Uv_new / (np.linalg.norm(Uv_new) + 1e-15)
            
            # Update Vv: Vv[j*p+k] = sum_{i} T[i,j,k] * Uv[i*m+j] * Wv[k*n+i]
            Vv_new = np.zeros(m * p)
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for i in range(n):
                        s += T[i, j, k] * Uv[i * m + j] * Wv[k * n + i]
                    Vv_new[j * p + k] = s
            Vv = Vv_new / (np.linalg.norm(Vv_new) + 1e-15)
            
            # Update Wv: Wv[k*n+i] = sum_{j} T[i,j,k] * Uv[i*m+j] * Vv[j*p+k]
            Wv_new = np.zeros(p * n)
            for k in range(p):
                for i in range(n):
                    s = 0.0
                    for j in range(m):
                        s += T[i, j, k] * Uv[i * m + j] * Vv[j * p + k]
                    Wv_new[k * n + i] = s
            Wv = Wv_new / (np.linalg.norm(Wv_new) + 1e-15)
        
        # Check if this rank-1 component is useful
        # Compute the residual after subtracting this component
        # T[i,j,k] -= Uv[i*m+j] * Vv[j*p+k] * Wv[k*n+i]
        
        # Only add if it significantly reduces the residual
        residual_before = np.sum(T * T)
        T_new = T.copy()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_new[i, j, k] -= Uv[i * m + j] * Vv[j * p + k] * Wv[k * n + i]
        residual_after = np.sum(T_new * T_new)
        
        if residual_after < residual_before * 0.99:  # Significant reduction
            U_cols.append(Uv)
            V_cols.append(Vv)
            W_cols.append(Wv)
            T = T_new
            
            # Check if we've captured the full tensor
            if np.max(np.abs(T)) < 1e-6:
                break
    
    if len(U_cols) > 0 and len(U_cols) <= target_rank:
        rank = len(U_cols)
        U = np.column_stack(U_cols)
        V = np.column_stack(V_cols)
        W = np.column_stack(W_cols)
        
        # Verify
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U3[i, j, :] * V3[j, k, :] * W3[k, i, :])
                    if abs(val - 1.0) > 1e-4:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        
        if ok:
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist(),
                "rank": rank
            }
    
    return None

def _structured_4x4x4():
    """
    Attempt to construct a known lower-rank decomposition for 4x4x4.
    The best known upper bound for 4x4x4 matrix multiplication rank is 49 (Strassen-like).
    For 4x4x8, since the output splits into two independent 4x4x4 blocks,
    rank should be 2 * rank(4x4x4) = 98.
    
    However, we may be able to do better by sharing factors across the two blocks.
    
    Let's try a structured approach based on Strassen's 7-rank 2x2x2 decomposition
    extended to 4x4x4.
    """
    # Strassen 2x2x2 has rank 7
    # For 4x4x4, we can think of it as four 2x2x2 subproblems
    # but with sharing, we might get rank < 4*7 = 28
    
    # Actually, the best known for 4x4x4 is 49 (from the literature)
    # Let's just use the schoolbook for now and try to improve
    
    n = m = p = 4
    rank = n * m * p  # 64 for schoolbook
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
    return U, V, W, rank

def construct():
    n, m, p = DIMENSIONS
    deadline = START_TIME + TIME_LIMIT
    
    # Start with schoolbook as fallback
    school = _schoolbook(n, m, p)
    best = school
    
    # Try greedy construction with various target ranks
    # The theoretical lower bound is ceil(log2(n*m*p)) / log2(7) * 7 or similar
    # For 4x4x8 = 128, schoolbook rank is 128
    # Lower bound is at least n*m*p / (some factor)
    
    # Try greedy with different target ranks
    target_ranks = [64, 72, 80, 96, 98, 100, 112, 120, 128]
    
    for tr in target_ranks:
        if time.time() > deadline - 5:
            break
        result = _greedy_construct(n, m, p, tr, deadline)
        if result is not None and result["rank"] < best["rank"]:
            best = result
    
    # Try random search with lower ranks
    for tr in [64, 72, 80, 96, 98]:
        if time.time() > deadline - 5:
            break
        result = _try_random_search(n, m, p, tr, num_trials=50, deadline=deadline)
        if result is not None and result["rank"] < best["rank"]:
            best = result
    
    # Return in the expected format
    return {"U": best["U"], "V": best["V"], "W": best["W"]}
# EVOLVE-BLOCK-END
