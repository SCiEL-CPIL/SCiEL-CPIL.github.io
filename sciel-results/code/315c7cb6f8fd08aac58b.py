"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    T = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    # Check diagonal entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if abs(T[a, b, c] - 1.0) > 1e-9:
                    return False
    # Check off-diagonal entries are zero
    T_check = T.copy()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_check[a, b, c] = 0
    if np.max(np.abs(T_check)) > 1e-9:
        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Strategy: Fix V to identity-like pattern to decouple middle dimension
    # If V[j*p+k, r] = 1 when j == r (for r < m), else 0
    # Then we need U[i*m+j, r] * W[k*n+i, r] to satisfy:
    # For each (i,j,k): sum_r U[i*m+j, r] * delta(j,r) * W[k*n+i, r] = 1
    # => U[i*m+j, j] * W[k*n+i, j] = 1 for all i,j,k
    # This forces U[i*m+j, j] * W[k*n+i, j] = 1, which means
    # U[i*m+j, j] = 1/W[k*n+i, j] for all i,k, which is only possible
    # if W[k*n+i, j] is constant in i and k for each j.
    # So W[k*n+i, j] = c_j, U[i*m+j, j] = 1/c_j
    # This gives rank m = 3, but we need to check if this actually works.
    # Let's verify: T[i*m+j, j'*p+k, k*n+i] = sum_r U[i*m+j,r]*V[j'*p+k,r]*W[k*n+i,r]
    # V[j'*p+k, r] = 1 if j'==r else 0
    # So T[i*m+j, j'*p+k, k*n+i] = U[i*m+j, j'] * W[k*n+i, j'] if j' is in [0,m)
    # We need this to be 1 when j'=j and 0 otherwise.
    # When j'=j: U[i*m+j, j] * W[k*n+i, j] = (1/c_j) * c_j = 1 ✓
    # When j'!=j: U[i*m+j, j'] * W[k*n+i, j'] 
    # But U[i*m+j, j'] for j'!=j is 0 in our construction!
    # So we need U[i*m+j, r] = 0 for r != j.
    # Then V[j'*p+k, r] = delta(j', r), so:
    # T[i*m+j, j'*p+k, k*n+i] = U[i*m+j, j'] * 1 * W[k*n+i, j']
    # = delta(j,j') * (1/c_j) * c_j = delta(j,j')
    # This works! Rank = m = 3.
    
    # But wait - this only works if we can set U[i*m+j, r] = 0 for r != j
    # and U[i*m+j, j] = 1/c_j, W[k*n+i, j] = c_j.
    # Let's try c_j = 1 for all j:
    U_try = [[0] * m for _ in range(n * m)]
    V_try = [[0] * m for _ in range(m * p)]
    W_try = [[0] * m for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            U_try[i * m + j][j] = 1
    for j in range(m):
        for k in range(p):
            V_try[j * p + k][j] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                W_try[k * n + i][j] = 1
    
    if _verify(U_try, V_try, W_try, n, m, p):
        return {"U": U_try, "V": V_try, "W": W_try}
    
    # The simple identity trick doesn't work because W needs to be same for all k
    # Let's try a more general approach: use the structure more carefully.
    # 
    # General approach: We want to find rank-r decomposition.
    # Fix V to have a specific structure that makes the problem separable.
    # 
    # Alternative: Use the fact that for fixed j, the sub-tensor is (2,1,7)
    # which has rank 14 (schoolbook) or potentially less.
    # 
    # Let's try a randomized optimization approach.
    # We'll parameterize U, V, W and minimize the error.
    
    def _optimize_rank(target_rank, max_iters=200, lr=0.01, seed=42):
        """Try to find a rank-target_rank decomposition using gradient descent."""
        rng = np.random.RandomState(seed)
        
        # Initialize randomly
        U_init = rng.randn(n * m, target_rank) * 0.1
        V_init = rng.randn(m * p, target_rank) * 0.1
        W_init = rng.randn(p * n, target_rank) * 0.1
        
        # Build target tensor (sparse representation)
        # Target: T[i*m+j, j*p+k, k*n+i] = 1
        target_indices = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target_indices.append((i * m + j, j * p + k, k * n + i))
        
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Compute current tensor
            T = np.einsum('ar,br,cr->abc', U, V, W)
            
            # Compute error
            err = 0.0
            for idx in target_indices:
                err += (T[idx] - 1.0) ** 2
            
            # Also penalize off-diagonal entries (sample a few)
            # This is expensive, so we'll do it less frequently
            if it % 10 == 0:
                # Sample some off-diagonal positions
                for _ in range(100):
                    a = rng.randint(n * m)
                    b = rng.randint(m * p)
                    c = rng.randint(p * n)
                    # Check if this is a diagonal position
                    is_diag = False
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                if a == i*m+j and b == j*p+k and c == k*n+i:
                                    is_diag = True
                    if not is_diag:
                        err += T[a, b, c] ** 2
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_err < 1e-12:
                break
            
            # Compute gradients
            # dT/dU = V @ W^T (for each a)
            # grad_U[a, r] = sum_{b,c} T[a,b,c] * V[b,r] * W[c,r] - sum_{(i,j,k): a=i*m+j} V[b,r]*W[c,r]
            
            # Efficient gradient computation
            # T = einsum('ar,br,cr->abc', U, V, W)
            # dL/dU[a,r] = sum_{b,c} (T[a,b,c] - target[a,b,c]) * V[b,r] * W[c,r]
            
            # Build target tensor
            T_target = np.zeros((n*m, m*p, p*n))
            for idx in target_indices:
                T_target[idx] = 1.0
            
            diff = T - T_target
            
            # grad_U[a,r] = sum_{b,c} diff[a,b,c] * V[b,r] * W[c,r]
            # = einsum('abc,br,cr->ar', diff, V, W)
            grad_U = np.einsum('abc,br,cr->ar', diff, V, W)
            grad_V = np.einsum('abc,ar,cr->br', diff, U, W)
            grad_W = np.einsum('abc,ar,br->cr', diff, U, V)
            
            # Update with adaptive learning rate
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Normalize to prevent blowup
            for arr in [U, V, W]:
                norm = np.linalg.norm(arr)
                if norm > 1e6:
                    arr /= norm
        
        return best_U, best_V, best_W, best_err
    
    # Try various ranks
    for target_rank in [14, 15, 16, 17, 18, 20, 24, 28, 30, 32, 35, 38, 40]:
        if time.time() > deadline - 10:
            break
        
        U_opt, V_opt, W_opt, err = _optimize_rank(target_rank, max_iters=300, seed=target_rank)
        
        if err < 1e-10:
            # Verify
            if _verify(U_opt.tolist(), V_opt.tolist(), W_opt.tolist(), n, m, p):
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_result = {
                        "U": U_opt.tolist(),
                        "V": V_opt.tolist(),
                        "W": W_opt.tolist()
                    }
                break
        
        # Also try with different seeds
        if time.time() > deadline - 10:
            break
        U_opt2, V_opt2, W_opt2, err2 = _optimize_rank(target_rank, max_iters=200, seed=target_rank+100)
        if err2 < 1e-10:
            if _verify(U_opt2.tolist(), V_opt2.tolist(), W_opt2.tolist(), n, m, p):
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_result = {
                        "U": U_opt2.tolist(),
                        "V": V_opt2.tolist(),
                        "W": W_opt2.tolist()
                    }
                break
    
    return best_result
# EVOLVE-BLOCK-END
