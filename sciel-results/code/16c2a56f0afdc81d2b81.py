import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition is valid using vectorized computation."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Compute all tensor entries: T[u,v,w] = sum_c U[u,c]*V[v,c]*W[w,c]
    # We only need to check specific entries
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # Check the 1s: for each (i,j,k), T[i*m+j, j*p+k, k*n+i] should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U3[i,j,:] * V3[j,k,:] * W3[k,i,:])
                if abs(val - 1.0) > tol:
                    return False
    
    # Check the 0s: all other (u,v,w) combinations should be 0
    # Build a set of valid (u,v,w) indices
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid.add((i*m+j, j*p+k, k*n+i))
    
    # Vectorized check for 0s - compute full tensor and check
    T = np.einsum('uc,vc,wc->uvw', U, V, W)
    for u in range(n*m):
        for v in range(m*p):
            for w in range(p*n):
                if (u, v, w) not in valid:
                    if abs(T[u,v,w]) > tol:
                        return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _try_optimize(n, m, p, rank, seed, max_iter=2000, tol=1e-10):
    """Try to find a valid decomposition of given rank using optimization with both 1 and 0 constraints."""
    rng = np.random.RandomState(seed)
    
    # Initialize with random values
    U = rng.randn(n * m, rank) * 0.1
    V = rng.randn(m * p, rank) * 0.1
    W = rng.randn(p * n, rank) * 0.1
    
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Precompute which (i,j,k) triples are valid (multiplication entries)
    valid_ijk = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid_ijk.add((i, j, k))
    
    # Build target tensor: 1 for valid entries, 0 for others
    target = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i, j, k] = 1.0
    
    # Also need to track the full tensor for 0-constraints
    # The full tensor T[u,v,w] where u=i*m+j, v=j*p+k, w=k*n+i
    # For invalid (u,v,w), T should be 0
    
    for it in range(max_iter):
        # Compute the multiplication result (only valid entries matter for 1-constraint)
        result_ijk = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        
        # Compute error for 1-constraints
        err_ones = np.sum((result_ijk - target) ** 2)
        
        # Compute error for 0-constraints: need to check all (u,v,w) not in support
        # T[u,v,w] = sum_c U[u,c]*V[v,c]*W[w,c]
        # This is expensive, so we approximate by checking a subset or using the full tensor
        T_full = np.einsum('uc,vc,wc->uvw', U, V, W)
        
        # Build mask for valid (u,v,w)
        valid_mask = np.zeros((n*m, m*p, p*n), dtype=bool)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i*m + j
                    v = j*p + k
                    w = k*n + i
                    valid_mask[u, v, w] = True
        
        # Error for 0-constraints
        zero_mask = ~valid_mask
        err_zeros = np.sum(T_full[zero_mask] ** 2)
        
        total_err = err_ones + err_zeros
        
        if total_err < best_err:
            best_err = total_err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if total_err < tol:
            break
        
        # Compute gradients for both constraints
        # For 1-constraints:
        diff_ijk = result_ijk - target  # (n, m, p)
        
        dU_ones = np.einsum('ijk,jkc,kic->ijc', diff_ijk, V3, W3)
        dV_ones = np.einsum('ijk,ijc,kic->jkc', diff_ijk, U3, W3)
        dW_ones = np.einsum('ijk,ijc,jkc->kic', diff_ijk, U3, V3)
        
        # For 0-constraints: gradient of sum T[u,v,w]^2 for invalid (u,v,w)
        # dT/du = V[v,c]*W[w,c], etc.
        # dU[u,c] = sum_{v,w: invalid} 2*T[u,v,w]*V[v,c]*W[w,c]
        # This is expensive. Use a simplified approach:
        # Only consider a random subset of invalid entries for gradient
        
        # Simplified: use the fact that the 0-constraint gradient can be computed as
        # dU[u,c] = sum_{v,w not valid} 2*T[u,v,w]*V[v,c]*W[w,c]
        # But this is O(n*m * m*p * p*n * rank) which is too slow
        
        # Alternative: use a penalty approach with only the 1-constraints,
        # and rely on the structure to naturally satisfy 0-constraints
        
        # For now, just use 1-constraint gradients and hope the 0-constraints are satisfied
        dU = dU_ones
        dV = dV_ones
        dW = dW_ones
        
        # Gradient descent with adaptive step size
        lr = 0.005 / (1 + it * 0.002)
        U -= lr * dU.reshape(n*m, rank)
        V -= lr * dV.reshape(m*p, rank)
        W -= lr * dW.reshape(p*n, rank)
        
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
    
    if best_err < 1e-6:
        return best_U, best_V, best_W, best_err
    return None

def _try_structured(n, m, p, rank, seed):
    """Try a structured initialization based on block patterns."""
    rng = np.random.RandomState(seed)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Use a mix of random and structured initialization
    for c in range(rank):
        if c % 4 == 0:
            U[:, c] = rng.randn(n * m) * 0.3
            V[:, c] = rng.randn(m * p) * 0.3
            W[:, c] = rng.randn(p * n) * 0.3
        elif c % 4 == 1:
            # Sign-structured
            U[:, c] = rng.choice([-1, 1], size=n*m) * 0.3
            V[:, c] = rng.choice([-1, 1], size=m*p) * 0.3
            W[:, c] = rng.choice([-1, 1], size=p*n) * 0.3
        elif c % 4 == 2:
            # Sparse: only a few non-zero entries per column
            nz_u = rng.choice(n*m, size=min(3, n*m), replace=False)
            nz_v = rng.choice(m*p, size=min(3, m*p), replace=False)
            nz_w = rng.choice(p*n, size=min(3, p*n), replace=False)
            U[nz_u, c] = rng.randn(len(nz_u))
            V[nz_v, c] = rng.randn(len(nz_v))
            W[nz_w, c] = rng.randn(len(nz_w))
        else:
            # Small random
            U[:, c] = rng.randn(n * m) * 0.1
            V[:, c] = rng.randn(m * p) * 0.1
            W[:, c] = rng.randn(p * n) * 0.1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U, "V": V, "W": W}
    
    # Lower bound: max(n*p, m*p, n*m) = max(18, 24, 12) = 24
    min_rank = max(n*p, m*p, n*m)
    
    # Try ranks from min_rank upward
    max_rank_to_try = min(50, n * m * p)
    
    for rank in range(min_rank, max_rank_to_try + 1):
        if time.time() > deadline - 20:
            break
        
        # Try multiple seeds for each rank
        max_seeds = 4 if rank < 30 else 2
        for seed in range(max_seeds):
            if time.time() > deadline - 20:
                break
            
            # Method 1: Pure optimization
            result = _try_optimize(n, m, p, rank, seed, max_iter=1200)
            if result is not None:
                U_opt, V_opt, W_opt, err = result
                # Round to nearest 0.5 if close
                U_round = np.round(U_opt * 2) / 2
                V_round = np.round(V_opt * 2) / 2
                W_round = np.round(W_opt * 2) / 2
                
                U_list = U_round.tolist()
                V_list = V_round.tolist()
                W_list = W_round.tolist()
                
                if _verify(U_list, V_list, W_list, n, m, p):
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = rank
                    return best_result
            
            # Method 2: Structured initialization + optimization
            if time.time() > deadline - 20:
                break
            U_struct, V_struct, W_struct = _try_structured(n, m, p, rank, seed)
            
            # Optimize from structured init with both constraints
            U3 = U_struct.reshape(n, m, rank)
            V3 = V_struct.reshape(m, p, rank)
            W3 = W_struct.reshape(p, n, rank)
            
            target = np.ones((n, m, p))
            best_err = np.inf
            for it in range(800):
                result_ijk = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
                err = np.sum((result_ijk - target) ** 2)
                
                if err < best_err:
                    best_err = err
                
                if err < 1e-10:
                    break
                
                diff = result_ijk - target
                dU = np.einsum('ijk,jkc,kic->ijc', diff, V3, W3)
                dV = np.einsum('ijk,ijc,kic->jkc', diff, U3, W3)
                dW = np.einsum('ijk,ijc,jkc->kic', diff, U3, V3)
                
                lr = 0.003 / (1 + it * 0.001)
                U_struct -= lr * dU.reshape(n*m, rank)
                V_struct -= lr * dV.reshape(m*p, rank)
                W_struct -= lr * dW.reshape(p*n, rank)
                
                U3 = U_struct.reshape(n, m, rank)
                V3 = V_struct.reshape(m, p, rank)
                W3 = W_struct.reshape(p, n, rank)
            
            if best_err < 1e-6:
                U_round = np.round(U_struct * 2) / 2
                V_round = np.round(V_struct * 2) / 2
                W_round = np.round(W_struct * 2) / 2
                
                U_list = U_round.tolist()
                V_list = V_round.tolist()
                W_list = W_round.tolist()
                
                if _verify(U_list, V_list, W_list, n, m, p):
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = rank
                    return best_result
    
    return best_result
