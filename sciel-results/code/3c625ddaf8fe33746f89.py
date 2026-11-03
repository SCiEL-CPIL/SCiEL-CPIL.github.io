import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def _build_target_tensor(n, m, p):
    """Build the target matrix multiplication tensor T[i,j,k] = 1 if i==k else 0."""
    T = np.zeros((n, m, p))
    for i in range(n):
        for k in range(p):
            for j in range(m):
                if i == k:
                    T[i, j, k] = 1.0
    return T

def _residual_norm(T, U, V, W, rank):
    """Compute ||T - sum_c U[:,c]⊗V[:,c]⊗W[:,c]||_F"""
    R = np.zeros_like(T)
    for c in range(rank):
        R += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
    return np.linalg.norm(T - R)

def _greedy_random_search(T, n, m, p, max_rank, time_budget, rng):
    """
    Greedy randomized search for low-rank tensor decomposition.
    T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
    """
    n, m, p = DIMENSIONS
    target = T.copy()
    
    best_rank = max_rank
    best_U = None
    best_V = None
    best_W = None
    
    # Start with schoolbook as fallback
    sb_rank = n * m * p
    sb_U = np.zeros((n*m, sb_rank))
    sb_V = np.zeros((m*p, sb_rank))
    sb_W = np.zeros((p*n, sb_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                sb_U[i*m+j, col] = 1.0
                sb_V[j*p+k, col] = 1.0
                sb_W[k*n+i, col] = 1.0
                col += 1
    best_rank = sb_rank
    best_U, best_V, best_W = sb_U, sb_V, sb_W
    
    # Try various target ranks from low to high
    target_ranks = list(range(80, max_rank + 1))
    
    for target_r in target_ranks:
        if time.time() > time_budget:
            break
        
        # Multiple random restarts
        found = False
        for restart in range(3):
            if time.time() > time_budget:
                break
            
            # Initialize with random sparse terms
            U = np.zeros((n*m, target_r))
            V = np.zeros((m*p, target_r))
            W = np.zeros((p*n, target_r))
            
            # Random initialization: each term picks random indices
            for c in range(target_r):
                # Randomly pick which (i,j) for U, (j,k) for V, (k,i) for W
                # Use structured random: bias toward diagonal structure
                U[i*m+j, c] = 1.0 if rng.random() < 0.3 else 0.0
                # Actually let's do proper random init
            U = np.zeros((n*m, target_r))
            V = np.zeros((m*p, target_r))
            W = np.zeros((p*n, target_r))
            
            for c in range(target_r):
                # Each factor entry is random in {-1, 0, 1} with some sparsity
                U[:, c] = rng.choice([-1, 0, 1], size=n*m, p=[0.2, 0.6, 0.2])
                V[:, c] = rng.choice([-1, 0, 1], size=m*p, p=[0.2, 0.6, 0.2])
                W[:, c] = rng.choice([-1, 0, 1], size=p*n, p=[0.2, 0.6, 0.2])
            
            # Gauss-Newton refinement
            for iteration in range(50):
                if time.time() > time_budget:
                    break
                
                # Compute residual
                R = np.zeros((n, m, p))
                for c in range(target_r):
                    R += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
                resid = T - R
                
                # Check convergence
                rnorm = np.linalg.norm(resid)
                if rnorm < 1e-10:
                    found = True
                    break
                
                # Gauss-Newton: for each term c, solve for optimal U[:,c], V[:,c], W[:,c]
                # Given V[:,c] and W[:,c], optimal U[:,c] satisfies:
                # U[i,c] * sum_jk V[j,c]*W[k,c]*resid[i,j,k] ... 
                # Actually the tensor equation is T[i,j,k] = sum_c U[i,c]*V[j,c]*W[k,c]
                # For fixed V[:,c] and W[:,c], U[i,c] should minimize:
                # sum_{j,k} (resid[i,j,k] - U[i,c]*V[j,c]*W[k,c])^2
                # Optimal U[i,c] = sum_{j,k} resid[i,j,k]*V[j,c]*W[k,c] / sum_{j,k} V[j,c]^2*W[k,c]^2
                
                denom_U = np.sum(V * V, axis=0) * np.sum(W * W, axis=0)  # shape (target_r,)
                # Actually: for term c, the "projection" of resid onto the i-slice
                # U[i,c] = sum_{j,k} resid[i,j,k] * V[j,c] * W[k,c] / (sum_j V[j,c]^2 * sum_k W[k,c]^2)
                
                # Compute numerator: sum_{j,k} resid[i,j,k]*V[j,c]*W[k,c] for each i,c
                # This is: for each c, (resid * V[:,:,c] * W[:,:,c]) summed over j,k for each i
                # = einsum('ijk,jc,kc->ic', resid, V, W)
                num_U = np.einsum('ijk,jc,kc->ic', resid, V, W)
                
                # Denominator for U: for each c, sum_j V[j,c]^2 * sum_k W[k,c]^2
                sumV2 = np.sum(V * V, axis=0)  # (target_r,)
                sumW2 = np.sum(W * W, axis=0)  # (target_r,)
                denom_U = sumV2 * sumW2  # (target_r,)
                
                # Similarly for V: V[j,c] = sum_{i,k} resid[i,j,k]*U[i,c]*W[k,c] / (sum_i U[i,c]^2 * sum_k W[k,c]^2)
                num_V = np.einsum('ijk,ic,kc->jc', resid, U, W)
                sumU2 = np.sum(U * U, axis=0)
                denom_V = sumU2 * sumW2
                
                # Similarly for W: W[k,c] = sum_{i,j} resid[i,j,k]*U[i,c]*V[j,c] / (sum_i U[i,c]^2 * sum_j V[j,c]^2)
                num_W = np.einsum('ijk,ic,jc->kc', resid, U, V)
                denom_W = sumU2 * sumV2
                
                # Update
                mask_U = denom_U > 1e-15
                mask_V = denom_V > 1e-15
                mask_W = denom_W > 1e-15
                
                U[:, mask_U] = num_U[:, mask_U] / denom_U[mask_U]
                V[:, mask_V] = num_V[:, mask_V] / denom_V[mask_V]
                W[:, mask_W] = num_W[:, mask_W] / denom_W[mask_W]
            
            if found:
                if target_r < best_rank:
                    best_rank = target_r
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                break  # Found for this target_r, no need more restarts
        
        # If we found a valid decomposition at this rank, we can stop
        # (since we're going from low to high)
        if found:
            break
    
    # Convert to list format
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return U_list, V_list, W_list, best_rank

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                expected = 1.0 if i == k else 0.0
                if abs(total - expected) > 1e-8:
                    return False
    return True

def _schoolbook():
    """Schoolbook decomposition of rank n*m*p = 144."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_based():
    """Fallback: Strassen-based rank-114 decomposition."""
    # 2x2x2 Strassen
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 1, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, -1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)
    
    # Kronecker product for 4x4x4
    U4 = np.kron(U2, U2)  # 16x49
    V4 = np.kron(V2, V2)  # 16x49
    W4 = np.kron(W2, W2)  # 16x49
    
    # Assemble 4x4x9: two 4x4x4 blocks + one 4x4x1 schoolbook
    total_rank = 49 + 49 + 16
    U_full = np.zeros((16, total_rank))
    V_full = np.zeros((36, total_rank))
    W_full = np.zeros((36, total_rank))
    
    # Block 1: cols 0-3
    U_full[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V_full[j*9+k, 0:49] = V4[j*4+k, :]
    for k in range(4):
        for i in range(4):
            W_full[k*4+i, 0:49] = W4[k*4+i, :]
    
    # Block 2: cols 4-7
    off = 49
    U_full[:, off:off+49] = U4
    for j in range(4):
        for k in range(4):
            V_full[j*9+(k+4), off:off+49] = V4[j*4+k, :]
    for k in range(4):
        for i in range(4):
            W_full[(k+4)*4+i, off:off+49] = W4[k*4+i, :]
    
    # Block 3: col 8 (schoolbook 4x4x1)
    off2 = 98
    for i in range(4):
        for j in range(4):
            for k in range(1):
                col = off2 + i*4 + j
                U_full[i*4+j, col] = 1.0
                V_full[j*9+8, col] = 1.0
                W_full[8*4+i, col] = 1.0
    
    return U_full.tolist(), V_full.tolist(), W_full.tolist()

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    n, m, p = DIMENSIONS
    T = _build_target_tensor(n, m, p)
    
    # First, get the Strassen-based rank-114 as a safe fallback
    U_str, V_str, W_str = _strassen_based()
    if _verify_decomposition(U_str, V_str, W_str, n, m, p):
        best_result = {"U": U_str, "V": V_str, "W": W_str}
        best_rank = 114
    else:
        best_result = _schoolbook()
        best_rank = 144
    
    # Try randomized search for lower rank
    rng = np.random.default_rng(42)
    
    # Try ranks from 100 up to 113
    for target_r in range(100, best_rank):
        if time.time() > deadline:
            break
        
        # Multiple random restarts for this rank
        for restart in range(5):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.zeros((n*m, target_r))
            V = np.zeros((m*p, target_r))
            W = np.zeros((p*n, target_r))
            
            for c in range(target_r):
                U[:, c] = rng.choice([-1, 0, 1], size=n*m, p=[0.25, 0.5, 0.25])
                V[:, c] = rng.choice([-1, 0, 1], size=m*p, p=[0.25, 0.5, 0.25])
                W[:, c] = rng.choice([-1, 0, 1], size=p*n, p=[0.25, 0.5, 0.25])
            
            # Gauss-Newton iterations
            converged = False
            for iteration in range(100):
                if time.time() > deadline:
                    break
                
                # Compute residual
                R = np.zeros((n, m, p))
                for c in range(target_r):
                    R += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
                resid = T - R
                rnorm = np.linalg.norm(resid)
                
                if rnorm < 1e-10:
                    converged = True
                    break
                
                # Gauss-Newton update
                sumU2 = np.sum(U * U, axis=0)
                sumV2 = np.sum(V * V, axis=0)
                sumW2 = np.sum(W * W, axis=0)
                
                num_U = np.einsum('ijk,jc,kc->ic', resid, V, W)
                num_V = np.einsum('ijk,ic,kc->jc', resid, U, W)
                num_W = np.einsum('ijk,ic,jc->kc', resid, U, V)
                
                denom_U = sumV2 * sumW2
                denom_V = sumU2 * sumW2
                denom_W = sumU2 * sumV2
                
                mask_U = denom_U > 1e-12
                mask_V = denom_V > 1e-12
                mask_W = denom_W > 1e-12
                
                U[:, mask_U] = num_U[:, mask_U] / denom_U[mask_U]
                V[:, mask_V] = num_V[:, mask_V] / denom_V[mask_V]
                W[:, mask_W] = num_W[:, mask_W] / denom_W[mask_W]
            
            if converged:
                # Verify with exact check
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = target_r
                    break  # Found at this rank, try lower
        
        # If we found a valid one at this rank, try lower
        if best_rank == target_r:
            continue
        else:
            # Didn't converge at this rank, try next higher
            continue
    
    return best_result
