"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 6
    dim_v = m * p  # 27
    dim_w = p * n  # 18
    
    start_time = time.time()
    deadline = start_time + 140  # leave 10s margin
    
    # Schoolbook decomposition as fallback
    schoolbook_rank = n * m * p
    U_sb = np.zeros((dim_u, schoolbook_rank))
    V_sb = np.zeros((dim_v, schoolbook_rank))
    W_sb = np.zeros((dim_w, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = schoolbook_rank
    
    def full_verify(U, V, W, tol=1e-10):
        """Full verification of all entries."""
        r = U.shape[1]
        # Check all non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m+j
                    b = j*p+k
                    c = k*n+i
                    val = np.sum(U[a,:] * V[b,:] * W[c,:])
                    if abs(val - 1) > tol:
                        return False
        # Check all zero entries
        for a in range(dim_u):
            for b in range(dim_v):
                for c in range(dim_w):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if not (j == j2 and k == k2 and i == i2):
                        val = np.sum(U[a,:] * V[b,:] * W[c,:])
                        if abs(val) > tol:
                            return False
        return True
    
    def solve_for_U(V, W, rank):
        """Solve linear system for U given V and W."""
        rows = []
        b_vec = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m+j
                    b_idx = j*p+k
                    c_idx = k*n+i
                    row = np.zeros(dim_u * rank)
                    for d in range(rank):
                        row[a*rank + d] = V[b_idx, d] * W[c_idx, d]
                    rows.append(row)
                    b_vec.append(1.0)
        A = np.array(rows)
        b = np.array(b_vec)
        try:
            U_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            return U_flat.reshape(dim_u, rank)
        except:
            return None
    
    def solve_for_V(U, W, rank):
        """Solve linear system for V given U and W."""
        rows = []
        b_vec = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m+j
                    b_idx = j*p+k
                    c_idx = k*n+i
                    row = np.zeros(dim_v * rank)
                    for d in range(rank):
                        row[b_idx*rank + d] = U[a, d] * W[c_idx, d]
                    rows.append(row)
                    b_vec.append(1.0)
        A = np.array(rows)
        b = np.array(b_vec)
        try:
            V_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            return V_flat.reshape(dim_v, rank)
        except:
            return None
    
    def solve_for_W(U, V, rank):
        """Solve linear system for W given U and V."""
        rows = []
        b_vec = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m+j
                    b_idx = j*p+k
                    c_idx = k*n+i
                    row = np.zeros(dim_w * rank)
                    for d in range(rank):
                        row[c_idx*rank + d] = U[a, d] * V[b_idx, d]
                    rows.append(row)
                    b_vec.append(1.0)
        A = np.array(rows)
        b = np.array(b_vec)
        try:
            W_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            return W_flat.reshape(dim_w, rank)
        except:
            return None
    
    def als_step(U, V, W, rank, max_iters=100, tol=1e-12):
        """Alternating least squares."""
        for it in range(max_iters):
            if time.time() > deadline:
                break
            U_new = solve_for_U(V, W, rank)
            if U_new is None:
                break
            V_new = solve_for_V(U_new, W, rank)
            if V_new is None:
                break
            W_new = solve_for_W(U_new, V_new, rank)
            if W_new is None:
                break
            
            # Check convergence
            dU = np.max(np.abs(U_new - U))
            dV = np.max(np.abs(V_new - V))
            dW = np.max(np.abs(W_new - W))
            U, V, W = U_new, V_new, W_new
            if dU < tol and dV < tol and dW < tol:
                break
        return U, V, W
    
    def try_rank(rank, num_trials=20, seed_base=0):
        """Try to find a decomposition at given rank."""
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            seed = seed_base + trial
            rng = np.random.RandomState(seed)
            
            # Initialize with random values
            U = rng.randn(dim_u, rank)
            V = rng.randn(dim_v, rank)
            W = rng.randn(dim_w, rank)
            
            # Run ALS
            U, V, W = als_step(U, V, W, rank)
            
            # Verify
            if full_verify(U, V, W, tol=1e-10):
                return U, V, W
        
        return None
    
    # Try ranks from schoolbook down
    # For 2x3 x 3x9, let's try a range of ranks
    min_rank_to_try = max(1, schoolbook_rank - 20)
    
    for rank in range(schoolbook_rank, min_rank_to_try - 1, -1):
        if time.time() > deadline:
            break
        
        result = try_rank(rank, num_trials=30, seed_base=rank*100)
        if result is not None:
            U, V, W = result
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            best_rank = rank
            # Continue trying lower ranks
            continue
        else:
            # Try a few more with different seeds before giving up
            result = try_rank(rank, num_trials=20, seed_base=rank*100 + 1000)
            if result is not None:
                U, V, W = result
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = rank
                continue
            else:
                # Give up on this rank and lower
                break
    
    return best_result
# EVOLVE-BLOCK-END
