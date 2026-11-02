"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit (150s wall-clock kill)
    
    dim_u = n * m  # 8
    dim_v = m * p  # 16
    dim_w = p * n  # 8
    
    # The multiplication tensor T[i,j,k] = 1 for all valid i,j,k
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
    
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    def verify_exact(U, V, W):
        """Check if decomposition is exact using high-precision rational-like check."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        r = U_arr.shape[1]
        for (i, j, k) in triples:
            u_row = U_arr[i*m+j, :]
            v_row = V_arr[j*p+k, :]
            w_row = W_arr[k*n+i, :]
            s = np.dot(u_row * v_row, w_row)
            if abs(s - 1.0) > 1e-8:
                return False
        return True
    
    def verify_with_matrices(A, B, U, V, W):
        """Verify by actually computing matrix multiplication."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        r = U_arr.shape[1]
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                total = 0.0
                for j in range(m):
                    for rr in range(r):
                        total += U_arr[i*m+j, rr] * A[i,j] * V_arr[j*p+k, rr] * B[j,k] * W_arr[k*n+i, rr]
                C_computed[i,k] = total
        C_expected = A @ B
        return np.allclose(C_computed, C_expected, atol=1e-8)
    
    def solve_for_W_given_UV(U, V, rank):
        """
        Given U (dim_u x rank) and V (dim_v x rank), solve for W (dim_w x rank).
        For each (i,k): sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        Let a_r(i,k) = sum_j U[i*m+j, r] * V[j*p+k, r]
        Then we need: sum_r a_r(i,k) * W[k*n+i, r] = 1
        
        This is a single linear equation per (i,k) pair.
        We use minimum-norm solution: W_row = a / (a^T a)
        """
        W = np.zeros((dim_w, rank))
        for i in range(n):
            for k in range(p):
                a = np.zeros(rank)
                for j in range(m):
                    a += U[i*m+j, :] * V[j*p+k, :]
                norm_sq = np.dot(a, a)
                if norm_sq < 1e-14:
                    return None
                W[k*n+i, :] = a / norm_sq
        return W
    
    def solve_for_UVW_rank(rank, seed):
        """
        Try to find a rank-rank decomposition using randomized approach.
        Strategy: Randomly initialize U and V, solve for W, verify.
        If that fails, use a more sophisticated approach with partial W fixing.
        """
        rng = np.random.RandomState(seed)
        
        # Approach 1: Random U, V, solve W
        for attempt in range(20):
            U = rng.randn(dim_u, rank)
            V = rng.randn(dim_v, rank)
            W = solve_for_W_given_UV(U, V, rank)
            if W is not None and verify_exact(U, V, W):
                return U, V, W
        
        # Approach 2: Fix W to standard basis-like structure, solve for U, V
        # W has dim_w=8 rows, rank columns
        # Try sparse W patterns
        for w_pattern in range(5):
            W = np.zeros((dim_w, rank))
            if w_pattern == 0:
                # Identity-like: W[k*n+i, k*n+i] = 1 if k*n+i < rank
                for idx in range(min(dim_w, rank)):
                    W[idx, idx] = 1.0
            elif w_pattern == 1:
                # All ones in first column
                W[:, 0] = 1.0
            elif w_pattern == 2:
                # Random sparse
                for idx in range(dim_w):
                    col = rng.randint(rank)
                    W[idx, col] = 1.0
            elif w_pattern == 3:
                # Block structure
                for idx in range(dim_w):
                    col = idx % rank
                    W[idx, col] = 1.0
            else:
                # Random dense
                W = rng.randn(dim_w, rank)
            
            # Now solve for U and V given W
            # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # This is a system of bilinear equations. Use alternating minimization.
            
            U = rng.randn(dim_u, rank)
            V = rng.randn(dim_v, rank)
            
            for iteration in range(100):
                # Update U: for each (i,j), fix V and W, solve for U[i*m+j, :]
                for i in range(n):
                    for j in range(m):
                        # sum_r U[i*m+j, r] * (sum_k V[j*p+k, r] * W[k*n+i, r]) = 1
                        # Let b_r = sum_k V[j*p+k, r] * W[k*n+i, r]
                        # Then sum_r U[i*m+j, r] * b_r = 1
                        b = np.zeros(rank)
                        for k in range(p):
                            b += V[j*p+k, :] * W[k*n+i, :]
                        norm_sq = np.dot(b, b)
                        if norm_sq > 1e-14:
                            U[i*m+j, :] = b / norm_sq
                        else:
                            # No good solution, keep current
                            pass
                
                # Update V: for each (j,k), fix U and W, solve for V[j*p+k, :]
                for j in range(m):
                    for k in range(p):
                        # sum_r V[j*p+k, r] * (sum_i U[i*m+j, r] * W[k*n+i, r]) = 1
                        # Let c_r = sum_i U[i*m+j, r] * W[k*n+i, r]
                        # Then sum_r V[j*p+k, r] * c_r = 1
                        c = np.zeros(rank)
                        for i in range(n):
                            c += U[i*m+j, :] * W[k*n+i, :]
                        norm_sq = np.dot(c, c)
                        if norm_sq > 1e-14:
                            V[j*p+k, :] = c / norm_sq
                        else:
                            pass
                
                # Check convergence
                if verify_exact(U, V, W):
                    return U, V, W
        
        return None
    
    def solve_for_UV_given_W(W, rank, seed):
        """
        Given W, solve for U and V using alternating minimization.
        """
        rng = np.random.RandomState(seed)
        U = rng.randn(dim_u, rank)
        V = rng.randn(dim_v, rank)
        
        for iteration in range(200):
            # Update U
            for i in range(n):
                for j in range(m):
                    b = np.zeros(rank)
                    for k in range(p):
                        b += V[j*p+k, :] * W[k*n+i, :]
                    norm_sq = np.dot(b, b)
                    if norm_sq > 1e-14:
                        U[i*m+j, :] = b / norm_sq
            
            # Update V
            for j in range(m):
                for k in range(p):
                    c = np.zeros(rank)
                    for i in range(n):
                        c += U[i*m+j, :] * W[k*n+i, :]
                    norm_sq = np.dot(c, c)
                    if norm_sq > 1e-14:
                        V[j*p+k, :] = c / norm_sq
            
            if verify_exact(U, V, W):
                return U, V, W
        
        return None
    
    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(dim_u)]
        V = [[0] * rank for _ in range(dim_v)]
        W = [[0] * rank for _ in range(dim_w)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_rank = n * m * p  # 32
    best_result = None
    
    # Try ranks from low to high
    # Lower bound: max(n*m, m*p, n*p) = max(8, 16, 8) = 16
    for rank in range(16, best_rank + 1):
        if time.time() > deadline - 15:
            break
        
        found = False
        
        # Method 1: Random U, V, solve W
        max_random_attempts = 30 if rank <= 20 else 15
        for attempt in range(max_random_attempts):
            if time.time() > deadline - 10:
                break
            
            seed = rank * 1000 + attempt
            result = solve_for_UVW_rank(rank, seed)
            if result is not None:
                U, V, W = result
                best_rank = rank
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                found = True
                break
        
        if found:
            break
        
        # Method 2: Structured W patterns with alternating minimization
        if time.time() > deadline - 10:
            break
        
        for w_seed in range(10):
            if time.time() > deadline - 8:
                break
            
            # Create structured W
            W = np.zeros((dim_w, rank))
            rng_w = np.random.RandomState(rank * 100 + w_seed)
            
            if w_seed < 3:
                # Sparse diagonal-like
                for idx in range(dim_w):
                    col = idx % rank
                    W[idx, col] = 1.0
            elif w_seed < 6:
                # Random sparse
                for idx in range(dim_w):
                    col = rng_w.randint(rank)
                    W[idx, col] = 1.0
            else:
                # Random dense
                W = rng_w.randn(dim_w, rank)
            
            result = solve_for_UV_given_W(W, rank, seed=rank * 200 + w_seed)
            if result is not None:
                U, V, W_sol = result
                best_rank = rank
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W_sol.tolist()}
                found = True
                break
        
        if found:
            break
    
    if best_result is not None:
        return best_result
    else:
        return schoolbook()
# EVOLVE-BLOCK-END
