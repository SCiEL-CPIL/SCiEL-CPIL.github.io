import numpy as np
import time
import random

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Shapes
    dim_U = n * m  # 12
    dim_V = m * p  # 16
    dim_W = p * n  # 12
    
    # Target rank
    target_rank = 47
    
    # Precompute tensor T as a list of (I, J, K) indices where T=1
    # T[I, J, K] = 1 if I=i*m+j, J=j*p+k, K=k*n+i
    T_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I_idx = i * m + j
                J_idx = j * p + k
                K_idx = k * n + i
                T_indices.append((I_idx, J_idx, K_idx))
                
    # Also need the "zero" constraints? 
    # The tensor is zero everywhere else.
    # We must ensure the decomposition is 0 at all other (I, J, K).
    # Total number of entries in T is dim_U * dim_V * dim_W = 12 * 16 * 12 = 2304.
    # Non-zeros are 48.
    # So 2256 zeros.
    
    # We will formulate the ALS step to satisfy both non-zero and zero constraints.
    # For a fixed I, the equation is sum_r U[I,r] * V[J,r] * W[K,r] = T[I,J,K] for all J,K.
    
    def verify_solution(U, V, W, R):
        """
        Verifies if U, V, W (as lists of lists of floats) exactly decompose T.
        Assumes entries are close to half-integers.
        """
        # Convert to scaled integers
        U_int = np.round(np.array(U) * 2)
        V_int = np.round(np.array(V) * 2)
        W_int = np.round(np.array(W) * 2)
        
        # Check if they are actually integers (i.e. original was half-integer)
        # If U, V, W were already integers, U_int is even.
        # The formula is sum_r (U_int[r]/2) * (V_int[r]/2) * (W_int[r]/2) = T
        # sum_r U_int[r] * V_int[r] * W_int[r] = 8 * T
        
        # Check non-zero entries
        for (I_idx, J_idx, K_idx) in T_indices:
            val = 0
            for r in range(R):
                val += U_int[I_idx][r] * V_int[J_idx][r] * W_int[K_idx][r]
            if val != 8:
                return False
                
        # Check a sample of zero entries to save time? 
        # No, we must check all to be safe, but 2256 * 47 is ~100k ops, which is fast.
        # However, iterating in Python is slow.
        # Let's use NumPy for verification.
        T_approx = np.einsum('ir, jr, kr -> ijk', U_int, V_int, W_int)
        
        # T_approx should be 8 where T is 1, and 0 elsewhere.
        # Create full T array
        T_full = np.zeros((dim_U, dim_V, dim_W), dtype=int)
        for (I_idx, J_idx, K_idx) in T_indices:
            T_full[I_idx, J_idx, K_idx] = 8
            
        if not np.array_equal(T_approx, T_full):
            return False
            
        return True

    def run_als(R, max_iters, time_limit):
        """
        Runs ALS for rank R.
        Returns (U, V, W) if successful, else None.
        """
        start_time = time.time()
        
        # Initialize with random small integers
        # Using {-1, 0, 1} often works well for tensor decomps
        U = np.random.randint(-1, 2, (dim_U, R)).astype(float)
        V = np.random.randint(-1, 2, (dim_V, R)).astype(float)
        W = np.random.randint(-1, 2, (dim_W, R)).astype(float)
        
        for it in range(max_iters):
            if time.time() > time_limit:
                return None
            
            # Update U
            # For each I, solve A_U * U[I, :] = b_U
            # A_U[J*K, r] = V[J, r] * W[K, r]
            # We can precompute the V*W product matrix for all J,K pairs?
            # V is (16, R), W is (12, R).
            # V_W = np.einsum('jr, kr -> jkr', V, W) -> (16, 12, R)
            # Flatten to (192, R)
            V_W = np.einsum('jr, kr -> jkr', V, W)
            A_U = V_W.reshape(-1, R)
            
            for i in range(dim_U):
                # b_U is T[i, :, :] flattened
                # T[i, J, K] is 1 if (i, J, K) in T_indices else 0
                # We can precompute T_flat for each i
                # But T is sparse.
                # Let's just build b_U
                b_U = np.zeros(dim_V * dim_W)
                for (I_idx, J_idx, K_idx) in T_indices:
                    if I_idx == i:
                        b_U[J_idx * dim_W + K_idx] = 1.0
                
                # Solve
                try:
                    U[i, :], residuals, rank, s = np.linalg.lstsq(A_U, b_U, rcond=None)
                except:
                    return None
            
            # Update V
            U_W = np.einsum('ir, kr -> ikr', U, W)
            A_V = U_W.reshape(-1, R)
            
            for j in range(dim_V):
                b_V = np.zeros(dim_U * dim_W)
                for (I_idx, J_idx, K_idx) in T_indices:
                    if J_idx == j:
                        b_V[I_idx * dim_W + K_idx] = 1.0
                try:
                    V[j, :], residuals, rank, s = np.linalg.lstsq(A_V, b_V, rcond=None)
                except:
                    return None
            
            # Update W
            U_V = np.einsum('ir, jr -> ijr', U, V)
            A_W = U_V.reshape(-1, R)
            
            for k in range(dim_W):
                b_W = np.zeros(dim_U * dim_V)
                for (I_idx, J_idx, K_idx) in T_indices:
                    if K_idx == k:
                        b_W[I_idx * dim_V + J_idx] = 1.0
                try:
                    W[k, :], residuals, rank, s = np.linalg.lstsq(A_W, b_W, rcond=None)
                except:
                    return None
            
            # Check for convergence to integer/half-integer solution
            # Periodically check (e.g., every 5 iterations) to save time
            if it % 5 == 0:
                # Round to nearest half-integer
                U_round = np.round(U * 2) / 2.0
                V_round = np.round(V * 2) / 2.0
                W_round = np.round(W * 2) / 2.0
                
                # Verify
                if verify_solution(U_round.tolist(), V_round.tolist(), W_round.tolist(), R):
                    return U_round, V_round, W_round
        
        return None

    # Main search loop
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Try rank 47
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Use different random seeds
        np.random.seed(attempts)
        res = run_als(target_rank, max_iters=50, time_limit=deadline)
        if res is not None:
            U, V, W = res
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            
    # Fallback to Schoolbook Rank 48
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
# EVOLVE-BLOCK-END
