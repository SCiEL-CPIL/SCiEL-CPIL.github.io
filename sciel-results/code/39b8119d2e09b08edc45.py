import numpy as np
import time
import itertools
import random

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # 1. Generate Baseline (Schoolbook Rank 72)
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best_U = U_sb.tolist()
    best_V = V_sb.tolist()
    best_W = W_sb.tolist()
    best_rank = rank_sb

    def verify(U, V, W, n, m, p):
        """Verify exact decomposition with integer/half-integer tolerance."""
        rank = len(U[0]) if U else 0
        if rank == 0:
            return False
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def solve_linear_system_for_pattern(pattern, rank):
        """
        Given a pattern of non-zero indices for each term c, 
        solve for the coefficients.
        
        pattern: list of (i, j, k) tuples for each c in 0..rank-1.
        The tensor equation is:
        sum_c u_c[i,j] v_c[j,k] w_c[k,i] = 1 for all i,j,k
        
        For a fixed pattern (support), the values are constrained.
        This is still non-linear generally, but if we restrict to 
        "diagonal" patterns where u, v, w are 1 or 0, we can check.
        
        However, a more robust approach for small ranks is to use 
        a randomized algebraic method or simply check if a known 
        decomposition exists.
        
        Since general tensor decomposition is hard, we will use a 
        "randomized linear algebra" approach:
        We fix the supports (which entries are non-zero) and then 
        use a simple iterative solver or check if the system 
        sum u*v*w = 1 has a solution.
        
        Actually, for small matrices, we can try to find a 
        "rank-1" like structure or use known identities.
        
        Let's try a different structural approach:
        Decompose the multiplication into blocks.
        """
        pass # Placeholder, we will use a different method below

    def try_strassen_like_decomposition(rank_limit):
        """
        Attempts to find a decomposition using a block-structured approach.
        This is a heuristic search over "Strassen-like" formulas.
        """
        # This is a complex search space. 
        # Instead, we will use a "Randomized Tensor Decomposition" 
        # with a focus on sparse supports.
        
        # We know rank 72 is trivial. 
        # We suspect rank < 72 is possible.
        # Let's try to find a rank 60-70 decomposition by 
        # randomly initializing and using a "local search" on the 
        # support structure (sparsity) combined with value optimization.
        
        # But first, let's check if a simple "half-integer" solution 
        # exists for lower ranks using a different optimizer: 
        # Alternating Least Squares (ALS) with sparsity constraints?
        
        pass

    def optimize_with_als(rank, seed, time_limit):
        """
        Uses Alternating Least Squares (ALS) to optimize the values 
        for a fixed rank. This is often more robust than L-BFGS 
        for tensor decompositions because it handles the 
        multi-linearity better by fixing two factors and solving 
        for the third (which becomes a linear problem if we 
        consider the tensor equation appropriately, though here 
        it's cubic, so it's not strictly linear in one factor 
        unless we fix the others and treat it as a quadratic? 
        No, T = U V W. If we fix V and W, T is linear in U.
        Wait, the equation is sum_c U_ic V_jc W_kc = 1.
        If we fix V and W, the term for a specific (i,j,k) is 
        U_i * (V_j * W_k). This is linear in U_i.
        So we can solve for U given V, W.
        Similarly for V and W.
        """
        if time.time() > time_limit:
            return None
            
        np.random.seed(seed)
        
        # Initialize
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        W = np.random.randn(p * n, rank) * 0.5
        
        # To make the system solvable, we need to be careful with 
        # scaling. 
        # Let's define the residual R[i,j,k] = 1 - sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
        
        # ALS Step for U:
        # For each (i,j), we want to minimize sum_k (1 - sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c])^2
        # Let A_c = V[j*p+k,c] W[k*n+i,c] for a fixed k? No, k varies.
        # The term is sum_k (Target[i,j,k] - sum_c U_idx,c * M_c(i,j,k))^2
        # where M_c(i,j,k) = V[j*p+k,c] * W[k*n+i,c]
        # This is a linear least squares problem for U_idx.
        # We can solve for all U rows simultaneously by forming a big linear system.
        
        def solve_U(V, W):
            # Build matrix A and vector b for the linear system A * U_flat = b
            # U_flat is (n*m*rank,)
            # For each (i,j,k), the contribution to the residual is:
            # (1 - sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c])
            # We want to minimize sum_{i,j,k} (1 - sum_c U_i,c M_c)^2
            # This is equivalent to solving a least squares problem.
            # Let's construct the system.
            
            # Number of equations: n*m*p
            # Number of unknowns: n*m*rank
            # This system is underdetermined if n*m*p < n*m*rank, i.e., p < rank.
            # Here p=6, rank might be > 6. So we need regularization or 
            # we just find the minimum norm solution.
            
            # However, constructing the full matrix might be large.
            # n*m*p = 72. n*m*rank = 12*rank.
            # If rank=60, unknowns=720. Equations=72.
            # We can use np.linalg.lstsq.
            
            # Let's build A (72, 720) and b (72,).
            num_eq = n * m * p
            num_unk = n * m * rank
            
            A = np.zeros((num_eq, num_unk))
            b = np.ones(num_eq)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Equation: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                        # Unknowns: U[i*m+j, c] for c in 0..rank-1
                        # Coefficients: V[j*p+k, c] * W[k*n+i, c]
                        
                        u_base = i * m + j
                        for c in range(rank):
                            A[eq_idx, u_base * rank + c] = V[j * p + k, c] * W[k * n + i, c]
                        b[eq_idx] = 1.0
                        eq_idx += 1
            
            # Solve A * U_flat = b
            try:
                U_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                return None
            
            return U_flat.reshape(n * m, rank)

        def solve_V(U, W):
            num_eq = n * m * p
            num_unk = m * p * rank
            
            A = np.zeros((num_eq, num_unk))
            b = np.ones(num_eq)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        v_base = j * p + k
                        for c in range(rank):
                            A[eq_idx, v_base * rank + c] = U[i * m + j, c] * W[k * n + i, c]
                        b[eq_idx] = 1.0
                        eq_idx += 1
            
            try:
                V_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                return None
            
            return V_flat.reshape(m * p, rank)

        def solve_W(U, V):
            num_eq = n * m * p
            num_unk = p * n * rank
            
            A = np.zeros((num_eq, num_unk))
            b = np.ones(num_eq)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        w_base = k * n + i
                        for c in range(rank):
                            A[eq_idx, w_base * rank + c] = U[i * m + j, c] * V[j * p + k, c]
                        b[eq_idx] = 1.0
                        eq_idx += 1
            
            try:
                W_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                return None
            
            return W_flat.reshape(p * n, rank)

        # Iterate ALS
        for it in range(50):
            if time.time() > time_limit:
                return None
            
            U = solve_U(V, W)
            if U is None: continue
            V = solve_V(U, W)
            if V is None: continue
            W = solve_W(U, V)
            if W is None: continue
            
            # Check convergence / error
            errors = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(rank):
                            s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                        errors += (s - 1.0) ** 2
                        
            if errors < 1e-6:
                # Round to nearest half-integer
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                if verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
                    return U_rounded, V_rounded, W_rounded
                    
        return None

    # Strategy:
    # 1. Try ALS for ranks 71, 70, ... down to 60.
    # ALS is much faster than L-BFGS for this structure.
    
    ranks_to_try = list(range(71, 59, -1))
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
            
        found = False
        # Try a few seeds
        for seed in [42, 123, 999, 7, 13]:
            if time.time() > deadline:
                break
            if found:
                break
                
            t_limit = time.time() + 5  # 5s per ALS attempt
            if t_limit > deadline:
                t_limit = deadline
                
            res = optimize_with_als(r, seed, t_limit)
            if res is not None:
                U_r, V_r, W_r = res
                if r < best_rank:
                    best_U = U_r.tolist()
                    best_V = V_r.tolist()
                    best_W = W_r.tolist()
                    best_rank = r
                found = True
                break # Move to next rank if found

    return {
        "U": best_U,
        "V": best_V,
        "W": best_W
    }
