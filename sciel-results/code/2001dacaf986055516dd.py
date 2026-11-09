import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin for 120s limit
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i*m + j
                idx_v = j*p + k
                idx_w = k*n + i
                col = i*m*p + j*p + k
                U_sb[idx_u, col] = 1
                V_sb[idx_v, col] = 1
                W_sb[idx_w, col] = 1
                
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    best_rank = rank_sb

    def verify_solution(U, V, W, tol=1e-9):
        """Check if the decomposition is exact and entries are integer/half-integer."""
        # Check exactness of tensor contraction
        # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # We need T[i,j,k] == 1 for all i,j,k
        # Using einsum for speed
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
        try:
            U_reshaped = U.reshape(n, m, -1)
            V_reshaped = V.reshape(m, p, -1)
            W_reshaped = W.reshape(p, n, -1)
            
            # Contract: sum over r
            # Result shape: (n, m, p)
            # T[i, j, k] = sum_r U_reshaped[i, j, r] * V_reshaped[j, k, r] * W_reshaped[k, i, r]
            # This is a bit tricky with einsum indices
            # U: i, j, r
            # V: j, k, r
            # W: k, i, r
            # Result: i, j, k
            # einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
            T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
            
            if not np.allclose(T, 1.0, atol=tol):
                return False
        except Exception:
            return False

        # Check integer/half-integer constraint
        for mat in [U, V, W]:
            # Check if val is close to a half-integer
            closest_half_int = np.round(mat * 2) / 2
            if not np.allclose(mat, closest_half_int, atol=1e-5):
                return False
        return True

    def try_rank_als(r, seed=42, max_iters=2000):
        """
        Attempt to find a rank-r decomposition using ALS.
        Returns (U, V, W) if found, else None.
        """
        if time.time() > deadline:
            return None

        rng = np.random.RandomState(seed)
        
        # Initialize
        U = rng.randn(n * m, r)
        V = rng.randn(m * p, r)
        W = rng.randn(p * n, r)
        
        # Normalize initial vectors to prevent numerical issues
        for i in range(n*m):
            norm = np.linalg.norm(U[i, :])
            if norm > 1e-6: U[i, :] /= norm
        for j in range(m*p):
            norm = np.linalg.norm(V[j, :])
            if norm > 1e-6: V[j, :] /= norm
        for k in range(p*n):
            norm = np.linalg.norm(W[k, :])
            if norm > 1e-6: W[k, :] /= norm

        prev_err = np.inf
        for it in range(max_iters):
            if time.time() > deadline - 5:
                return None

            # Update U
            # For each i, j:
            # M[k, r] = V[j*p+k, r] * W[k*n+i, r]
            # Solve M @ u = ones(p)
            for i in range(n):
                for j in range(m):
                    # Construct M efficiently
                    # V rows: indices j*p to j*p+p-1
                    # W rows: indices k*n+i for k in 0..p-1
                    # M is p x r
                    # M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    # We can build this with a loop over k, or vectorized if we index W correctly
                    # W_indices = [k*n + i for k in range(p)]
                    # M = V[j*p:(j+1)*p, :] * W[W_indices, :]
                    W_indices = np.array([k*n + i for k in range(p)])
                    M = V[j*p:(j+1)*p, :] * W[W_indices, :]
                    
                    try:
                        u, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        U[i*m+j, :] = u
                    except:
                        pass

            # Update V
            # For each j, k:
            # M[i, r] = U[i*m+j, r] * W[k*n+i, r]
            # Solve M @ v = ones(n)
            for j in range(m):
                for k in range(p):
                    # U indices: i*m + j for i in 0..n-1
                    # W rows: k*n + i for i in 0..n-1
                    U_indices = np.array([i*m + j for i in range(n)])
                    W_rows = W[k*n:(k+1)*n, :] # This is n x r, but rows are i=0..n-1
                    # M[i, :] = U[U_indices[i], :] * W_rows[i, :]
                    M = U[U_indices, :] * W_rows
                    
                    try:
                        v, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                        V[j*p+k, :] = v
                    except:
                        pass

            # Update W
            # For each k, i:
            # M[j, r] = U[i*m+j, r] * V[j*p+k, r]
            # Solve M @ w = ones(m)
            for k in range(p):
                for i in range(n):
                    # U rows: i*m + j for j in 0..m-1
                    # V rows: j*p + k for j in 0..m-1
                    U_rows = U[i*m:(i+1)*m, :] # m x r
                    V_indices = np.array([j*p + k for j in range(m)])
                    # M[j, :] = U_rows[j, :] * V[V_indices[j], :]
                    M = U_rows * V[V_indices, :]
                    
                    try:
                        w, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                        W[k*n+i, :] = w
                    except:
                        pass

            # Calculate error
            # Use einsum for speed
            U_reshaped = U.reshape(n, m, r)
            V_reshaped = V.reshape(m, p, r)
            W_reshaped = W.reshape(p, n, r)
            T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
            max_err = np.max(np.abs(T - 1.0))

            # Check convergence
            if max_err < 1e-10:
                # Try to round to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                if verify_solution(U_rounded, V_rounded, W_rounded):
                    return U_rounded, V_rounded, W_rounded
                else:
                    # If rounding fails, the solution is not rational with denominator 2.
                    # We cannot use it.
                    pass
            
            if max_err < 1e-6 and max_err < prev_err * 0.9:
                # Potential candidate, but need to check integer constraint at the end
                pass
                
            prev_err = max_err

        # Final check after max iterations
        U_reshaped = U.reshape(n, m, r)
        V_reshaped = V.reshape(m, p, r)
        W_reshaped = W.reshape(p, n, r)
        T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        max_err = np.max(np.abs(T - 1.0))

        if max_err < 1e-10:
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            if verify_solution(U_rounded, V_rounded, W_rounded):
                return U_rounded, V_rounded, W_rounded

        return None

    # Strategy:
    # 1. Try to find a decomposition with rank < 60.
    # The known complex rank is 16. The integer rank might be higher.
    # We will try ranks from 16 upwards.
    # If we find a valid one, we return it.
    # If we reach 60, we return the schoolbook.
    
    # To save time, we will try a limited number of seeds for each rank.
    # If low ranks fail, we might just return 60.
    
    # Let's try ranks 16 to 30.
    # If 16 is the answer, we want to find it.
    
    found_solution = False
    for r in range(16, 31):
        if time.time() > deadline - 10:
            break
            
        # Try a few seeds for this rank
        # Use more seeds for lower ranks as they are more likely to be the target if they exist
        num_seeds = 5 if r < 20 else 3
        seeds = [42, 123, 999, 1234, 5678][:num_seeds]
        
        for seed in seeds:
            if time.time() > deadline - 5:
                break
                
            res = try_rank_als(r, seed=seed, max_iters=1000)
            if res is not None:
                U_found, V_found, W_found = res
                # Update best
                best_U = U_found
                best_V = V_found
                best_W = W_found
                best_rank = r
                found_solution = True
                # If we found a solution, we can try to go lower? 
                # No, we are iterating upwards. So this is the lowest found so far.
                # We can break and return immediately?
                # Yes, because we want the *minimum* rank.
                # If we found rank r, any rank < r was already tried and failed.
                # So r is the minimum.
                break
        
        if found_solution:
            break
    
    # If no solution found, use schoolbook
    if not found_solution:
        best_U = U_sb
        best_V = V_sb
        best_W = W_sb
        best_rank = rank_sb

    # Convert best result to lists
    U_list = [[float(x) for x in row] for row in best_U.tolist()]
    V_list = [[float(x) for x in row] for row in best_V.tolist()]
    W_list = [[float(x) for x in row] for row in best_W.tolist()]
    
    return {"U": U_list, "V": V_list, "W": W_list}
