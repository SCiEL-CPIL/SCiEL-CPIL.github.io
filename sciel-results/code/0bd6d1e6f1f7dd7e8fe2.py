"""Matrix multiplication 3x5x7 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    trivial_rank = n * m * p  # 105
    
    # Build the trivial decomposition
    U = np.zeros((n * m, trivial_rank), dtype=float)
    V = np.zeros((m * p, trivial_rank), dtype=float)
    W = np.zeros((p * n, trivial_rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Precompute valid triples (a, b, c) where a=i*m+j, b=j*p+k, c=k*n+i
    # These are the non-zero entries of the multiplication tensor T
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                triples.append((i * m + j, j * p + k, k * n + i))
                
    # Convert to arrays for fast indexing
    a_idx = np.array([t[0] for t in triples])
    b_idx = np.array([t[1] for t in triples])
    c_idx = np.array([t[2] for t in triples])
    
    def verify_fast(U_mat, V_mat, W_mat, tol=1e-8):
        """Vectorized check: sum_t U[a,t]*V[b,t]*W[c,t] == 1 for all triples"""
        U_vals = U_mat[a_idx, :]
        V_vals = V_mat[b_idx, :]
        W_vals = W_mat[c_idx, :]
        products = U_vals * V_vals * W_vals
        sums = products.sum(axis=1)
        if np.any(np.abs(sums - 1.0) > tol):
            return False
        return True

    def try_rank(r, max_iters=50, n_seeds=20):
        """
        Attempt to find a rank-r decomposition using gradient descent.
        Objective: minimize sum_{(a,b,c) in triples} (sum_t U[a,t]V[b,t]W[c,t] - 1)^2
                 + sum_{(a,b,c) NOT in triples} (sum_t U[a,t]V[b,t]W[c,t])^2
        """
        if r >= trivial_rank:
            return None
        
        # Precompute the "zero" triples (all triples that are not in the multiplication tensor)
        # Total possible triples is (n*m)*(m*p)*(p*n) which is huge (15*35*21 = 11025).
        # We only need to check the non-zeros for the "1" constraint.
        # The "0" constraint is implicit if we just minimize the error on the 1s and 
        # assume the model is sparse? No, we must ensure the other entries are 0.
        # However, for small r, the model might not be able to fit the 1s without 
        # leaking into the 0s.
        
        # To be efficient, we only check the 1s in the objective during optimization,
        # and verify the 0s at the end. If the 0s are not 0, the solution is invalid.
        
        for seed in range(n_seeds):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(seed * 7919 + 13)
            
            # Initialize with small random values
            U_mat = np.random.randn(n * m, r) * 0.1
            V_mat = np.random.randn(m * p, r) * 0.1
            W_mat = np.random.randn(p * n, r) * 0.1
            
            lr = 0.01
            best_obj = np.inf
            best_state = None
            
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                
                # Compute current sums on the 1-triples
                U_vals = U_mat[a_idx, :]
                V_vals = V_mat[b_idx, :]
                W_vals = W_mat[c_idx, :]
                sums = (U_vals * V_vals * W_vals).sum(axis=1)
                
                # Error
                err = sums - 1.0
                obj = np.sum(err ** 2)
                
                if obj < best_obj:
                    best_obj = obj
                    best_state = (U_mat.copy(), V_mat.copy(), W_mat.copy())
                    
                if obj < 1e-12:
                    # Check if it's a valid solution (zeros elsewhere)
                    # This check is expensive, so only do it if we think we found it
                    # For now, assume if 1s are fit, we can clean up.
                    # We need to check the full tensor.
                    pass
                
                # Compute gradients
                # dU[a,t] = sum_{(a,b,c) in triples} err[a] * V[b,t] * W[c,t]
                # This is a matrix multiplication: 
                # G_U[a,t] = sum_c' ... 
                # Let's vectorize.
                # We have err (N,), V_vals (N, r), W_vals (N, r).
                # We want G_U (N, r) such that G_U[a,t] = sum_{n} err[n] * V_vals[n,t] * W_vals[n,t]
                # This is not a standard matmul because of the element-wise product of V and W.
                # We can compute P = V_vals * W_vals (N, r).
                # Then G_U = P.T @ (err * 1) ? No.
                # G_U[a,t] = sum_n err[n] * P[n,t] where P is indexed by the triple n.
                # But U_mat is indexed by 'a'. Multiple triples can have the same 'a'.
                # So we need to sum over all triples that share the same 'a'.
                
                # Efficient way:
                # Create a matrix E of shape (N, r) where E[n,t] = err[n] * V_vals[n,t] * W_vals[n,t]
                # Then G_U[a,t] = sum_{n: a_idx[n]==a} E[n,t]
                
                # We can use np.add.at or bincount-like logic.
                # Since N is small (105), we can just loop or use advanced indexing.
                # Let's use a sparse-like accumulation.
                
                E_U = err[:, None] * V_vals * W_vals
                # Accumulate to G_U
                G_U = np.zeros((n * m, r))
                np.add.at(G_U, a_idx, E_U)
                
                E_V = err[:, None] * U_vals * W_vals
                G_V = np.zeros((m * p, r))
                np.add.at(G_V, b_idx, E_V)
                
                E_W = err[:, None] * U_vals * V_vals
                G_W = np.zeros((p * n, r))
                np.add.at(G_W, c_idx, E_W)
                
                # Gradient step
                U_mat -= lr * G_U
                V_mat -= lr * G_V
                W_mat -= lr * G_W
                
                # Simple regularization to prevent explosion
                U_mat *= 0.99
                V_mat *= 0.99
                W_mat *= 0.99
                
            # After optimization, check the best state
            if best_state is not None:
                U_best, V_best, W_best = best_state
                # Try to clean up to integers/half-integers
                # Check if the current state is close to integer/half-integer
                U_check = np.round(U_best * 2) / 2
                V_check = np.round(V_best * 2) / 2
                W_check = np.round(W_best * 2) / 2
                
                if verify_fast(U_check, V_check, W_check, tol=1e-6):
                    # We must also verify the zeros!
                    # Construct full tensor and check
                    # This is expensive, but we only do it for promising candidates.
                    # For now, assume if the 1s are perfect, the zeros are likely 0 
                    # if the rank is low enough and the solution is sparse.
                    # However, we must be careful.
                    # Let's do a full verification.
                    if full_verify(U_check, V_check, W_check):
                        return (U_check, V_check, W_check)
                        
        return None

    def full_verify(U_mat, V_mat, W_mat, tol=1e-6):
        """
        Verify that the tensor defined by U,V,W matches the multiplication tensor exactly.
        T[a,b,c] = sum_t U[a,t] V[b,t] W[c,t]
        Should be 1 if (a,b,c) in triples, 0 otherwise.
        """
        # Check 1s
        U_vals = U_mat[a_idx, :]
        V_vals = V_mat[b_idx, :]
        W_vals = W_mat[c_idx, :]
        sums_1 = (U_vals * V_vals * W_vals).sum(axis=1)
        if np.any(np.abs(sums_1 - 1.0) > tol):
            return False
            
        # Check 0s
        # Generate all possible (a,b,c)
        all_a = np.arange(n * m)
        all_b = np.arange(m * p)
        all_c = np.arange(p * n)
        
        # We cannot iterate over all (15*35*21) = 11025 triples easily in a loop 
        # if we do it many times, but it's fast in numpy.
        # Create meshgrid
        A_grid, B_grid, C_grid = np.meshgrid(all_a, all_b, all_c, indexing='ij')
        A_flat = A_grid.flatten()
        B_flat = B_grid.flatten()
        C_flat = C_grid.flatten()
        
        # Identify which ones are in triples
        # Create a set of valid triples
        valid_set = set(triples)
        # This is slow in python. 
        # Alternative: Compute the full tensor T = U @ V^T? No.
        # T[a,b,c] = sum_t U[a,t] V[b,t] W[c,t]
        # This is a 3-way tensor contraction.
        
        # We can compute the full tensor T of shape (n*m, m*p, p*n)
        # T = np.einsum('at,bt,ct->abc', U_mat, V_mat, W_mat)
        # This might be memory heavy: 15 * 35 * 21 * 8 bytes = ~0.9 MB. Fine.
        
        T = np.einsum('at,bt,ct->abc', U_mat, V_mat, W_mat)
        
        # Check all entries
        # Expected: 1 at triples, 0 elsewhere.
        # Create expected tensor
        T_expected = np.zeros((n * m, m * p, p * n))
        for a, b, c in triples:
            T_expected[a, b, c] = 1.0
            
        if np.any(np.abs(T - T_expected) > tol):
            return False
            
        return True

    # Strategy:
    # 1. Return trivial solution if no better found.
    U_best = U.copy()
    V_best = V.copy()
    W_best = W.copy()
    best_rank = trivial_rank
    
    # Try a few ranks below 105. 
    # Given the difficulty, we try 104, 103.
    # If we find one, we return it.
    for r in range(trivial_rank - 1, trivial_rank - 5, -1):
        if time.time() > deadline - 20:
            break
        result = try_rank(r, max_iters=20, n_seeds=5)
        if result is not None:
            U_best, V_best, W_best = result
            best_rank = r
            break # Found a better one, stop searching lower for now to save time
            
    # Convert to lists for return
    U_list = U_best.tolist()
    V_list = V_best.tolist()
    W_list = W_best.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
