import numpy as np
import time

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Lower bound heuristic: start from a reasonable guess
    # For 4x5x6, schoolbook is 120.
    # Known lower bounds are lower, but we search upwards.
    min_rank = 60
    max_rank = 119
    
    best_result = None
    
    # Precompute ones matrix for J
    J = np.ones((n, n))
    
    # We iterate R from low to high to find minimal rank
    for R in range(min_rank, max_rank + 1):
        if time.time() > deadline:
            break
            
        # Number of trials per rank
        # Reduce trials for higher R if time is tight, but R is the main cost driver in linear algebra
        num_trials = 500 if R < 80 else 200
        
        for trial in range(num_trials):
            if time.time() > deadline:
                break
                
            # Generate random sparse U and V
            # U: (n*m, R) -> reshape to (n, m, R)
            # V: (m*p, R) -> reshape to (m, p, R)
            
            # Use density ~ 0.5 for sparsity
            # Entries are 1 or -1
            U_flat = np.zeros((n * m, R))
            V_flat = np.zeros((m * p, R))
            
            # Randomly set some entries to 1 or -1
            # To increase chance of integer solutions, keep entries simple
            mask_U = np.random.rand(n * m, R) < 0.5
            mask_V = np.random.rand(m * p, R) < 0.5
            
            U_flat[mask_U] = np.random.choice([-1, 1], size=mask_U.sum())
            V_flat[mask_V] = np.random.choice([-1, 1], size=mask_V.sum())
            
            U = U_flat.reshape(n, m, R)
            V = V_flat.reshape(m, p, R)
            
            # Check if W exists
            valid = True
            W_parts = []
            
            for k in range(p):
                # Compute A_k: (n, R)
                # A_k[i, r] = sum_j U[i, j, r] * V[j, k, r]
                # This is element-wise product sum over j
                # U[:, :, r] is (n, m), V[:, k, r] is (m,)
                # So A_k = U * V[:, k, :] (broadcast) summed over axis 1
                # U shape (n, m, R), V[:, k, :] shape (m, R)
                # Product: (n, m, R) * (1, m, R) -> (n, m, R)
                # Sum over axis 1 -> (n, R)
                A_k = np.sum(U * V[:, k, :], axis=1)
                
                # We need to solve A_k * W_k^T = J
                # Where W_k is (n, R). Let X = W_k^T (R, n).
                # A_k (n, R) * X (R, n) = J (n, n)
                
                # Check consistency: rank(A_k) == rank([A_k, J])
                # Since J is all ones, we just need to check if 1 is in the column space of A_k?
                # No, A_k * x = 1_vector for each column of X.
                # So we need to solve A_k * x = 1 for x.
                # This requires 1 to be in the column space of A_k.
                
                # Use least squares to find x
                # If the solution is exact, residual should be 0.
                try:
                    # Solve A_k x = 1
                    # A_k is (n, R). 1 is (n,)
                    x, residuals, rank, sv = np.linalg.lstsq(A_k, np.ones(n), rcond=None)
                    
                    # Check if solution is exact
                    # If residuals is empty, check norm(A_k x - 1)
                    if residuals.size > 0:
                        if residuals[0] > 1e-9:
                            valid = False
                            break
                    else:
                        if np.linalg.norm(A_k @ x - np.ones(n)) > 1e-9:
                            valid = False
                            break
                            
                    # We found a particular solution x for one column.
                    # But we need W_k such that A_k W_k^T = J.
                    # Since all columns of J are identical (1), the solution for all columns is the same x.
                    # So W_k^T = x * 1^T? No.
                    # W_k is (n, R). W_k^T is (R, n).
                    # Column i of W_k^T is x.
                    # So W_k^T = np.tile(x, (1, n))
                    # W_k = np.tile(x, (1, n)).T -> (n, R)
                    
                    # Check if entries are integer/half-integer
                    # x should be rational with small denominator
                    # Check if 2*x is close to integers
                    W_k_part = np.tile(x, (1, n)).T
                    W_parts.append(W_k_part)
                    
                except np.linalg.LinAlgError:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Construct full W
            # W shape: (p*n, R)
            # Index: k*n + i
            W_flat = np.zeros((p * n, R))
            for k in range(p):
                W_flat[k*n:(k+1)*n, :] = W_parts[k]
                
            # Verify exactness with original U, V, W
            # Check T = sum_r U[:,r] otimes V[:,r] otimes W[:,r]
            # T[i,j,k] = 1 for all i,j,k
            # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
            
            # Efficient check:
            # For each r, compute outer products? No, R is small enough.
            # We can check a few random indices or all indices if fast.
            # n=4, m=5, p=6. Total 120 indices.
            # R <= 120.
            # 120 * 120 ops is trivial.
            
            exact = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for r in range(R):
                            val += U[i, j, r] * V[j, k, r] * W_flat[k*n + i, r]
                        if abs(val - 1.0) > 1e-9:
                            exact = False
                            break
                    if not exact:
                        break
                if not exact:
                    break
            
            if exact:
                # Convert to lists
                U_list = U_flat.tolist()
                V_list = V_flat.tolist()
                W_list = W_flat.tolist()
                
                # Check if entries are integer/half-integer
                # They are sums of 1/-1, so they are integers.
                # But lstsq might give floats.
                # We should round them?
                # The problem says "integer/half-integer entries".
                # If the decomposition is exact, the values should be very close to such numbers.
                # Let's round to nearest 0.5
                U_rounded = [[round(x * 2) / 2 for x in row] for row in U_list]
                V_rounded = [[round(x * 2) / 2 for x in row] for row in V_list]
                W_rounded = [[round(x * 2) / 2 for x in row] for row in W_list]
                
                # Re-verify with rounded values
                U_r = np.array(U_rounded)
                V_r = np.array(V_rounded)
                W_r = np.array(W_rounded)
                
                exact_r = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for r in range(R):
                                val += U_r[i*m+j, r] * V_r[j*p+k, r] * W_r[k*n+i, r]
                            if abs(val - 1.0) > 1e-9:
                                exact_r = False
                                break
                        if not exact_r:
                            break
                    if not exact_r:
                        break
                
                if exact_r:
                    return {"U": U_rounded, "V": V_rounded, "W": W_rounded}
                    
    # Fallback: Schoolbook
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
