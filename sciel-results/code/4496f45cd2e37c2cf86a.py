import numpy as np
import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
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
    
    start_time = time.time()
    deadline = start_time + 110 # 110s limit to be safe
    
    # Precompute target tensor indices and values
    # T[alpha, beta, gamma] = 1 if j_alpha == j_beta else 0
    # alpha = i*m + j
    # beta = j'*p + k
    # gamma = k*n + i
    # We only need to check consistency for all valid triples (i, j, k) where j_alpha == j_beta.
    # Actually, the tensor is defined by the multiplication.
    # T[i, j, k] in standard notation (A[i,j], B[j,k], C[i,k]) is 1.
    # In our flattened indices:
    # alpha = i*m + j
    # beta = j*p + k
    # gamma = k*n + i
    # The condition for a non-zero term in the tensor T is that the 'j' index in alpha matches the 'j' index in beta.
    # j_alpha = alpha % m
    # j_beta = beta // p
    # So T[alpha, beta, gamma] = 1 if (alpha % m) == (beta // p) else 0.
    # Note: gamma is determined by i and k.
    # i = alpha // m
    # k = beta % p
    # gamma should be k*n + i.
    # Let's verify: gamma = k*n + i.
    # The tensor is sparse. We can define a function to get the target value.
    
    def get_target(alpha, beta, gamma):
        j_a = alpha % m
        j_b = beta // p
        if j_a != j_b:
            return 0
        # Check if gamma is consistent with i and k
        i = alpha // m
        k = beta % p
        expected_gamma = k * n + i
        if gamma != expected_gamma:
            return 0
        return 1

    def compute_residual(U, V, W):
        # Compute sum_r U[alpha, r] V[beta, r] W[gamma, r] - T[alpha, beta, gamma]
        # We only need to check entries where T is non-zero?
        # No, we need the Frobenius norm over the whole tensor or at least the support.
        # But the tensor is huge (10 * 25 * 10 = 2500 entries).
        # Actually, n=2, m=5, p=5.
        # alpha in [0, 10), beta in [0, 25), gamma in [0, 10).
        # Total entries 2500. This is small.
        # We can just iterate over all alpha, beta, gamma.
        residual = 0.0
        for alpha in range(n * m):
            for beta in range(m * p):
                for gamma in range(p * n):
                    t_val = get_target(alpha, beta, gamma)
                    # sum_r U[alpha, r] V[beta, r] W[gamma, r]
                    # Efficiently:
                    # tmp = U[alpha] * V[beta]
                    # val = np.dot(tmp, W[gamma])
                    # This is O(R) per entry. Total O(N*M*P*R).
                    # 2500 * R. If R=50, 125k ops. Very fast.
                    tmp = U[alpha] * V[beta]
                    val = np.dot(tmp, W[gamma])
                    diff = val - t_val
                    residual += diff * diff
        return residual

    def polish_and_verify(U, V, W, tol=1e-6):
        # Snap to nearest 0.5
        U_p = np.round(U * 2) / 2
        V_p = np.round(V * 2) / 2
        W_p = np.round(W * 2) / 2
        
        # Verify exactness
        # We need integer/half-integer entries.
        # Check if the reconstructed tensor matches the target exactly.
        for alpha in range(n * m):
            for beta in range(m * p):
                for gamma in range(p * n):
                    t_val = get_target(alpha, beta, gamma)
                    tmp = U_p[alpha] * V_p[beta]
                    val = np.dot(tmp, W_p[gamma])
                    # val should be exactly t_val (0 or 1)
                    if abs(val - t_val) > 1e-9:
                        return False
        return True

    def als_solve(R, max_iters=100, tol=1e-8):
        # Initialize randomly
        # Use small integers to start? Or random floats.
        # Random floats are better for ALS.
        U = np.random.randn(n * m, R)
        V = np.random.randn(m * p, R)
        W = np.random.randn(p * n, R)
        
        # Scale to avoid large numbers initially
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        prev_res = float('inf')
        for it in range(max_iters):
            # Update U
            for alpha in range(n * m):
                # Target vector for U[alpha] is T_alpha * (V \otimes W)
                # But T is sparse.
                # We want to minimize sum_{beta, gamma} (sum_r U[alpha, r] V[beta, r] W[gamma, r] - T[alpha, beta, gamma])^2
                # This is a linear least squares problem for U[alpha] given V, W.
                # Let M be a matrix of size (m*p * p*n) x R? No.
                # For fixed alpha, we have variables U[alpha, r].
                # The term is sum_r U[alpha, r] (V[beta, r] W[gamma, r]).
                # So we construct a matrix A_alpha of size (m*p * p*n) x R where
                # A_alpha[beta * p*n + gamma, r] = V[beta, r] * W[gamma, r]
                # And b_alpha[beta * p*n + gamma] = T[alpha, beta, gamma]
                # Then solve A_alpha U[alpha]^T = b_alpha.
                
                A_alpha = np.zeros((m * p * p * n, R))
                b_alpha = np.zeros(m * p * p * n)
                idx = 0
                for beta in range(m * p):
                    for gamma in range(p * n):
                        A_alpha[idx, :] = V[beta] * W[gamma]
                        b_alpha[idx] = get_target(alpha, beta, gamma)
                        idx += 1
                
                # Solve least squares
                # Use np.linalg.lstsq
                U[alpha], _, _, _ = np.linalg.lstsq(A_alpha, b_alpha, rcond=None)
            
            # Update V
            for beta in range(m * p):
                A_beta = np.zeros((n * m * p * n, R))
                b_beta = np.zeros(n * m * p * n)
                idx = 0
                for alpha in range(n * m):
                    for gamma in range(p * n):
                        A_beta[idx, :] = U[alpha] * W[gamma]
                        b_beta[idx] = get_target(alpha, beta, gamma)
                        idx += 1
                V[beta], _, _, _ = np.linalg.lstsq(A_beta, b_beta, rcond=None)
            
            # Update W
            for gamma in range(p * n):
                A_gamma = np.zeros((n * m * m * p, R))
                b_gamma = np.zeros(n * m * m * p)
                idx = 0
                for alpha in range(n * m):
                    for beta in range(m * p):
                        A_gamma[idx, :] = U[alpha] * V[beta]
                        b_gamma[idx] = get_target(alpha, beta, gamma)
                        idx += 1
                W[gamma], _, _, _ = np.linalg.lstsq(A_gamma, b_gamma, rcond=None)
            
            # Check convergence
            res = compute_residual(U, V, W)
            if abs(prev_res - res) < tol:
                break
            prev_res = res
            
        return U, V, W, prev_res

    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school

    # Try ranks from lower bound to schoolbook
    # Lower bound for matrix multiplication tensor rank is max(nm, mp, np)
    # Here max(10, 25, 10) = 25.
    for R in range(25, best_rank):
        if time.time() > deadline:
            break
            
        found = False
        attempts = 0
        max_attempts = 10 # ALS is expensive, limit attempts
        
        while not found and attempts < max_attempts and time.time() < deadline:
            attempts += 1
            U, V, W, res = als_solve(R)
            
            # If residual is small, try to polish
            if res < 1e-4:
                if polish_and_verify(U, V, W):
                    found = True
                    best_rank = R
                    best_U = U.tolist()
                    best_V = V.tolist()
                    best_W = W.tolist()
                    break
        
        if found:
            # If we found a solution for R, we can stop if we want the minimal rank?
            # The loop goes 25, 26...
            # If we find 25, it's the best.
            # If we find 30, we update best_rank to 30 and continue to 31?
            # No, if we found 30, we should check if 29 was missed?
            # The loop is `for R in range(25, best_rank)`.
            # If best_rank becomes 30, the range stops at 29.
            # So it will try 25..29.
            # If it finds 30, it means 25..29 failed.
            # So we can break?
            # Yes, because we are iterating R in increasing order.
            # The first R we find is the minimal R in the range [25, current_best).
            break

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
