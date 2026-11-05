import numpy as np
import time

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    np_ = n * p
    
    # Target tensor T flattened
    # Indices: alpha (0..nm-1), beta (0..mp-1), gamma (0..np_-1)
    # alpha = i*m + j
    # beta = j*p + k
    # gamma = k*n + i  (Note: W[k*n+i, r] in prompt? "output factor follows ... W[k*n+i,r]")
    # Wait, the prompt says: "output factor follows the official trace convention W[k*n+i,r]"
    # This likely refers to the indexing of W in the returned dictionary.
    # The decomposition is:
    # C[i, k] = sum_c U[i, j, c] V[j, k, c] W[k, i, c]?
    # Or is it:
    # C[i, k] = sum_c ( sum_j U[i, j, c] A[i, j] ) ( sum_j V[j, k, c] B[j, k] )? No.
    
    # Let's stick to the standard tensor definition:
    # T[i, j, k] = 1 if i in [0,n), j in [0,m), k in [0,p)
    # The rank is the min r such that T = sum_c u_c v_c w_c
    # u_c is in R^(n*m), v_c in R^(m*p), w_c in R^(n*p)
    # u_c[i, j] = U[i*m+j, c]
    # v_c[j, k] = V[j*p+k, c]
    # w_c[i, k] = W[i*p+k, c]?
    
    # The prompt says: "output factor follows the official trace convention W[k*n+i,r]"
    # This suggests W is indexed by k*n + i.
    # So W[k*n + i, c].
    # This corresponds to w_c[k, i] or w_c[i, k]?
    # Usually w_c is a vector in the output space.
    # If the output is C[i, k], the index is i*p + k.
    # If W is indexed by k*n + i, then W[k*n + i] corresponds to coefficient for C[i, k]?
    # Let's assume the standard mapping:
    # U row: i*m + j
    # V row: j*p + k
    # W row: k*n + i  (as per prompt)
    
    # The tensor entry T[i, j, k] must satisfy:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # for all i, j, k.
    
    # Let's verify this matches C[i, k] = sum_j A[i, j] B[j, k].
    # LHS: sum_c (U[i*m+j, c] V[j*p+k, c] W[k*n+i, c])
    # This sums over c.
    # This does NOT sum over j.
    # In the multiplication, C[i, k] = sum_j A[i, j] B[j, k].
    # The term A[i, j] B[j, k] contributes to C[i, k].
    # So the coefficient of A[i, j] B[j, k] in C[i, k] is 1.
    # The coefficient of A[i, j] B[j, k] in C[i', k'] is 0 if (i', k') != (i, k).
    
    # So the condition is:
    # For a fixed triple (i, j, k):
    # The product A[i, j] B[j, k] should appear in C[i, k] with coeff 1.
    # It should not appear in any other C[i', k'].
    
    # The decomposition form is:
    # C[i, k] = sum_c W[k*n+i, c] * ( sum_{i', j} U[i'*m+j, c] A[i', j] ) * ( sum_{j', k} V[j'*p+k, c] B[j', k] )?
    # No, that's not right. The standard decomposition is:
    # C[i, k] = sum_c P_c(A) Q_c(B)
    # Where P_c, Q_c are linear forms.
    # P_c(A) = sum_{i', j} U_c[i', j] A[i', j]
    # Q_c(B) = sum_{j', k} V_c[j', k] B[j', k]
    # And we need to combine them to get C[i, k].
    # Usually, it is:
    # C[i, k] = sum_c W_c[i, k] P_c(A) Q_c(B)
    
    # In this case, the coefficient of A[i, j] B[j, k] in C[i, k] is:
    # sum_c W_c[i, k] U_c[i, j] V_c[j, k]
    # This must be 1.
    
    # The coefficient of A[i, j] B[j, k] in C[i', k'] (where (i', k') != (i, k)) is:
    # sum_c W_c[i', k'] U_c[i, j] V_c[j, k]
    # This must be 0.
    
    # So the tensor equation is:
    # sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1
    # if we are looking at the term for A[i, j] B[j, k] contributing to C[i, k].
    
    # What about terms where indices don't match?
    # e.g. A[i, j] B[j', k] with j != j'.
    # These do not multiply in the standard algorithm.
    # In the decomposition, P_c(A) includes A[i, j] and Q_c(B) includes B[j', k].
    # Their product is in P_c Q_c.
    # The coefficient in C[i, k] is sum_c W_c[i, k] U_c[i, j] V_c[j', k].
    # This must be 0 for j != j'.
    
    # So the constraints are:
    # 1. For all i, j, k: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1
    # 2. For all i, j, j', k with j != j': sum_c U[i*m+j, c] V[j'*p+k, c] W[k*n+i, c] = 0
    #    (And similarly for mismatches in i or k if they affect the output index)
    #    Actually, if i' != i or k' != k, the output is different.
    #    The term A[i, j] B[j, k] only contributes to C[i, k].
    #    So for any (i, j, j', k), the coefficient of A[i, j] B[j', k] in C[i', k'] is:
    #    sum_c W_c[i', k'] U_c[i, j] V_c[j', k]
    #    This must be 0 unless (i', k') == (i, k) AND j == j'.
    
    # So we have a system of equations for U, V, W.
    # We can solve this using ALS as described.
    
    # Let's define the target tensor T in the flattened index space.
    # The variables are U (nm x r), V (mp x r), W (np_ x r).
    # Note: W index is k*n + i. So row index of W is gamma = k*n + i.
    # U row index alpha = i*m + j.
    # V row index beta = j*p + k.
    
    # The equation for a specific combination of indices (i, j, j', k) is:
    # sum_c U[alpha, c] V[beta, c] W[gamma, c] = Target[i, j, j', k]
    # where alpha = i*m + j, beta = j'*p + k, gamma = k*n + i.
    # Wait, gamma depends on i and k.
    # In the sum for C[i', k'], the weight is W_c[i', k'].
    # So the equation for the coefficient of A[i, j] B[j', k] in C[i, k] is:
    # sum_c U[i*m+j, c] V[j'*p+k, c] W[k*n+i, c] = 1 if j==j', else 0.
    
    # So for each i, j, j', k, we have a linear constraint on the vectors u_c, v_c, w_c.
    # Actually, it's a cubic form.
    
    # ALS Step 1: Fix V, W. Solve for U.
    # For each alpha (i, j), and for each output (i_out, k_out)?
    # No, the equation links alpha, beta, gamma.
    # If we fix V and W, the equation for a specific (beta, gamma) pair?
    # No, for a fixed alpha, we have many equations.
    # Let's group by alpha.
    # For a fixed alpha (i, j), the unknowns are U[alpha, c] for c=1..r.
    # The equations involving U[alpha, c] are:
    # sum_c U[alpha, c] * V[beta, c] * W[gamma, c] = T[alpha, beta, gamma]
    # Here beta and gamma vary.
    # beta = j'*p + k.
    # gamma = k*n + i. (Note: i is from alpha, k is from beta? No, k is in both beta and gamma?)
    # Wait, gamma = k*n + i. i is from alpha. k is from beta?
    # beta = j'*p + k. So k is determined by beta.
    # So for a fixed alpha (i, j) and a fixed beta (j', k), gamma is fixed as k*n + i.
    # So the equation is:
    # sum_c U[alpha, c] * V[beta, c] * W[gamma, c] = T[alpha, beta]
    # where T[alpha, beta] = 1 if j == j', else 0.
    
    # This is a linear system for U[alpha, *].
    # Let M be a matrix of size (num_betas) x r.
    # M[beta, c] = V[beta, c] * W[gamma(beta, alpha), c].
    # Note: gamma depends on alpha.
    # So for each alpha, we construct M_alpha.
    # M_alpha[beta, c] = V[beta, c] * W[k(beta)*n + i(alpha), c].
    # RHS_alpha[beta] = 1 if j(alpha) == j'(beta), else 0.
    # Solve M_alpha * u_alpha = RHS_alpha.
    
    # This works!
    
    # Start time
    start_time = time.time()
    deadline = 120.0 # 120s limit, keep margin
    
    # Try rank 19 first
    ranks_to_try = [19, 20, 21, 22, 23, 24]
    
    best_result = None
    best_error = float('inf')
    
    import random
    np.random.seed(42)
    
    for r in ranks_to_try:
        # Multiple restarts
        num_restarts = 50
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            # Initialize
            U = np.random.randn(nm, r) * 0.1
            V = np.random.randn(mp, r) * 0.1
            W = np.random.randn(np_, r) * 0.1
            
            # ALS Loop
            max_iters = 200
            prev_error = float('inf')
            
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                
                # Update U
                for alpha in range(nm):
                    i = alpha // m
                    j = alpha % m
                    M_alpha = np.zeros((mp, r))
                    for beta in range(mp):
                        j_prime = beta // p
                        k = beta % p
                        gamma = k * n + i
                        M_alpha[beta, :] = V[beta, :] * W[gamma, :]
                    
                    rhs = np.zeros(mp)
                    for beta in range(mp):
                        j_prime = beta // p
                        if j == j_prime:
                            rhs[beta] = 1.0
                    
                    # Solve least squares
                    try:
                        u_alpha = np.linalg.lstsq(M_alpha, rhs, rcond=None)[0]
                        U[alpha, :] = u_alpha
                    except:
                        pass
                
                # Update V
                for beta in range(mp):
                    j_prime = beta // p
                    k = beta % p
                    M_beta = np.zeros((nm, r))
                    for alpha in range(nm):
                        i = alpha // m
                        j = alpha % m
                        gamma = k * n + i
                        M_beta[alpha, :] = U[alpha, :] * W[gamma, :]
                    
                    rhs = np.zeros(nm)
                    for alpha in range(nm):
                        i = alpha // m
                        j = alpha % m
                        if j == j_prime:
                            rhs[alpha] = 1.0
                        # Note: If j != j_prime, rhs is 0.
                        # But wait, does A[i, j] B[j', k] contribute to C[i, k] if j != j'? No.
                        # So rhs is 1 if j == j_prime, else 0.
                    
                    try:
                        v_beta = np.linalg.lstsq(M_beta, rhs, rcond=None)[0]
                        V[beta, :] = v_beta
                    except:
                        pass
                
                # Update W
                for gamma in range(np_):
                    k = gamma // n
                    i = gamma % n
                    M_gamma = np.zeros((nm * mp, r)) # ?
                    # Equation: sum_c U[alpha, c] V[beta, c] W[gamma, c] = T[alpha, beta]
                    # W[gamma, c] is the unknown.
                    # sum_c W[gamma, c] ( U[alpha, c] V[beta, c] ) = T[alpha, beta]
                    # This is a system for W[gamma, *].
                    # Variables: W[gamma, c].
                    # Coefficients: C[alpha, beta, c] = U[alpha, c] * V[beta, c].
                    # There are nm * mp equations.
                    # rhs[alpha, beta] = 1 if j(alpha) == j'(beta) AND i(alpha) == i(gamma) AND k(beta) == k(gamma).
                    # Wait, gamma is fixed (i, k).
                    # So we only care about equations where the output is C[i, k].
                    # The output index is gamma = k*n + i.
                    # The target T[alpha, beta] is non-zero only if alpha corresponds to i and beta corresponds to k?
                    # No, T[alpha, beta] = 1 if j(alpha) == j'(beta).
                    # And it contributes to C[i, k] where i is from alpha and k is from beta.
                    # So for a fixed gamma (i, k), we only include equations where i(alpha) == i(gamma) and k(beta) == k(gamma).
                    
                    M_gamma = np.zeros((m * m, r)) # Only relevant alphas and betas?
                    # Alphas with i(alpha) == i: m of them (varying j).
                    # Betas with k(beta) == k: m of them (varying j').
                    # So m * m = 9 equations.
                    
                    idx = 0
                    rhs_gamma = np.zeros(m * m)
                    for j in range(m):
                        alpha = i * m + j
                        for j_prime in range(m):
                            beta = j_prime * p + k
                            for c in range(r):
                                M_gamma[idx, c] = U[alpha, c] * V[beta, c]
                            
                            if j == j_prime:
                                rhs_gamma[idx] = 1.0
                            else:
                                rhs_gamma[idx] = 0.0
                            idx += 1
                    
                    try:
                        w_gamma = np.linalg.lstsq(M_gamma, rhs_gamma, rcond=None)[0]
                        W[gamma, :] = w_gamma
                    except:
                        pass
                
                # Calculate Error
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        alpha = i * m + j
                        for j_prime in range(m):
                            for k in range(p):
                                beta = j_prime * p + k
                                gamma = k * n + i
                                val = np.dot(U[alpha, :], V[beta, :] * W[gamma, :])
                                target = 1.0 if j == j_prime else 0.0
                                error += (val - target) ** 2
                
                if error < 1e-8:
                    best_result = (r, U, V, W)
                    best_error = error
                    break
                if error > prev_error:
                    # Diverging or stuck
                    pass
                prev_error = error
            
            if best_result is not None:
                break
        
        if best_result is not None:
            break

    if best_result is not None:
        r, U, V, W = best_result
        # Round to half-integers?
        # Check if rounding helps
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        # Verify rounded
        error_rounded = 0.0
        for i in range(n):
            for j in range(m):
                alpha = i * m + j
                for j_prime in range(m):
                    for k in range(p):
                        beta = j_prime * p + k
                        gamma = k * n + i
                        val = np.dot(U_rounded[alpha, :], V_rounded[beta, :] * W_rounded[gamma, :])
                        target = 1.0 if j == j_prime else 0.0
                        error_rounded += (val - target) ** 2
        
        if error_rounded < 1e-8:
            U, V, W = U_rounded, V_rounded, W_rounded
        
        # Convert to lists
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback to schoolbook
        rank = nm * mp # Wait, rank is nm*p? No, rank is number of terms.
        # Schoolbook rank is n*m*p = 24.
        rank = 24
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(np_)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
