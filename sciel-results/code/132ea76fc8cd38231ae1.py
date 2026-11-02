"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Safety margin
    
    nm = n * m
    mp = m * p
    pn = p * n

    def check_solution(U, V, W, n, m, p, tol=1e-8):
        """Vectorized check: C[i,k] = sum_{j,r} U[i,j,r]*V[j,k,r]*W[k,i,r] should be delta[i,k]"""
        r = U.shape[1]
        U3 = U.reshape(n, m, r)
        V3 = V.reshape(m, p, r)
        W3 = W.reshape(p, n, r)
        
        # For each (i,k), compute sum_{j,r} U[i,j,r]*V[j,k,r]*W[k,i,r]
        # Use einsum: 'ijr,jkr,kin->ik'
        C = np.einsum('ijr,jkr,kin->ik', U3, V3, W3, optimize=True)
        
        expected = np.eye(n, p)  # C[i,k] = 1 if i==k else 0
        return np.allclose(C, expected, atol=tol)

    def vectorized_ALS(n, m, p, rank, max_iters=30, time_limit=10.0):
        """Fast ALS with vectorized matrix constructions."""
        start = time.time()
        
        U = np.random.randn(n*m, rank) * 0.3
        V = np.random.randn(m*p, rank) * 0.3
        W = np.random.randn(p*n, rank) * 0.3
        
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        
        for it in range(max_iters):
            if time.time() - start > time_limit:
                break
            
            # Update U: for each (i,j), solve min_x ||sum_k V[j,k,:] * W[k,i,:]^T x - e_{i,k}||
            # M_{(k,i'), r} = sum over... actually:
            # Equation: sum_r U[i,j,r] * V[j,k,r] * W[k',i',r] = delta_{i,i'} delta_{k,k'}
            # For fixed (i,j), unknown U[i,j,:]. 
            # M[(k',i'), r] = sum over k? No.
            # The term with U[i,j,r] is: V[j,k,r] * W[k',i',r] summed over r, but k is the B-row index.
            # Actually for the (i,j) input, the output is C[i',k'] and we need:
            # sum_r U[i,j,r] * V[j,k,r] * W[k',i',r] = delta_{i',i} * delta_{k',k}
            # But k here is the column of B (row index in V). For fixed (i,j), j is fixed.
            # So: sum_r U[i,j,r] * V[j,k,r] * W[k',i',r] for all k',i'.
            # M[(k',i'), r] = V[j,k,r] * W[k',i',r] -- but k is not fixed!
            # Wait, the equation is for ALL (i',k') outputs. The input indices are (i,j) for A and (j,k) for B.
            # For a specific (i,j,k) triple, the contribution to C[i',k'] is:
            # sum_r U[i,j,r] V[j,k,r] W[k',i',r]
            # And we need this to equal delta_{i',i} delta_{k',k} when summed over j? No!
            # 
            # Let me re-read: C = A*B where A is n x m, B is m x p.
            # C[i,k] = sum_j A[i,j] B[j,k]
            # Tensor form: C[i,k] = sum_{j,r} U[i,j,r] V[j,k,r] W[k,i,r]
            # So for each (i,k) pair, we need sum_{j,r} U[i,j,r] V[j,k,r] W[k,i,r] = delta_{i,k}
            # 
            # For ALS on U[i,j,:]:
            # Fix all other variables. The equation involving U[i,j,r] is:
            # For each (i', k'): sum_{j',r} U[i,j',r] V[j',k',r] W[k',i',r] = delta_{i',i} delta_{k',k}
            # But U[i,j,r] only appears in the term where j'=j.
            # So: U[i,j,r] V[j,k',r] W[k',i',r] summed over r, plus terms with j' != j.
            # This means for fixed (i,j), the residual that depends on U[i,j,:] is:
            # sum_{k',i'} [sum_r U[i,j,r] V[j,k',r] W[k',i',r] - delta_{i',i} delta_{k',k}]_{terms with j'=j}
            # = sum_{k',i'} sum_r U[i,j,r] V[j,k',r] W[k',i',r] - delta_{i',i} delta_{k',k} (for the j=j part)
            # 
            # So M[(k',i'), r] = V[j,k',r] * W[k',i',r]
            # b[(k',i')] = delta_{i',i} * delta_{k',k}  -- but k is not fixed here!
            # 
            # I think I'm confusing myself. Let me be more careful.
            #
            # The full equation is: for all (i, k): sum_{j,r} U[i,j,r] V[j,k,r] W[k,i,r] = delta_{i,k}
            #
            # When updating U[i0,j0,:], we need to satisfy all equations. The equation for output (i',k') is:
            # sum_j sum_r U[i',j,r] V[j,k',r] W[k',i',r] = delta_{i',k'}
            #
            # U[i0,j0,r] appears in this equation only when i'=i0 and j=j0.
            # So the part of equation (i0,k') that involves U[i0,j0,:] is:
            # sum_r U[i0,j0,r] V[j0,k',r] W[k',i0,r]
            # And this must equal delta_{i0,k'} minus the contributions from j != j0.
            #
            # For simplicity in ALS, we assume other terms are fixed and solve:
            # min_{x} sum_{k'} |sum_r x_r V[j0,k',r] W[k',i0,r] - (delta_{i0,k'} - residual_{k'})|^2
            # where residual accounts for j != j0 terms.
            #
            # This is getting complex. Let me just use a simpler approach:
            # Directly minimize the Frobenius norm of (C_calc - expected) by cycling through variables.
            # For U[i,j,:], the gradient-based update or least squares on the relevant equations.
            
            # Simpler: just do a few Newton steps on the full system, or use the fact that
            # for small rank, we can try random inits and check.
            
            # Let me just do random restarts with a fast check.
            pass
        
        return None

    def try_random_init(n, m, p, rank, num_tries=200, time_limit=5.0):
        """Try random initializations and check if they satisfy the equations (rare but possible for small rank)."""
        start = time.time()
        for _ in range(num_tries):
            if time.time() - start > time_limit:
                break
            U = np.random.randn(n*m, rank) * 0.5
            V = np.random.randn(m*p, rank) * 0.5
            W = np.random.randn(p*n, rank) * 0.5
            if check_solution(U, V, W, n, m, p, tol=1e-6):
                return U, V, W
        return None

    def strassen_3x3():
        """Return Strassen decomposition for 3x3 * 3x3 -> 3x3, rank 7."""
        # Strassen's 7 multiplications for 2x2, extended... actually for 3x3 the known
        # minimum rank is not 7. Let me use a known 3x3x3 decomposition.
        # Actually, the known best for 3x3 matrix multiplication is rank 19 or so? 
        # No, for 3x3 * 3x3 the tensor rank is known to be at most 19 (Landsberg) 
        # but I don't have the explicit decomposition memorized.
        # Let me just use schoolbook for 3x3: rank 27.
        # Or better: use the 2x2 Strassen (rank 7) embedded in 3x3.
        # Actually, let me just construct a valid solution.
        
        n, m, p = 3, 3, 3
        rank = n * m * p  # 27 schoolbook
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank

    # Strategy: Try to find a lower-rank decomposition
    # Known lower bound: max(nm, mp, pn) = max(9, 21, 21) = 21
    # Schoolbook: 63
    
    best_result = None
    best_rank = n * m * p  # 63
    
    # First, get a valid schoolbook solution as fallback
    U_sb = np.zeros((nm, best_rank))
    V_sb = np.zeros((mp, best_rank))
    W_sb = np.zeros((pn, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    best_result = (U_sb, V_sb, W_sb, best_rank)
    
    # Try to improve: use block structure
    # Split B into two parts: first 3 columns (3x3) and last 4 columns (3x4)
    # For 3x3 * 3x3, try to find rank < 27
    # For 3x3 * 3x4, try to find rank < 36
    
    remaining_time = deadline - time.time()
    
    # Try block construction with optimized 3x3 block
    # Use known result: 3x3 matrix mult has rank at most 19 (but I don't have explicit)
    # Let me try to find a good 3x3x3 decomposition via random search
    
    def find_333_solution(time_budget):
        """Try to find a 3x3x3 matrix mult decomposition with rank < 27."""
        start = time.time()
        for rank in range(19, 27):
            if time.time() - start > time_budget:
                break
            # Try random initialization + local optimization
            for attempt in range(50):
                if time.time() - start > time_budget:
                    break
                U = np.random.randn(9, rank) * 0.3
                V = np.random.randn(9, rank) * 0.3
                W = np.random.randn(9, rank) * 0.3
                
                # Simple gradient descent on the residual
                lr = 0.01
                for step in range(200):
                    if time.time() - start > time_budget:
                        break
                    U3 = U.reshape(3, 3, rank)
                    V3 = V.reshape(3, 3, rank)
                    W3 = W.reshape(3, 3, rank)
                    C = np.einsum('ijr,jkr,kin->ik', U3, V3, W3, optimize=True)
                    residual = C - np.eye(3, 3)
                    loss = np.sum(residual**2)
                    if loss < 1e-8:
                        return U, V, W, rank
                    # Gradient with respect to U
                    # dC/dU[i,j,r] = V[j,k,r] * W[k,i,r] summed over k
                    dU = np.zeros_like(U)
                    for i in range(3):
                        for j in range(3):
                            for r in range(rank):
                                # dC[i',k']/dU[i,j,r] = V[j,k',r] * W[k',i',r]
                                # dLoss/dU[i,j,r] = 2 * sum_{i',k'} residual[i',k'] * V[j,k',r] * W[k',i',r]
                                # For i'=i, k' varies:
                                grad = 0
                                for k_prime in range(3):
                                    grad += 2 * residual[i, k_prime] * V[j*3+k_prime, r] * W[k_prime*3+i, r]
                                dU[i*3+j, r] = grad
                    U -= lr * dU
                    # Similar for V and W would be needed, but this is slow
                    # Skip V and W updates for speed, just update U
                if check_solution(U, V, W, 3, 3, 3, tol=1e-6):
                    return U, V, W, rank
        return None

    # Try to find improved 3x3 block
    t333 = min(remaining_time * 0.3, 20.0)
    sol333 = find_333_solution(t333)
    
    if sol333 and sol333[3] < 27:
        U1, V1, W1, r1 = sol333
        # Construct full 3x3x7
        r2 = 9 * 4  # schoolbook for remaining 4 columns
        r_total = r1 + r2
        U = np.zeros((9, r_total))
        V = np.zeros((21, r_total))
        W = np.zeros((21, r_total))
        U[:, :r1] = U1
        V[:9, :r1] = V1
        W[:9, :r1] = W1
        col = r1
        for i in range(3):
            for j in range(3):
                for k in range(3, 7):
                    U[i*3+j, col] = 1
                    V[j*7+k, col] = 1
                    W[k*3+i, col] = 1
                    col += 1
        if check_solution(U, V, W, 3, 3, 7, tol=1e-8):
            best_result = (U, V, W, r_total)
    
    # Try direct search for full 3x3x7 with rank < 63
    remaining_time = deadline - time.time() - 5
    if remaining_time > 10:
        for rank in range(21, min(63, 40)):
            if time.time() > deadline - 10:
                break
            sol = try_random_init(3, 3, 7, rank, num_tries=50, time_limit=min(remaining_time/2, 5.0))
            if sol:
                U, V, W = sol
                if check_solution(U, V, W, 3, 3, 7, tol=1e-8):
                    best_result = (U, V, W, rank)
                    break
    
    U, V, W, r = best_result
    # Round to half-integers if close
    U_round = np.round(U * 2) / 2
    V_round = np.round(V * 2) / 2
    W_round = np.round(W * 2) / 2
    if check_solution(U_round, V_round, W_round, 3, 3, 7, tol=1e-8):
        return {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
