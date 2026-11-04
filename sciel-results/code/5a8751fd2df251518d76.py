import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 3, 5
    
    # Schoolbook decomposition: rank 45
    # C_ik = sum_j A_ij * B_jk
    # Tensor: T[i,j,k] = 1 for all i,j,k (the multiplication tensor)
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if (i,j,k) is a valid triple
    
    # Schoolbook: for each (i,j,k), one term
    # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
    
    # Build schoolbook
    best_rank = n * m * p  # 45
    best_U = np.zeros((n*m, best_rank))
    best_V = np.zeros((m*p, best_rank))
    best_W = np.zeros((p*n, best_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i*m+j, col] = 1.0
                best_V[j*p+k, col] = 1.0
                best_W[k*n+i, col] = 1.0
                col += 1
    
    # Try to find lower rank decompositions
    # Known: rank(M_{3,3,3}) = 23 (Landsberg)
    # For M_{3,3,5}, we can split into M_{3,3,3} + M_{3,3,2}
    # rank(M_{3,3,2}) <= 3*3*2 = 18 (schoolbook), but maybe less?
    # Actually rank(M_{n,m,p}) for p=2: rank <= n*m*2, but can we do better?
    # For M_{3,3,2}: C is 3x2, A is 3x3, B is 3x2
    # C_ik = sum_j A_ij B_jk
    # This is like two independent 3x3 multiplications sharing A
    # Rank is at most 3*3*2 = 18
    # Can it be less? The tensor has 3*3*2 = 18 non-zero entries in the "support"
    # Actually each output C_ik depends on all 3 j's, so we need at least... 
    # Let me think: we need sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all i,j,k
    # For fixed k, we have a 3x3 matrix multiplication problem
    # The two k values share the same A (rows of U)
    # So it's not simply 2x rank(M_{3,3,1})
    # rank(M_{3,3,1}) = 3 (since C_i = sum_j A_ij B_j, which is a linear map)
    # Actually M_{n,m,1}: C_i = sum_j A_ij B_j. This is rank n (each C_i is a linear form)
    # Wait, the tensor T[i,j,k] for p=1: T[i,j,0] = 1
    # We need sum_r U[i,j,r] V[j,0,r] W[0,i,r] = 1
    # This is just a rank-1 decomposition of the all-ones matrix in (i,j) for each... 
    # No, it's a 3-tensor. Let me just use schoolbook for p=2: rank 18.
    
    # Strategy: Try to find rank R decompositions for R from 44 down to some minimum
    # Use the block structure: first 3 columns of B/C handled by part 1, last 2 by part 2
    
    # For a given total rank R, let r1 = R - 18 (terms for k=0,1,2) and r2 = 18 (terms for k=3,4)
    # But r1 could be less than 27 if we find a better decomposition for M_{3,3,3}
    
    # Let's try different splits
    # Part 2 (k=3,4): always use schoolbook, rank 18
    # Part 1 (k=0,1,2): try ranks from 26 down to 20
    
    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    if not np.isclose(val, 1.0, atol=1e-8):
                        return False
        return True
    
    def verify_rounded(U, V, W):
        """Verify with rounding to half-integers."""
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        return verify(U_r, V_r, W_r), U_r, V_r, W_r
    
    # Try to find a decomposition with total rank R
    # For each R, we split into r1 terms for k in {0,1,2} and r2 = 18 terms for k in {3,4}
    # r1 = R - 18
    
    # Objective: minimize sum over all (i,j,k) of (sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] - delta_{k<3} * 1)^2
    # Wait, for k in {0,1,2}, the target is 1. For k in {3,4}, the target is 1.
    # But the part 1 terms should only contribute to k in {0,1,2} and part 2 only to k in {3,4}
    # So we enforce: for part 1 terms, V[j,3,r] = V[j,4,r] = 0 and W[3,i,r] = W[4,i,r] = 0
    # And for part 2 terms, V[j,0,r] = V[j,1,r] = V[j,2,r] = 0 and W[0,i,r] = W[1,i,r] = W[2,i,r] = 0
    
    # This makes the problem separable!
    # Part 1: find rank r1 decomposition of M_{3,3,3}
    # Part 2: find rank 18 decomposition of M_{3,3,2} (which is schoolbook)
    
    # So really, we just need to find a rank r1 decomposition of M_{3,3,3} for r1 < 27
    # And then combine with the schoolbook M_{3,3,2}
    
    # Let's focus on finding low-rank decompositions of M_{3,3,3}
    # Known rank is 23, so let's try r1 from 26 down to 23
    
    # For M_{3,3,3}: T[i,j,k] = 1 for i,j,k in {0,1,2}
    # U is 9 x r1, V is 9 x r1, W is 9 x r1
    
    def solve_m333(r1, seed):
        """Try to find rank r1 decomposition of M_{3,3,3}."""
        np.random.seed(seed)
        
        U = np.random.randn(9, r1) * 0.5
        V = np.random.randn(9, r1) * 0.5
        W = np.random.randn(9, r1) * 0.5
        
        # Objective: sum_{i,j,k in {0,1,2}} (sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] - 1)^2
        def objective(x):
            U_f = x[:9*r1].reshape(9, r1)
            V_f = x[9*r1:18*r1].reshape(9, r1)
            W_f = x[18*r1:].reshape(9, r1)
            
            err = 0.0
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        val = np.dot(U_f[i*3+j], V_f[j*3+k] * W_f[k*3+i])
                        err += (val - 1.0)**2
            return err
        
        from scipy.optimize import minimize
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', 
                          options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.fun < 1e-6:
                U_sol = res.x[:9*r1].reshape(9, r1)
                V_sol = res.x[9*r1:18*r1].reshape(9, r1)
                W_sol = res.x[18*r1:].reshape(9, r1)
                
                # Try rounding to half-integers
                ok, U_r, V_r, W_r = verify_rounded(U_sol, V_sol, W_sol)
                if ok:
                    return U_r, V_r, W_r
                
                # Check if exact without rounding
                if verify(U_sol, V_sol, W_sol):
                    return U_sol, V_sol, W_sol
        except Exception:
            pass
        
        return None
    
    # Try to find M_{3,3,3} decomposition
    m333_sol = None
    for r1 in range(26, 19, -1):  # Try ranks 26, 25, ..., 20
        if time.time() > deadline - 40:
            break
        
        for seed in range(10):
            if time.time() > deadline - 40:
                break
            
            sol = solve_m333(r1, seed)
            if sol is not None:
                m333_sol = (r1, sol)
                break
        
        if m333_sol is not None:
            break
    
    # Build final solution
    if m333_sol is not None:
        r1, (U1, V1, W1) = m333_sol
        r2 = 18
        total_rank = r1 + r2
        
        # Build full U, V, W
        U_full = np.zeros((n*m, total_rank))
        V_full = np.zeros((m*p, total_rank))
        W_full = np.zeros((p*n, total_rank))
        
        # Part 1: M_{3,3,3} terms (columns 0 to r1-1)
        # U1 is 9 x r1, maps (i,j) for i,j in {0,1,2}
        # V1 is 9 x r1, maps (j,k) for j,k in {0,1,2}
        # W1 is 9 x r1, maps (k,i) for k,i in {0,1,2}
        
        for i in range(3):
            for j in range(3):
                U_full[i*3+j, :r1] = U1[i*3+j, :]
        
        for j in range(3):
            for k in range(3):
                V_full[j*5+k, :r1] = V1[j*3+k, :]
        
        for k in range(3):
            for i in range(3):
                W_full[k*3+i, :r1] = W1[k*3+i, :]
        
        # Part 2: M_{3,3,2} schoolbook terms (columns r1 to r1+r2-1)
        col = r1
        for i in range(3):
            for j in range(3):
                for k in range(3, 5):
                    U_full[i*3+j, col] = 1.0
                    V_full[j*5+k, col] = 1.0
                    W_full[k*3+i, col] = 1.0
                    col += 1
        
        # Verify
        if verify(U_full, V_full, W_full):
            return {
                "U": U_full.tolist(),
                "V": V_full.tolist(),
                "W": W_full.tolist()
            }
        
        # Try rounding
        ok, U_r, V_r, W_r = verify_rounded(U_full, V_full, W_full)
        if ok:
            return {
                "U": U_r.tolist(),
                "V": V_r.tolist(),
                "W": W_r.tolist()
            }
    
    # Fallback: schoolbook rank 45
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
