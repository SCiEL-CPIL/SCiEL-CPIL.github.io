"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 7
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Schoolbook as fallback
    rank_schoolbook = n * m * p
    def schoolbook():
        U = [[0] * rank_schoolbook for _ in range(n * m)]
        V = [[0] * rank_schoolbook for _ in range(m * p)]
        W = [[0] * rank_schoolbook for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = rank_schoolbook
    
    def verify(U, V, W, r):
        """Verify the decomposition: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True
    
    def try_rank_linear(r, max_restarts=30, seed=42):
        """
        Slice-and-fiber approach: 
        For the multiplication tensor T[i,j,k] = 1 for all i,j,k,
        we need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        
        Key insight: For fixed W, the constraint becomes linear in U and V.
        Specifically, for each k, define W_k[i,j] = W[k*n+i, c] * V[j*p+k, c] summed over c...
        
        Actually, let me think more carefully.
        
        For a fixed rank r, we have:
        sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
        
        Let's fix the columns of W. For each c, W_c is a vector of length p*n = 21.
        W_c[k*n+i] gives the coefficient for output (i,k) at rank term c.
        
        Alternative: Fix U and W, solve for V. Or fix V and W, solve for U.
        
        Actually, here's a cleaner approach:
        For each k (0 to p-1), and for each i (0 to n-1), the equation is:
        sum_j sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        
        Let's define for each k and c: 
        a_{k,c,j} = V[j*p+k, c]  (length m for each k,c)
        b_{k,c,i} = W[k*n+i, c]  (length n for each k,c)
        
        Then the equation for output (i,k) is:
        sum_c b_{k,c,i} * sum_j U[i*m+j,c] * a_{k,c,j} = 1
        
        Let's define for each c and k: 
        M_c[k] = sum_j U[i*m+j,c] * V[j*p+k,c] ... no, this depends on i.
        
        Let me try yet another approach. 
        
        For fixed c, define:
        - u_c: vector in R^{n*m} (the c-th column of U)
        - v_c: vector in R^{m*p} (the c-th column of V)  
        - w_c: vector in R^{p*n} (the c-th column of W)
        
        The constraint is: sum_c (u_c ⊗ v_c ⊗ w_c)[i,j,k] = 1 for all i,j,k
        
        where (u_c ⊗ v_c ⊗ w_c)[i,j,k] = u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i]
        
        Let me try: fix U and V randomly, then for each output index (i,k), we need:
        sum_c W[k*n+i,c] * (sum_j U[i*m+j,c] * V[j*p+k,c]) = 1
        
        Define S[i,k,c] = sum_j U[i*m+j,c] * V[j*p+k,c]
        Then we need: sum_c W[k*n+i,c] * S[i,k,c] = 1 for all i,k
        
        This is a linear system in W! For each (i,k), we have one equation with r unknowns W[k*n+i,c].
        But wait, W[k*n+i,c] appears in the equation for output (i,k) with coefficient S[i,k,c].
        
        Actually, for a fixed (i,k), the equation is:
        sum_c W[k*n+i,c] * S[i,k,c] = 1
        
        This gives us n*p = 21 equations, each with r unknowns. But the unknowns are W[k*n+i,c] for all i,k,c.
        So we have 21 * r unknowns and 21 equations. This is underdetermined if r > 1.
        
        Wait, I'm confusing myself. Let me re-index.
        
        The unknowns are W[a,c] where a = k*n+i ranges from 0 to p*n-1, and c ranges from 0 to r-1.
        Total unknowns: p*n*r = 21*r
        
        The equations: for each (i,k), sum_c W[k*n+i,c] * S[i,k,c] = 1
        Total equations: n*p = 21
        
        So we have 21 equations and 21*r unknowns. If r >= 1, this is underdetermined.
        We can solve it by least squares or by setting most W entries to 0.
        
        But wait, S[i,k,c] depends on U and V which are also unknowns. So we can't fix U and V and solve for W independently.
        
        Let me try a different approach: alternating minimization.
        
        1. Initialize U, V, W randomly.
        2. Fix U and V, solve for W using least squares.
        3. Fix V and W, solve for U using least squares.
        4. Fix U and W, solve for V using least squares.
        5. Repeat until convergence.
        
        For step 2: Given U and V, compute S[i,k,c] = sum_j U[i*m+j,c] * V[j*p+k,c] for all i,k,c.
        Then solve: for each (i,k), sum_c W[k*n+i,c] * S[i,k,c] = 1
        This is a linear system. We can write it as:
        Let W be reshaped to (p*n, r). For each (i,k), the row of the system is S[i,k,:] (length r).
        The full system is: A @ W_vec = b, where A is (n*p) x (p*n*r) and b is (n*p) x 1.
        
        Actually, it's simpler: for each (i,k), we have one equation:
        dot(W[k*n+i, :], S[i,k, :]) = 1
        
        This can be solved by setting W[k*n+i, :] = S[i,k, :] / (S[i,k,:] @ S[i,k,:]) if the norm is nonzero.
        But this only works if we can satisfy all equations simultaneously, which requires that the S[i,k,:] vectors are consistent.
        
        Hmm, this is getting complicated. Let me just try the alternating least squares approach directly.
        
        For fixed U and V:
        For each (i,k), we need W[k*n+i, :] such that dot(W[k*n+i, :], S[i,k, :]) = 1
        The minimum-norm solution is W[k*n+i, :] = S[i,k, :] / ||S[i,k, :]||^2
        
        For fixed V and W:
        For each (i,j,k), the contribution to the equation for output (i,k) from term c is:
        U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        Summing over j and c: sum_c W[k*n+i,c] * (sum_j U[i*m+j,c] * V[j*p+k,c]) = 1
        
        Let T[i,k,c] = sum_j V[j*p+k,c] * W[k*n+i,c] ... no, that's not right either.
        
        Let me just implement the alternating approach numerically.
        """
        if time.time() > deadline:
            return None
        
        rng = np.random.RandomState(seed)
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
            
            # Initialize
            U = rng.randn(n * m, r)
            V = rng.randn(m * p, r)
            W = rng.randn(p * n, r)
            
            # Alternating minimization
            for it in range(200):
                if time.time() > deadline:
                    return None
                
                # Step 1: Fix U and V, solve for W
                # For each (i,k): sum_c W[k*n+i,c] * (sum_j U[i*m+j,c] * V[j*p+k,c]) = 1
                S = np.zeros((n, p, r))
                for i in range(n):
                    for k in range(p):
                        for c in range(r):
                            S[i, k, c] = sum(U[i*m+j, c] * V[j*p+k, c] for j in range(m))
                
                # For each (i,k), solve: dot(W[k*n+i, :], S[i,k, :]) = 1
                # Minimum norm solution: W[k*n+i, :] = S[i,k, :] / (S[i,k,:] @ S[i,k,:])
                for i in range(n):
                    for k in range(p):
                        idx = k * n + i
                        norm_sq = np.dot(S[i, k, :], S[i, k, :])
                        if norm_sq > 1e-12:
                            W[idx, :] = S[i, k, :] / norm_sq
                        else:
                            # If S is zero, we can't satisfy the equation. Set W to something.
                            W[idx, :] = 0
                
                # Step 2: Fix V and W, solve for U
                # For each (i,k): sum_c W[k*n+i,c] * (sum_j U[i*m+j,c] * V[j*p+k,c]) = 1
                # => sum_j sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                # Let R[i,j,c] = V[j*p+k,c] * W[k*n+i,c] for each k... 
                # Actually, for fixed (i,k), the equation is:
                # sum_j sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1
                # Let Q[i,k,j,c] = V[j*p+k,c] * W[k*n+i,c]
                # Then: sum_j sum_c U[i*m+j,c] * Q[i,k,j,c] = 1
                # This is a linear system in U[i*m+0,c], ..., U[i*m+m-1,c] for each c.
                # But it's coupled across c.
                
                # For fixed (i,k), define for each j: 
                # sum_c U[i*m+j,c] * Q[i,k,j,c] = t_{i,k,j}
                # and sum_j t_{i,k,j} = 1
                
                # This is underdetermined. Let's use least squares.
                # For each (i,k), we have m*r unknowns U[i*m+0,0], ..., U[i*m+m-1,r-1]
                # and one equation: sum_{j,c} U[i*m+j,c] * Q[i,k,j,c] = 1
                
                # Minimum norm solution: U[i*m+j,c] = Q[i,k,j,c] / (sum_{j',c'} Q[i,k,j',c']^2)
                for i in range(n):
                    for k in range(p):
                        Q = np.zeros((m, r))
                        for j in range(m):
                            for c in range(r):
                                Q[j, c] = V[j*p+k, c] * W[k*n+i, c]
                        norm_sq = np.sum(Q * Q)
                        if norm_sq > 1e-12:
                            for j in range(m):
                                for c in range(r):
                                    U[i*m+j, c] = Q[j, c] / norm_sq
                        else:
                            for j in range(m):
                                for c in range(r):
                                    U[i*m+j, c] = 0
                
                # Step 3: Fix U and W, solve for V
                # For each (i,k): sum_j sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                # => sum_c sum_j U[i*m+j,c] * W[k*n+i,c] * V[j*p+k,c] = 1
                # Let P[i,k,j,c] = U[i*m+j,c] * W[k*n+i,c]
                # Then: sum_{j,c} V[j*p+k,c] * P[i,k,j,c] = 1
                for k in range(p):
                    for j in range(m):
                        for c in range(r):
                            # V[j*p+k,c] appears in equations for all i
                            pass
                
                # For fixed (j,k), V[j*p+k,c] appears in the equation for output (i,k) for all i.
                # The equation for output (i,k) is: sum_{j',c'} V[j'*p+k,c'] * U[i*m+j',c'] * W[k*n+i,c'] = 1
                # For fixed (j,k), the contribution of V[j*p+k,c] to the equation for (i,k) is:
                # V[j*p+k,c] * U[i*m+j,c] * W[k*n+i,c]
                
                # This is more complex. Let's just do a simple update:
                # For each (j,k,c), V[j*p+k,c] should be proportional to the sum over i of 
                # U[i*m+j,c] * W[k*n+i,c] weighted by the residual.
                
                # Actually, let's just do a gradient step for V.
                # Residual for (i,k): res[i,k] = 1 - sum_{j,c} U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                # Gradient of sum_{i,k} res[i,k]^2 with respect to V[j*p+k,c] is:
                # -2 * sum_i res[i,k] * U[i*m+j,c] * W[k*n+i,c]
                
                lr = 0.01
                for j in range(m):
                    for k in range(p):
                        for c in range(r):
                            grad = 0.0
                            for i in range(n):
                                res_ik = 1.0 - sum(U[i*m+j2, c2] * V[j2*p+k, c2] * W[k*n+i, c2] for j2 in range(m) for c2 in range(r))
                                grad -= 2.0 * res_ik * U[i*m+j, c] * W[k*n+i, c]
                            V[j*p+k, c] -= lr * grad
                
                # Check convergence
                max_res = 0.0
                for i in range(n):
                    for k in range(p):
                        val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for j in range(m) for c in range(r))
                        max_res = max(max_res, abs(val - 1.0))
                
                if max_res < 1e-8:
                    # Found a solution
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    return U_list, V_list, W_list
        
        return None
    
    def try_rank_gradient(r, max_iters=3000, seed=42):
        """Gradient-based search for rank r decomposition."""
        if time.time() > deadline:
            return None
        
        rng = np.random.RandomState(seed)
        
        U = rng.randn(n * m, r)
        V = rng.randn(m * p, r)
        W = rng.randn(p * n, r)
        
        def compute_residual(U, V, W):
            res = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    val = 0.0
                    for j in range(m):
                        for c in range(r):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    res[i, k] = val - 1.0
            return res
        
        def residual_norm(U, V, W):
            res = compute_residual(U, V, W)
            return np.linalg.norm(res)
        
        best_norm = residual_norm(U, V, W)
        best_U = U.copy()
        best_V = V.copy()
        best_W = W.copy()
        
        lr = 0.01
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            if best_norm < 1e-10:
                return best_U.tolist(), best_V.tolist(), best_W.tolist()
            
            # Compute gradients
            res = compute_residual(U, V, W)
            
            # Gradient with respect to U[i*m+j, c]:
            # d/dU[i*m+j,c] sum_{i',k'} res[i',k']^2 = 2 * sum_{k'} res[i,k'] * V[j*p+k',c] * W[k'*n+i,c]
            dU = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    for c in range(r):
                        grad = 0.0
                        for k in range(p):
                            grad += 2.0 * res[i, k] * V[j*p+k, c] * W[k*n+i, c]
                        dU[i*m+j, c] = grad
            
            # Gradient with respect to V[j*p+k, c]:
            dV = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    for c in range(r):
                        grad = 0.0
                        for i in range(n):
                            grad += 2.0 * res[i, k] * U[i*m+j, c] * W[k*n+i, c]
                        dV[j*p+k, c] = grad
            
            # Gradient with respect to W[k*n+i, c]:
            dW = np.zeros_like(W)
            for i in range(n):
                for k in range(p):
                    for c in range(r):
                        grad = 0.0
                        for j in range(m):
                            grad += 2.0 * res[i, k] * U[i*m+j, c] * V[j*p+k, c]
                        dW[k*n+i, c] = grad
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            new_norm = residual_norm(U, V, W)
            if new_norm < best_norm:
                best_norm = new_norm
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            else:
                # Revert
                U = best_U.copy()
                V = best_V.copy()
                W = best_W.copy()
                lr *= 0.5
                if lr < 1e-6:
                    break
        
        if best_norm < 1e-8:
            return best_U.tolist(), best_V.tolist(), best_W.tolist()
        return None
    
    # Try ranks from lower bound upward
    lower_bound = max(n * m, m * p, p * n)
    
    for r in range(lower_bound, rank_schoolbook + 1):
        if time.time() > deadline:
            break
        
        # Try linear approach first
        result = try_rank_linear(r, max_restarts=20, seed=r)
        if result is not None:
            U_list, V_list, W_list = result
            U_out = [[float(x) for x in row] for row in U_list]
            V_out = [[float(x) for x in row] for row in V_list]
            W_out = [[float(x) for x in row] for row in W_list]
            if verify(U_out, V_out, W_out, r):
                best_rank = r
                best_result = {"U": U_out, "V": V_out, "W": W_out}
                break
        
        # If linear didn't work, try gradient
        result = try_rank_gradient(r, max_iters=1000, seed=r)
        if result is not None:
            U_list, V_list, W_list = result
            U_out = [[float(x) for x in row] for row in U_list]
            V_out = [[float(x) for x in row] for row in V_list]
            W_out = [[float(x) for x in row] for row in W_list]
            if verify(U_out, V_out, W_out, r):
                best_rank = r
                best_result = {"U": U_out, "V": V_out, "W": W_out}
                break
    
    return best_result
# EVOLVE-BLOCK-END
