import numpy as np
import time

def construct():
    n, m, p = 2, 3, 3
    deadline = time.time() + 140
    
    # Schoolbook rank is 18. Try to find rank 17.
    # The tensor T has entries T[i,j,k] = 1 for all (i,j,k) in [n]x[m]x[p]
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Try explicit rank-17 construction using Winograd-style trick
    # For the j-dimension (size 3), use: a0*b0 + a1*b1 + a2*b2
    # = ((a0+a1+a2)*(b0+b1+b2) + (a0-a2)*(b0-b2) + 2*(a0*b0 + a1*b1 + a2*b2 - ...))
    # Actually, let's use: sum_{j=0}^2 a_j b_j where we have freedom in how U,V,W couple
    
    # Key insight: U depends on (i,j), V on (j,k), W on (k,i)
    # So for fixed (i,k), we need sum_j U[i,j,r] * V[j,k,r] * W[k,i,r] summed over r to give 1
    # But W[k,i,r] is common factor for all j in the inner sum!
    # So: sum_r W[k,i,r] * (sum_j U[i,j,r] * V[j,k,r]) = 1
    
    # This means for each (i,k), the vector w_{ik} = [W[k,i,r]]_r must be such that
    # sum_r W[k,i,r] * dot(U_i[:,r], V_k[:,r]) = 1 where U_i is row i of U reshaped, etc.
    
    # Let me try a direct construction. For rank 17, we need 17 terms.
    # Start with schoolbook (18 terms) and try to merge two terms.
    
    # Schoolbook: for each (i,j,k), one term with U[i*m+j, r]=1, V[j*p+k, r]=1, W[k*n+i, r]=1
    # where r = i*m*p + j*p + k
    
    # To reduce rank, we need to find two (i,j,k) pairs that can share a rank-1 term
    # This requires that for some (i1,j1,k1) and (i2,j2,k2), we can write
    # U[i1*m+j1, r] * V[j1*p+k1, r] * W[k1*n+i1, r] + U[i2*m+j2, r] * V[j2*p+k2, r] * W[k2*n+i2, r]
    # as a single rank-1 term, which is generally impossible unless there's structure.
    
    # Alternative: use the fact that the tensor has symmetries.
    # Let me try a parametrized family and solve.
    
    # For rank r, we have variables U (6*r), V (9*r), W (6*r) = 21*r variables
    # Constraints: 18 equations (one for each (i,j,k))
    # For r=17: 357 variables, 18 equations - very underdetermined
    
    # Let me try a structured approach: assume W has a simple form.
    # If W[k*n+i, r] = 1 for all k,i,r, then we need sum_r U[i*m+j,r]*V[j*p+k,r] = 1
    # This is a matrix equation: for each (i,k), sum_j U[i,j,:] * V[j,k,:] (element-wise product summed) = 1
    
    # Let U[i,j,r] and V[j,k,r] be such that for each (i,k):
    # sum_r sum_j U[i,j,r] * V[j,k,r] = 1
    
    # If we set U[i,j,r] = u_{i,j,r} and V[j,k,r] = v_{j,k,r}, this is hard.
    
    # Let me try: U[i,j,r] = a_{i,r} * b_{j,r}, V[j,k,r] = c_{j,r} * d_{k,r}, W[k,i,r] = e_{k,r} * f_{i,r}
    # Then sum_r a_{i,r} b_{j,r} c_{j,r} d_{k,r} e_{k,r} f_{i,r} = sum_r (a_{i,r} f_{i,r}) (b_{j,r} c_{j,r}) (d_{k,r} e_{k,r}) = 1
    # This factors as (A_i) (B_j) (C_k) = 1 for all i,j,k, which means A_i=B_j=C_k=1 (or all zero with sum=1, impossible)
    # So this ansatz gives rank 1 only if A_i B_j C_k = 1, i.e., A_i=1, B_j=1, C_k=1, rank 1. Not useful.
    
    # Let me just try numerical optimization for rank 17 with a good initialization.
    
    for r in [17, 18]:
        if time.time() > deadline - 30:
            break
        
        # Multiple random restarts
        max_restarts = 50 if r == 17 else 20
        for restart in range(max_restarts):
            if time.time() > deadline - 25:
                break
            
            seed = restart * 1000003 + r * 7919 + 42
            np.random.seed(seed)
            
            # Initialize with schoolbook-like structure
            U = np.zeros((n*m, r))
            V = np.zeros((m*p, r))
            W = np.zeros((p*n, r))
            
            if r >= 18:
                # Schoolbook
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if col < r:
                                U[i*m+j, col] = 1
                                V[j*p+k, col] = 1
                                W[k*n+i, col] = 1
                                col += 1
                if col <= r:
                    # Verify
                    ok = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = 0
                                for rr in range(r):
                                    val += U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr]
                                if abs(val - 1) > 1e-10:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            break
                    if ok:
                        U_list = [[float(val) for val in row] for row in U]
                        V_list = [[float(val) for val in row] for row in V]
                        W_list = [[float(val) for val in row] for row in W]
                        return {"U": U_list, "V": V_list, "W": W_list}
            else:
                # Random init for r=17
                U = np.random.randn(n*m, r) * 0.5
                V = np.random.randn(m*p, r) * 0.5
                W = np.random.randn(p*n, r) * 0.5
            
            # Use gradient descent on the residual
            def compute_residual(params):
                U_f = params[:n*m*r].reshape(n*m, r)
                V_f = params[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
                W_f = params[n*m*r+m*p*r:].reshape(p*n, r)
                res = np.empty(n*m*p)
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            res[idx] = np.sum(U_f[i*m+j, :] * V_f[j*p+k, :] * W_f[k*n+i, :]) - 1
                            idx += 1
                return res
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            res0 = compute_residual(x0)
            cost0 = np.sum(res0**2)
            
            # Simple gradient descent with line search
            lr = 0.01
            best_cost = cost0
            best_x = x0.copy()
            stagnant = 0
            
            for step in range(2000):
                if time.time() > deadline - 20:
                    break
                
                # Compute gradient numerically (expensive but r=17, 21*17=357 vars)
                # Use finite differences for gradient
                grad = np.zeros_like(x0)
                eps = 1e-5
                # Only compute gradient every few steps to save time
                if step % 5 == 0 or step < 50:
                    for idx in range(0, len(x0), 10):  # Sample gradient
                        for d in range(10):
                            if idx + d >= len(x0):
                                break
                            x_plus = x0.copy()
                            x_plus[idx+d] += eps
                            x_minus = x0.copy()
                            x_minus[idx+d] -= eps
                            grad[idx+d] = (np.sum(compute_residual(x_plus)**2) - np.sum(compute_residual(x_minus)**2)) / (2*eps)
                
                # Update
                x0 = x0 - lr * grad
                res = compute_residual(x0)
                cost = np.sum(res**2)
                
                if cost < best_cost:
                    best_cost = cost
                    best_x = x0.copy()
                    stagnant = 0
                    # Adaptive learning rate
                    if cost < 1e-10:
                        break
                else:
                    stagnant += 1
                    lr *= 0.9
                    x0 = best_x.copy()
                
                if stagnant > 50:
                    break
                
                if cost < 1e-12:
                    break
            
            if best_cost < 1e-10:
                U_sol = best_x[:n*m*r].reshape(n*m, r)
                V_sol = best_x[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
                W_sol = best_x[n*m*r+m*p*r:].reshape(p*n, r)
                
                # Verify
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_sol[i*m+j, :] * V_sol[j*p+k, :] * W_sol[k*n+i, :])
                            max_err = max(max_err, abs(val - 1))
                
                if max_err < 1e-9:
                    U_list = [[float(val) for val in row] for row in U_sol]
                    V_list = [[float(val) for val in row] for row in V_sol]
                    W_list = [[float(val) for val in row] for row in W_sol]
                    return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook rank 18
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
