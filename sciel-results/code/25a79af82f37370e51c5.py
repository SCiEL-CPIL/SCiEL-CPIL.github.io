import time
import numpy as np
from fractions import Fraction
import scipy.optimize

DIMENSIONS = [3, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 36

    # Build schoolbook decomposition as fallback
    U_school = np.zeros((n * m, schoolbook_rank), dtype=object)
    V_school = np.zeros((m * p, schoolbook_rank), dtype=object)
    W_school = np.zeros((p * n, schoolbook_rank), dtype=object)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = Fraction(1)
                V_school[j * p + k, col] = Fraction(1)
                W_school[k * n + i, col] = Fraction(1)
                col += 1

    def verify_exact(U_mat, V_mat, W_mat, rank):
        """Verify exact decomposition using Fractions"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(rank):
                        s += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                    if s != Fraction(1):
                        return False
        return True

    def compute_tensor_and_grad(params, rank):
        """
        params: flat array of size (n*m + m*p + p*n) * rank
        Returns: error, gradients
        """
        U_f = params[:n*m*rank].reshape(n*m, rank)
        V_f = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W_f = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
        
        T = np.zeros((n, m, p))
        for r in range(rank):
            u_2d = U_f[:, r].reshape(n, m)
            v_2d = V_f[:, r].reshape(m, p)
            w_2d = W_f[:, r].reshape(p, n)
            T += np.einsum('ij,jk,ki->ijk', u_2d, v_2d, w_2d)
            
        err = np.sum((T - 1.0)**2)
        
        dU = np.zeros_like(U_f)
        dV = np.zeros_like(V_f)
        dW = np.zeros_like(W_f)
        
        diff = 2.0 * (T - 1.0)
        
        for r in range(rank):
            u_2d = U_f[:, r].reshape(n, m)
            v_2d = V_f[:, r].reshape(m, p)
            w_2d = W_f[:, r].reshape(p, n)
            
            dU_2d = np.einsum('ijk,jk,ki->ij', diff, v_2d, w_2d)
            dV_2d = np.einsum('ijk,ij,ki->jk', diff, u_2d, w_2d)
            dW_2d = np.einsum('ijk,ij,jk->ki', diff, u_2d, v_2d)
            
            dU[:, r] = dU_2d.flatten()
            dV[:, r] = dV_2d.flatten()
            dW[:, r] = dW_2d.flatten()
            
        grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
        return err, grad

    def try_rank(R, seed):
        """Try to find a rank R decomposition"""
        np.random.seed(seed)
        # Smaller initialization to help convergence
        init_U = np.random.randn(n*m, R) * 0.1
        init_V = np.random.randn(m*p, R) * 0.1
        init_W = np.random.randn(p*n, R) * 0.1
        
        init_params = np.concatenate([init_U.flatten(), init_V.flatten(), init_W.flatten()])
        
        def fun(x):
            err, _ = compute_tensor_and_grad(x, R)
            return err
            
        def jac(x):
            _, grad = compute_tensor_and_grad(x, R)
            return grad

        if time.time() > deadline - 10:
            return None

        try:
            res = scipy.optimize.minimize(
                fun, init_params, jac=jac, method='L-BFGS-B', 
                options={'maxiter': 500, 'gtol': 1e-10, 'disp': False}
            )
        except Exception:
            return None

        # Allow slightly higher error for snapping, rely on exact verification
        if res.fun < 1e-4:
            U_f = res.x[:n*m*R].reshape(n*m, R)
            V_f = res.x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
            W_f = res.x[n*m*R + m*p*R:].reshape(p*n, R)
            
            U_rat = np.zeros((n*m, R), dtype=object)
            V_rat = np.zeros((m*p, R), dtype=object)
            W_rat = np.zeros((p*n, R), dtype=object)
            
            def snap(val):
                best_frac = None
                best_err = float('inf')
                # Expand denominators to 10
                for d in range(1, 11):
                    num = round(val * d)
                    frac = Fraction(num, d)
                    err = abs(float(frac) - val)
                    if err < best_err:
                        best_err = err
                        best_frac = frac
                return best_frac

            for i in range(n*m):
                for r in range(R):
                    U_rat[i,r] = snap(U_f[i,r])
            for i in range(m*p):
                for r in range(R):
                    V_rat[i,r] = snap(V_f[i,r])
            for i in range(p*n):
                for r in range(R):
                    W_rat[i,r] = snap(W_f[i,r])
            
            if verify_exact(U_rat, V_rat, W_rat, R):
                return U_rat, V_rat, W_rat, R
        
        return None

    best_result = None
    
    # Search ranks from low to high
    # Start at 21, the known lower bound for 3x3x4
    for R in range(21, 36):
        if time.time() > deadline - 20:
            break
            
        found = False
        # More seeds for lower ranks
        num_seeds = 100 if R < 25 else 50
        for seed in range(num_seeds):
            if time.time() > deadline - 5:
                break
                
            result = try_rank(R, seed)
            if result:
                U_rat, V_rat, W_rat, rank = result
                best_result = (U_rat, V_rat, W_rat, rank)
                found = True
                break
        
        if found:
            break

    if best_result:
        U_rat, V_rat, W_rat, rank = best_result
        return {"U": [[float(x) for x in row] for row in U_rat], 
                "V": [[float(x) for x in row] for row in V_rat], 
                "W": [[float(x) for x in row] for row in W_rat]}
    else:
        return {"U": [[float(x) for x in row] for row in U_school], 
                "V": [[float(x) for x in row] for row in V_school], 
                "W": [[float(x) for x in row] for row in W_school]}
