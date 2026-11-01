import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    R_school = n * m * p  # 18
    
    # Build the target tensor T[i, j, k] = 1 for all (i,j,k)
    T = np.ones((n, m, p))
    
    # Build schoolbook solution as fallback
    U_school = np.zeros((n * m, R_school))
    V_school = np.zeros((m * p, R_school))
    W_school = np.zeros((p * n, R_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = R_school
    
    start_time = time.time()
    deadline = 130  # Safety margin before 150s kill

    def compute_error(U, V, W, target_rank):
        """Compute the error of the tensor decomposition."""
        U_r = U.reshape(n, m, target_rank)
        V_r = V.reshape(m, p, target_rank)
        W_r = W.reshape(p, n, target_rank)
        result = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        return np.linalg.norm(result - T)

    def als_decompose(U, V, W, target_rank, max_iters=200, tol=1e-10):
        """Alternating Least Squares refinement."""
        for iter_ in range(max_iters):
            if time.time() - start_time > deadline:
                return U, V, W, None
            
            U_r = U.reshape(n, m, target_rank)
            V_r = V.reshape(m, p, target_rank)
            W_r = W.reshape(p, n, target_rank)
            
            # Solve for U
            for r in range(target_rank):
                # U[:,r] = argmin_U sum_{i,j,k} (T[i,j,k] - U[i,j,r]*V[j,k,r]*W[k,i,r])^2
                # Linear in U[:,r]: sum_{j,k} V[j,k,r]*W[k,i,r] * U[i,j,r] = T[i,j,k] - sum_{r'!=r} U[i,j,r']*V[j,k,r']*W[k,i,r]
                # This is a system: A @ u = b where A[i,j] = V[j,p+k,r]*W[k,n+i,r]
                # Actually, for each (i,j), we have: sum_k V[j,k,r]*W[k,i,r]*U[i,j,r] = T[i,j,k] - sum_{r'!=r} U[i,j,r']*V[j,k,r']*W[k,i,r]
                # But this couples over k... Let me think differently.
                
                # The residual for rank r: R[i,j,k] = T[i,j,k] - sum_{r'} U[i,j,r']*V[j,k,r']*W[k,i,r']
                # We want to minimize sum_{i,j,k} (R[i,j,k])^2 with respect to U[:,r]
                # d/dU[i,j,r] sum_{i',j',k'} (R[i',j',k'])^2 = -2 * sum_{k'} R[i,j,k'] * V[j,k',r] * W[k',i,r]
                # So the update is: U[:,r] += lr * sum_k R[:,j,k] * V[j,k,r] * W[k,i,r]
                
                # Actually, ALS solves the linear system exactly. For U[:,r]:
                # The contribution to the tensor from rank r is: U_r[i,j] * V_r[j,k] * W_r[k,i]
                # We want to find U_r that best fits the residual after removing other ranks.
                
                # Let's compute the residual without rank r:
                result_other = np.zeros((n, m, p))
                for r2 in range(target_rank):
                    if r2 == r:
                        continue
                    result_other += U_r[:,:,r2] * V_r[:,:,r2] * W_r[:,:,r2]
                
                residual_r = T - result_other
                
                # Now solve for U_r: minimize sum_{i,j,k} (residual_r[i,j,k] - U_r[i,j]*V_r[j,k,r]*W_r[k,i,r])^2
                # This is a linear least squares problem. For each (i,j), we have:
                # sum_k residual_r[i,j,k] = U_r[i,j] * sum_k V_r[j,k,r]*W_r[k,i,r]
                # So U_r[i,j] = sum_k residual_r[i,j,k] / sum_k V_r[j,k,r]*W_r[k,i,r]
                
                denom = np.einsum('jk,ki->i', V_r[:,:,r], W_r[:,:,r])  # shape (n,)
                # Wait, this doesn't look right. Let me re-derive.
                
                # We want to minimize: sum_{i,j,k} (residual_r[i,j,k] - U_r[i,j]*V_r[j,k,r]*W_r[k,i,r])^2
                # Taking derivative w.r.t. U_r[i,j]:
                # -2 * sum_k residual_r[i,j,k] * V_r[j,k,r] * W_r[k,i,r] + 2 * U_r[i,j] * sum_k V_r[j,k,r]^2 * W_r[k,i,r]^2 = 0
                # So: U_r[i,j] = sum_k residual_r[i,j,k] * V_r[j,k,r] * W_r[k,i,r] / sum_k V_r[j,k,r]^2 * W_r[k,i,r]^2
                
                num = np.einsum('ijk,jk,ki->ij', residual_r, V_r[:,:,r], W_r[:,:,r])
                denom = np.einsum('jk,ki->j', V_r[:,:,r]**2, W_r[:,:,r]**2)  # This is wrong shape
                # Actually denom should be: for each (i,j), sum_k V_r[j,k,r]^2 * W_r[k,i,r]^2
                denom = np.einsum('jk,ki->ij', V_r[:,:,r]**2, W_r[:,:,r]**2)
                
                # Avoid division by zero
                mask = denom > 1e-12
                U_r[:,:,r] = np.zeros_like(U_r[:,:,r])
                U_r[:,:,r][mask] = num[mask] / denom[mask]
            
            # Solve for V
            for r in range(target_rank):
                result_other = np.zeros((n, m, p))
                for r2 in range(target_rank):
                    if r2 == r:
                        continue
                    result_other += U_r[:,:,r2] * V_r[:,:,r2] * W_r[:,:,r2]
                
                residual_r = T - result_other
                
                # Minimize: sum_{i,j,k} (residual_r[i,j,k] - U_r[i,j,r]*V_r[j,k,r]*W_r[k,i,r])^2
                # V_r[j,k,r] = sum_i residual_r[i,j,k] * U_r[i,j,r] * W_r[k,i,r] / sum_i U_r[i,j,r]^2 * W_r[k,i,r]^2
                
                num = np.einsum('ijk,ij,ki->jk', residual_r, U_r[:,:,r], W_r[:,:,r])
                denom = np.einsum('ij,ki->jk', U_r[:,:,r]**2, W_r[:,:,r]**2)
                
                mask = denom > 1e-12
                V_r[:,:,r] = np.zeros_like(V_r[:,:,r])
                V_r[:,:,r][mask] = num[mask] / denom[mask]
            
            # Solve for W
            for r in range(target_rank):
                result_other = np.zeros((n, m, p))
                for r2 in range(target_rank):
                    if r2 == r:
                        continue
                    result_other += U_r[:,:,r2] * V_r[:,:,r2] * W_r[:,:,r2]
                
                residual_r = T - result_other
                
                # Minimize: sum_{i,j,k} (residual_r[i,j,k] - U_r[i,j,r]*V_r[j,k,r]*W_r[k,i,r])^2
                # W_r[k,i,r] = sum_j residual_r[i,j,k] * U_r[i,j,r] * V_r[j,k,r] / sum_j U_r[i,j,r]^2 * V_r[j,k,r]^2
                
                num = np.einsum('ijk,ij,jk->ki', residual_r, U_r[:,:,r], V_r[:,:,r])
                denom = np.einsum('ij,jk->ki', U_r[:,:,r]**2, V_r[:,:,r]**2)
                
                mask = denom > 1e-12
                W_r[:,:,r] = np.zeros_like(W_r[:,:,r])
                W_r[:,:,r][mask] = num[mask] / denom[mask]
            
            # Compute error
            err = compute_error(U_r.reshape(n*m, target_rank), V_r.reshape(m*p, target_rank), W_r.reshape(p*n, target_rank), target_rank)
            if err < tol:
                break
        
        return U_r.reshape(n*m, target_rank), V_r.reshape(m*p, target_rank), W_r.reshape(p*n, target_rank), err

    def gradient_descent(target_rank, max_iters=1000, lr=0.01, seed=None):
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()
            
        U = rng.randn(n * m, target_rank) * 0.1
        V = rng.randn(m * p, target_rank) * 0.1
        W = rng.randn(p * n, target_rank) * 0.1
        
        for iter_ in range(max_iters):
            if time.time() - start_time > deadline:
                return None, None, None, None
            
            U_r = U.reshape(n, m, target_rank)
            V_r = V.reshape(m, p, target_rank)
            W_r = W.reshape(p, n, target_rank)
            
            result = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
            residual = T - result
            
            # Gradients
            grad_U = np.einsum('ijk,jkr,kir->ijr', residual, V_r, W_r)
            grad_V = np.einsum('ijk,ijr,kir->jkr', residual, U_r, W_r)
            grad_W = np.einsum('ijk,ijr,jkr->kir', residual, U_r, V_r)
            
            # Update with adaptive learning rate
            U -= lr * grad_U.reshape(n * m, target_rank)
            V -= lr * grad_V.reshape(m * p, target_rank)
            W -= lr * grad_W.reshape(p * n, target_rank)
            
            if iter_ % 100 == 0:
                err = compute_error(U, V, W, target_rank)
                if err < 1e-6:
                    break
                
        err = compute_error(U, V, W, target_rank)
        return U, V, W, err

    def try_rank(target_rank):
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() - start_time > deadline:
                return False
            
            seed = restart * 1000 + target_rank
            U, V, W, error = gradient_descent(target_rank, max_iters=500, lr=0.05, seed=seed)
            
            if U is None:
                return False
                
            if error < 1e-5:
                # ALS refinement
                U, V, W, err_als = als_decompose(U, V, W, target_rank, max_iters=100, tol=1e-12)
                
                if err_als is not None and err_als < 1e-10:
                    # High-precision polish
                    for _ in range(50):
                        if time.time() - start_time > deadline:
                            return False
                        U_r = U.reshape(n, m, target_rank)
                        V_r = V.reshape(m, p, target_rank)
                        W_r = W.reshape(p, n, target_rank)
                        result = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
                        residual = T - result
                        err = np.linalg.norm(residual)
                        if err < 1e-12:
                            break
                        
                        grad_U = np.einsum('ijk,jkr,kir->ijr', residual, V_r, W_r)
                        grad_V = np.einsum('ijk,ijr,kir->jkr', residual, U_r, W_r)
                        grad_W = np.einsum('ijk,ijr,jkr->kir', residual, U_r, V_r)
                        
                        U -= 0.001 * grad_U.reshape(n * m, target_rank)
                        V -= 0.001 * grad_V.reshape(m * p, target_rank)
                        W -= 0.001 * grad_W.reshape(p * n, target_rank)
                
                err = compute_error(U, V, W, target_rank)
                if err < 1e-8:
                    # Round to half-integers
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    # Verify rounded
                    err_rounded = compute_error(U_rounded, V_rounded, W_rounded, target_rank)
                    if err_rounded < 1e-6:
                        nonlocal best_U, best_V, best_W, best_rank
                        best_U = U_rounded
                        best_V = V_rounded
                        best_W = W_rounded
                        best_rank = target_rank
                        return True
        return False

    # Try ranks from 14 to 17 (18 is schoolbook)
    for target_rank in range(14, R_school):
        if try_rank(target_rank):
            break

    # Final verification
    def verify_solution(U, V, W, rank):
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    max_err = max(max_err, abs(val - T[i, j, k]))
        return max_err

    if best_rank < R_school:
        err = verify_solution(best_U, best_V, best_W, best_rank)
        if err > 1e-6:
            best_U, best_V, best_W = U_school, V_school, W_school
            best_rank = R_school

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
