"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
from itertools import product
import scipy.optimize

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a low-rank exact tensor decomposition for multiplying a 2x4 matrix by a 4x5 matrix.
    Uses optimization to find a low-rank approximation, then snaps to half-integers
    and verifies exactly. Includes a discrete local search for refinement.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    def schoolbook_decomposition():
        rank_schoolbook = n * m * p
        U = np.zeros((n*m, rank_schoolbook))
        V = np.zeros((m*p, rank_schoolbook))
        W = np.zeros((p*n, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W
    
    U_school, V_school, W_school = schoolbook_decomposition()
    
    def verify_half_integer(U, V, W):
        """
        Verify that U, V, W with half-integer entries exactly decompose the tensor.
        """
        U_int = (2 * U).astype(int)
        V_int = (2 * V).astype(int)
        W_int = (2 * W).astype(int)
        
        if not np.allclose(U * 2, U_int) or not np.allclose(V * 2, V_int) or not np.allclose(W * 2, W_int):
            return False
            
        # Check if sum_r U V W == 8 for all i,j,k
        # Using integer arithmetic for speed and exactness
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(U.shape[1]):
                        val += U_int[i*m+j, r] * V_int[j*p+k, r] * W_int[k*n+i, r]
                    if val != 8:
                        return False
        return True

    def loss_function(x, R):
        """
        Loss function for optimization.
        """
        U_size = n * m * R
        V_size = m * p * R
        W_size = p * n * R
        
        U = x[:U_size].reshape(n*m, R)
        V = x[U_size:U_size+V_size].reshape(m*p, R)
        W = x[U_size+V_size:].reshape(p*n, R)
        
        loss = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    loss += (val - 1.0)**2
        return loss

    def grad_function(x, R):
        """
        Gradient of the loss function.
        """
        U_size = n * m * R
        V_size = m * p * R
        W_size = p * n * R
        
        U = x[:U_size].reshape(n*m, R)
        V = x[U_size:U_size+V_size].reshape(m*p, R)
        W = x[U_size+V_size:].reshape(p*n, R)
        
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # residual = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] - 1
                    uvw = U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
                    residual = np.sum(uvw) - 1.0
                    
                    # dL/dU[i,j,r] = 2 * residual * V[j,k,r] * W[k,i,r]
                    dU[i*m+j, :] += 2 * residual * V[j*p+k, :] * W[k*n+i, :]
                    dV[j*p+k, :] += 2 * residual * U[i*m+j, :] * W[k*n+i, :]
                    dW[k*n+i, :] += 2 * residual * U[i*m+j, :] * V[j*p+k, :]
        
        return np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])

    def discrete_local_search(U_int, V_int, W_int, max_iters=50):
        """
        Perform local search on integer-scaled matrices (entries are 2 * value).
        Goal: Minimize sum_{i,j,k} (sum_r U V W - 8)^2.
        """
        R = U_int.shape[1]
        
        # Precompute indices for faster access? 
        # For now, simple loops.
        
        def calc_error():
            err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for r in range(R):
                            val += U_int[i*m+j, r] * V_int[j*p+k, r] * W_int[k*n+i, r]
                        err += (val - 8)**2
            return err

        current_error = calc_error()
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            improved = False
            
            # Try improving U
            for i in range(n):
                for j in range(m):
                    for r in range(R):
                        orig_val = U_int[i*m+j, r]
                        # Try orig_val - 1 and orig_val + 1
                        for delta in [-1, 1]:
                            new_val = orig_val + delta
                            U_int[i*m+j, r] = new_val
                            new_error = calc_error()
                            if new_error < current_error:
                                current_error = new_error
                                improved = True
                            else:
                                U_int[i*m+j, r] = orig_val
                                
            # Try improving V
            for j in range(m):
                for k in range(p):
                    for r in range(R):
                        orig_val = V_int[j*p+k, r]
                        for delta in [-1, 1]:
                            new_val = orig_val + delta
                            V_int[j*p+k, r] = new_val
                            new_error = calc_error()
                            if new_error < current_error:
                                current_error = new_error
                                improved = True
                            else:
                                V_int[j*p+k, r] = orig_val

            # Try improving W
            for k in range(p):
                for i in range(n):
                    for r in range(R):
                        orig_val = W_int[k*n+i, r]
                        for delta in [-1, 1]:
                            new_val = orig_val + delta
                            W_int[k*n+i, r] = new_val
                            new_error = calc_error()
                            if new_error < current_error:
                                current_error = new_error
                                improved = True
                            else:
                                W_int[k*n+i, r] = orig_val

            if not improved:
                break
                
        return current_error

    best_rank = n * m * p
    best_decomp = (U_school, V_school, W_school)
    
    # Try ranks from lower bound up to schoolbook
    # Lower bound is max(nm, mp, pn) = 20
    candidate_ranks = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38]
    
    for target_rank in candidate_ranks:
        if target_rank >= best_rank:
            continue
        if time.time() > deadline - 20:
            break
            
        found = False
        # Try more random initializations with gradient-based optimization
        num_restarts = 50
        for restart in range(num_restarts):
            if time.time() > deadline - 20:
                break
                
            # Random initialization
            U_init = np.random.randn(n*m, target_rank)
            V_init = np.random.randn(m*p, target_rank)
            W_init = np.random.randn(p*n, target_rank)
            
            # Scale to help optimization
            U_init /= np.sqrt(target_rank)
            V_init /= np.sqrt(target_rank)
            W_init /= np.sqrt(target_rank)
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = scipy.optimize.minimize(
                    loss_function, x0, args=(target_rank,), 
                    jac=grad_function,
                    method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
            except Exception:
                continue
                
            # Check if we got close to a solution
            if res.fun < 1e-6:
                # Extract U, V, W
                U_opt = res.x[:n*m*target_rank].reshape(n*m, target_rank)
                V_opt = res.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_opt = res.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                
                # Snap to nearest integer (scaled by 2)
                U_snap_int = np.round(U_opt * 2).astype(int)
                V_snap_int = np.round(V_opt * 2).astype(int)
                W_snap_int = np.round(W_opt * 2).astype(int)
                
                # Run local search
                err = discrete_local_search(U_snap_int, V_snap_int, W_snap_int)
                
                if err == 0:
                    # Convert back to float for return
                    U_final = U_snap_int / 2.0
                    V_final = V_snap_int / 2.0
                    W_final = W_snap_int / 2.0
                    
                    # Final verification
                    if verify_half_integer(U_final, V_final, W_final):
                        found = True
                        best_rank = target_rank
                        best_decomp = (U_final, V_final, W_final)
                        break
        
        if found:
            break
    
    if best_rank == n * m * p:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    else:
        U_best, V_best, W_best = best_decomp
        return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
# EVOLVE-BLOCK-END
