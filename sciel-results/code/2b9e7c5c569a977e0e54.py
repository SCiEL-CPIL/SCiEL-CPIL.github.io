"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import svd

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # Fallback: Schoolbook decomposition
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for col in range(rank):
            i = col // (m * p)
            j = (col // p) % m
            k = col % p
            
            U[i * m + j, col] = 1
            V[j * p + k, col] = 1
            W[k * n + i, col] = 1
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    # Try to find a lower rank decomposition using optimization
    # Rank 32 is a plausible target for 3x3x4 (vs 36 schoolbook)
    # We search for integer/half-integer solutions.
    
    start_time = time.time()
    timeout = 100  # Leave 20s margin
    
    # 1. Try exact integer search for low ranks? No, space is too large.
    # 2. Try continuous relaxation and snap to grid.
    
    def evaluate_decomposition(U, V, W):
        # Check if U*V*W approximates the multiplication tensor
        # Tensor T[i,j,k] = delta_{i,i} * delta_{j,k} ? No.
        # C = A B. C_{ik} = sum_j A_{ij} B_{jk}.
        # Tensor M_{ikj} = sum U_{ij} V_{jk} W_{ik}
        # We need sum_c U_{ij,c} V_{jk,c} W_{ik,c} = delta_{i,i?} No.
        # Target: for each i, k: C_{ik} = sum_j A_{ij} B_{jk}
        # So sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c} should be 1 if j matches? 
        # No, the sum over c is for a specific i,j,k.
        # The term in the sum is A_{ij} B_{jk}. 
        # So we need: sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c} = 1 for all i,j,k?
        # No, that would mean the coefficient of A_{ij} B_{jk} in C_{ik} is 1.
        # Yes, C_{ik} = sum_j (sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c}) A_{ij} B_{jk}.
        # So the inner sum must be 1 for all i,j,k.
        
        # Let's verify:
        # C_{ik} = sum_c W_{k*n+i, c} (sum_j U_{i*m+j, c} A_{ij}) (sum_k' V_{j*p+k', c} B_{jk'})
        # This expands to sum_c sum_j sum_k' W_{k*n+i, c} U_{i*m+j, c} V_{j*p+k', c} A_{ij} B_{jk'}
        # We want this to equal sum_j A_{ij} B_{jk}.
        # So the coefficient of A_{ij} B_{jk} in the expansion must be 1 if k'=k, 0 otherwise.
        # Wait, the index in W is k*n+i. The index in V is j*p+k'.
        # The term A_{ij} B_{jk'} appears in the sum over c and j (fixed j).
        # Coeff of A_{ij} B_{jk'} in C_{ik} is sum_c U_{i*m+j, c} V_{j*p+k', c} W_{k*n+i, c}.
        # We need this to be 1 if k'=k, 0 if k' != k.
        # So for all i, j, k: sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c} = 1.
        # And for all i, j, k, k' != k: sum_c U_{i*m+j, c} V_{j*p+k', c} W_{k*n+i, c} = 0.
        
        error = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target = 1
                    val = 0
                    for c in range(len(U[0])):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    error += (val - target)**2
                    for k2 in range(p):
                        if k2 == k: continue
                        target = 0
                        val = 0
                        for c in range(len(U[0])):
                            val += U[i*m+j][c] * V[j*p+k2][c] * W[k*n+i][c]
                        error += val**2
        return error

    def snap_to_grid(matrix):
        # Snap entries to nearest 0.5
        return np.round(np.array(matrix) * 2) / 2.0

    # Optimization strategy:
    # Parameterize U, V, W with rank r.
    # Minimize error.
    # If error is close to 0, snap to grid and verify.
    
    best_result = None
    best_rank = n * m * p
    
    # Try ranks from 32 upwards
    for rank in range(32, best_rank + 1):
        if time.time() - start_time > timeout:
            break
            
        # Initialize with random small values? Or schoolbook subset?
        # Schoolbook has rank 36. For rank 32, we need to merge some terms.
        # Random init might get stuck.
        # Let's try a few random restarts for rank 32.
        
        for restart in range(5):
            if time.time() - start_time > timeout:
                break
                
            U = np.random.randn(n*m, rank) * 0.5
            V = np.random.randn(m*p, rank) * 0.5
            W = np.random.randn(p*n, rank) * 0.5
            
            # Flatten for scipy
            def fun(params):
                U_p = params[:n*m*rank].reshape(n*m, rank)
                V_p = params[n*m*rank:2*n*m*rank].reshape(m*p, rank)
                W_p = params[2*n*m*rank:].reshape(p*n, rank)
                return evaluate_decomposition(U_p, V_p, W_p)
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Use Nelder-Mead for speed, or BFGS? Nelder-Mead is robust for non-smooth.
            # But error function is smooth (sum of squares).
            try:
                res = minimize(fun, x0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
                if res.fun < 1e-8:
                    # Success! Snap to grid
                    U_p = res.x[:n*m*rank].reshape(n*m, rank)
                    V_p = res.x[n*m*rank:2*n*m*rank].reshape(m*p, rank)
                    W_p = res.x[2*n*m*rank:].reshape(p*n, rank)
                    
                    U_s = snap_to_grid(U_p)
                    V_s = snap_to_grid(V_p)
                    W_s = snap_to_grid(W_p)
                    
                    if evaluate_decomposition(U_s, V_s, W_s) < 1e-6:
                        # Verify exactness
                        if evaluate_decomposition(U_s, V_s, W_s) < 1e-10:
                            best_result = {'U': U_s.tolist(), 'V': V_s.tolist(), 'W': W_s.tolist()}
                            best_rank = rank
                            break # Found lower rank, break restarts
            except Exception:
                pass
            
        if best_result and best_rank <= rank:
            break

    if best_result:
        return best_result
    else:
        return schoolbook()

# EVOLVE-BLOCK-END
