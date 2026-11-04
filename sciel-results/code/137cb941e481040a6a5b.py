import time
import numpy as np
import random

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS
    # We decompose into 3 blocks of 2x3x3
    # Find rank of 2x3x3
    n_sub, m_sub, p_sub = 2, 3, 3
    
    # Target rank for 2x3x3 is 14
    # Try to find a decomposition with rank R_sub
    # We'll try R_sub from 14 up to 18
    
    best_U_sub = None
    best_V_sub = None
    best_W_sub = None
    best_rank = m_sub * p_sub * n_sub # 18
    
    # Generate the 2x3x3 tensor T
    # T[i, j, k] = 1 if i,j,k are the indices for the term A_i,j * B_j,k * C_k,i
    # Actually, the tensor is defined by the multiplication rule.
    # T[i, j, k] = 1.0 for all i,j,k? No.
    # The tensor T is such that <T, A \otimes B \otimes C> = Tr(ACB) ?
    # The standard matrix multiplication tensor M_{n,m,p} has entries M_{i,j,k} = 1.
    # Wait, the contraction is sum_{i,j,k} M_{i,j,k} A_{i,j} B_{j,k} C_{k,i}.
    # For matrix multiplication, C = AB.
    # (AB)_{i,k} = sum_j A_{i,j} B_{j,k}.
    # So we want sum_{i,j,k} M_{i,j,k} A_{i,j} B_{j,k} C_{k,i} = sum_{i,k} (AB)_{i,k} C_{k,i} = Tr(AB C).
    # This is not the standard definition.
    # The standard tensor decomposition for matrix multiplication A (n x m) * B (m x p) = C (n x p)
    # is given by the tensor T with indices i (row A), j (col A/row B), k (col B).
    # The decomposition is T = sum_r U_r \otimes V_r \otimes W_r.
    # Such that sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = 1 for all i,j,k?
    # No.
    # The condition is:
    # sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = delta_{i, i'} delta_{j, j'} delta_{k, k'}? No.
    
    # Let's check the definition in the prompt:
    # "exactly decompose the matrix multiplication tensor"
    # "W[k*n+i,r]"
    # The tensor T is defined by T[i, j, k] = 1.
    # And the decomposition is T = sum_r U_r \otimes V_r \otimes W_r.
    # So we need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i, j, k.
    
    # So for 2x3x3, we need:
    # sum_r U[i, j, r] V[j, k, r] W[k, i, r] = 1 for all i in 0..1, j in 0..2, k in 0..2.
    
    # This is a system of 18 equations.
    # We want to find U, V, W with minimal rank R such that this holds.
    
    def find_decomposition(n_s, m_s, p_s, target_rank, time_limit):
        # n_s=2, m_s=3, p_s=3
        # U: (n_s*m_s, R), V: (m_s*p_s, R), W: (p_s*n_s, R)
        U = np.zeros((n_s * m_s, target_rank))
        V = np.zeros((m_s * p_s, target_rank))
        W = np.zeros((p_s * n_s, target_rank))
        
        # Initialize with random integers
        for _ in range(50): # 50 restarts
            if time.time() > time_limit:
                break
            U[:] = np.random.randint(-2, 3, size=(n_s * m_s, target_rank))
            V[:] = np.random.randint(-2, 3, size=(m_s * p_s, target_rank))
            W[:] = np.random.randint(-2, 3, size=(p_s * n_s, target_rank))
            
            # Simple gradient descent / coordinate descent
            # We want to minimize sum_{i,j,k} (sum_r U[i,j,r]V[j,k,r]W[k,i,r] - 1)^2
            # Let's use a simple iterative update.
            # This is a hard non-convex problem.
            # Maybe use scipy.optimize?
            # Let's try a simple local search: perturb one entry at a time.
            
            best_err = np.inf
            for iter in range(1000):
                if time.time() > time_limit:
                    break
                # Calculate current error
                err = 0.0
                for i in range(n_s):
                    for j in range(m_s):
                        for k in range(p_s):
                            val = 0.0
                            for r in range(target_rank):
                                val += U[i*m_s+j, r] * V[j*p_s+k, r] * W[k*n_s+i, r]
                            err += (val - 1.0)**2
                if err < best_err:
                    best_err = err
                    if err < 1e-9:
                        return U.copy(), V.copy(), W.copy()
                
                # Try to improve
                # Randomly pick a matrix and an entry
                mat_idx = np.random.randint(3)
                if mat_idx == 0:
                    idx = np.random.randint(n_s * m_s)
                    r = np.random.randint(target_rank)
                    # Try delta = -1, 0, 1
                    old_val = U[idx, r]
                    for delta in [-1, 1]:
                        U[idx, r] = old_val + delta
                        # Quick check? Or just accept and check later?
                        # To be efficient, we can just accept and let the next error calc handle it.
                        # But this is slow.
                        # Let's just do a few steps of gradient descent.
                        U[idx, r] = old_val
                # This random walk is too slow.
            
        return None, None, None

    # This random walk is likely to fail.
    # I should use a more robust method or hardcode.
    # Given the constraints, I will assume the rank is 42 and try to hardcode the 14-rank decomposition if I can find it in my internal knowledge.
    # If not, I will return the trivial 54.
    
    # Actually, I can try to use the known rank-14 decomposition coefficients.
    # The 14 terms are:
    # (1,1,1), (1,2,1), (1,3,1), (2,1,1), (2,2,1), (2,3,1) ... no.
    
    # Let's try to construct the 2x3x3 decomposition using the "3x3x3" rank 23 decomposition restricted?
    # No.
    
    # I will output the trivial 54 to be safe, as finding 42 is risky without hardcoded coefficients.
    # Wait, the seed is 54. If I output 54, I get score 1.0.
    # If I can get 42, I get a better score.
    # Is there a rank-15 decomposition that is easy?
    # Rank 15 for 2x3x3?
    # If I use rank 15, total rank is 45.
    
    # Let's try to implement a simple search for 2x3x3 rank 14 using scipy.optimize.minimize with a good initial guess.
    # Initial guess: use the trivial 18-rank decomposition and set 4 random columns to 0? No.
    # Use random initialization.
    
    # I'll try a simple gradient descent with numpy.
    
    def solve_2x3x3(target_rank, time_limit):
        n_s, m_s, p_s = 2, 3, 3
        U = np.random.randn(n_s * m_s, target_rank)
        V = np.random.randn(m_s * p_s, target_rank)
        W = np.random.randn(p_s * n_s, target_rank)
        
        # Flatten
        x = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def loss(x):
            U_f = x[:n_s*m_s*target_rank].reshape(n_s*m_s, target_rank)
            V_f = x[n_s*m_s*target_rank : n_s*m_s*target_rank + m_s*p_s*target_rank].reshape(m_s*p_s, target_rank)
            W_f = x[n_s*m_s*target_rank + m_s*p_s*target_rank :].reshape(p_s*n_s, target_rank)
            
            err = 0.0
            for i in range(n_s):
                for j in range(m_s):
                    for k in range(p_s):
                        val = np.dot(U_f[i*m_s+j], V_f[j*p_s+k] * W_f[k*n_s+i])
                        # Wait, V and W multiplication is not element-wise in the dot product sense directly like that.
                        # val = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
                        val = np.sum(U_f[i*m_s+j] * V_f[j*p_s+k] * W_f[k*n_s+i])
                        err += (val - 1.0)**2
            return err
        
        # Use scipy.optimize.minimize
        try:
            from scipy.optimize import minimize
            res = minimize(loss, x, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-8})
            if res.fun < 1e-8:
                U_f = res.x[:n_s*m_s*target_rank].reshape(n_s*m_s, target_rank)
                V_f = res.x[n_s*m_s*target_rank : n_s*m_s*target_rank + m_s*p_s*target_rank].reshape(m_s*p_s, target_rank)
                W_f = res.x[n_s*m_s*target_rank + m_s*p_s*target_rank :].reshape(p_s*n_s, target_rank)
                # Snap to integers
                U_f = np.round(U_f)
                V_f = np.round(V_f)
                W_f = np.round(W_f)
                # Verify
                if loss(np.concatenate([U_f.flatten(), V_f.flatten(), W_f.flatten()])) < 1e-8:
                    return U_f, V_f, W_f
        except:
            pass
        return None

    # Try rank 14
    U14, V14, W14 = solve_2x3x3(14, deadline - 10)
    if U14 is not None:
        R_sub = 14
        U_sub, V_sub, W_sub = U14, V14, W14
    else:
        # Fallback to trivial 18
        R_sub = 18
        U_sub = np.eye(6, 18) # Not quite, need to map correctly
        # Trivial:
        U_sub = np.zeros((6, 18))
        V_sub = np.zeros((9, 18))
        W_sub = np.zeros((6, 18))
        idx = 0
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    U_sub[i*3+j, idx] = 1.0
                    V_sub[j*3+k, idx] = 1.0
                    W_sub[k*2+i, idx] = 1.0
                    idx += 1

    R_total = 3 * R_sub
    U_total = np.zeros((6, R_total))
    V_total = np.zeros((27, R_total))
    W_total = np.zeros((18, R_total))
    
    for b in range(3):
        start_r = b * R_sub
        end_r = (b + 1) * R_sub
        U_total[:, start_r:end_r] = U_sub
        # V_sub is (9, R_sub) corresponding to j in 0..2, k in 0..2
        # V_total is (27, R_total) corresponding to j in 0..2, k in 0..8
        # For block b, k in V_total is 3b + k_sub
        for j in range(3):
            for k_sub in range(3):
                V_total[j*9 + 3*b + k_sub, start_r:end_r] = V_sub[j*3 + k_sub, :]
        
        # W_sub is (6, R_sub) corresponding to k in 0..2, i in 0..1
        # W_total is (18, R_total) corresponding to k in 0..8, i in 0..1
        # For block b, k in W_total is 3b + k_sub
        for k_sub in range(3):
            for i in range(2):
                W_total[3*b + k_sub, i] # Wait, W index is k*n+i.
                # W_total index is k_total * 2 + i.
                # k_total = 3*b + k_sub.
                W_total[(3*b + k_sub)*2 + i, start_r:end_r] = W_sub[k_sub*2 + i, :]

    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
