"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 145s to stay safely under 150s
    
    n, m, p = DIMENSIONS
    # Schoolbook decomposition of rank n*m*p
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    
    best_rank = schoolbook_rank
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    
    # Strategy: Decompose 2x3x6 into two 2x3x3 blocks.
    # If we find a rank R decomposition for 2x3x3, we can get rank 2R for 2x3x6.
    # Known rank for 2x3x3 is 12. Schoolbook is 18.
    # So if we find R < 18, we improve on 36.
    
    def construct_2x3x3_decomposition(rank):
        """
        Construct a rank-r decomposition for 2x3x3 matrix multiplication.
        Returns (U, V, W) as numpy arrays or None.
        """
        n2, m2, p2 = 2, 3, 3
        num_U = n2 * m2  # 6
        num_V = m2 * p2  # 9
        num_W = p2 * n2  # 6
        
        def compute_tensor(U, V, W):
            """Compute the full tensor T[i,j,k] = sum_r U[i*m2+j,r]*V[j*p2+k,r]*W[k*n2+i,r]"""
            T = np.zeros((n2, m2, p2))
            for i in range(n2):
                for j in range(m2):
                    for k in range(p2):
                        T[i,j,k] = np.sum(U[i*m2+j, :] * V[j*p2+k, :] * W[k*n2+i, :])
            return T
        
        def compute_errors(U, V, W):
            """Compute errors = T - 1"""
            T = compute_tensor(U, V, W)
            return T - 1.0
        
        def objective_and_grad(x):
            """Flattened objective and gradient for scipy optimizer"""
            U = x[:num_U*rank].reshape(num_U, rank)
            V = x[num_U*rank:(num_U+num_V)*rank].reshape(num_V, rank)
            W = x[(num_U+num_V)*rank:].reshape(num_W, rank)
            
            errors = compute_errors(U, V, W)
            loss = np.sum(errors ** 2)
            
            # Compute gradients
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n2):
                for j in range(m2):
                    for k in range(p2):
                        e = errors[i,j,k]
                        for r in range(rank):
                            grad_U[i*m2+j, r] += 2 * e * V[j*p2+k, r] * W[k*n2+i, r]
                            grad_V[j*p2+k, r] += 2 * e * U[i*m2+j, r] * W[k*n2+i, r]
                            grad_W[k*n2+i, r] += 2 * e * U[i*m2+j, r] * V[j*p2+k, r]
            
            grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            return loss, grad
        
        def verify(U, V, W, tol=1e-8):
            """Check if decomposition is valid"""
            errors = compute_errors(U, V, W)
            return np.max(np.abs(errors)) < tol
        
        best_result = None
        best_loss = float('inf')
        
        # Generate multiple initializations
        inits = []
        
        # Random initializations
        for seed in range(20):
            np.random.seed(seed)
            U = np.random.randn(num_U, rank) * 0.3
            V = np.random.randn(num_V, rank) * 0.3
            W = np.random.randn(num_W, rank) * 0.3
            inits.append((U, V, W))
        
        # Structured initialization: try to respect the block structure
        # For 2x3x3, we can try initializations where each rank-1 term
        # contributes to specific (i,j,k) entries
        for seed in range(10):
            np.random.seed(1000 + seed)
            U = np.zeros((num_U, rank))
            V = np.zeros((num_V, rank))
            W = np.zeros((num_W, rank))
            
            # Assign each rank term to contribute to a subset of tensor entries
            for r in range(rank):
                # Randomly choose which i, j, k this term affects
                i_choices = np.random.choice(n2, size=random.randint(1, n2), replace=False)
                j_choices = np.random.choice(m2, size=random.randint(1, m2), replace=False)
                k_choices = np.random.choice(p2, size=random.randint(1, p2), replace=False)
                
                for i in i_choices:
                    for j in j_choices:
                        U[i*m2+j, r] = np.random.choice([-1, 1]) * np.random.uniform(0.5, 1.5)
                
                for j in j_choices:
                    for k in k_choices:
                        V[j*p2+k, r] = np.random.choice([-1, 1]) * np.random.uniform(0.5, 1.5)
                
                for k in k_choices:
                    for i in i_choices:
                        W[k*n2+i, r] = np.random.choice([-1, 1]) * np.random.uniform(0.5, 1.5)
            
            inits.append((U, V, W))
        
        # Try each initialization with L-BFGS
        for idx, (U0, V0, W0) in enumerate(inits):
            if time.time() > deadline - 10:
                break
            
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            try:
                result = minimize(
                    objective_and_grad,
                    x0,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
                )
                
                x_opt = result.x
                U_opt = x_opt[:num_U*rank].reshape(num_U, rank)
                V_opt = x_opt[num_U*rank:(num_U+num_V)*rank].reshape(num_V, rank)
                W_opt = x_opt[(num_U+num_V)*rank:].reshape(num_W, rank)
                
                if result.fun < best_loss:
                    best_loss = result.fun
                    best_result = (U_opt.copy(), V_opt.copy(), W_opt.copy())
                
                # Check if we found a valid solution
                if verify(U_opt, V_opt, W_opt):
                    return U_opt, V_opt, W_opt
                    
            except Exception as e:
                continue
        
        # If we have a good but not perfect result, do a polish pass
        if best_result is not None:
            U_b, V_b, W_b = best_result
            errors = compute_errors(U_b, V_b, W_b)
            max_err = np.max(np.abs(errors))
            
            if max_err < 1e-4:
                # Polish with tighter tolerance
                x0 = np.concatenate([U_b.flatten(), V_b.flatten(), W_b.flatten()])
                try:
                    result = minimize(
                        objective_and_grad,
                        x0,
                        method='L-BFGS-B',
                        jac=True,
                        options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 100}
                    )
                    
                    x_opt = result.x
                    U_opt = x_opt[:num_U*rank].reshape(num_U, rank)
                    V_opt = x_opt[num_U*rank:(num_U+num_V)*rank].reshape(num_V, rank)
                    W_opt = x_opt[(num_U+num_V)*rank:].reshape(num_W, rank)
                    
                    if verify(U_opt, V_opt, W_opt):
                        return U_opt, V_opt, W_opt
                        
                except Exception:
                    pass
        
        return None

    # Try ranks from 12 up to 17
    # 12 is the known minimal rank
    for R in range(12, 18):
        if time.time() > deadline - 10:
            break
            
        decomp_2x3x3 = construct_2x3x3_decomposition(R)
        
        if decomp_2x3x3 is not None:
            U_2x3x3, V_2x3x3, W_2x3x3 = decomp_2x3x3
            
            # Construct 2x3x6 decomposition from two copies
            # U (6, 2R), V (18, 2R), W (12, 2R)
            R_total = 2 * R
            U_2x3x6 = np.zeros((n * m, R_total))
            V_2x3x6 = np.zeros((m * p, R_total))
            W_2x3x6 = np.zeros((p * n, R_total))
            
            # First copy (columns 0 to R-1)
            U_2x3x6[:, 0:R] = U_2x3x3
            V_2x3x6[0:9, 0:R] = V_2x3x3
            W_2x3x6[0:6, 0:R] = W_2x3x3
            
            # Second copy
            U_2x3x6[:, R:2*R] = U_2x3x3
            V_2x3x6[9:18, R:2*R] = V_2x3x3
            W_2x3x6[6:12, R:2*R] = W_2x3x3
            
            # Verify
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(R_total):
                            val += U_2x3x6[i*m+j, r] * V_2x3x6[j*p+k, r] * W_2x3x6[k*n+i, r]
                        if abs(val - 1.0) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_U = U_2x3x6.tolist()
                best_V = V_2x3x6.tolist()
                best_W = W_2x3x6.tolist()
                best_rank = R_total
                break  # Found a valid decomposition, stop searching for higher ranks

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
