"""Rewritten program for matrix multiplication 3,5,6 rank-minimal decomposition."""
import time
import numpy as np
import random
from scipy.optimize import linprog

DIMENSIONS = [3, 5, 6]

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor as a flat vector
    T_target = np.zeros((nm, mp, pn), dtype=float)
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                d = k * n + i
                T_target[a, b, d] = 1.0
                target_indices.append((a, b, d))
    
    T_flat = np.zeros(nm * mp * pn, dtype=float)
    for idx, (a, b, d) in enumerate(target_indices):
        T_flat[a * mp * pn + b * pn + d] = 1.0
    
    def verify(U, V, W, rank):
        """Verify exact decomposition using numpy einsum."""
        if time.time() > deadline:
            return False
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        T_computed = np.einsum('ac,bc,dc->abd', U_np, V_np, W_np)
        if not np.allclose(T_computed, T_target, atol=1e-9):
            return False
        return True
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W, rank
    
    best_U, best_V, best_W, best_rank = schoolbook()
    
    if time.time() > deadline - 5:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    def try_lp_rank(rank, max_iters=3):
        """
        Use LP to find sparse factor matrices.
        Variables: x_U (nm*rank), x_V (mp*rank), x_W (pn*rank)
        Constraint: sum_c U[a,c]*V[b,c]*W[d,c] = T[a,b,d] for all (a,b,d)
        This is bilinear, so we linearize by fixing two matrices.
        
        Strategy: Fix U and V randomly, solve for W via least squares.
        Then fix U and W, solve for V. Alternate.
        Use L1 regularization to promote sparsity.
        """
        if time.time() > deadline - 10:
            return None
        
        for trial in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Random sparse initialization
            U = np.zeros((nm, rank))
            V = np.zeros((mp, rank))
            W = np.zeros((pn, rank))
            
            for c in range(rank):
                # Each column has a few non-zero entries
                nz_u = random.randint(2, min(5, nm))
                nz_v = random.randint(2, min(5, mp))
                nz_w = random.randint(2, min(5, pn))
                
                u_idx = random.sample(range(nm), nz_u)
                v_idx = random.sample(range(mp), nz_v)
                w_idx = random.sample(range(pn), nz_w)
                
                for idx in u_idx:
                    U[idx, c] = random.choice([-1, 1])
                for idx in v_idx:
                    V[idx, c] = random.choice([-1, 1])
                for idx in w_idx:
                    W[idx, c] = random.choice([-1, 1])
            
            # Alternate least squares with sparsification
            for iteration in range(50):
                if time.time() > deadline - 10:
                    break
                
                # Solve for W: T[a,b,d] = sum_c U[a,c]*V[b,c]*W[d,c]
                # For each d: T[:, :, d] = sum_c (U[:, c] * V[:, c]) * W[d, c]
                # This is: for each (a,b), T[a,b,d] = sum_c M[a,b,c] * W[d,c]
                # where M[a,b,c] = U[a,c] * V[b,c]
                
                # Build linear system for W
                # Variables: W[d, c] for d in range(pn), c in range(rank)
                # Constraints: for each (a,b,d): sum_c U[a,c]*V[b,c]*W[d,c] = T[a,b,d]
                
                # Instead of full LP, use iterative refinement
                # For each d, we have: sum_c (U[:,c] * V[:,c])^T * T[:, :, d] = W[d,c]
                # This is underdetermined, so use least squares
                
                M = np.einsum('ac,bc->abc', U, V)  # (nm, mp, rank)
                M_flat = M.reshape(nm * mp, rank)
                
                # For each d, solve: M_flat @ W[d, :] = T[:, :, d].flatten()
                for d in range(pn):
                    target_d = T_target[:, :, d].flatten()
                    # Least squares: min ||M_flat @ w - target_d||^2
                    # Add L1 penalty for sparsity
                    try:
                        w_sol, residuals, rank_val, sv = np.linalg.lstsq(M_flat, target_d, rcond=None)
                        W[d, :] = w_sol
                    except:
                        pass
                
                # Sparsify W
                W[np.abs(W) < 0.1] = 0
                
                # Solve for V similarly
                N = np.einsum('ac,dc->adc', U, W)  # (nm, pn, rank)
                N_flat = N.reshape(nm * pn, rank)
                
                for b in range(mp):
                    target_b = T_target[:, b, :].flatten()
                    try:
                        v_sol, residuals, rank_val, sv = np.linalg.lstsq(N_flat, target_b, rcond=None)
                        V[b, :] = v_sol
                    except:
                        pass
                
                # Sparsify V
                V[np.abs(V) < 0.1] = 0
                
                # Solve for U similarly
                P = np.einsum('bc,dc->bdc', V, W)  # (mp, pn, rank)
                P_flat = P.reshape(mp * pn, rank)
                
                for a in range(nm):
                    target_a = T_target[a, :, :].flatten()
                    try:
                        u_sol, residuals, rank_val, sv = np.linalg.lstsq(P_flat, target_a, rcond=None)
                        U[a, :] = u_sol
                    except:
                        pass
                
                # Sparsify U
                U[np.abs(U) < 0.1] = 0
                
                # Check convergence
                T_computed = np.einsum('ac,bc,dc->abd', U, V, W)
                err = np.max(np.abs(T_computed - T_target))
                if err < 1e-9:
                    # Convert to list format
                    U_list = [[float(x) for x in row] for row in U]
                    V_list = [[float(x) for x in row] for row in V]
                    W_list = [[float(x) for x in row] for row in W]
                    if verify(U_list, V_list, W_list, rank):
                        return U_list, V_list, W_list, rank
            
            # After iterations, try rounding to half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            U_list = [[float(x) for x in row] for row in U_rounded]
            V_list = [[float(x) for x in row] for row in V_rounded]
            W_list = [[float(x) for x in row] for row in W_rounded]
            
            if verify(U_list, V_list, W_list, rank):
                return U_list, V_list, W_list, rank
        
        return None
    
    def try_lp_rank_fixed(rank, max_trials=5):
        """
        Alternative LP approach: Fix U and V, solve for W exactly.
        For each d, we need: sum_c U[a,c]*V[b,c]*W[d,c] = T[a,b,d] for all (a,b)
        This is a linear system in W[d, c].
        """
        if time.time() > deadline - 10:
            return None
        
        for trial in range(max_trials):
            if time.time() > deadline - 10:
                break
            
            # Initialize U and V sparsely
            U = np.zeros((nm, rank))
            V = np.zeros((mp, rank))
            
            for c in range(rank):
                nz_u = random.randint(3, min(6, nm))
                nz_v = random.randint(3, min(6, mp))
                
                u_idx = random.sample(range(nm), nz_u)
                v_idx = random.sample(range(mp), nz_v)
                
                for idx in u_idx:
                    U[idx, c] = random.choice([-1, 1])
                for idx in v_idx:
                    V[idx, c] = random.choice([-1, 1])
            
            # For each d, solve linear system for W[d, :]
            W = np.zeros((pn, rank))
            
            # Build coefficient matrix: for each (a,b), M[a,b,c] = U[a,c]*V[b,c]
            M = np.einsum('ac,bc->abc', U, V)  # (nm, mp, rank)
            M_flat = M.reshape(nm * mp, rank)
            
            success = True
            for d in range(pn):
                target_d = T_target[:, :, d].flatten()
                # Solve M_flat @ w = target_d
                try:
                    w_sol, residuals, rank_val, sv = np.linalg.lstsq(M_flat, target_d, rcond=None)
                    W[d, :] = w_sol
                except:
                    success = False
                    break
            
            if not success:
                continue
            
            # Sparsify W
            W[np.abs(W) < 0.1] = 0
            
            # Verify
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            
            if verify(U_list, V_list, W_list, rank):
                return U_list, V_list, W_list, rank
            
            # Try rounding W to half-integers
            W_rounded = np.round(W * 2) / 2
            W_list2 = [[float(x) for x in row] for row in W_rounded]
            
            if verify(U_list, V_list, W_list2, rank):
                return U_list, V_list, W_list2, rank
        
        return None
    
    # Try ranks from 85 down to 70
    for rank in range(85, 69, -1):
        if time.time() > deadline - 15:
            break
        
        # Try LP-based approach
        result = try_lp_rank_fixed(rank, max_trials=3)
        if result is not None:
            best_U, best_V, best_W, best_rank = result
            break
        
        if time.time() > deadline - 15:
            continue
        
        # Try alternating least squares
        result = try_lp_rank(rank, max_iters=2)
        if result is not None:
            best_U, best_V, best_W, best_rank = result
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
