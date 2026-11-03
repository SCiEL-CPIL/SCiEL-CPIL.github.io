"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize
from itertools import product

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """Find rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    schoolbook_rank = n * m * p  # 36
    
    def verify_decomposition(U, V, W, rank, tol=1e-9):
        """Verify that the decomposition exactly represents matrix multiplication."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def verify_numpy(U, V, W, rank, tol=1e-9):
        """Faster verification using numpy."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def snap_to_half(x):
        """Snap values to nearest half-integer."""
        return np.round(2.0 * x) / 2.0
    
    def schoolbook():
        rank = schoolbook_rank
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
        return U, V, W, rank
    
    def objective_and_grad(params, rank):
        """
        Objective: sum of (sum_c U_ijc * V_jkc * W_kic - 1)^2 over all (i,j,k).
        params: flat vector of length (n*m + m*p + p*n) * rank
        Layout: U (n*m*rank), V (m*p*rank), W (p*n*rank)
        """
        U = params[0:n*m*rank].reshape(n*m, rank)
        V = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
        
        # Compute residual
        res = 0.0
        # For efficiency, use numpy
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    res += (val - 1.0) ** 2
        
        return res
    
    def try_rank_opt(rank, n_starts=5, max_iters=200):
        """
        Try to find a decomposition of given rank using optimization.
        Uses alternating minimization + local optimization.
        """
        if time.time() > deadline - 20:
            return None
        
        best_params = None
        best_obj = float('inf')
        
        for start in range(n_starts):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(42 + start * 7 + rank)
            
            # Initialize with structured random values
            # Use a mix of zeros, ±1, and small random values
            U_init = np.zeros((n*m, rank))
            V_init = np.zeros((m*p, rank))
            W_init = np.zeros((p*n, rank))
            
            # Pattern: for each (i,j,k), assign a random column
            for c in range(rank):
                # Assign some entries
                for i in range(n):
                    for j in range(m):
                        if np.random.random() < 0.5:
                            U_init[i*m+j, c] = np.random.choice([-1, 0, 1])
                        elif np.random.random() < 0.3:
                            U_init[i*m+j, c] = np.random.uniform(-0.5, 0.5)
                
                for j in range(m):
                    for k in range(p):
                        if np.random.random() < 0.5:
                            V_init[j*p+k, c] = np.random.choice([-1, 0, 1])
                        elif np.random.random() < 0.3:
                            V_init[j*p+k, c] = np.random.uniform(-0.5, 0.5)
                
                for k in range(p):
                    for i in range(n):
                        if np.random.random() < 0.5:
                            W_init[k*n+i, c] = np.random.choice([-1, 0, 1])
                        elif np.random.random() < 0.3:
                            W_init[k*n+i, c] = np.random.uniform(-0.5, 0.5)
            
            params_init = np.concatenate([
                U_init.flatten(), V_init.flatten(), W_init.flatten()
            ])
            
            # Alternating minimization
            U = U_init.copy()
            V = V_init.copy()
            W = W_init.copy()
            
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    break
                
                # Fix V, W; solve for U
                # For each (i,j): sum_k,c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                # This is a linear system in U[i*m+j, :]
                # Actually, for each (i,j), we have p equations (one per k) and rank unknowns
                # Wait, that's not right. The constraint is:
                # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                # If we fix V and W, for a given (i,j), we have p equations in rank unknowns
                # (the U[i*m+j, c] for c=0..rank-1). But p=6 and rank may be < 6.
                # So this is overdetermined. Use least squares.
                
                # Better approach: fix two, optimize one via gradient or least squares.
                
                # Let's fix V and W, and for each (i,j), solve least squares:
                # For each (i,j): minimize sum_k (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
                # This is a quadratic in U[i*m+j, :], solvable analytically.
                
                U_new = U.copy()
                for i in range(n):
                    for j in range(m):
                        # Build A (p x rank) and b (p,)
                        A = np.zeros((p, rank))
                        b = np.ones(p)
                        for k in range(p):
                            A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        # Solve A x = b in least squares
                        try:
                            U_new[i*m+j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                
                V_new = V.copy()
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n*m, rank))
                        b = np.ones(n*m)
                        for i in range(n):
                            for jj in range(m):
                                A[i*m+jj, :] = U_new[i*m+jj, :] * W[k*n+i, :]
                        try:
                            V_new[j*p+k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                
                W_new = W.copy()
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((n*m, rank))
                        b = np.ones(n*m)
                        for jj in range(m):
                            for ii in range(n):
                                A[ii*m+jj, :] = U_new[ii*m+jj, :] * V_new[jj*p+k, :]
                        try:
                            W_new[k*n+i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                
                U, V, W = U_new, V_new, W_new
                
                # Check convergence
                if it % 10 == 0:
                    obj = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                                obj += (val - 1.0) ** 2
                    
                    if obj < best_obj:
                        best_obj = obj
                        best_params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                    
                    if obj < 1e-12:
                        # Try to snap to half-integers
                        U_snap = snap_to_half(U)
                        V_snap = snap_to_half(V)
                        W_snap = snap_to_half(W)
                        if verify_numpy(U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), rank, tol=1e-8):
                            return (U_snap.tolist(), V_snap.tolist(), W_snap.tolist())
                    
                    if time.time() > deadline - 20:
                        break
        
        # After alternating, try to polish the best solution
        if best_params is not None and best_obj < 1e-6:
            U = best_params[0:n*m*rank].reshape(n*m, rank)
            V = best_params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
            W = best_params[n*m*rank + m*p*rank:].reshape(p*n, rank)
            
            # Final snap attempt
            U_snap = snap_to_half(U)
            V_snap = snap_to_half(V)
            W_snap = snap_to_half(W)
            if verify_numpy(U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), rank, tol=1e-8):
                return (U_snap.tolist(), V_snap.tolist(), W_snap.tolist())
            
            # Try small perturbations around the solution
            for _ in range(20):
                if time.time() > deadline - 10:
                    break
                U_pert = U + np.random.randn(n*m, rank) * 1e-4
                V_pert = V + np.random.randn(m*p, rank) * 1e-4
                W_pert = W + np.random.randn(p*n, rank) * 1e-4
                U_s = snap_to_half(U_pert)
                V_s = snap_to_half(V_pert)
                W_s = snap_to_half(W_pert)
                if verify_numpy(U_s.tolist(), V_s.tolist(), W_s.tolist(), rank, tol=1e-8):
                    return (U_s.tolist(), V_s.tolist(), W_s.tolist())
        
        return None
    
    # Try ranks from 18 up to 36
    for rank in range(18, 37):
        if time.time() > deadline - 30:
            break
        
        result = try_rank_opt(rank, n_starts=3, max_iters=50)
        if result is not None:
            U, V, W = result
            if verify_decomposition(U, V, W, rank):
                return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U_fb, V_fb, W_fb, rank_fb = schoolbook()
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
