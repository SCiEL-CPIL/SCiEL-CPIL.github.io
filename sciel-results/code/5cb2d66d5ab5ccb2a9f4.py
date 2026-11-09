from scipy.optimize import minimize
import numpy as np
import time
from fractions import Fraction
from itertools import product
from collections import defaultdict

DIMENSIONS = [2, 3, 3]
GLOBAL_DEADLINE = 145.0 # 5s safety margin before 150s kill

def _verify(U, V, W, n, m, p):
    """
    Verify if the decomposition is exact.
    U: (n*m, R)
    V: (m*p, R)
    W: (p*n, R)
    Check: sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] == 1 for all i,j,k
    """
    R = U.shape[1]
    U_mat = U.reshape(n, m, R)
    V_mat = V.reshape(m, p, R)
    W_mat = W.reshape(p, n, R)
    
    T = np.einsum('ijr,jkr,kir->ijk', U_mat, V_mat, W_mat)
    error = np.linalg.norm(T - 1.0)
    return error

def _snap_to_half(arr):
    return np.round(arr * 2) / 2

def _try_rank(R, n, m, p, time_limit=10, num_restarts=10):
    # Initialize
    # Start with random or scaled schoolbook?
    # Random is better for escaping local minima, but schoolbook might be a good basin.
    # Let's try a few random restarts.
    
    best_error = float('inf')
    best_factors = None
    
    start_time = time.time()
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
        if time.time() > GLOBAL_DEADLINE:
            break
            
        # Random initialization
        # Scale to 1
        U = np.random.randn(n * m, R)
        V = np.random.randn(m * p, R)
        W = np.random.randn(p * n, R)
        
        # Flatten
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def objective(x):
            U_flat, V_flat, W_flat = x[:n*m*R], x[n*m*R:n*m*R+m*p*R], x[n*m*R+m*p*R:]
            U_mat = U_flat.reshape(n, m, R)
            V_mat = V_flat.reshape(m, p, R)
            W_mat = W_flat.reshape(p, n, R)
            
            T = np.einsum('ijr,jkr,kir->ijk', U_mat, V_mat, W_mat)
            err = np.sum((T - 1.0)**2)
            return err
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12})
            if res.success:
                x_opt = res.x
                U_flat, V_flat, W_flat = x_opt[:n*m*R], x_opt[n*m*R:n*m*R+m*p*R], x_opt[n*m*R+m*p*R:]
                U_opt = U_flat.reshape(n*m, R)
                V_opt = V_flat.reshape(m*p, R)
                W_opt = W_flat.reshape(p*n, R)
                
                err = _verify(U_opt, V_opt, W_opt, n, m, p)
                if err < best_error:
                    best_error = err
                    best_factors = (U_opt, V_opt, W_opt)
                    
                    # Try snapping
                    U_snap = _snap_to_half(U_opt)
                    V_snap = _snap_to_half(V_opt)
                    W_snap = _snap_to_half(W_opt)
                    err_snap = _verify(U_snap, V_snap, W_snap, n, m, p)
                    if err_snap < 1e-6:
                        return U_snap, V_snap, W_snap
        except Exception:
            pass
            
    # If we found a good solution but didn't snap perfectly, try to snap the best one
    if best_factors is not None and best_error < 1e-4:
        U_opt, V_opt, W_opt = best_factors
        U_snap = _snap_to_half(U_opt)
        V_snap = _snap_to_half(V_opt)
        W_snap = _snap_to_half(W_opt)
        err_snap = _verify(U_snap, V_snap, W_snap, n, m, p)
        if err_snap < 1e-6:
            return U_snap, V_snap, W_snap
            
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    # Try structured patterns for U and V, then solve for W
    # For 2x3x3, try ranks 16 down to 14
    
    for R in [16, 15, 14]:
        if time.time() - start_time > GLOBAL_DEADLINE - 10:
            break
        
        # Generate candidate U and V patterns
        # Strategy: Use sparse patterns with entries from {-1, 0, 1}
        # Try different structural patterns
        
        # Pattern 1: Random sparse patterns
        np.random.seed(42)
        for trial in range(50):
            if time.time() - start_time > GLOBAL_DEADLINE - 5:
                break
            
            # Generate sparse U and V with entries in {-1, 0, 1}
            U_pattern = np.zeros((n * m, R), dtype=int)
            V_pattern = np.zeros((m * p, R), dtype=int)
            
            # Fill with sparse random patterns
            for col in range(R):
                # U: each column has ~3-5 nonzeros
                num_nonzero_u = np.random.randint(3, 6)
                rows_u = np.random.choice(n * m, num_nonzero_u, replace=False)
                for row in rows_u:
                    U_pattern[row, col] = np.random.choice([-1, 1])
                
                # V: each column has ~3-5 nonzeros
                num_nonzero_v = np.random.randint(3, 6)
                rows_v = np.random.choice(m * p, num_nonzero_v, replace=False)
                for row in rows_v:
                    V_pattern[row, col] = np.random.choice([-1, 1])
            
            # Convert to Fraction for exact arithmetic
            U_frac = [[Fraction(U_pattern[i, j]) for j in range(R)] for i in range(n * m)]
            V_frac = [[Fraction(V_pattern[i, j]) for j in range(R)] for i in range(m * p)]
            
            # Build linear system for W
            # For each (i, j, k), we need:
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # This is linear in W.
            
            # Group by W index (k*n+i)
            # For each output position (i, j, k), the equation involves W[k*n+i, r]
            # for all r where U[i*m+j, r] * V[j*p+k, r] != 0
            
            # Build system: for each (i, j, k), equation is:
            # sum_r (U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r] = 1
            
            # Unknowns are W[k*n+i, r] for all k, i, r
            # But W[k*n+i, r] appears in equations for all j
            
            # Let's index W as W_vec[w_idx] where w_idx = (k*n+i)*R + r
            
            num_w = p * n * R
            A = [[Fraction(0) for _ in range(num_w)] for _ in range(n * m * p)]
            b = [Fraction(1)] * (n * m * p)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r in range(R):
                            coeff = U_frac[i * m + j][r] * V_frac[j * p + k][r]
                            if coeff != 0:
                                w_idx = (k * n + i) * R + r
                                A[eq_idx][w_idx] = coeff
                        eq_idx += 1
            
            # Solve
            try:
                x = _solve_linear_system(A, b)
                if x is not None:
                    # Reconstruct W
                    W_frac = [[Fraction(0) for _ in range(R)] for _ in range(p * n)]
                    for k in range(p):
                        for i in range(n):
                            for r in range(R):
                                w_idx = (k * n + i) * R + r
                                W_frac[k * n + i][r] = x[w_idx]
                    
                    # Verify
                    U_np = np.array(U_pattern, dtype=float)
                    V_np = np.array(V_pattern, dtype=float)
                    W_np = np.array([[float(W_frac[i][j]) for j in range(R)] for i in range(p * n)])
                    err = _verify(U_np, V_np, W_np, n, m, p)
                    
                    if err < 1e-10:
                        # Convert to list format
                        U_list = [[float(U_frac[i][j]) for j in range(R)] for i in range(n * m)]
                        V_list = [[float(V_frac[i][j]) for j in range(R)] for i in range(m * p)]
                        W_list = [[float(W_frac[i][j]) for j in range(R)] for i in range(p * n)]
                        return {"U": U_list, "V": V_list, "W": W_list}
            except Exception:
                continue
        
        # Pattern 2: Structured patterns based on known constructions
        # For 2x3x3, try patterns inspired by 3x3x3 rank-16 constructions
        if time.time() - start_time > GLOBAL_DEADLINE - 10:
            break
        
        for trial in range(30):
            if time.time() - start_time > GLOBAL_DEADLINE - 5:
                break
            
            # Create structured U and V
            U_pattern = np.zeros((n * m, R), dtype=int)
            V_pattern = np.zeros((m * p, R), dtype=int)
            
            # Use a structured approach: each column corresponds to a "term"
            # in the multiplication
            for col in range(R):
                # Choose which (i, j) and (j, k) pairs are involved
                # For 2x3x3, i in {0,1}, j in {0,1,2}, k in {0,1,2}
                
                # Random selection with structure
                i_val = np.random.randint(0, n)
                j_val = np.random.randint(0, m)
                k_val = np.random.randint(0, p)
                
                U_pattern[i_val * m + j_val, col] = 1
                V_pattern[j_val * p + k_val, col] = 1
                
                # Sometimes add extra terms
                if np.random.random() < 0.3:
                    i2 = np.random.randint(0, n)
                    j2 = np.random.randint(0, m)
                    k2 = np.random.randint(0, p)
                    if i2 * m + j2 != i_val * m + j_val:
                        U_pattern[i2 * m + j2, col] = np.random.choice([-1, 1])
                    if j2 * p + k2 != j_val * p + k_val:
                        V_pattern[j2 * p + k2, col] = np.random.choice([-1, 1])
            
            # Convert to Fraction
            U_frac = [[Fraction(U_pattern[i, j]) for j in range(R)] for i in range(n * m)]
            V_frac = [[Fraction(V_pattern[i, j]) for j in range(R)] for i in range(m * p)]
            
            # Build and solve linear system (same as above)
            num_w = p * n * R
            A = [[Fraction(0) for _ in range(num_w)] for _ in range(n * m * p)]
            b = [Fraction(1)] * (n * m * p)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r in range(R):
                            coeff = U_frac[i * m + j][r] * V_frac[j * p + k][r]
                            if coeff != 0:
                                w_idx = (k * n + i) * R + r
                                A[eq_idx][w_idx] = coeff
                        eq_idx += 1
            
            try:
                x = _solve_linear_system(A, b)
                if x is not None:
                    W_frac = [[Fraction(0) for _ in range(R)] for _ in range(p * n)]
                    for k in range(p):
                        for i in range(n):
                            for r in range(R):
                                w_idx = (k * n + i) * R + r
                                W_frac[k * n + i][r] = x[w_idx]
                    
                    U_np = np.array(U_pattern, dtype=float)
                    V_np = np.array(V_pattern, dtype=float)
                    W_np = np.array([[float(W_frac[i][j]) for j in range(R)] for i in range(p * n)])
                    err = _verify(U_np, V_np, W_np, n, m, p)
                    
                    if err < 1e-10:
                        U_list = [[float(U_frac[i][j]) for j in range(R)] for i in range(n * m)]
                        V_list = [[float(V_frac[i][j]) for j in range(R)] for i in range(m * p)]
                        W_list = [[float(W_frac[i][j]) for j in range(R)] for i in range(p * n)]
                        return {"U": U_list, "V": V_list, "W": W_list}
            except Exception:
                continue
    
    # Fallback to schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
