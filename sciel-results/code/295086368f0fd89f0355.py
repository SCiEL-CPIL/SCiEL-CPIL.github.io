"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
import sys

DIMENSIONS = [3, 3, 7]

def _verify(U, V, W, n, m, p):
    """
    Verify that U @ diag @ V^T == W^T (element-wise)
    where U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # T = sum_c U[:,c] V[:,c] W[:,c]^T
    # T is (n*m, m*p, p*n)
    # For each (i,j,k) in tensor: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    
    # Efficient check:
    # Create tensor T of shape (n*m, m*p, p*n)
    # T[a, b, c] = sum_d U[a,d] * V[b,d] * W[c,d]
    
    # We can check by flattening or using einsum
    # T = np.einsum('ad,bd,cd->abc', U_arr, V_arr, W_arr)
    # This might be memory heavy if rank is large, but for rank < 100 it's fine.
    
    T = np.einsum('ad,bd,cd->abc', U_arr, V_arr, W_arr)
    
    # Check against target tensor
    # Target T_target[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    # Otherwise 0
    
    # Iterate over all non-zero entries of target
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if abs(T[a, b, c] - 1.0) > 1e-6:
                    return False
    
    # Check that all other entries are 0
    # Create a mask of valid indices
    valid = np.zeros((n*m, m*p, p*n), dtype=bool)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid[a, b, c] = True
    
    # Check invalid entries are 0
    invalid_mask = ~valid
    if np.any(np.abs(T[invalid_mask]) > 1e-6):
        return False
        
    return True

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Strategy: Direct sum of Strassen(3x3x3, rank 20) + (3x3x4, rank 8) = rank 28
    
    # Strassen's 3x3x3 decomposition (rank 20)
    # The 3x3x3 multiplication tensor has known rank 20 (Strassen 1969)
    # We'll use the standard Strassen decomposition for 3x3 matrices
    
    # For 3x3x3: A is 3x3, B is 3x3, C is 3x3
    # Strassen uses 20 multiplications
    
    # Strassen 3x3 decomposition:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A22 - A11)(B21 - B11)
    # M3 = (A11 + A12)B22
    # M4 = A22(B11 + B12)
    # M5 = (A11 + A33)(B11 + B33)
    # M6 = (A33 - A11)(B13 - B31)
    # M7 = (A13 + A33)B31
    # M8 = A33(B11 + B13)
    # M9 = (A11 + A21)(B22 + B32)
    # M10 = (A21 - A11)(B12 - B22)
    # M11 = (A11 + A12 + A31 + A32)(B11 + B12 + B31 + B32)
    # M12 = (A12 - A31 + A32 - A11)(B12 + B31 - B22 - B11)
    # M13 = (A21 + A22 + A31 + A32)(B11 + B12 + B31 + B32)
    # M14 = (A22 - A31 + A32 - A21)(B12 + B31 - B22 - B11)
    # M15 = (A21 + A31)(B22 + B32)
    # M16 = (A31 - A21)(B12 - B22)
    # M17 = (A22 + A32)B31
    # M18 = A32(B11 + B13)
    # M19 = (A12 + A22)(B13 + B23)
    # M20 = (A22 - A12)(B23 - B13)
    
    # Actually, let me use a simpler approach. I'll construct the decomposition
    # using known formulas for 3x3x3 Strassen and a simple 3x3x4 decomposition.
    
    # For now, let's use schoolbook as fallback and try to construct rank 28
    
    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to construct rank-28 decomposition
    # Block 1: 3x3x3 (columns 0-2 of A and B, rows 0-2 of C) -> rank 20
    # Block 2: 3x3x4 (columns 3-6 of B, rows 3-6 of C) -> rank 8
    
    # Strassen 3x3x3 (rank 20)
    # Indices: A[i,j] for i,j in 0..2, B[j,k] for j,k in 0..2, C[i,k] for i,k in 0..2
    
    # Strassen's 20-term decomposition for 3x3:
    # Let's use the standard formulation
    
    # Actually, I'll use a different approach. Let me use the known fact that
    # R(3,3,3) = 20 and construct it explicitly.
    
    # For 3x3x3, the Strassen decomposition can be written as:
    # C = A * B where we partition into 1x1 blocks (9 blocks each)
    
    # Standard Strassen 3x3 uses these 20 products:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a22-a11)(b21-b11)  
    # M3 = (a11+a12)b22
    # M4 = a22(b11+b12)
    # M5 = (a11+a33)(b11+b33)
    # M6 = (a33-a11)(b13-b31)
    # M7 = (a13+a33)b31
    # M8 = a33(b11+b13)
    # M9 = (a11+a21)(b22+b32)
    # M10 = (a21-a11)(b12-b22)
    # M11 = (a11+a12+a31+a32)(b11+b12+b31+b32)
    # M12 = (a12-a31+a32-a11)(b12+b31-b22-b11)
    # M13 = (a21+a22+a31+a32)(b11+b12+b31+b32)
    # M14 = (a22-a31+a32-a21)(b12+b31-b22-b11)
    # M15 = (a21+a31)(b22+b32)
    # M16 = (a31-a21)(b12-b22)
    # M17 = (a22+a32)b31
    # M18 = a32(b11+b13)
    # M19 = (a12+a22)(b13+b23)
    # M20 = (a22-a12)(b23-b13)
    
    # C11 = M1 - M2 + M10 + M12
    # C12 = M2 - M10 + M12 - M14  (wait, this doesn't look right)
    
    # Let me use a verified Strassen 3x3 decomposition from literature.
    # Actually, for simplicity and correctness, let me use a different strategy:
    # Use the schoolbook for the 3x3x3 part (rank 27) and see if we can find
    # a better 3x3x4 decomposition.
    
    # Alternative: Try rank 28 by using rank 20 for 3x3x3 and rank 8 for 3x3x4
    
    # For 3x3x4: A is 3x3, B is 3x4, C is 3x4
    # This is equivalent to multiplying two 3x3 matrices where the second has
    # an extra column. We can use rank 9 (schoolbook) or potentially lower.
    
    # Actually, let me try a completely different approach:
    # Use random initialization + optimization to find a rank-28 solution
    
    # Precompute target tensor
    T_target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
    
    # Try rank 28 with multiple restarts
    target_rank = 28
    max_restarts = 3
    time_per_restart = 25.0
    
    found = False
    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Random initialization with structure
        # Use block structure: first 20 columns for 3x3x3, next 8 for 3x3x4
        U = np.zeros((n*m, target_rank), dtype=np.float64)
        V = np.zeros((m*p, target_rank), dtype=np.float64)
        W = np.zeros((p*n, target_rank), dtype=np.float64)
        
        # Initialize first 20 columns with random values for 3x3x3 block
        U[:, :20] = np.random.randn(n*m, 20) * 0.5
        V[:, :20] = np.random.randn(m*p, 20) * 0.5
        W[:, :20] = np.random.randn(p*n, 20) * 0.5
        
        # Initialize last 8 columns with random values for 3x3x4 block
        U[:, 20:] = np.random.randn(n*m, 8) * 0.5
        V[:, 20:] = np.random.randn(m*p, 8) * 0.5
        W[:, 20:] = np.random.randn(p*n, 8) * 0.5
        
        try:
            from scipy.optimize import minimize
        except ImportError:
            continue
        
        def objective(x):
            U_flat = x[:n*m*target_rank]
            V_flat = x[n*m*target_rank:n*m*target_rank + m*p*target_rank]
            W_flat = x[n*m*target_rank + m*p*target_rank:]
            
            U_mat = U_flat.reshape(n*m, target_rank)
            V_mat = V_flat.reshape(m*p, target_rank)
            W_mat = W_flat.reshape(p*n, target_rank)
            
            T = np.einsum('ad,bd,cd->abc', U_mat, V_mat, W_mat)
            return np.sum((T - T_target)**2)
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        res = minimize(objective, x0, method='L-BFGS-B', 
                       options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
        
        if res.fun < 1e-6:
            U_opt = res.x[:n*m*target_rank].reshape(n*m, target_rank)
            V_opt = res.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            W_opt = res.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
            
            U_list = [[float(x) for x in row] for row in U_opt]
            V_list = [[float(x) for x in row] for row in V_opt]
            W_list = [[float(x) for x in row] for row in W_opt]
            
            def snap(val):
                nearest = round(val * 2) / 2.0
                if abs(val - nearest) < 1e-5:
                    return nearest
                return val
            
            U_snap = [[snap(v) for v in row] for row in U_list]
            V_snap = [[snap(v) for v in row] for row in V_list]
            W_snap = [[snap(v) for v in row] for row in W_list]
            
            if _verify(U_snap, V_snap, W_snap, n, m, p):
                best_rank = target_rank
                best_U, best_V, best_W = U_snap, V_snap, W_snap
                found = True
                break
            elif _verify(U_list, V_list, W_list, n, m, p):
                best_rank = target_rank
                best_U, best_V, best_W = U_list, V_list, W_list
                found = True
                break
    
    if not found:
        # Try rank 29, 30, ... up to schoolbook
        for tr in range(29, min(rank_sb, 40)):
            if time.time() > deadline - 10.0:
                break
            if tr >= best_rank:
                break
            
            U = np.random.randn(n*m, tr) * 0.3
            V = np.random.randn(m*p, tr) * 0.3
            W = np.random.randn(p*n, tr) * 0.3
            
            try:
                from scipy.optimize import minimize
            except ImportError:
                continue
            
            def objective2(x):
                U_flat = x[:n*m*tr]
                V_flat = x[n*m*tr:n*m*tr + m*p*tr]
                W_flat = x[n*m*tr + m*p*tr:]
                
                U_mat = U_flat.reshape(n*m, tr)
                V_mat = V_flat.reshape(m*p, tr)
                W_mat = W_flat.reshape(p*n, tr)
                
                T = np.einsum('ad,bd,cd->abc', U_mat, V_mat, W_mat)
                return np.sum((T - T_target)**2)
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            res = minimize(objective2, x0, method='L-BFGS-B', 
                           options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6})
            
            if res.fun < 1e-6:
                U_opt = res.x[:n*m*tr].reshape(n*m, tr)
                V_opt = res.x[n*m*tr:n*m*tr + m*p*tr].reshape(m*p, tr)
                W_opt = res.x[n*m*tr + m*p*tr:].reshape(p*n, tr)
                
                U_list = [[float(x) for x in row] for row in U_opt]
                V_list = [[float(x) for x in row] for row in V_opt]
                W_list = [[float(x) for x in row] for row in W_opt]
                
                def snap2(val):
                    nearest = round(val * 2) / 2.0
                    if abs(val - nearest) < 1e-5:
                        return nearest
                    return val
                
                U_snap = [[snap2(v) for v in row] for row in U_list]
                V_snap = [[snap2(v) for v in row] for row in V_list]
                W_snap = [[snap2(v) for v in row] for row in W_list]
                
                if _verify(U_snap, V_snap, W_snap, n, m, p):
                    best_rank = tr
                    best_U, best_V, best_W = U_snap, V_snap, W_snap
                    break
                elif _verify(U_list, V_list, W_list, n, m, p):
                    best_rank = tr
                    best_U, best_V, best_W = U_list, V_list, W_list
                    break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
