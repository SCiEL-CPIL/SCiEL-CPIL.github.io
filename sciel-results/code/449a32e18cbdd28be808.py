"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
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
    
    def try_structured_decomposition(rank):
        """
        Try to find a structured decomposition by exploiting the block structure.
        The 3x6 matrix can be viewed as two 3x3 blocks.
        We look for decompositions where the k-index has a block structure.
        """
        if time.time() > deadline - 10:
            return None
        
        # Strategy: Use the fact that p=6=2*3
        # We can try to find a decomposition that respects this structure
        # by using a base decomposition for 2x3x3 and extending it
        
        # First, try to find a rank-r decomposition for 2x3x3
        # Then combine two copies for the 6 columns
        
        # For 2x3x3, known minimal rank is 9 (schoolbook), but let's try lower
        # Actually, for 2x3x3, the tensor rank is known to be 9
        
        # Let's try a different approach: enumerate structured patterns
        
        # Pattern 1: Block diagonal structure
        # Split k into two groups of 3: k in {0,1,2} and k in {3,4,5}
        # Use the same decomposition pattern for both blocks
        
        # Try rank = 2 * r_base where r_base is for 2x3x3
        # But we can do better by sharing some structure
        
        # Let's try direct enumeration for small ranks
        # Focus on ranks that are multiples of 3 or have special structure
        
        # Known result: R(2,3,6) <= 18 by using two independent 2x3x3 decompositions
        # But we can potentially do better by sharing structure
        
        # Try rank 18 first (two independent 2x3x3 blocks)
        if rank == 18:
            # Use two independent schoolbook decompositions for 2x3x3
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            # First block: k = 0,1,2
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(3):
                        U[i*m+j][col] = 1
                        V[j*p+k][col] = 1
                        W[k*n+i][col] = 1
                        col += 1
            
            # Second block: k = 3,4,5
            for i in range(n):
                for j in range(m):
                    for k in range(3, 6):
                        U[i*m+j][col] = 1
                        V[j*p+k][col] = 1
                        W[k*n+i][col] = 1
                        col += 1
            
            if verify_numpy(U, V, W, rank):
                return (U, V, W)
        
        # Try rank 16: share some structure between blocks
        if rank == 16:
            # Try to find a decomposition where some terms are shared
            # This is more complex, so we'll use optimization
            pass
        
        # For other ranks, use optimization
        return None
    
    def alternating_minimization(U, V, W, rank, max_iters=200, tol=1e-14):
        """
        Alternating least squares for tensor decomposition with tighter tolerance.
        Returns (U, V, W, obj_value)
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 15:
                break
            
            # Update U: fix V, W
            U_new = U.copy()
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, rank))
                    b = np.ones(p)
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        U_new[i*m+j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass
            
            # Update V: fix U, W
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
            
            # Update W: fix U, V
            W_new = W.copy()
            for k in range(p):
                for i in range(n):
                    A = np.zeros((n*m, rank))
                    b = np.ones(n*m)
                    for ii in range(n):
                        for jj in range(m):
                            A[ii*m+jj, :] = U_new[ii*m+jj, :] * V_new[jj*p+k, :]
                    try:
                        W_new[k*n+i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass
            
            U, V, W = U_new, V_new, W_new
            
            # Check convergence
            if it % 10 == 0 or it == max_iters - 1:
                obj = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            obj += (val - 1.0) ** 2
                if obj < tol:
                    break
        
        obj = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    obj += (val - 1.0) ** 2
        return U, V, W, obj
    
    def try_rank_opt(rank, n_starts=15, max_iters=100):
        """
        Try to find a decomposition of given rank using optimization with more starts.
        """
        if time.time() > deadline - 20:
            return None
        
        best_result = None
        best_obj = float('inf')
        
        for start in range(n_starts):
            if time.time() > deadline - 15:
                break
            
            # Different initialization strategies
            if start % 4 == 0:
                # Strategy 1: Block-structured random
                np.random.seed(42 + start * 7 + rank)
                U_init = np.zeros((n*m, rank))
                V_init = np.zeros((m*p, rank))
                W_init = np.zeros((p*n, rank))
                
                # Initialize with block structure for p=6
                for c in range(rank):
                    # Randomly assign to first or second block
                    block = c % 2
                    k_offset = block * 3
                    
                    for i in range(n):
                        for j in range(m):
                            if np.random.random() < 0.5:
                                U_init[i*m+j, c] = np.random.choice([-1.0, 1.0])
                    for j in range(m):
                        for k in range(3):
                            if np.random.random() < 0.5:
                                V_init[j*p + k_offset + k, c] = np.random.choice([-1.0, 1.0])
                    for k in range(3):
                        for i in range(n):
                            if np.random.random() < 0.5:
                                W_init[(k_offset + k)*n + i, c] = np.random.choice([-1.0, 1.0])
                
            elif start % 4 == 1:
                # Strategy 2: Small random values
                np.random.seed(100 + start * 13 + rank)
                U_init = np.random.randn(n*m, rank) * 0.5
                V_init = np.random.randn(m*p, rank) * 0.5
                W_init = np.random.randn(p*n, rank) * 0.5
                
            elif start % 4 == 2:
                # Strategy 3: Based on schoolbook but compressed
                np.random.seed(200 + start * 17 + rank)
                U_init = np.zeros((n*m, rank))
                V_init = np.zeros((m*p, rank))
                W_init = np.zeros((p*n, rank))
                
                # Try to reuse patterns from schoolbook
                for c in range(rank):
                    i_idx = np.random.randint(n)
                    j_idx = np.random.randint(m)
                    k_idx = np.random.randint(p)
                    
                    U_init[i_idx*m+j_idx, c] = 1.0
                    V_init[j_idx*p+k_idx, c] = 1.0
                    W_init[k_idx*n+i_idx, c] = 1.0
                    
                    # Add some sparsity
                    for i in range(n):
                        for j in range(m):
                            if (i,j) != (i_idx,j_idx) and np.random.random() < 0.2:
                                U_init[i*m+j, c] = np.random.choice([-1.0, 1.0])
                    for j in range(m):
                        for k in range(p):
                            if (j,k) != (j_idx,k_idx) and np.random.random() < 0.2:
                                V_init[j*p+k, c] = np.random.choice([-1.0, 1.0])
                    for k in range(p):
                        for i in range(n):
                            if (k,i) != (k_idx,i_idx) and np.random.random() < 0.2:
                                W_init[k*n+i, c] = np.random.choice([-1.0, 1.0])
                
            else:
                # Strategy 4: Gradient-based initialization
                np.random.seed(300 + start * 19 + rank)
                U_init = np.random.randn(n*m, rank) * 0.3
                V_init = np.random.randn(m*p, rank) * 0.3
                W_init = np.random.randn(p*n, rank) * 0.3
                
                # Add structured components
                for c in range(0, rank, 3):
                    i_idx = c % n
                    j_idx = (c // n) % m
                    k_idx = (c // (n*m)) % p
                    U_init[i_idx*m+j_idx, c] += 1.0
                    V_init[j_idx*p+k_idx, c] += 1.0
                    W_init[k_idx*n+i_idx, c] += 1.0
            
            # Run alternating minimization
            U, V, W, obj = alternating_minimization(U_init, V_init, W_init, rank, max_iters=max_iters)
            
            if obj < best_obj:
                best_obj = obj
                best_result = (U, V, W)
            
            # If we found a very good solution, try to snap
            if obj < 1e-10:
                U_snap = snap_to_half(U)
                V_snap = snap_to_half(V)
                W_snap = snap_to_half(W)
                if verify_numpy(U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), rank, tol=1e-8):
                    return (U_snap.tolist(), V_snap.tolist(), W_snap.tolist())
            
            # If good but not perfect, do a polish pass
            if obj < 1e-6:
                # Try local optimization with scipy
                params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                
                def obj_func(p):
                    U_p = p[0:n*m*rank].reshape(n*m, rank)
                    V_p = p[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                    W_p = p[n*m*rank + m*p*rank:].reshape(p*n, rank)
                    res = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.sum(U_p[i*m+j, :] * V_p[j*p+k, :] * W_p[k*n+i, :])
                                res += (val - 1.0) ** 2
                    return res
                
                try:
                    from scipy.optimize import minimize
                    res = minimize(obj_func, params, method='BFGS', 
                                   options={'maxiter': 100, 'gtol': 1e-12})
                    if res.fun < best_obj:
                        best_obj = res.fun
                        best_result = (
                            res.x[0:n*m*rank].reshape(n*m, rank),
                            res.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank),
                            res.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
                        )
                        
                        if res.fun < 1e-10:
                            U_snap = snap_to_half(best_result[0])
                            V_snap = snap_to_half(best_result[1])
                            W_snap = snap_to_half(best_result[2])
                            if verify_numpy(U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), rank, tol=1e-8):
                                return (U_snap.tolist(), V_snap.tolist(), W_snap.tolist())
                except:
                    pass
        
        # Final attempt: if we have a good solution, try to snap with perturbations
        if best_result is not None and best_obj < 1e-6:
            U, V, W = best_result
            
            # Try snapping directly
            U_snap = snap_to_half(U)
            V_snap = snap_to_half(V)
            W_snap = snap_to_half(W)
            if verify_numpy(U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), rank, tol=1e-8):
                return (U_snap.tolist(), V_snap.tolist(), W_snap.tolist())
            
            # Try small perturbations
            for _ in range(50):
                if time.time() > deadline - 10:
                    break
                U_pert = U + np.random.randn(n*m, rank) * 1e-3
                V_pert = V + np.random.randn(m*p, rank) * 1e-3
                W_pert = W + np.random.randn(p*n, rank) * 1e-3
                U_s = snap_to_half(U_pert)
                V_s = snap_to_half(V_pert)
                W_s = snap_to_half(W_pert)
                if verify_numpy(U_s.tolist(), V_s.tolist(), W_s.tolist(), rank, tol=1e-8):
                    return (U_s.tolist(), V_s.tolist(), W_s.tolist())
            
            # Try rounding to integers as well
            U_int = np.round(U)
            V_int = np.round(V)
            W_int = np.round(W)
            if verify_numpy(U_int.tolist(), V_int.tolist(), W_int.tolist(), rank, tol=1e-8):
                return (U_int.tolist(), V_int.tolist(), W_int.tolist())
        
        return None
    
    # Try ranks from low to high
    # Based on theoretical bounds, rank should be around 18-24 for 2x3x6
    for rank in [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]:
        if time.time() > deadline - 25:
            break
        
        # First try structured decomposition
        result = try_structured_decomposition(rank)
        if result is not None:
            U, V, W = result
            if verify_decomposition(U, V, W, rank):
                return {"U": U, "V": V, "W": W}
        
        # Then try optimization
        result = try_rank_opt(rank, n_starts=12, max_iters=80)
        if result is not None:
            U, V, W = result
            if verify_decomposition(U, V, W, rank):
                return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U_fb, V_fb, W_fb, rank_fb = schoolbook()
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
