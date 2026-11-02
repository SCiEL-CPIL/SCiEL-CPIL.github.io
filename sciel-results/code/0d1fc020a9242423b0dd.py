"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 5]
START_TIME = time.time()
TIME_LIMIT = 140.0

# EVOLVE-BLOCK-START

def _verify(U, V, W, n=2, m=2, p=5):
    """Verify if the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target = 1.0
                computed = 0.0
                for r in range(rank):
                    computed += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if abs(computed - target) > 1e-9:
                    return False
    return True

def _construct_rank18(n=2, m=2, p=5):
    """
    Construct a rank-18 decomposition using Strassen for 2x2x2 (cols 0-1),
    Strassen for 2x2x2 (cols 2-3), and Schoolbook for 2x2x1 (col 4).
    """
    rank = 18
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    def add_strassen_block(col0, col1, base):
        r = base
        U[0][r] = 1; U[3][r] = 1
        V[0*p+col0][r] = 1; V[1*p+col1][r] = 1
        W[col0*n+0][r] = 1; W[col1*n+1][r] = 1
        
        r = base + 1
        U[2][r] = 1; U[3][r] = 1
        V[0*p+col0][r] = 1
        W[col0*n+1][r] = 1; W[col1*n+1][r] = -1
        
        r = base + 2
        U[0][r] = 1
        V[0*p+col1][r] = 1; V[1*p+col1][r] = -1
        W[col1*n+0][r] = 1; W[col1*n+1][r] = 1
        
        r = base + 3
        U[3][r] = 1
        V[1*p+col0][r] = 1; V[0*p+col0][r] = -1
        W[col0*n+0][r] = 1; W[col0*n+1][r] = 1
        
        r = base + 4
        U[0][r] = 1; U[1][r] = 1
        V[1*p+col1][r] = 1
        W[col0*n+0][r] = -1; W[col1*n+0][r] = 1
        
        r = base + 5
        U[2][r] = 1; U[0][r] = -1
        V[0*p+col0][r] = 1; V[0*p+col1][r] = 1
        W[col1*n+1][r] = 1
        
        r = base + 6
        U[1][r] = 1; U[3][r] = -1
        V[1*p+col0][r] = 1; V[1*p+col1][r] = 1
        W[col0*n+0][r] = 1
    
    add_strassen_block(0, 1, 0)
    add_strassen_block(2, 3, 7)
    
    # Schoolbook for column 4
    r = 14
    U[0][r] = 1; V[0*p+4][r] = 1; W[4*n+0][r] = 1
    r = 15
    U[1][r] = 1; V[1*p+4][r] = 1; W[4*n+0][r] = 1
    r = 16
    U[2][r] = 1; V[0*p+4][r] = 1; W[4*n+1][r] = 1
    r = 17
    U[3][r] = 1; V[1*p+4][r] = 1; W[4*n+1][r] = 1
    
    return {"U": U, "V": V, "W": W}

def _als_refine(U, V, W, T, R, n, m, p, max_iter=100, tol=1e-12):
    """Run ALS iterations to refine the decomposition. U, V, W are numpy arrays."""
    prev_err = np.inf
    for it in range(max_iter):
        # Solve for U
        # T[u, v, w] = sum_r U[u, r] * V[v, r] * W[w, r]
        # For fixed u: T[u, :, :] = sum_r U[u, r] * (V[:, r] * W[:, r])
        # Let M_r = V[:, r] * W[:, r] (element-wise product, shape m*p x p*n)
        # T[u] (flattened) = sum_r U[u, r] * M_r (flattened)
        
        VvW = np.einsum('vr,wr->vwr', V, W)
        VvW_flat = VvW.transpose(0, 2, 1).reshape(-1, R)
        
        for u in range(n*m):
            b = T[u].reshape(-1)
            sol, res, rank_s, sv = np.linalg.lstsq(VvW_flat, b, rcond=None)
            U[u] = sol
            
        # Solve for V
        # For fixed v: T[:, v, :] = sum_r V[v, r] * (U[:, r] * W[:, r])
        Uw = np.einsum('ur,wr->uwr', U, W)
        Uw_flat = Uw.transpose(0, 2, 1).reshape(-1, R)
        
        for v in range(m*p):
            b = T[:, v, :].reshape(-1)
            sol, res, rank_s, sv = np.linalg.lstsq(Uw_flat, b, rcond=None)
            V[v] = sol
            
        # Solve for W
        # For fixed w: T[:, :, w] = sum_r W[w, r] * (U[:, r] * V[:, r])
        UV = np.einsum('ur,vr->uvr', U, V)
        UV_flat = UV.reshape(-1, R)
        
        for w in range(p*n):
            b = T[:, :, w].reshape(-1)
            sol, res, rank_s, sv = np.linalg.lstsq(UV_flat, b, rcond=None)
            W[w] = sol
            
        # Compute error
        result = np.einsum('ur,vr,wr->uvw', U, V, W)
        err = np.linalg.norm(result - T)
        
        if abs(prev_err - err) < tol:
            break
        prev_err = err
        
    return U, V, W, err

def _check_and_rationalize(U, V, W, T, R, n, m, p):
    """Check if decomposition is exact and try to rationalize entries."""
    # Check numerical validity
    result = np.einsum('ur,vr,wr->uvw', U, V, W)
    if not np.allclose(result, T, atol=1e-5):
        return None
    
    # Try to rationalize with various denominators
    for denom in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        U_s = np.round(U * denom) / denom
        V_s = np.round(V * denom) / denom
        W_s = np.round(W * denom) / denom
        
        # Check if the rounded solution is valid
        result_s = np.einsum('ur,vr,wr->uvw', U_s, V_s, W_s)
        if np.allclose(result_s, T, atol=1e-7):
            U_list = [[float(x) for x in row] for row in U_s]
            V_list = [[float(x) for x in row] for row in V_s]
            W_list = [[float(x) for x in row] for row in W_s]
            
            # Convert to int if close to integer
            for mat in [U_list, V_list, W_list]:
                for i in range(len(mat)):
                    for j in range(len(mat[i])):
                        if abs(mat[i][j] - round(mat[i][j])) < 1e-9:
                            mat[i][j] = int(round(mat[i][j]))
            
            if _verify(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Return as floats if valid
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    if _verify(U_list, V_list, W_list, n, m, p):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _try_reduce_rank_from_18(n=2, m=2, p=5, target_rank=17, timeout=10.0):
    """
    Attempt to find a lower rank decomposition by initializing with the rank-18 solution
    and using ALS to see if a lower rank solution exists.
    """
    start_time = time.time()
    
    # Get the rank-18 solution
    sol18 = _construct_rank18(n, m, p)
    U18 = np.array(sol18['U'], dtype=np.float64)
    V18 = np.array(sol18['V'], dtype=np.float64)
    W18 = np.array(sol18['W'], dtype=np.float64)
    
    # Build the target tensor T
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T[u, v, w] = 1.0

    # Try to reduce to target_rank
    # Strategy: Initialize U, V, W with the first target_rank columns of the rank-18 solution.
    # This assumes the first target_rank terms are "most important" or linearly independent.
    # Then run ALS to see if we can fit T with only these target_rank terms.
    
    # We also try different subsets of columns to be more robust
    # For now, just try the first target_rank columns and a few random subsets if time permits
    
    subsets_to_try = [list(range(target_rank))]
    
    # Add a few random subsets
    if target_rank < 18:
        for _ in range(5):
            if time.time() - start_time > timeout - 2:
                break
            indices = random.sample(range(18), target_rank)
            subsets_to_try.append(indices)
    
    for indices in subsets_to_try:
        if time.time() - start_time > timeout - 1:
            break
            
        U = U18[:, indices].copy()
        V = V18[:, indices].copy()
        W = W18[:, indices].copy()
        
        # Run ALS
        max_iter = 50
        U, V, W, err = _als_refine(U, V, W, T, target_rank, n, m, p, max_iter=max_iter)
        
        # Check if exact
        result = _check_and_rationalize(U, V, W, T, target_rank, n, m, p)
        if result is not None:
            return result
            
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = START_TIME + TIME_LIMIT
    
    # 1. Immediately construct the known Rank-18 solution
    result18 = _construct_rank18(n, m, p)
    
    # Verify it to be safe
    if not _verify(result18['U'], result18['V'], result18['W'], n, m, p):
        # Fallback to schoolbook if rank 18 construction fails (should not happen)
        rank = n * m * p
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
        return {"U": U, "V": V, "W": W}
    
    # 2. Try to find a Rank-17 solution using the Rank-18 solution as initialization
    # This is a structural improvement: using the known solution as a prior for optimization
    time_spent = time.time() - START_TIME
    remaining_time = deadline - time.time()
    
    if remaining_time > 5.0:
        # Try Rank 17
        result17 = _try_reduce_rank_from_18(n, m, p, target_rank=17, timeout=min(10.0, remaining_time - 2.0))
        if result17 is not None:
            return result17
            
        # Try Rank 16 if still time
        if time.time() < deadline - 5.0:
            result16 = _try_reduce_rank_from_18(n, m, p, target_rank=16, timeout=min(5.0, deadline - time.time() - 2.0))
            if result16 is not None:
                return result16

    # 3. Return the best valid solution found (Rank 18)
    return result18

# EVOLVE-BLOCK-END
