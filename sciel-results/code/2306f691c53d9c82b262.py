"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

import time
import random
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    # Shapes: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # Tensor T[i, j, k] = 1 for all i,j,k
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}, rank

    best_result, best_rank = schoolbook()
    
    # Try block decomposition first (known good result: rank 42)
    def strassen_2x2x2_terms():
        """Return list of 7 tuples (U_dict, V_dict, W_dict) for 2x2x2 Strassen."""
        terms = []
        # r=1: M1 = (A00+A11)(B00+B11), C: C00+=1, C11+=1
        U = {(0,0):1, (1,1):1}
        V = {(0,0):1, (1,1):1}
        W = {(0,0):1, (1,1):1}
        terms.append((U, V, W))
        # r=2: M2 = (A10+A11)B00, C: C10+=1
        U = {(1,0):1, (1,1):1}
        V = {(0,0):1}
        W = {(0,1):1}
        terms.append((U, V, W))
        # r=3: M3 = A00(B01-B11), C: C01+=1
        U = {(0,0):1}
        V = {(0,1):1, (1,1):-1}
        W = {(1,0):1}
        terms.append((U, V, W))
        # r=4: M4 = A11(B10-B00), C: C00+=1, C10+=1
        U = {(1,1):1}
        V = {(1,0):1, (0,0):-1}
        W = {(0,0):1, (0,1):1}
        terms.append((U, V, W))
        # r=5: M5 = (A00+A01)B11, C: C00-=1, C01+=1
        U = {(0,0):1, (0,1):1}
        V = {(1,1):1}
        W = {(0,0):-1, (1,0):1}
        terms.append((U, V, W))
        # r=6: M6 = (A10-A00)(B00+B01), C: C11+=1
        U = {(1,0):1, (0,0):-1}
        V = {(0,0):1, (0,1):1}
        W = {(1,1):1}
        terms.append((U, V, W))
        # r=7: M7 = (A01-A11)(B10+B11), C: C00+=1, C11+=1
        U = {(0,1):1, (1,1):-1}
        V = {(1,0):1, (1,1):1}
        W = {(0,0):1, (1,1):1}
        terms.append((U, V, W))
        return terms

    def build_block_decomposition():
        n, m, p = DIMENSIONS
        m_block = 2
        p_block = 2
        m_blocks = m // m_block  # 2
        p_blocks = p // p_block  # 3
        rank = m_blocks * p_blocks * 7
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        strassen_terms = strassen_2x2x2_terms()
        col = 0
        for jb in range(m_blocks):
            for kb in range(p_blocks):
                for (U_dict, V_dict, W_dict) in strassen_terms:
                    for (i, j_local), val in U_dict.items():
                        U[i * m + (jb * m_block + j_local)][col] = val
                    for (j_local, k_local), val in V_dict.items():
                        V[(jb * m_block + j_local) * p + (kb * p_block + k_local)][col] = val
                    for (k_local, i), val in W_dict.items():
                        W[(kb * p_block + k_local) * n + i][col] = val
                    col += 1
        return {"U": U, "V": V, "W": W}

    def check_decomposition(U, V, W):
        r = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for rr in range(r):
                        s += U[i*m+j][rr] * V[j*p+k][rr] * W[k*n+i][rr]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    # Initialize with block decomposition
    if time.time() < deadline - 20:
        block_decomp = build_block_decomposition()
        if check_decomposition(block_decomp["U"], block_decomp["V"], block_decomp["W"]):
            best_result = block_decomp
            best_rank = len(block_decomp["U"][0])

    # New Strategy: ALS with Rational Rounding
    # We try to find a lower rank decomposition by optimizing a rank-r initialization.
    # We start from the block decomposition (rank 42) and try to project it down to lower ranks.
    # Or better: start from a random rank-r initialization and run ALS.
    
    def round_to_half_int(x, tol=1e-5):
        """Round x to the nearest half-integer if close, else return x as is (or 0 if very small)."""
        if abs(x) < tol:
            return 0
        # Check if close to integer or half-integer
        nearest_int = round(x)
        if abs(x - nearest_int) < tol:
            return int(nearest_int)
        
        nearest_half = round(x * 2) / 2.0
        if abs(x - nearest_half) < tol:
            # Return as float or int? The task says integer/half-integer entries.
            # Returning float is usually fine if it's exactly 0.5, 1.5 etc.
            # But to be safe for exactness, let's return float.
            return nearest_half
        return x

    def als_optimize(U_init, V_init, W_init, target_rank, max_iters=200, tol=1e-8):
        """
        Run ALS to minimize error for a given target rank.
        Returns optimized U, V, W and final error.
        """
        n, m, p = DIMENSIONS
        dim_u = n * m
        dim_v = m * p
        dim_w = p * n
        
        # Flatten indices for tensor T
        # T[i, j, k] = 1
        # We represent the tensor as a 3D array for efficient ALS updates?
        # Or just loop. Given small sizes, looping is fine.
        # But for ALS, we need to solve linear systems.
        
        U = np.array(U_init, dtype=float)
        V = np.array(V_init, dtype=float)
        W = np.array(W_init, dtype=float)
        
        # Ensure shapes match target_rank
        if U.shape[1] != target_rank:
             # This function assumes inputs are already of target_rank shape
             # We will handle initialization outside
             pass
             
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
                
            # Update U
            # For each (i, j), U[i*m+j, :] minimizes sum_k (1 - sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r])^2
            # This is linear in U[i*m+j, :]
            # Let u_r = U[i*m+j, r]
            # Error^2 = sum_k (1 - sum_r u_r (V[j*p+k, r] W[k*n+i, r]))^2
            # Let A_rk = V[j*p+k, r] W[k*n+i, r]
            # Minimize sum_k (1 - sum_r u_r A_rk)^2
            # This is || 1 - A^T u ||^2
            # Solution: u = (A A^T)^-1 A 1
            
            # Precompute A for all (i, j)?
            # A is of shape (rank, p). A[r, k] = V[j*p+k, r] * W[k*n+i, r]
            
            for i in range(n):
                for j in range(m):
                    idx_u = i * m + j
                    # Construct A (rank x p)
                    A = np.zeros((target_rank, p))
                    for k in range(p):
                        idx_v = j * p + k
                        idx_w = k * n + i
                        A[:, k] = V[idx_v, :] * W[idx_w, :]
                    
                    # Solve (A A^T) u = A 1
                    # Note: A is (rank, p). We want to project the vector of 1s (length p) onto the column space of A^T?
                    # Wait.
                    # Term in sum: sum_r u_r A[r, k]
                    # Let b_k = 1.
                    # We want u such that A^T u approx b.
                    # Normal equations: (A A^T) u = A b
                    # A A^T is (rank x rank). A b is (rank x 1).
                    
                    AtA = A @ A.T
                    Ab = A @ np.ones(p)
                    
                    # Add small regularization for stability
                    AtA += 1e-8 * np.eye(target_rank)
                    
                    try:
                        u_new = np.linalg.solve(AtA, Ab)
                    except np.linalg.LinAlgError:
                        u_new = np.linalg.lstsq(AtA, Ab, rcond=None)[0]
                    
                    U[idx_u, :] = u_new
            
            # Update V
            # For each (j, k), V[j*p+k, :] minimizes sum_i (1 - sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r])^2
            # Linear in V[j*p+k, :]
            # Let v_r = V[j*p+k, r]
            # Error^2 = sum_i (1 - sum_r (U[i*m+j, r] W[k*n+i, r]) v_r)^2
            # Let B_ri = U[i*m+j, r] W[k*n+i, r]
            # Minimize sum_i (1 - sum_r B_ri v_r)^2
            # Solution: v = (B^T B)^-1 B^T 1
            
            for j in range(m):
                for k in range(p):
                    idx_v = j * p + k
                    B = np.zeros((target_rank, n))
                    for i in range(n):
                        idx_u = i * m + j
                        idx_w = k * n + i
                        B[:, i] = U[idx_u, :] * W[idx_w, :]
                    
                    BtB = B.T @ B
                    Bt1 = B.T @ np.ones(n)
                    
                    BtB += 1e-8 * np.eye(target_rank)
                    
                    try:
                        v_new = np.linalg.solve(BtB, Bt1)
                    except np.linalg.LinAlgError:
                        v_new = np.linalg.lstsq(BtB, Bt1, rcond=None)[0]
                    
                    V[idx_v, :] = v_new
            
            # Update W
            # For each (k, i), W[k*n+i, :] minimizes sum_j (1 - sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r])^2
            # Linear in W[k*n+i, :]
            # Let w_r = W[k*n+i, r]
            # Error^2 = sum_j (1 - sum_r (U[i*m+j, r] V[j*p+k, r]) w_r)^2
            # Let C_rj = U[i*m+j, r] V[j*p+k, r]
            # Minimize sum_j (1 - sum_r C_rj w_r)^2
            # Solution: w = (C C^T)^-1 C 1
            
            for k in range(p):
                for i in range(n):
                    idx_w = k * n + i
                    C = np.zeros((target_rank, m))
                    for j in range(m):
                        idx_u = i * m + j
                        idx_v = j * p + k
                        C[:, j] = U[idx_u, :] * V[idx_v, :]
                    
                    CtC = C @ C.T
                    C1 = C @ np.ones(m)
                    
                    CtC += 1e-8 * np.eye(target_rank)
                    
                    try:
                        w_new = np.linalg.solve(CtC, C1)
                    except np.linalg.LinAlgError:
                        w_new = np.linalg.lstsq(CtC, C1, rcond=None)[0]
                    
                    W[idx_w, :] = w_new
            
            # Check convergence
            # Calculate error
            err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx_u = i * m + j
                        idx_v = j * p + k
                        idx_w = k * n + i
                        s = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                        err += (1 - s) ** 2
            
            if err < tol:
                break
        
        return U, V, W, err

    def try_rank_reduction(start_rank, target_rank, max_iters=100):
        """
        Try to find a valid decomposition of target_rank.
        We initialize with random noise added to a scaled version of a valid decomposition?
        Or just random?
        Random is risky. Let's try to project the current best (block decomp) into a lower rank subspace?
        Actually, ALS from random init often fails for matrix multiplication tensors due to symmetry.
        
        Better strategy: 
        1. Start with the known valid decomposition (rank 42).
        2. Try to find a linear transformation that maps rank 42 to rank R < 42 such that the decomposition holds.
        This is equivalent to finding a rank-R tensor in the variety.
        
        Alternative: 
        Use the "splitting" idea. 
        If we can find a decomposition of rank R, we can try to merge columns.
        
        Let's stick to ALS from a smart initialization.
        Initialization: Take the block decomposition, and try to "compress" it by selecting a subset of columns? No, that breaks exactness.
        
        Let's try ALS with a rank slightly higher than target, then prune?
        
        Actually, for 2x4x6, the known rank is 24? 
        Schoolbook: 48.
        Block Strassen: 42.
        Is it possible to do better?
        The rank of 2x4x6 is at most 24?
        Wait, rank(2x4x6) <= rank(2x2x2) * 2 * 3 = 7 * 6 = 42.
        But we can do better by mixing blocks.
        
        Let's try ALS from a random initialization with target_rank = best_rank - 1, etc.
        """
        if time.time() > deadline - 30:
            return None
            
        n, m, p = DIMENSIONS
        dim_u = n * m
        dim_v = m * p
        dim_w = p * n
        
        # Random initialization
        U = np.random.randn(dim_u, target_rank)
        V = np.random.randn(dim_v, target_rank)
        W = np.random.randn(dim_w, target_rank)
        
        # Scale to help convergence?
        # The tensor entries are 1.
        # If U, V, W are ~ N(0,1), then sum_r U V W is ~ N(0, sqrt(R)).
        # We want it to be 1. So scale by 1/sqrt(R) maybe?
        scale = 1.0 / np.sqrt(target_rank)
        U *= scale
        V *= scale
        W *= scale
        
        U, V, W, err = als_optimize(U, V, W, target_rank, max_iters=max_iters)
        
        if err < 1e-6:
            # Try to round to half-integers
            U_rounded = np.vectorize(round_to_half_int)(U)
            V_rounded = np.vectorize(round_to_half_int)(V)
            W_rounded = np.vectorize(round_to_half_int)(W)
            
            # Check if rounded version is still valid
            # Convert to list of lists for check_decomposition
            U_list = U_rounded.tolist()
            V_list = V_rounded.tolist()
            W_list = W_rounded.tolist()
            
            if check_decomposition(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}
            
            # If rounding failed, return the float version? 
            # The task says "integer/half-integer entries". 
            # If we return floats that are not half-integers, it might be invalid.
            # But if the error is < 1e-6, it's likely very close to a half-integer solution.
            # Let's force round to nearest half-integer and hope the error was small enough.
            # If check fails, it's not a valid solution.
            
        return None

    # Try to reduce rank from best_rank down to some limit
    # We know 42 is valid. Can we do 41? 40? ...
    # ALS is stochastic. We might need multiple restarts.
    
    current_best_rank = best_rank
    # We'll try a few steps down.
    # Since ALS is slow, we limit the number of attempts.
    
    for target_rank in range(current_best_rank - 1, max(10, current_best_rank - 10), -1):
        if time.time() > deadline - 40:
            break
            
        # Try a few random restarts
        found = False
        for restart in range(3):
            if time.time() > deadline - 20:
                break
                
            res = try_rank_reduction(current_best_rank, target_rank, max_iters=50)
            if res is not None:
                best_result = res
                current_best_rank = target_rank
                found = True
                break
        
        if found:
            continue
        else:
            # If we failed to reduce by 1, maybe we can't reduce further easily?
            # Or maybe we just got unlucky.
            # Given the time budget, we might stop here.
            # But let's try one more rank down if we have time?
            # Actually, if rank R fails, R-1 is harder. 
            # Let's break if we fail to reduce by 1.
            break

    return best_result

# EVOLVE-BLOCK-END
