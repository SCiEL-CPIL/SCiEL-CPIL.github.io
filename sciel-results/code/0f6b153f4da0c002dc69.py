import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def _strassen_2x2():
    """
    Return U, V, W for 2x2 * 2x2 matrix multiplication with rank 7.
    Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A11(B12-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A22(B21-B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1; W[1, 6] = 1
    
    return U, V, W

def _strassen_4x4():
    """
    Construct rank-49 decomposition for 4x4 * 4x4 using Kronecker product
    of two 2x2 * 2x2 Strassen decompositions.
    T_{4,4,4} = T_{2,2,2} (outer) x T_{2,2,2} (inner)
    """
    U1, V1, W1 = _strassen_2x2()
    U2, V2, W2 = _strassen_2x2()
    
    n1, m1, p1, r1 = 2, 2, 2, 7
    n2, m2, p2, r2 = 2, 2, 2, 7
    n, m, p = 4, 4, 4
    rank = r1 * r2 # 49
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for c1 in range(r1):
        for c2 in range(r2):
            c = c1 * r2 + c2
            
            # U[i, j, c] = U1[i1, j1, c1] * U2[i2, j2, c2]
            # i = n1*i1 + i2, j = m1*j1 + j2
            for i1 in range(n1):
                for j1 in range(m1):
                    u1_val = U1[i1 * m1 + j1, c1]
                    if u1_val == 0: continue
                    for i2 in range(n2):
                        for j2 in range(m2):
                            u2_val = U2[i2 * m2 + j2, c2]
                            if u2_val == 0: continue
                            i = n1 * i1 + i2
                            j = m1 * j1 + j2
                            U[i * m + j, c] = u1_val * u2_val

            # V[j, k, c] = V1[j1, k1, c1] * V2[j2, k2, c2]
            # j = m1*j1 + j2, k = p1*k1 + k2
            for j1 in range(m1):
                for k1 in range(p1):
                    v1_val = V1[j1 * p1 + k1, c1]
                    if v1_val == 0: continue
                    for j2 in range(m2):
                        for k2 in range(p2):
                            v2_val = V2[j2 * p2 + k2, c2]
                            if v2_val == 0: continue
                            j = m1 * j1 + j2
                            k = p1 * k1 + k2
                            V[j * p + k, c] = v1_val * v2_val

            # W[k, i, c] = W1[k1, i1, c1] * W2[k2, i2, c2]
            # k = p1*k1 + k2, i = n1*i1 + i2
            for k1 in range(p1):
                for i1 in range(n1):
                    w1_val = W1[k1 * n1 + i1, c1]
                    if w1_val == 0: continue
                    for k2 in range(p2):
                        for i2 in range(n2):
                            w2_val = W2[k2 * n2 + i2, c2]
                            if w2_val == 0: continue
                            k = p1 * k1 + k2
                            i = n1 * i1 + i2
                            W[k * n + i, c] = w1_val * w2_val

    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > 1e-8:
                    return False
    return True

def _try_random_projection(U, V, W, target_rank, n, m, p, rng):
    """
    Try to project the rank-98 decomposition onto a lower-rank subspace.
    Fix U structure, solve for V and W.
    """
    if target_rank >= U.shape[1]:
        return None
    
    # Strategy: randomly select target_rank columns from the rank-98 solution
    # and try to adjust them to maintain exactness.
    
    # Alternative: use random linear combinations of the original columns
    # to create new columns, then solve for scaling.
    
    # Build the tensor equation: sum_c U_ic V_jc W_kc = delta_jk
    # For a given set of rank-r columns, this is a system of equations.
    
    # Let's try: pick r random linear combinations of the original 98 columns
    # U_new = U @ P where P is (98, r)
    # Then we need V_new, W_new such that the tensor equation holds.
    
    # This is nonlinear, so instead let's try a different approach:
    # Keep U fixed as the first r columns, then solve for V and W.
    
    U_fixed = U[:, :target_rank]
    
    # For each (i,j) pair, U_fixed[i*m+j, c] is known for c=0..r-1
    # We need: sum_c U_fixed[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_jk
    
    # This is still bilinear. Let's try fixing V as well and solving for W,
    # or use an alternating approach.
    
    # Simpler: just try random subsets of columns and check if they work.
    # This is unlikely to work for large reductions, but let's try small ones.
    
    return None

def construct():
    t_start = time.time()
    DEADLINE = t_start + 130
    
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    try:
        # Get 4x4x4 decomposition (rank 49)
        U4, V4, W4 = _strassen_4x4()
        
        # Construct 4x4x8 decomposition (rank 98)
        rank_half = 49
        rank_total = 98
        
        U = np.zeros((n * m, rank_total), dtype=float)  # 16 x 98
        V = np.zeros((m * p, rank_total), dtype=float)  # 32 x 98
        W = np.zeros((p * n, rank_total), dtype=float)  # 32 x 98
        
        # Left half (k in 0..3)
        U[:, 0:49] = U4
        for j in range(4):
            for k in range(4):
                V[j*8 + k, 0:49] = V4[j*4 + k, :]
        for k in range(4):
            for i in range(4):
                W[k*8 + i, 0:49] = W4[k*4 + i, :]
        
        # Right half (k in 4..7)
        U[:, 49:98] = U4
        for j in range(4):
            for k in range(4):
                V[j*8 + 4 + k, 49:98] = V4[j*4 + k, :]
        for k in range(4):
            for i in range(4):
                W[(k+4)*8 + i, 49:98] = W4[k*4 + i, :]
        
        # Verify
        if not _verify(U, V, W, n, m, p):
            raise ValueError("Strassen 4x4x8 verification failed")
        
        # Try to reduce rank using subspace embedding / random projection
        best_rank = rank_total
        best_result = (U, V, W)
        
        # Try various target ranks below 98
        for target_rank in range(rank_total - 1, max(40, rank_total - 30), -1):
            if time.time() > DEADLINE - 10:
                break
            
            # Method: randomly permute and select columns, then try to
            # find a valid decomposition by solving linear systems.
            
            # Approach: Use the fact that the tensor is determined by
            # the trilinear form. Try random rank-r subspaces.
            
            # For each trial, create a random (98, target_rank) matrix P
            # and set U_new = U @ P. Then try to solve for V_new, W_new.
            
            # Since the problem is bilinear in V and W, we can fix V_new
            # randomly and solve for W_new, or vice versa.
            
            for trial in range(5):
                if time.time() > DEADLINE - 5:
                    break
                
                rng = np.random.default_rng(trial * 1000 + target_rank)
                
                # Create random projection
                P = rng.standard_normal((rank_total, target_rank))
                
                U_new = U @ P  # (16, target_rank)
                
                # Now solve for V_new and W_new.
                # The constraint: sum_c U_new[i*m+j, c] * V_new[j*p+k, c] * W_new[k*n+i, c] = delta_jk
                # 
                # For fixed U_new, this is a system of 4*4*8 = 128 equations
                # in the entries of V_new (32*target_rank) and W_new (32*target_rank).
                # Total unknowns: 64*target_rank
                # This is underdetermined for target_rank < 2, so it should have solutions.
                # But we need integer/half-integer solutions.
                
                # Let's try: fix V_new to be random, then solve for W_new linearly.
                V_new = rng.standard_normal((m * p, target_rank))
                
                # For each (i, k), the equation is:
                # sum_j sum_c U_new[i*m+j, c] * V_new[j*p+k, c] * W_new[k*n+i, c] = delta_jk
                # Wait, this isn't right. Let me re-derive.
                
                # The tensor equation is: for all i, j, k:
                # sum_c U_new[i*m+j, c] * V_new[j*p+k, c] * W_new[k*n+i, c] = delta_{jk}
                
                # For fixed i, k: sum_c [sum_j ... ] no, j is summed in the original matmul.
                # Actually, the equation is per (i,j,k) triple, not summed over j.
                
                # So for each (i, j, k): sum_c U_new[i*m+j, c] * V_new[j*p+k, c] * W_new[k*n+i, c] = delta_{jk}
                
                # For fixed (j, k): this is a linear equation in the products U_new[i*m+j,c]*W_new[k*n+i,c]
                # But it's coupled across i.
                
                # Let's vectorize. For fixed (j,k), define:
                # a_{i,c} = U_new[i*m+j, c] * W_new[k*n+i, c]
                # Then sum_c a_{i,c} * V_new[j*p+k, c] = delta_{jk} for all i.
                # This is: V_new[j*p+k, :] @ diag(W_new[k*n+i,:]) ... no.
                
                # Actually: for each i, sum_c U_new[i*m+j, c] * W_new[k*n+i, c] * V_new[j*p+k, c] = delta_{jk}
                # Let b_{i,c} = U_new[i*m+j, c] * W_new[k*n+i, c]
                # Then: sum_c b_{i,c} * V_new[j*p+k, c] = delta_{jk} for all i
                # This is a system: for each i, the dot product of b_i with V_new[j*p+k,:] equals delta_{jk}
                
                # This is nonlinear because b depends on W_new.
                
                # Alternative approach: use ALS (alternating least squares)
                # Initialize V_new and W_new randomly, then iterate:
                # 1. Fix V_new, solve for W_new (linear system)
                # 2. Fix W_new, solve for V_new (linear system)
                
                V_new = rng.standard_normal((m * p, target_rank))
                W_new = rng.standard_normal((p * n, target_rank))
                
                for als_iter in range(20):
                    if time.time() > DEADLINE - 3:
                        break
                    
                    # Fix V_new, solve for W_new
                    # For each (i, k): sum_j sum_c U_new[i*m+j, c] * V_new[j*p+k, c] * W_new[k*n+i, c] = delta_{jk}
                    # Wait, the equation is per (i,j,k), not summed over j.
                    # sum_c U_new[i*m+j, c] * V_new[j*p+k, c] * W_new[k*n+i, c] = delta_{jk}
                    
                    # For fixed (j, k), this is a linear equation in W_new[k*n+0, :], ..., W_new[k*n+n-1, :]
                    # Actually, W_new[k*n+i, c] appears with coefficient U_new[i*m+j, c] * V_new[j*p+k, c]
                    
                    # So for fixed (j, k): sum_c [sum_i delta... ] no.
                    
                    # Let me re-index. For fixed j, k:
                    # For each i: sum_c U_new[i*m+j, c] * V_new[j*p+k, c] * W_new[k*n+i, c] = delta_{jk}
                    
                    # Let v_c = V_new[j*p+k, c]
                    # Then for each i: sum_c U_new[i*m+j, c] * v_c * W_new[k*n+i, c] = delta_{jk}
                    
                    # This is: for each i, the vector (U_new[i*m+j, :] * v) dotted with W_new[k*n+i, :] = delta_{jk}
                    
                    # So we have n equations (one per i) in the target_rank unknowns W_new[k*n+i, c] for c=0..r-1
                    # But each equation only involves one row of W_new.
                    
                    # Actually, for fixed (j,k), the equations for different i are independent in W_new!
                    # For each i: sum_c (U_new[i*m+j, c] * V_new[j*p+k, c]) * W_new[k*n+i, c] = delta_{jk}
                    
                    # This is a single linear equation in W_new[k*n+i, :] (a vector of length target_rank).
                    # We need this to hold for all i, j, k.
                    
                    # So for each (i, j, k): one linear equation.
                    # Total: n*m*p = 128 equations.
                    # Unknowns: W_new has p*n*target_rank = 32*target_rank entries.
                    
                    # But the equations are not all independent. For fixed (j,k), the n equations
                    # (one per i) constrain the n rows W_new[k*n+i, :] for i=0..n-1.
                    
                    # Let's build the linear system for W_new given V_new.
                    # Unknowns: W_new flattened, size p*n*target_rank
                    # Equation for (i,j,k): sum_c U_new[i*m+j, c] * V_new[j*p+k, c] * W_new[k*n+i, c] = delta_{jk}
                    
                    # This is a linear equation in W_new[k*n+i, :].
                    # Coefficient for W_new[k*n+i, c] is U_new[i*m+j, c] * V_new[j*p+k, c]
                    
                    # So for each (i,j,k), we have one equation involving only W_new[k*n+i, :].
                    # Group by (k, i): for each (k,i), we have m equations (one per j):
                    # sum_c (U_new[i*m+j, c] * V_new[j*p+k, c]) * W_new[k*n+i, c] = delta_{jk}
                    
                    # This is a system of m equations in target_rank unknowns (W_new[k*n+i, c]).
                    # If m > target_rank, this is overdetermined and may not have a solution.
                    # If m <= target_rank, it may have solutions.
                    
                    # m = 4, so if target_rank >= 4, we have a chance.
                    
                    # Let's solve using least squares.
                    W_new = np.zeros((p * n, target_rank))
                    for k in range(p):
                        for i in range(n):
                            # Build the m x target_rank system
                            A_sys = np.zeros((m, target_rank))
                            b_sys = np.zeros(m)
                            for j in range(m):
                                for c in range(target_rank):
                                    A_sys[j, c] = U_new[i*m+j, c] * V_new[j*p+k, c]
                                b_sys[j] = 1.0 if j == k else 0.0
                            
                            # Solve A_sys @ x = b_sys for x = W_new[k*n+i, :]
                            try:
                                x, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                                W_new[k*n+i, :] = x
                            except:
                                break
                    
                    # Now fix W_new, solve for V_new similarly
                    V_new = np.zeros((m * p, target_rank))
                    for j in range(m):
                        for k in range(p):
                            # For each (j,k), we have n equations (one per i):
                            # sum_c U_new[i*m+j, c] * V_new[j*p+k, c] * W_new[k*n+i, c] = delta_{jk}
                            # Group by (j,k): n equations in target_rank unknowns (V_new[j*p+k, c])
                            A_sys = np.zeros((n, target_rank))
                            b_sys = np.zeros(n)
                            for i in range(n):
                                for c in range(target_rank):
                                    A_sys[i, c] = U_new[i*m+j, c] * W_new[k*n+i, c]
                                b_sys[i] = 1.0 if j == k else 0.0
                            
                            try:
                                x, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                                V_new[j*p+k, :] = x
                            except:
                                break
                    
                    # Check convergence
                    if als_iter > 0:
                        # Compute residual
                        max_err = 0.0
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    val = 0.0
                                    for c in range(target_rank):
                                        val += U_new[i*m+j, c] * V_new[j*p+k, c] * W_new[k*n+i, c]
                                    expected = 1.0 if j == k else 0.0
                                    max_err = max(max_err, abs(val - expected))
                        if max_err < 1e-6:
                            # Check if entries are close to integer/half-integer
                            # Round and check
                            U_round = np.round(U_new * 2) / 2
                            V_round = np.round(V_new * 2) / 2
                            W_round = np.round(W_new * 2) / 2
                            
                            max_err_round = 0.0
                            for i in range(n):
                                for j in range(m):
                                    for k in range(p):
                                        val = 0.0
                                        for c in range(target_rank):
                                            val += U_round[i*m+j, c] * V_round[j*p+k, c] * W_round[k*n+i, c]
                                        expected = 1.0 if j == k else 0.0
                                        max_err_round = max(max_err_round, abs(val - expected))
                            
                            if max_err_round < 1e-6:
                                best_rank = target_rank
                                best_result = (U_round, V_round, W_round)
                                break
                        break
                    else:
                        # Check if the solution is close to integer/half-integer
                        U_round = np.round(U_new * 2) / 2
                        V_round = np.round(V_new * 2) / 2
                        W_round = np.round(W_new * 2) / 2
                        
                        max_err_round = 0.0
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    val = 0.0
                                    for c in range(target_rank):
                                        val += U_round[i*m+j, c] * V_round[j*p+k, c] * W_round[k*n+i, c]
                                    expected = 1.0 if j == k else 0.0
                                    max_err_round = max(max_err_round, abs(val - expected))
                        
                        if max_err_round < 1e-6:
                            best_rank = target_rank
                            best_result = (U_round, V_round, W_round)
                            break
                else:
                    continue
                break
            
            if best_rank < rank_total:
                break
        
        if best_rank < rank_total:
            U_best, V_best, W_best = best_result
            if _verify(U_best, V_best, W_best, n, m, p):
                return {
                    "U": U_best.tolist(),
                    "V": V_best.tolist(),
                    "W": W_best.tolist()
                }
        
        # Return the rank-98 solution
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
        
    except Exception:
        pass
    
    # Fallback: Schoolbook (rank 128)
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
