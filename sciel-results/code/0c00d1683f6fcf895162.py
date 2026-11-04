"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 130s deadline, 20s safety margin
    
    # Strategy:
    # 1. Construct a known valid rank-56 decomposition using 8 Strassen blocks (2x2x2).
    # 2. Attempt to reduce the rank by merging columns via linear algebra.
    #    Specifically, we look for a rank-r decomposition (r < 56) that satisfies
    #    sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c] = T.
    #    This is equivalent to finding vectors u_c, v_c, w_c such that the tensor sum matches.
    #    We can formulate this as a linear system if we fix U and V, or use iterative methods.
    #    However, a more robust approach for small ranks is to use the fact that the tensor
    #    has a specific structure.
    
    # First, build the rank-56 baseline.
    rank_base = 56
    U_base = np.zeros((n*m, rank_base))
    V_base = np.zeros((m*p, rank_base))
    W_base = np.zeros((p*n, rank_base))
    
    # Strassen decomposition for 2x2x2 blocks.
    # We have blocks of j in {0,1}, {2,3} and k in {0,1}, {2,3}, {4,5}, {6,7}.
    # There are 2 (j-blocks) * 4 (k-blocks) = 8 blocks.
    # Each block has 7 terms.
    
    # Define the 7 Strassen patterns for U, V, W in a local 2x2x2 space.
    # Indices: i in {0,1}, j_local in {0,1}, k_local in {0,1}
    # U pattern (2x2):
    U_pats = [
        [[1, 0], [0, 1]],  # 1: a00+a11
        [[0, 0], [1, 1]],  # 2: a10+a11
        [[1, 0], [0, 0]],  # 3: a00
        [[0, 0], [0, 1]],  # 4: a11
        [[1, 1], [0, 0]],  # 5: a00+a01
        [[-1, 0], [1, 0]], # 6: a10-a00
        [[0, 1], [0, -1]]  # 7: a01-a11
    ]
    
    # V pattern (2x2):
    V_pats = [
        [[1, 0], [0, 1]],  # 1: b00+b11
        [[1, 0], [0, 0]],  # 2: b00
        [[0, 1], [0, -1]], # 3: b01-b11
        [[0, -1], [1, 0]], # 4: b10-b00
        [[0, 0], [0, 1]],  # 5: b11
        [[1, 1], [0, 0]],  # 6: b00+b01
        [[0, 1], [1, 1]]   # 7: b10+b11
    ]
    
    # W pattern (2x2): Note W index is k*n + i.
    # The contribution to C[i,k] is sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # For Strassen:
    # C00 = t1 + t4 - t5 + t7
    # C01 = t3 + t5
    # C10 = t2 + t4
    # C11 = t1 - t2 + t3 + t6
    
    # W_pats[t][k][i]
    W_pats = [
        [[1, 0], [0, 1]],  # 1: C00+=1, C11+=1
        [[0, 1], [-1, 0]], # 2: C10+=1, C11-=-1 -> W[0][1]=1, W[1][1]=-1
        [[0, 0], [1, 1]],  # 3: C01+=1, C11+=1 -> W[1][0]=1, W[1][1]=1
        [[1, 1], [0, 0]],  # 4: C00+=1, C10+=1 -> W[0][0]=1, W[0][1]=1
        [[-1, 0], [1, 0]], # 5: C00-=-1, C01+=1 -> W[0][0]=-1, W[1][0]=1
        [[0, 0], [0, 1]],  # 6: C11+=1 -> W[1][1]=1
        [[1, 0], [0, 0]]   # 7: C00+=1 -> W[0][0]=1
    ]
    
    # Verify patterns mentally or via code later.
    # Let's populate the base matrices.
    col = 0
    for j0 in [0, 2]:
        for k0 in [0, 2, 4, 6]:
            for t in range(7):
                # Fill U
                for i in range(2):
                    for j_local in range(2):
                        val = U_pats[t][i][j_local]
                        if val != 0:
                            U_base[i*m + (j0 + j_local), col] = val
                # Fill V
                for j_local in range(2):
                    for k_local in range(2):
                        val = V_pats[t][j_local][k_local]
                        if val != 0:
                            V_base[(j0 + j_local)*p + (k0 + k_local), col] = val
                # Fill W
                for k_local in range(2):
                    for i in range(2):
                        val = W_pats[t][k_local][i]
                        if val != 0:
                            W_base[(k0 + k_local)*n + i, col] = val
                col += 1
    
    # Verify base solution
    def verify(U, V, W):
        n, m, p = DIMENSIONS
        # T[i,j,k] should be 1
        # Check sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] == 1
        # We can check a few random points or all.
        # Efficient check:
        # For each c, form outer products? No, just iterate.
        # Since rank is small, we can do this quickly.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(U.shape[1]):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if not verify(U_base, V_base, W_base):
        # Fallback to schoolbook if Strassen construction failed
        rank_school = n*m*p
        U_school = np.zeros((n*m, rank_school))
        V_school = np.zeros((m*p, rank_school))
        W_school = np.zeros((p*n, rank_school))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i*m+j, c] = 1
                    V_school[j*p+k, c] = 1
                    W_school[k*n+i, c] = 1
                    c += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

    best_U, best_V, best_W = U_base, V_base, W_base
    best_rank = rank_base
    
    # Attempt to reduce rank.
    # We can try to find a linear combination of columns that cancels out.
    # Or, use a random projection approach:
    # Suppose we want rank r. We pick random matrices R_U (n*m x r), R_V (m*p x r).
    # Then we need W such that sum_c R_U[:,c] ⊗ R_V[:,c] ⊗ W[:,c] = T.
    # This is a linear system for W?
    # Let T be the tensor.
    # T[i,j,k] = sum_c R_U[i*m+j, c] R_V[j*p+k, c] W[k*n+i, c]
    # This is not linear in W if R_U, R_V are fixed?
    # Actually, for fixed R_U, R_V, the equation is:
    # sum_c W[k*n+i, c] * (sum_j ... no)
    # The term is sum_c (R_U[i*m+j, c] * R_V[j*p+k, c]) * W[k*n+i, c]
    # This is linear in W!
    # For each (k,i), we have a vector of coefficients for W[k*n+i, :].
    # Wait, the sum is over c.
    # Let A_{k,i,j} = sum_c R_U[i*m+j, c] R_V[j*p+k, c]
    # Then T[i,j,k] = sum_c ... wait.
    # T[i,j,k] = sum_c R_U[i*m+j, c] R_V[j*p+k, c] W[k*n+i, c]
    # This couples j and k and i.
    # If we fix U and V, the equation for W is:
    # For all i,j,k: sum_c (U[i*m+j, c] V[j*p+k, c]) W[k*n+i, c] = 1
    # This is a system of n*m*p equations.
    # Unknowns are W (p*n * r).
    # This is a linear system!
    # Let's formulate it.
    # For each (i,j,k), we have an equation.
    # The unknowns are W[0,0]...W[p*n-1, r-1].
    # Let x be the vectorized W.
    # The coefficient for W[k*n+i, c] in the equation for (i,j,k) is U[i*m+j, c] * V[j*p+k, c].
    # So we can build a matrix M of size (n*m*p) x (p*n*r).
    # M[row, col] where row = i*m*p + j*p + k, col = (k*n+i)*r + c.
    # M[row, col] = U[i*m+j, c] * V[j*p+k, c] if (k*n+i, c) matches col, else 0.
    
    # We can solve this system using least squares.
    # If the system has a solution, we found a valid W for the given U, V.
    # We can randomize U and V to find such a solution.
    
    # To make it faster, we can use the fact that we start from a valid solution (U_base, V_base, W_base).
    # We can try to perturb U and V and re-solve for W.
    # Or, we can try to reduce the rank by projecting the columns.
    
    # Let's try a specific search:
    # Try ranks from 55 down to 40.
    # For each rank, try a few random initializations of U and V (scaled down).
    # Solve for W.
    # Check if W is valid.
    
    # Optimization:
    # Instead of building the full M matrix (64 x 16r), which is small, we can just build it.
    # 64 rows, 16*56 = 896 cols. Very small.
    
    def try_reduce_rank(target_rank, num_trials=50):
        n, m, p = DIMENSIONS
        size_W = p * n
        num_eqs = n * m * p
        
        for trial in range(num_trials):
            if time.time() > deadline - 10:
                return None
            
            # Initialize U and V randomly
            # Use small values to avoid overflow?
            np.random.seed(trial)
            U = np.random.randn(n*m, target_rank) * 0.5
            V = np.random.randn(m*p, target_rank) * 0.5
            
            # Build linear system M W = 1
            # M is num_eqs x (size_W * target_rank)
            M = np.zeros((num_eqs, size_W * target_rank))
            
            # Precompute indices
            # row = i * m * p + j * p + k
            # col = (k * n + i) * target_rank + c
            
            # We can fill M more efficiently by iterating c, then i,j,k
            for c in range(target_rank):
                # For each c, we have vectors U_c (n*m), V_c (m*p)
                # Contribution to M:
                # For each i, j, k:
                # M[i*m*p + j*p + k, (k*n+i)*target_rank + c] += U_c[i*m+j] * V_c[j*p+k]
                
                # Vectorization:
                # U_c is shape (n*m). Reshape to (n, m).
                # V_c is shape (m*p). Reshape to (m, p).
                # We need to fill M rows.
                
                # This loop is 2*4*8 = 64 iterations per c.
                # Total 64 * target_rank operations.
                for i in range(n):
                    for j in range(m):
                        u_val = U[i*m + j, c]
                        if u_val == 0: continue
                        for k in range(p):
                            v_val = V[j*p + k, c]
                            if v_val == 0: continue
                            row = i * m * p + j * p + k
                            col = (k * n + i) * target_rank + c
                            M[row, col] += u_val * v_val
            
            # Solve M W = 1
            # Use least squares
            try:
                W_flat, residuals, rank_M, s = np.linalg.lstsq(M, np.ones(num_eqs), rcond=None)
            except:
                continue
            
            # Check residual
            # if residuals[0] > 1e-6: continue # lstsq returns sum of squares? No, it's the residual sum of squares.
            # Actually, let's just check the reconstruction error.
            
            W = W_flat.reshape(size_W, target_rank)
            
            # Verify
            # We can verify by checking the max error.
            # err = max |sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] - 1|
            
            # Efficient verification:
            # For each i, j:
            #   vec_W = W.T @ ... no.
            #   Let's just do the triple sum. It's 64 * r ops.
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for c in range(target_rank):
                            s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        err = abs(s - 1.0)
                        if err > max_err:
                            max_err = err
            
            if max_err < 1e-6:
                return U, V, W
        
        return None

    # Try to reduce rank
    # Start from 55 down to 48
    for r in range(rank_base - 1, 47, -1):
        if time.time() > deadline - 20:
            break
            
        # Increase trials for lower ranks? Or keep constant.
        # Let's give more time to higher ranks first.
        res = try_reduce_rank(r, num_trials=20)
        
        if res is not None:
            U, V, W = res
            best_U, best_V, best_W = U, V, W
            best_rank = r
            print(f"Found valid decomposition with rank {r}")
            # We can continue to see if we can go lower, but we have a valid one.
            # Break? No, try to go lower.
            
    # Final check
    if time.time() < deadline - 5:
        # One last check with a different random seed strategy?
        # Maybe try rank 48 with more trials?
        pass

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
