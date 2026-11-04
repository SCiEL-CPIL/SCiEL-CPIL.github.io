import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook rank 24 as fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = rank_school
    
    def verify(U, V, W, tol=1e-10):
        """Verify that U, V, W exactly decompose T."""
        rank = U.shape[1]
        reconstructed = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            reconstructed += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        return np.allclose(reconstructed, T, atol=tol)
    
    # Try Strassen-like decomposition for rank 22
    # Strategy: Use 2x2 Strassen for the (0,0) and (1,0) blocks, 
    # and handle remaining terms with outer products
    # 
    # Matrix A is 2x3, B is 3x4
    # C = A * B where C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # Let's partition:
    # A = [[a00, a01, a02],
    #      [a10, a11, a12]]
    # B = [[b00, b01, b02, b03],
    #      [b10, b11, b12, b13],
    #      [b20, b21, b22, b23]]
    #
    # C[0,0] = a00*b00 + a01*b10 + a02*b20
    # C[0,1] = a00*b01 + a01*b11 + a02*b21
    # C[0,2] = a00*b02 + a01*b12 + a02*b22
    # C[0,3] = a00*b03 + a01*b13 + a02*b23
    # C[1,0] = a10*b00 + a11*b10 + a12*b20
    # C[1,1] = a10*b01 + a11*b11 + a12*b21
    # C[1,2] = a10*b02 + a11*b12 + a12*b22
    # C[1,3] = a10*b03 + a11*b13 + a12*b23
    #
    # Total 8 output elements, each needs 3 products = 24 schoolbook
    #
    # Strassen idea for 2x2:
    # For 2x2 matrices, we can compute product with 7 multiplications:
    # m1 = (a00+a11)*(b00+b11)
    # m2 = (a10+a11)*b00
    # m3 = a00*(b11-b00)
    # m4 = a11*(b01-b00)  -- wait, let me use standard Strassen
    #
    # Standard Strassen for 2x2:
    # m1 = (a00+a11)*(b00+b11)
    # m2 = (a10+a11)*(b00+b11)  -- no
    #
    # Let me use the correct Strassen formulas:
    # m1 = (a00+a11)*(b00+b11)
    # m2 = (a10+a11)*b00
    # m3 = a00*(b11-b00)
    # m4 = a11*(b01-b00)  -- this isn't right either
    #
    # Correct Strassen:
    # m1 = (a00+a11)*(b00+b11)
    # m2 = (a10+a11)*(b00+b11)  -- no
    #
    # Let me just use a known rank-22 decomposition pattern.
    # Based on the observed improvement, let's try a structured approach.
    
    # For 2x3 * 3x4, one approach:
    # Group the 3 inner dimension into pairs
    # Use Strassen-like for (i,j) where i in {0,1}, j in {0,1}
    # Then handle j=2 separately
    
    # Let me try a direct rank-22 construction:
    # U indices: i*m+j = i*3+j, so U has 6 rows
    # V indices: j*p+k = j*4+k, so V has 12 rows  
    # W indices: k*n+i = k*2+i, so W has 8 rows
    
    # I'll attempt to find a valid rank-22 decomposition by solving
    # for coefficients given a support pattern.
    
    # Define candidate support patterns for rank 22
    # Each pattern is (u_idx, v_idx, w_idx) where u_idx in [0,5], v_idx in [0,11], w_idx in [0,7]
    
    # Let me try a systematic approach: start with schoolbook and try to merge
    # some terms using Strassen-like identities.
    
    # For now, let's try to verify if we can get rank 22 with a known structure.
    # The key insight: for each pair of rows in A and columns in B, 
    # we can use 7 multiplications instead of 4 for 2x2 blocks.
    
    # Partition B's columns into two groups of 2: {0,1} and {2,3}
    # For each group, compute 2x2 product using Strassen (7 mults each)
    # But A is 2x3, not 2x2, so we need to be careful.
    
    # Alternative: partition the inner dimension j into {0,1} and {2}
    # For j in {0,1}: 2x2 * 2x4 problem
    # For j=2: 2x1 * 1x4 problem
    
    # 2x2 * 2x4: can be done with 7*2 = 14 multiplications (Strassen for each 2x2 block of output)
    # Actually, 2x2 * 2x4 = (2x2*2x2) + (2x2*2x2), so 7+7=14
    # 2x1 * 1x4: 2*1*4 = 8 multiplications
    # Total: 14 + 8 = 22
    
    # Let me construct this:
    # For j in {0,1}, k in {0,1}: Strassen 2x2 block
    # m1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # m2 = (A[1,0]+A[1,1])*B[0,0]
    # m3 = A[0,0]*(B[1,1]-B[0,0])
    # m4 = A[1,1]*(B[0,1]-B[0,0])  -- need to fix Strassen formulas
    
    # Standard Strassen for C = A*B where A,B are 2x2:
    # m1 = (a00+a11)*(b00+b11)
    # m2 = (a10+a11)*(b00+b11)  -- NO
    #
    # Correct:
    # m1 = (a00+a11)*(b00+b11)
    # m2 = (a10+a11)*b00
    # m3 = a00*(b11-b00)
    # m4 = a11*(b01-b00)  -- still wrong
    #
    # Let me look this up properly:
    # m1 = (a00+a11)*(b00+b11)
    # m2 = (a10+a11)*(b00+b11)  -- this is not standard
    #
    # The standard Strassen:
    # m1 = (a00+a11)*(b00+b11)
    # m2 = (a10+a11)*b00
    # m3 = a00*(b11-b00)
    # m4 = a11*(b01-b00)
    # m5 = (a00+a10)*(b11+b10)
    # m6 = (a11-a10)*(b01+b00)
    # m7 = (a00-a11)*(b00-b11)
    #
    # C00 = m1 + m4 - m5 - m7
    # C01 = m5 + m6
    # C10 = m2 + m3
    # C11 = m1 - m2 + m3 + m7
    #
    # Let me verify:
    # C00 = a00*b00 + a11*b11 + a11*b01 - a11*b00 - a00*b11 - a10*b11 + a10*b10 + a00*b11 - a11*b00 + a11*b11
    # This is getting messy. Let me just use a known valid decomposition.
    
    # I'll try a randomized approach to find valid rank-22 patterns.
    
    def try_rank(r, max_iterations=1000):
        """Try to find a rank-r decomposition by random pattern search."""
        for iter in range(max_iterations):
            if time.time() > deadline - 10:
                break
            
            # Generate random support patterns
            patterns = []
            for _ in range(r):
                u = np.random.randint(0, n*m)
                v = np.random.randint(0, m*p)
                w = np.random.randint(0, p*n)
                patterns.append((u, v, w))
            
            # Build linear system: for each (i,j,k) with T[i*m+j, j*p+k, k*n+i] = 1,
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # This is nonlinear. Instead, fix U and V, solve for W, or use alternating minimization.
            
            # Alternative: use the fact that rank-1 tensors form a variety.
            # For a given pattern, we can set U, V, W to have 1s at the pattern positions
            # and solve for coefficients.
            
            # Let's use a different approach: start with schoolbook and try to reduce rank
            # by finding linear dependencies.
            
            if iter == 0:
                # Use a structured approach based on Strassen
                # For 2x3 * 3x4, partition inner dim into {0,1} and {2}
                
                # Part 1: j in {0,1}, use Strassen-like for 2x2 * 2x4
                # Split columns k into {0,1} and {2,3}
                
                # Block (i in {0,1}, j in {0,1}, k in {0,1}): 2x2*2x2, rank 7
                # Block (i in {0,1}, j in {0,1}, k in {2,3}): 2x2*2x2, rank 7
                # Block (i in {0,1}, j=2, k in {0,1,2,3}): 2x1*1x4, rank 8
                # Total: 7 + 7 + 8 = 22
                
                # Let me construct the 7 Strassen terms for 2x2*2x2:
                # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
                # m1 = (a00+a11)*(b00+b11)
                # m2 = (a10+a11)*b00
                # m3 = a00*(b11-b00)
                # m4 = a11*(b01-b00)
                # m5 = (a00+a10)*(b11+b10)
                # m6 = (a11-a10)*(b01+b00)
                # m7 = (a00-a11)*(b00-b11)
                
                # C00 = m1 + m4 - m5 - m7
                # C01 = m5 + m6
                # C10 = m2 + m3
                # C11 = m1 - m2 + m3 + m7
                
                # Let me verify C00:
                # m1 = a00*b00 + a00*b11 + a11*b00 + a11*b11
                # m4 = a11*b01 - a11*b00
                # m5 = a00*b11 + a00*b10 + a10*b11 + a10*b10
                # m7 = a00*b00 - a00*b11 - a11*b00 + a11*b11
                # C00 = m1 + m4 - m5 - m7
                # = (a00*b00 + a00*b11 + a11*b00 + a11*b11) + (a11*b01 - a11*b00) - (a00*b11 + a00*b10 + a10*b11 + a10*b10) - (a00*b00 - a00*b11 - a11*b00 + a11*b11)
                # = a00*b00 + a00*b11 + a11*b00 + a11*b11 + a11*b01 - a11*b00 - a00*b11 - a00*b10 - a10*b11 - a10*b10 - a00*b00 + a00*b11 + a11*b00 - a11*b11
                # = a00*b00 + a11*b01 - a00*b10 - a10*b11 - a10*b10 + a00*b11 + a11*b00
                # This doesn't simplify to a00*b00 + a10*b10. So my Strassen formulas are wrong.
                
                # Let me use the correct Strassen:
                # m1 = (a00+a11)*(b00+b11)
                # m2 = (a10+a11)*(b00+b11)  -- NO, this is not standard
                #
                # I'll use a verified set:
                # m1 = (a00+a11)*(b00+b11)
                # m2 = (a10+a11)*b00
                # m3 = a00*(b11-b00)
                # m4 = a11*(b01-b00)
                # m5 = (a00+a10)*(b11+b10)
                # m6 = (a11-a10)*(b01+b00)
                # m7 = (a00-a11)*(b00-b11)
                #
                # C00 = m1 + m4 - m5 - m7
                # C01 = m5 + m6
                # C10 = m2 + m3
                # C11 = m1 - m2 + m3 + m7
                #
                # I'll just use a known valid rank-22 decomposition from the literature
                # or try a numerical approach.
                
                pass
        
        return None
    
    # Given the complexity, let me try a more direct approach:
    # Use scipy to solve for a rank-22 decomposition.
    
    # For rank r, we have r unknown vectors u_r, v_r, w_r in R^6, R^12, R^8 respectively.
    # The constraint is: sum_r u_r ⊗ v_r ⊗ w_r = T
    # This is equivalent to: for each (i,j,k), sum_r U[i,r]*V[j,r]*W[k,r] = T[i,j,k]
    
    # This is a system of 6*12*8 = 576 equations in 6*r + 12*r + 8*r = 26*r unknowns.
    # For r=22: 576 equations, 572 unknowns. Underdetermined, should have solutions.
    
    # Let me try to solve this using least squares with random initialization.
    
    def solve_rank(r, n_init=50, tol=1e-10):
        """Solve for rank-r decomposition using alternating least squares."""
        best_err = np.inf
        best_U = None
        best_V = None
        best_W = None
        
        for init in range(n_init):
            if time.time() > deadline - 15:
                break
            
            # Random initialization
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            # Alternating least squares
            for it in range(200):
                if time.time() > deadline - 15:
                    break
                
                # Update U: for each i, solve min_U sum_j,k (T[i,j,k] - sum_r U[i,r]*V[j,r]*W[k,r])^2
                # This is: U[i,:] @ (V[:,j] * W[:,k]) = T[i,j,k]
                # Vectorize: for fixed i, the system is:
                # sum_r U[i,r] * (V[j,r] * W[k,r]) = T[i,j,k] for all j,k
                # Let M_i be a (m*p*p*n) x r matrix where M_i[(j,k), r] = V[j,r]*W[k,r]
                # Then U[i,:] @ M_i = T[i, :, :].flatten()
                
                # Similarly for V and W.
                
                # Update U
                for i in range(n*m):
                    # Build matrix: shape (m*p*p*n, r)
                    # M[(j,k), r] = V[j,r] * W[k,r]
                    M = np.zeros((m*p*p*n, r))
                    for j in range(m*p):
                        for k in range(p*n):
                            M[j*p*n+k, :] = V[j, :] * W[k, :]
                    # Solve U[i,:] @ M = T[i, :, :].flatten()
                    rhs = T[i, :, :].flatten()
                    # Least squares
                    U[i, :], residuals, rank, sv = np.linalg.lstsq(M, rhs, rcond=None)
                
                # Update V
                for j in range(m*p):
                    M = np.zeros((n*m*p*n, r))
                    for i in range(n*m):
                        for k in range(p*n):
                            M[i*p*n+k, :] = U[i, :] * W[k, :]
                    rhs = T[:, j, :].flatten()
                    V[j, :], residuals, rank, sv = np.linalg.lstsq(M, rhs, rcond=None)
                
                # Update W
                for k in range(p*n):
                    M = np.zeros((n*m*m*p, r))
                    for i in range(n*m):
                        for j in range(m*p):
                            M[i*m*p+j, :] = U[i, :] * V[j, :]
                    rhs = T[:, :, k].flatten()
                    W[k, :], residuals, rank, sv = np.linalg.lstsq(M, rhs, rcond=None)
                
                # Check convergence
                err = np.linalg.norm(T - np.einsum('ir,jr,kr->ijk', U, V, W))
                if err < best_err:
                    best_err = err
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                if err < tol:
                    break
            
            if best_err < tol:
                break
        
        if best_err < tol and best_U is not None:
            return best_U, best_V, best_W, best_err
        return None, None, None, best_err
    
    # Try rank 22
    if time.time() < deadline - 20:
        U22, V22, W22, err22 = solve_rank(22, n_init=20)
        if U22 is not None and err22 < 1e-10:
            best_rank = 22
            best_result = {
                "U": U22.tolist(),
                "V": V22.tolist(),
                "W": W22.tolist()
            }
            print(f"Found rank-22 decomposition with error {err22}")
        
        # Try rank 21
        if time.time() < deadline - 20:
            U21, V21, W21, err21 = solve_rank(21, n_init=20)
            if U21 is not None and err21 < 1e-10:
                best_rank = 21
                best_result = {
                    "U": U21.tolist(),
                    "V": V21.tolist(),
                    "W": W21.tolist()
                }
                print(f"Found rank-21 decomposition with error {err21}")
        
        # Try rank 20
        if time.time() < deadline - 20:
            U20, V20, W20, err20 = solve_rank(20, n_init=20)
            if U20 is not None and err20 < 1e-10:
                best_rank = 20
                best_result = {
                    "U": U20.tolist(),
                    "V": V20.tolist(),
                    "W": W20.tolist()
                }
                print(f"Found rank-20 decomposition with error {err20}")
    
    # Verify the best result
    U_best = np.array(best_result["U"])
    V_best = np.array(best_result["V"])
    W_best = np.array(best_result["W"])
    
    if not verify(U_best, V_best, W_best):
        # Fallback to schoolbook
        best_result = {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }
        best_rank = rank_school
    
    return best_result
