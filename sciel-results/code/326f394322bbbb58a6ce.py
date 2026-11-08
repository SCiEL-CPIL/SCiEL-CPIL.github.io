"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    n, m, p = DIMENSIONS

    def check_time():
        return time.time() > deadline

    def schoolbook():
        """Guaranteed valid schoolbook decomposition of rank n*m*p."""
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
        return {"U": U, "V": V, "W": W}

    def verify_fast(U, V, W, n, m, p):
        """Faster verification using numpy."""
        rank = len(U[0])
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check on-entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.einsum('r,r,r->', U_np[i*m+j], V_np[j*p+k], W_np[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        
        # Full tensor check
        T = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
        target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target[i*m+j, j*p+k, k*n+i] = 1.0
        if not np.allclose(T, target, atol=1e-9):
            return False
        return True

    def build_block_terms():
        """Return list of 11 terms for M_{2,3,2}."""
        strassen_terms = [
            {'U': [(0,0,1), (1,1,1)], 'V': [(0,0,1), (1,1,1)], 'W': [(0,0,1), (1,1,1)]},
            {'U': [(1,0,1), (1,1,1)], 'V': [(0,0,1)], 'W': [(0,1,1), (1,1,-1)]},
            {'U': [(0,0,1)], 'V': [(0,1,1), (1,1,-1)], 'W': [(1,0,1), (1,1,1)]},
            {'U': [(1,1,1)], 'V': [(1,0,1), (0,0,-1)], 'W': [(0,0,1), (0,1,1)]},
            {'U': [(0,0,1), (0,1,1)], 'V': [(1,1,1)], 'W': [(0,0,-1), (1,0,1)]},
            {'U': [(1,0,1), (0,0,-1)], 'V': [(0,0,1), (0,1,1)], 'W': [(1,1,1)]},
            {'U': [(0,1,1), (1,1,-1)], 'V': [(1,0,1), (1,1,1)], 'W': [(0,0,1)]}
        ]
        outer_terms = [
            {'U': [(0,2,1)], 'V': [(2,0,1)], 'W': [(0,0,1)]},
            {'U': [(0,2,1)], 'V': [(2,1,1)], 'W': [(1,0,1)]},
            {'U': [(1,2,1)], 'V': [(2,0,1)], 'W': [(0,1,1)]},
            {'U': [(1,2,1)], 'V': [(2,1,1)], 'W': [(1,1,1)]}
        ]
        return strassen_terms + outer_terms

    def build_block_terms_10():
        """Attempt a rank-10 decomposition for M_{2,3,2}.
        
        Strategy: Use a modified Strassen-like scheme that shares structure
        between the j=0,1 Strassen core and the j=2 outer products.
        
        For 2x3x2, we need to compute:
        C[0,0] = A00*B00 + A01*B10 + A02*B20
        C[0,1] = A00*B01 + A01*B11 + A02*B21
        C[1,0] = A10*B00 + A11*B10 + A12*B20
        C[1,1] = A10*B01 + A11*B11 + A12*B21
        
        We try to find 10 rank-1 terms.
        """
        # Try a systematic construction:
        # Use 7 Strassen terms for the 2x2x2 core (j=0,1)
        # and 3 shared terms for j=2 that exploit linearity
        #
        # Key idea: Instead of 4 separate terms for j=2, use 3 terms
        # that combine A02/A12 with B20/B21 using differences/sums.
        #
        # Let's try:
        # Term 8: U has A02+A12 (combined), V has B20, W has C00-C10 pattern
        # Term 9: U has A02+A12, V has B21, W has C01-C11 pattern  
        # Term 10: U has A02-A12, V has B20+B21, W has cross pattern
        #
        # Actually, let's try a different approach. Consider:
        # Define M1 = A02 + A12, M2 = A02 - A12
        # Define N1 = B20 + B21, N2 = B20 - B21
        # Then A02*B20 = (M1+M2)/2 * (N1+N2)/2 = (M1*N1 + M1*N2 + M2*N1 + M2*N2)/4
        # This doesn't directly help for rank reduction.
        #
        # Let me try a direct algebraic approach. We need 10 vectors
        # u_t in R^6 (for A indices), v_t in R^6 (for B indices), w_t in R^4 (for C indices)
        # such that sum_t u_t ⊗ v_t ⊗ w_t = T
        #
        # T has 12 non-zero entries. Let's set up the system.
        # 
        # Actually, let me try a known construction. The tensor rank of 
        # 2x3x2 matrix multiplication... Let me try to find it by search.
        
        # Try: use the 7 Strassen terms, then find 3 more terms for j=2
        # that are "shared" in some way.
        #
        # Alternative: try a completely different 10-term construction.
        # 
        # Let me try to construct it directly. We have 4 output entries.
        # Each output is a sum of 3 products. Total 12 monomials.
        # 
        # A rank-1 term contributes at most 1 to each output (if we use 
        # structured vectors). With 10 terms, we need to cover 12 monomials
        # with 10 terms, so at least 2 terms must cover 2 monomials each
        # (via products of sums).
        
        # Try this construction:
        # Term 1: (A00+A11)*(B00+B11) -> contributes to C00, C01, C10, C11
        # Term 2: (A10+A11)*(B00) -> contributes to C10, C11  
        # ... this is getting complex. Let me just try to solve it numerically.
        
        return None  # Will use search instead

    def build_block_decomposition(rank_per_block):
        """Build full U,V,W for p=10 as 5 blocks of size 2."""
        num_blocks = p // 2
        block_size = 2
        total_rank = num_blocks * rank_per_block
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        terms = build_block_terms()
        col_offset = 0
        for b in range(num_blocks):
            for term in terms:
                col = col_offset
                for i, j, val in term['U']:
                    U[i * m + j][col] += val
                for j, k_local, val in term['V']:
                    k_global = b * block_size + k_local
                    V[j * p + k_global][col] += val
                for k_local, i, val in term['W']:
                    k_global = b * block_size + k_local
                    W[k_global * n + i][col] += val
                col_offset += 1
        return U, V, W

    # Try to find a rank-10 decomposition of M_{2,3,2} via numerical search
    def search_rank10_block():
        """Search for a rank-10 decomposition of the 2x3x2 multiplication tensor."""
        # The 2x3x2 tensor T[(i,j),(j,k),(k,i)] = 1 for i in {0,1}, j in {0,1,2}, k in {0,1}
        # We need to find 10 rank-1 tensors that sum to T.
        # 
        # U has shape (6, 10), V has shape (6, 10), W has shape (4, 10)
        # such that sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] = T
        #
        # Let's set up the system. T is a 6x6x4 tensor with 12 non-zero entries.
        # Flatten: T_flat[6*6*4] = T_flat[144]
        # Each rank-1 term contributes U[:,r] * V[:,r] * W[:,r] as a 6x6x4 tensor.
        #
        # This is a nonlinear system. Let's use a parameterized approach.
        # 
        # Strategy: Fix W to be a simple basis-like structure and solve for U, V.
        # Or use iterative methods.
        #
        # Let me try a structured approach. Assume W has a specific form.
        # W is 4x10. Let's say W[:,r] selects which output entries term r contributes to.
        #
        # For a rank-1 term u ⊗ v ⊗ w, the contribution to C[i,k] is u[i,j]*v[j,k]*w[i,k] summed over j.
        # Wait, no. The term is sum_j u[i*m+j] * v[j*p+k] * w[k*n+i].
        # For n=2, m=3, p=2: sum_j u[i*3+j] * v[j*2+k] * w[k*2+i]
        #
        # So for each term r, the contribution to C[i,k] is:
        # sum_{j=0}^{2} U[i*3+j, r] * V[j*2+k, r] * W[k*2+i, r]
        #
        # We need this to equal 1 if (i,j,k) is a valid index combination... wait, no.
        # The tensor T[(i,j),(j,k),(k,i)] = 1 means that for the full multiplication,
        # C[i,k] = sum_j A[i,j]*B[j,k].
        # The structure tensor has T[(i,j),(j,k),(k,i)] = 1.
        # So the decomposition must satisfy:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all valid (i,j,k)
        # and = 0 for all other combinations.
        #
        # For n=2, m=3, p=2: valid (i,j,k) are all i in {0,1}, j in {0,1,2}, k in {0,1}.
        # That's 12 entries that must equal 1.
        # All other entries in the 6x6x4 tensor must be 0.
        # Total entries: 6*6*4 = 144. 12 are 1, 132 are 0.
        #
        # Let's set up the equations. For each (a,b,c) in the 6x6x4 grid:
        # sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
        # where T[a,b,c] = 1 if a=i*3+j, b=j*2+k, c=k*2+i for valid i,j,k, else 0.
        #
        # Let me try a numerical approach with random initialization and gradient descent.
        
        import random
        random.seed(42)
        
        # Target tensor
        T = np.zeros((6, 6, 4))
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    T[i*3+j, j*2+k, k*2+i] = 1.0
        
        # Try multiple random initializations
        for attempt in range(200):
            if check_time():
                return None
            
            # Random initialization
            U = np.random.randn(6, 10) * 0.5
            V = np.random.randn(6, 10) * 0.5
            W = np.random.randn(4, 10) * 0.5
            
            # Simple gradient descent on the nonlinear objective
            # F = sum_{a,b,c} (sum_r U[a,r]*V[b,r]*W[c,r] - T[a,b,c])^2
            # 
            # This is expensive. Let's use a smarter approach.
            # 
            # Alternative: use the fact that this is a tensor decomposition problem.
            # We can fix W and solve for U, V, or use alternating minimization.
            
            # Let's try: fix W, solve for U and V using the structure.
            # For fixed W, the equations are bilinear in U and V.
            # 
            # Actually, let's try a different parameterization. 
            # Assume W has a specific sparse structure.
            
            # Try: W[:,r] is a unit vector or a simple combination.
            # There are 4 output entries. With 10 terms, we can assign terms to outputs.
            
            # Let me try: 5 terms for output (0,0), 5 for (0,1), 5 for (1,0), 5 for (1,1)
            # But that's 20 terms. Too many.
            
            # Better: each term can contribute to multiple outputs.
            
            # Let's try a structured W:
            # W[0, r] = w0_r, W[1, r] = w1_r, W[2, r] = w2_r, W[3, r] = w3_r
            # where indices correspond to (k=0,i=0), (k=0,i=1), (k=1,i=0), (k=1,i=1)
            
            # Try: W has 10 columns, each with at most 2 non-zero entries.
            
            # Let me just do a direct search with a simple heuristic.
            # Initialize with the known rank-11 solution and try to merge two terms.
            
            # Get the rank-11 solution
            terms = build_block_terms()
            U11 = np.zeros((6, 11))
            V11 = np.zeros((6, 11))
            W11 = np.zeros((4, 11))
            for r, term in enumerate(terms):
                for i, j, val in term['U']:
                    U11[i*3+j, r] += val
                for j, k, val in term['V']:
                    V11[j*2+k, r] += val
                for k, i, val in term['W']:
                    W11[k*2+i, r] += val
            
            # Try to merge the last two terms (the outer terms) with others.
            # Or try to find a linear dependency.
            
            # Actually, let's try a completely different approach:
            # Use the known rank-11 solution and try to perturb it to find rank 10.
            # This is unlikely to work directly.
            
            # Let me try to construct rank-10 from scratch with a specific structure.
            # 
            # Hypothesis: The rank of 2x3x2 matrix multiplication is 10.
            # 
            # Try this construction:
            # Use Strassen for the 2x2 part (j=0,1) with 7 terms.
            # For j=2, use 3 terms that share structure with the Strassen terms.
            #
            # Strassen terms (for j=0,1):
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)(B00)
            # M3 = (A00)(B11-B01)
            # M4 = (A11)(B10-B00)
            # M5 = (A00+A01)(B11)
            # M6 = (A10-A00)(B00+B01)
            # M7 = (A01-A11)(B10+B11)
            #
            # C00 = M1 - M4 - M7
            # C01 = M3 + M1
            # C10 = M2 + M4
            # C11 = M1 - M2 - M5
            #
            # Now for j=2, we need to add:
            # A02*B20 to C00
            # A02*B21 to C01
            # A12*B20 to C10
            # A12*B21 to C11
            #
            # Can we express these using 3 new terms that also modify the existing terms?
            #
            # Try:
            # M8 = (A02+A12)(B20+B21)
            # M9 = (A02)(B20)
            # M10 = (A12)(B21)
            #
            # Then:
            # C00 += A02*B20 = M9
            # C01 += A02*B21 = M8 - M9 - M10 + ... hmm, this doesn't work cleanly.
            #
            # Let me try:
            # M8 = (A02+A12)(B20)
            # M9 = (A02+A12)(B21)
            # M10 = (A02)(B20+B21)
            #
            # A02*B20 = M8 - (A12*B20) = M8 - (M8 - A02*B20)... circular.
            #
            # This approach isn't working. Let me try a numerical search.
            
            # Use scipy if available
            try:
                from scipy.optimize import minimize
                has_scipy = True
            except ImportError:
                has_scipy = False
            
            if not has_scipy:
                continue
            
            # Parameterize: 6*10 + 6*10 + 4*10 = 160 parameters
            # Objective: sum of squared errors
            # 
            # This is a large nonlinear optimization. Let's try a simpler approach:
            # Fix W to be a random matrix and solve for U, V using least squares.
            
            for sub_attempt in range(50):
                if check_time():
                    return None
                
                W_fixed = np.random.randn(4, 10)
                # For each (a,b), the equation is:
                # sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] for all c
                # This is: U[:,r]*V[:,r] * W[:,r]^T summed over r = T[:,:,c]
                # 
                # For fixed W, this is bilinear in U and V.
                # Let's use alternating minimization.
                
                U_curr = np.random.randn(6, 10) * 0.1
                V_curr = np.random.randn(6, 10) * 0.1
                
                for iteration in range(100):
                    # Solve for U given V, W
                    # For each a: sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
                    # This is: sum_r U[a,r] * (V[:,r] * W[:,r]) = T[a,:,:]
                    # Let M_r = V[:,r] * W[:,r] (element-wise, shape 6x4)
                    # Then sum_r U[a,r] * M_r = T[a,:,:]
                    # This is a linear system in U[a,:]:
                    # [M_0, M_1, ..., M_9] * U[a,:] = T[a,:,:]
                    # where the matrix is 24x10 (flattening M_r)
                    
                    A_mat = np.zeros((24, 10))
                    for r in range(10):
                        A_mat[:, r] = (V_curr[:, r] * W_fixed[:, r]).flatten()
                    
                    for a in range(6):
                        b_vec = T[a, :, :].flatten()
                        # Solve A_mat * U[a,:] = b_vec (least squares)
                        sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        U_curr[a, :] = sol
                    
                    # Solve for V given U, W
                    # For each b: sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
                    # sum_r V[b,r] * (U[:,r] * W[:,r]) = T[:,b,:]
                    # A_mat2 is 24x10
                    A_mat2 = np.zeros((24, 10))
                    for r in range(10):
                        A_mat2[:, r] = (U_curr[:, r] * W_fixed[:, r]).flatten()
                    
                    for b in range(6):
                        b_vec = T[:, b, :].flatten()
                        sol, _, _, _ = np.linalg.lstsq(A_mat2, b_vec, rcond=None)
                        V_curr[b, :] = sol
                    
                    # Check convergence
                    if iteration % 10 == 0:
                        # Compute error
                        T_recon = np.einsum('ar,br,cr->abc', U_curr, V_curr, W_fixed)
                        err = np.linalg.norm(T_recon - T)
                        if err < 1e-6:
                            # Found a solution!
                            return U_curr.tolist(), V_curr.tolist(), W_fixed.tolist()
                
                # Also try fixing U and solving for V, W
                # And other combinations
        
        return None
    
    # Build and verify the rank-55 block decomposition
    best_result = None
    try:
        U55, V55, W55 = build_block_decomposition(11)
        if verify_fast(U55, V55, W55, n, m, p):
            best_result = {"U": U55, "V": V55, "W": W55}
        else:
            best_result = schoolbook()
    except Exception:
        best_result = schoolbook()

    if check_time():
        return best_result

    # Try to find a rank-10 block decomposition
    try:
        rank10_result = search_rank10_block()
        if rank10_result is not None:
            U10, V10, W10 = rank10_result
            # Build full decomposition using rank-10 blocks
            num_blocks = p // 2
            total_rank = num_blocks * 10
            U_full = [[0] * total_rank for _ in range(n * m)]
            V_full = [[0] * total_rank for _ in range(m * p)]
            W_full = [[0] * total_rank for _ in range(p * n)]
            
            for b in range(num_blocks):
                for r in range(10):
                    col = b * 10 + r
                    for a in range(n * m):
                        U_full[a][col] += U10[a][r]
                    for a in range(m * 2):
                        j = a // 2
                        k_local = a % 2
                        k_global = b * 2 + k_local
                        V_full[j * p + k_global][col] += V10[a][r]
                    for a in range(2 * n):
                        k_local = a // n
                        i = a % n
                        k_global = b * 2 + k_local
                        W_full[k_global * n + i][col] += W10[a][r]
            
            if verify_fast(U_full, V_full, W_full, n, m, p):
                best_result = {"U": U_full, "V": V_full, "W": W_full}
    except Exception:
        pass

    return best_result
# EVOLVE-BLOCK-END
