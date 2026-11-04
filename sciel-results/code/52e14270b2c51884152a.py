"""Matrix multiplication 2x3x4 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 4]
DEADLINE = 150.0
START_TIME = time.time()

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 4
    
    # The multiplication tensor T[k, i, j] = 1 for all k in [0,p), i in [0,n), j in [0,m)
    # We seek U (n*m x rank), V (m*p x rank), W (p*n x rank) such that
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all (i,j,k)
    
    # Known lower bound for 2x3x4 matrix multiplication is 19.
    # The best known upper bound is 22 (from Strassen-style decomposition).
    # We'll try to find a rank-19 or rank-20 decomposition using linear algebra.
    
    # Strategy: For a given rank r, we need to find U, V, W with integer/half-integer entries.
    # The tensor decomposition condition is:
    # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    #
    # This is a system of polynomial equations. We can approach it by:
    # 1. Fixing W and solving for U, V (or vice versa)
    # 2. Using the fact that the tensor has a specific structure
    
    # Let's use a different approach: the multiplication tensor for 2x3x4
    # can be decomposed using the structure of the matrices.
    
    # Approach: Try rank 19, 20, 21, 22 in order.
    # For each rank, we set up the equations and try to solve.
    
    # The key insight: the tensor T is the "all-ones" tensor of shape (p, n, m) = (4, 2, 3).
    # T[k, i, j] = 1 for all valid indices.
    
    # A rank-r decomposition means:
    # T = sum_{r=0}^{R-1} u_r ⊗ v_r ⊗ w_r
    # where u_r is in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}
    
    # In matrix form: for each r, U[:,r] = u_r, V[:,r] = v_r, W[:,r] = w_r
    # And the condition is: for all (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is equivalent to: the tensor formed by the outer products equals the all-ones tensor.
    
    # Let's try a structured approach based on known results.
    # For 2x3x4, we can split as:
    # A (2x3) = [A_left (2x2) | A_right (2x1)]
    # B (3x4) = [B_top (2x4); B_bottom (1x4)]
    # C = A_left * B_top + A_right * B_bottom
    
    # A_left * B_top: 2x2 * 2x4 = two independent 2x2 * 2x2 multiplications
    #   Each 2x2*2x2 can be done in rank 7 (Strassen), so 14 total
    # A_right * B_bottom: 2x1 * 1x4 = outer product, rank 8
    # Total: 22
    
    # To go below 22, we need to share factors between the blocks.
    
    # Let me try to construct a rank-21 or rank-20 decomposition by sharing.
    
    # Actually, let's try a more systematic approach.
    # We'll use the fact that for the all-ones tensor, there might be a low-rank decomposition.
    
    # The all-ones tensor of shape (4,2,3) has a natural decomposition:
    # T[k,i,j] = 1 = sum over some structure
    
    # Let's try: T[k,i,j] = sum_{a=0}^{A-1} sum_{b=0}^{B-1} f_a(k) * g_b(i) * h_{a,b}(j)
    # But this is a different factorization.
    
    # Standard tensor rank decomposition:
    # T[k,i,j] = sum_{r=0}^{R-1} u_r[i,j] * v_r[j,k] * w_r[k,i]
    
    # For the all-ones tensor, one trivial decomposition is R = 4*2*3 = 24 (schoolbook).
    # We want to do better.
    
    # Let me try to use the structure more carefully.
    # Define:
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    
    # The constraint is: for all (i,j,k) in [0,2)x[0,3)x[0,4):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This gives 2*3*4 = 24 equations.
    # Unknowns: R * (n*m + m*p + p*n) = R * (6 + 12 + 8) = 26R
    
    # For R=19: 26*19 = 494 unknowns, 24 equations. Very underdetermined.
    # But the equations are nonlinear (trilinear).
    
    # Approach: Use a heuristic search.
    # 1. Start with the rank-22 Strassen decomposition.
    # 2. Try to merge factors to reduce rank.
    
    # Let's first construct the correct rank-22 decomposition and verify it.
    
    def build_strassen_rank22():
        """Build the rank-22 Strassen-based decomposition."""
        U = np.zeros((n * m, 22))
        V = np.zeros((m * p, 22))
        W = np.zeros((p * n, 22))
        
        def add_u(i, j, c, val):
            U[i * m + j, c] += val
        def add_v(j, k, c, val):
            V[j * p + k, c] += val
        def add_w(k, i, c, val):
            W[k * n + i, c] += val
        
        def add_strassen_block(a_indices, b_indices, c_indices, offset):
            """
            a_indices: list of 4 (i,j) pairs for a00,a01,a10,a11
            b_indices: list of 4 (j,k) pairs for b00,b01,b10,b11
            c_indices: list of 4 (k,i) pairs for C00,C01,C10,C11
            """
            a00, a01, a10, a11 = a_indices
            b00, b01, b10, b11 = b_indices
            C00, C01, C10, C11 = c_indices
            
            # m1 = (a00+a11)(b00+b11)
            add_u(a00[0], a00[1], offset+0, 1)
            add_u(a11[0], a11[1], offset+0, 1)
            add_v(b00[0], b00[1], offset+0, 1)
            add_v(b11[0], b11[1], offset+0, 1)
            add_w(C00[0], C00[1], offset+0, 1)
            add_w(C11[0], C11[1], offset+0, 1)
            
            # m2 = (a10+a11)(b00)
            add_u(a10[0], a10[1], offset+1, 1)
            add_u(a11[0], a11[1], offset+1, 1)
            add_v(b00[0], b00[1], offset+1, 1)
            add_w(C10[0], C10[1], offset+1, 1)
            add_w(C11[0], C11[1], offset+1, -1)
            
            # m3 = a00(b01-b11)
            add_u(a00[0], a00[1], offset+2, 1)
            add_v(b01[0], b01[1], offset+2, 1)
            add_v(b11[0], b11[1], offset+2, -1)
            add_w(C01[0], C01[1], offset+2, 1)
            add_w(C11[0], C11[1], offset+2, 1)
            
            # m4 = a11(b10-b00)
            add_u(a11[0], a11[1], offset+3, 1)
            add_v(b10[0], b10[1], offset+3, 1)
            add_v(b00[0], b00[1], offset+3, -1)
            add_w(C00[0], C00[1], offset+3, 1)
            add_w(C10[0], C10[1], offset+3, 1)
            
            # m5 = (a00+a01)(b11)
            add_u(a00[0], a00[1], offset+4, 1)
            add_u(a01[0], a01[1], offset+4, 1)
            add_v(b11[0], b11[1], offset+4, 1)
            add_w(C00[0], C00[1], offset+4, -1)
            add_w(C01[0], C01[1], offset+4, 1)
            
            # m6 = (a10-a00)(b00+b10)
            add_u(a10[0], a10[1], offset+5, 1)
            add_u(a00[0], a00[1], offset+5, -1)
            add_v(b00[0], b00[1], offset+5, 1)
            add_v(b10[0], b10[1], offset+5, 1)
            add_w(C11[0], C11[1], offset+5, 1)
            
            # m7 = (a01-a11)(b10+b11)
            add_u(a01[0], a01[1], offset+6, 1)
            add_u(a11[0], a11[1], offset+6, -1)
            add_v(b10[0], b10[1], offset+6, 1)
            add_v(b11[0], b11[1], offset+6, 1)
            add_w(C00[0], C00[1], offset+6, 1)
        
        # Block 1: A[0:2, 0:2] * B[0:2, 0:2]
        a1 = [(0,0), (0,1), (1,0), (1,1)]
        b1 = [(0,0), (0,1), (1,0), (1,1)]
        c1 = [(0,0), (1,0), (0,1), (1,1)]  # C[k,i]: C00=(k=0,i=0), C01=(k=1,i=0), C10=(k=0,i=1), C11=(k=1,i=1)
        add_strassen_block(a1, b1, c1, 0)
        
        # Block 2: A[0:2, 0:2] * B[0:2, 2:4]
        b2 = [(0,2), (0,3), (1,2), (1,3)]
        c2 = [(2,0), (3,0), (2,1), (3,1)]  # C02=(k=2,i=0), C03=(k=3,i=0), C12=(k=2,i=1), C13=(k=3,i=1)
        add_strassen_block(a1, b2, c2, 7)
        
        # Block 3: A[0:2, 2:3] * B[2:3, 0:4]
        # This is 2x1 * 1x4, rank 8
        offset = 14
        for i in range(2):
            for k in range(4):
                add_u(i, 2, offset, 1)
                add_v(2, k, offset, 1)
                add_w(k, i, offset, 1)
                offset += 1
        
        return U, V, W
    
    def verify(U, V, W):
        """Verify the decomposition is exact."""
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(R):
                        total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if not np.isclose(total, 1.0, atol=1e-9):
                        return False
        return True
    
    # First, build and verify the rank-22 decomposition
    U22, V22, W22 = build_strassen_rank22()
    
    if verify(U22, V22, W22):
        # Now try to reduce rank by merging factors
        # Strategy: look for pairs of factors that can be combined
        
        # Try to find a rank-21 decomposition by combining two factors
        # This requires finding U1, U2, V1, V2, W1, W2 such that
        # U1*V1*W1 + U2*V2*W2 can be replaced by a single factor
        # This is generally not possible unless there's special structure.
        
        # Alternative: use a numerical optimization approach
        # For rank r, we have 26r variables and 24 trilinear equations.
        # We can try to solve this using iterative methods.
        
        # Let's try a randomized approach for lower ranks
        best_result = {"U": U22.tolist(), "V": V22.tolist(), "W": W22.tolist()}
        best_rank = 22
        
        for target_rank in [21, 20, 19]:
            if time.time() - START_TIME > DEADLINE - 20:
                break
            
            found = False
            for attempt in range(100):
                if time.time() - START_TIME > DEADLINE - 10:
                    break
                
                # Initialize with random values
                np.random.seed(attempt * 12345 + target_rank)
                U = np.random.randn(n*m, target_rank)
                V = np.random.randn(m*p, target_rank)
                W = np.random.randn(p*n, target_rank)
                
                # Iteratively refine: fix two matrices, solve for the third
                # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                # If we fix U and V, then for each (k,i):
                #   sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * V[j*p+k, r]) = 1
                # This is a linear system in W for each (k,i) pair, but coupled through r.
                
                # Actually, let's use a different approach.
                # Fix W, then for each (i,j):
                #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
                # This is a linear system in the products U[i*m+j, r] * V[j*p+k, r]
                
                # Let's use a simpler iterative scheme:
                # 1. Fix V and W, solve for U (linear)
                # 2. Fix U and W, solve for V (linear)
                # 3. Fix U and V, solve for W (linear)
                # Repeat until convergence.
                
                for iteration in range(100):
                    # Solve for U: for each (i,j), we need
                    # sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1 for all k
                    # This is a system of p equations in n*m*target_rank unknowns... 
                    # Actually, for fixed (i,j), we have target_rank unknowns U[i*m+j, 0..R-1]
                    # and p equations. If p < target_rank, it's underdetermined.
                    
                    # Let's use least squares.
                    # For fixed (i,j): A_{k,r} = V[j*p+k, r] * W[k*n+i, r]
                    # We want: A @ U[i*m+j, :] = ones(p)
                    
                    for i in range(n):
                        for j in range(m):
                            A_mat = np.zeros((p, target_rank))
                            for k in range(p):
                                for r in range(target_rank):
                                    A_mat[k, r] = V[j*p+k, r] * W[k*n+i, r]
                            b_vec = np.ones(p)
                            # Least squares solution
                            if np.linalg.matrix_rank(A_mat) > 0:
                                U[i*m+j, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                    
                    # Solve for V: for each (j,k), we need
                    # sum_r V[j*p+k, r] * (U[i*m+j, r] * W[k*n+i, r]) = 1 for all i
                    for j in range(m):
                        for k in range(p):
                            A_mat = np.zeros((n, target_rank))
                            for i in range(n):
                                for r in range(target_rank):
                                    A_mat[i, r] = U[i*m+j, r] * W[k*n+i, r]
                            b_vec = np.ones(n)
                            if np.linalg.matrix_rank(A_mat) > 0:
                                V[j*p+k, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                    
                    # Solve for W: for each (k,i), we need
                    # sum_r W[k*n+i, r] * (U[i*m+j, r] * V[j*p+k, r]) = 1 for all j
                    for k in range(p):
                        for i in range(n):
                            A_mat = np.zeros((m, target_rank))
                            for j in range(m):
                                for r in range(target_rank):
                                    A_mat[j, r] = U[i*m+j, r] * V[j*p+k, r]
                            b_vec = np.ones(m)
                            if np.linalg.matrix_rank(A_mat) > 0:
                                W[k*n+i, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                
                # Check if we converged
                if verify(U, V, W):
                    # Round to nearest half-integer if possible
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    if verify(U_rounded, V_rounded, W_rounded):
                        best_result = {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                        best_rank = target_rank
                        found = True
                        break
                    else:
                        # Use the unrounded version if it's exact
                        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                        best_rank = target_rank
                        found = True
                        break
            
            if found:
                break
        
        return best_result
    else:
        # Fallback to schoolbook rank-24
        U_school = np.zeros((n * m, 24))
        V_school = np.zeros((m * p, 24))
        W_school = np.zeros((p * n, 24))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j, idx] = 1
                    V_school[j * p + k, idx] = 1
                    W_school[k * n + i, idx] = 1
                    idx += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
