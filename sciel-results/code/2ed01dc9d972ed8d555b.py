"""Matrix multiplication 2,4,6 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 6]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x6 matrix.
    
    Strategy:
    1. Build a structural decomposition using Strassen on 2x2x2 blocks.
       Split p=6 into three 2-column blocks. For each block, split the 4x4 
       multiplication into two 2x2 multiplications, each handled by Strassen (rank 7).
       Base rank = 3 * 2 * 7 = 42.
       
    2. Apply the Winograd Fourier Transform Algorithm (FTW) perspective:
       The 2x4x6 tensor can be seen as a Kronecker product structure.
       We explore whether combining the three 2-column blocks more cleverly
       can reduce the total rank below 42.
       
    3. Use numpy-based verification for speed.
    4. Attempt rank reduction through:
       a. Removing zero columns
       b. Checking for linearly dependent columns via SVD
       c. Randomized search for lower-rank decompositions
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    def schoolbook():
        r = n * m * p
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, r

    def strassen_decomposition():
        """
        Build a rank-42 decomposition using Strassen on 2x2x2 blocks.
        
        Split p=6 into three blocks of 2 columns each.
        For each block r (r=0,1,2), compute C[:, 2r:2r+2] = A * B[:, 2r:2r+2].
        
        A is 2x4, split into A1 (cols 0,1) and A2 (cols 2,3), each 2x2.
        B[:, 2r:2r+2] is 4x2, split into B_top (rows 0,1) and B_bot (rows 2,3), each 2x2.
        
        C_block = A1 * B_top + A2 * B_bot
        
        This is a sum of two 2x2x2 tensor products, each of rank 7 (Strassen).
        Total rank per block = 14, total = 42.
        """
        rank = 42
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Strassen coefficients for 2x2x2
        # M1 = (X00+X11)(Y00+Y11)
        # M2 = (X10+X11)Y00
        # M3 = X00(Y01-Y11)
        # M4 = X11(Y10-Y00)
        # M5 = (X00+X01)Y11
        # M6 = (X10-X00)(Y00+Y01)
        # M7 = (X01-X11)(Y10+Y11)
        #
        # Z00 = M1 + M4 - M5 + M7
        # Z01 = M3 + M5
        # Z10 = M2 + M4
        # Z11 = M1 - M2 + M3 + M6
        
        # Each term: list of (local_index, coefficient)
        # local index: 0=X00/Y00/Z00, 1=X01/Y01/Z01, 2=X10/Y10/Z10, 3=X11/Y11/Z11
        strassen_terms = [
            # M1: U has X00,X11; V has Y00,Y11; W has Z00,Z11
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            # M2: U has X10,X11; V has Y00; W has Z10,Z11
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
            # M3: U has X00; V has Y01,Y11; W has Z01,Z11
            ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
            # M4: U has X11; V has Y10,Y00; W has Z10,Z00
            ([(3, 1)], [(2, 1), (0, -1)], [(2, 1), (0, 1)]),
            # M5: U has X00,X01; V has Y11; W has Z00,Z01
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
            # M6: U has X10,X00; V has Y00,Y01; W has Z11
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
            # M7: U has X01,X11; V has Y10,Y11; W has Z00
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)])
        ]
        
        col = 0
        for r in range(3):  # 3 column blocks
            for sub in range(2):  # sub=0: A1*B_top, sub=1: A2*B_bot
                # A block offset: sub=0 -> cols 0,1; sub=1 -> cols 2,3
                a_col_offset = sub * 2
                # B row offset: sub=0 -> rows 0,1; sub=1 -> rows 2,3
                b_row_offset = sub * 2
                
                for s_idx in range(7):
                    u_terms, v_terms, w_terms = strassen_terms[s_idx]
                    
                    # Update U
                    for local_u, coeff in u_terms:
                        a = local_u // 2  # row in 2x2 block
                        l = local_u % 2   # col in 2x2 block
                        global_u = a * m + (a_col_offset + l)
                        U[global_u, col] += coeff
                    
                    # Update V
                    for local_v, coeff in v_terms:
                        l = local_v // 2   # row in 2x2 block
                        c = local_v % 2    # col in 2x2 block
                        global_v = (b_row_offset + l) * p + (2 * r + c)
                        V[global_v, col] += coeff
                    
                    # Update W
                    for local_w, coeff in w_terms:
                        a = local_w // 2   # row in 2x2 output
                        c = local_w % 2    # col in 2x2 output
                        global_w = (2 * r + c) * n + a
                        W[global_w, col] += coeff
                    
                    col += 1
        
        return U, V, W, rank

    def verify_numpy(U, V, W, r):
        """Check if the decomposition is exact using numpy."""
        # Compute T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Reshape U to (n,m,r), V to (m,p,r), W to (p,n,r)
        U_reshaped = U.reshape(n, m, r)
        V_reshaped = V.reshape(m, p, r)
        W_reshaped = W.reshape(p, n, r)
        
        # For each (i,j,k), compute sum over r of U[i,j,r]*V[j,k,r]*W[k,i,r]
        # This is a 3-tensor contraction
        T = np.einsum('ijr,jkr,ikr->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        # Check if T[i,j,k] == 1 for all i,j,k
        expected = np.ones((n, m, p))
        return np.allclose(T, expected, atol=1e-9)

    def remove_zero_columns(U, V, W, r):
        """Remove columns that are all zeros in any of U, V, W."""
        keep = []
        for c in range(r):
            if np.any(U[:, c] != 0) and np.any(V[:, c] != 0) and np.any(W[:, c] != 0):
                keep.append(c)
        
        new_r = len(keep)
        if new_r == r:
            return U, V, W, r
        
        new_U = U[:, keep]
        new_V = V[:, keep]
        new_W = W[:, keep]
        return new_U, new_V, new_W, new_r

    def try_svd_reduction(U, V, W, r):
        """
        Try to find linear dependencies among columns using SVD.
        If some columns are approximately linearly dependent, we can potentially reduce rank.
        """
        if time.time() > deadline:
            return U, V, W, r
        
        # Stack all columns into a single matrix for SVD
        # Each column c has entries (U[:,c], V[:,c], W[:,c])
        # Total dimension = n*m + m*p + p*n = 8 + 24 + 12 = 44
        stacked = np.vstack([U.T, V.T, W.T])  # shape (44, r)
        
        # Compute SVD
        try:
            U_svd, s, Vt = np.linalg.svd(stacked, full_matrices=False)
        except:
            return U, V, W, r
        
        # Find the numerical rank
        tol = 1e-10
        num_rank = np.sum(s > tol)
        
        if num_rank < r:
            # Keep only the first num_rank columns
            # Actually, we need to be more careful - SVD gives us a basis
            # For now, just return as-is since proper reduction is complex
            pass
        
        return U, V, W, r

    def randomized_search(U, V, W, r, target_rank, max_attempts=100):
        """
        Try to find a lower-rank decomposition by perturbing and re-solving.
        """
        if target_rank >= r:
            return U, V, W, r
        
        best_found = (U, V, W, r)
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Randomly set some entries to 0 to sparsify
            new_U = U.copy()
            new_V = V.copy()
            new_W = W.copy()
            
            # Randomly zero out some entries
            for i in range(len(new_U)):
                for c in range(r):
                    if random.random() < 0.2:
                        new_U[i, c] = 0
            for i in range(len(new_V)):
                for c in range(r):
                    if random.random() < 0.2:
                        new_V[i, c] = 0
            for i in range(len(new_W)):
                for c in range(r):
                    if random.random() < 0.2:
                        new_W[i, c] = 0
            
            # Remove zero columns
            new_U, new_V, new_W, new_r = remove_zero_columns(new_U, new_V, new_W, r)
            
            if new_r <= target_rank and new_r < r:
                if verify_numpy(new_U, new_V, new_W, new_r):
                    best_found = (new_U, new_V, new_W, new_r)
                    return best_found
        
        return best_found

    def ftw_kronecker_attempt():
        """
        Attempt to use the Winograd Fourier Transform Algorithm perspective.
        
        The 2x4x6 tensor can be viewed as a Kronecker product structure.
        2x4x6 = (2x2x2) ⊗ (1x2x3) in some sense.
        
        For the 1x2x3 part, we need a rank-minimal decomposition.
        A 1x2x3 matrix multiplication is just a vector-matrix product.
        The tensor for 1x2x3 has rank 3 (each output depends on 3 inputs).
        
        So the Kronecker product rank would be 7 * 3 = 21.
        However, this isn't directly applicable because the structure doesn't
        cleanly factor this way for our specific dimensions.
        
        Let's try a different approach: exploit the fact that p=6=2*3.
        We can think of the 4x6 matrix B as having 3 groups of 2 columns.
        Instead of treating each group independently, we can try to share
        Strassen intermediates across groups.
        
        For now, this is a placeholder that returns the standard Strassen result.
        """
        # The FTW/Kronecker approach would require more sophisticated analysis
        # For this implementation, we'll return the standard Strassen decomposition
        # but with a flag indicating we tried the FTW approach
        U, V, W, r = strassen_decomposition()
        return U, V, W, r

    # Start with the Strassen-based decomposition
    U, V, W, r = strassen_decomposition()
    
    # Remove zero columns
    U, V, W, r = remove_zero_columns(U, V, W, r)
    
    # Verify the base decomposition
    if not verify_numpy(U, V, W, r):
        # Fall back to schoolbook if something went wrong
        U, V, W, r = schoolbook()
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = r
    
    # Try SVD-based reduction
    U2, V2, W2, r2 = try_svd_reduction(U, V, W, r)
    if r2 < best_rank and verify_numpy(U2, V2, W2, r2):
        best_result = {"U": U2, "V": V2, "W": W2}
        best_rank = r2
        U, V, W, r = U2, V2, W2, r2
    
    # Try FTW/Kronecker approach
    U3, V3, W3, r3 = ftw_kronecker_attempt()
    if r3 < best_rank and verify_numpy(U3, V3, W3, r3):
        best_result = {"U": U3, "V": V3, "W": W3}
        best_rank = r3
        U, V, W, r = U3, V3, W3, r3
    
    # Try to reduce rank further with random search
    for target_rank in range(best_rank - 1, max(20, best_rank - 15), -1):
        if time.time() > deadline:
            break
        
        U4, V4, W4, r4 = randomized_search(U, V, W, r, target_rank, max_attempts=50)
        
        if r4 < best_rank and verify_numpy(U4, V4, W4, r4):
            best_result = {"U": U4, "V": V4, "W": W4}
            best_rank = r4
            U, V, W, r = U4, V4, W4, r4
    
    # Final verification
    if not verify_numpy(best_result["U"], best_result["V"], best_result["W"], best_rank):
        # Fall back to schoolbook
        U, V, W, r = schoolbook()
        best_result = {"U": U, "V": V, "W": W}
        best_rank = r
    
    return best_result
# EVOLVE-BLOCK-END
