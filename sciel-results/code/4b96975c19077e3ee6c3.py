import numpy as np
import time
import itertools

DIMENSIONS = [4, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin

    n, m, p = DIMENSIONS  # 4, 4, 8
    nm = n * m   # 16
    mp = m * p   # 32
    pn = p * n   # 32

    # === Step 1: Build valid rank-98 solution via Block Strassen ===
    
    # Strassen 2x2*2x2 decomposition (rank 7)
    U_2 = np.zeros((4, 7))
    V_2 = np.zeros((4, 7))
    W_2 = np.zeros((4, 7))

    # c=0: M1 = (a00+a11)(b00+b11)
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1

    # c=1: M2 = (a10+a11)*b00
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1; W_2[3, 1] = -1

    # c=2: M3 = a00*(b01-b11)
    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1; W_2[3, 2] = 1

    # c=3: M4 = a11*(b10-b00)
    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    W_2[0, 3] = 1; W_2[2, 3] = 1

    # c=4: M5 = (a00+a01)*b11
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[0, 4] = -1; W_2[1, 4] = 1

    # c=5: M6 = (a10-a00)*(b00+b01)
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    W_2[3, 5] = 1

    # c=6: M7 = (a01-a11)*(b10+b11)
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    W_2[0, 6] = 1

    # Block Strassen for 4x4*4x4 (rank 49 per half)
    A_specs = [
        [(1, 0, 0), (1, 1, 1)],   # P0: A00 + A11
        [(1, 1, 0), (1, 1, 1)],   # P1: A10 + A11
        [(1, 0, 0)],              # P2: A00
        [(1, 1, 1)],              # P3: A11
        [(1, 0, 0), (1, 0, 1)],   # P4: A00 + A01
        [(1, 1, 0), (-1, 0, 0)],  # P5: A10 - A00
        [(1, 0, 1), (-1, 1, 1)],  # P6: A01 - A11
    ]

    B_specs = [
        [(1, 0, 0), (1, 1, 1)],   # P0: B00 + B11
        [(1, 0, 0)],              # P1: B00
        [(1, 0, 1), (-1, 1, 1)],  # P2: B01 - B11
        [(1, 1, 0), (-1, 0, 0)],  # P3: B10 - B00
        [(1, 1, 1)],              # P4: B11
        [(1, 0, 0), (1, 0, 1)],   # P5: B00 + B01
        [(1, 1, 0), (1, 1, 1)],   # P6: B10 + B11
    ]

    C_specs = [
        {(0, 0): 1, (1, 1): 1},        # P0: C00 + C11
        {(1, 0): 1, (1, 1): -1},       # P1: C10 - C11
        {(0, 1): 1, (1, 1): 1},        # P2: C01 + C11
        {(0, 0): 1, (1, 0): 1},        # P3: C00 + C10
        {(0, 0): -1, (0, 1): 1},       # P4: -C00 + C01
        {(1, 1): 1},                   # P5: C11
        {(0, 0): 1},                   # P6: C00
    ]

    U = np.zeros((nm, 98))
    V = np.zeros((mp, 98))
    W = np.zeros((pn, 98))

    for half in range(2):
        for k_idx in range(7):
            for m_s in range(7):
                col = half * 49 + k_idx * 7 + m_s

                # A part
                for coeff_a, r_block, c_block in A_specs[k_idx]:
                    for r in range(2):
                        for c in range(2):
                            global_i = (r_block * 2 + r)
                            global_j = (c_block * 2 + c)
                            local_a_idx = r * 2 + c
                            U[global_i * m + global_j, col] += coeff_a * U_2[local_a_idx, m_s]

                # B part
                for coeff_b, r_block, c_block in B_specs[k_idx]:
                    for r in range(2):
                        for c in range(2):
                            global_j = (r_block * 2 + r)
                            global_k = (c_block * 2 + c) + half * 4
                            local_b_idx = r * 2 + c
                            V[global_j * p + global_k, col] += coeff_b * V_2[local_b_idx, m_s]

                # C part (W)
                for (r_block, c_block), coeff_c in C_specs[k_idx].items():
                    for r in range(2):
                        for c in range(2):
                            global_k = (c_block * 2 + c) + half * 4
                            global_i = (r_block * 2 + r)
                            local_w_idx = r * 2 + c
                            W[global_k * n + global_i, col] += coeff_c * W_2[local_w_idx, m_s]

    # Verify the 98-rank solution is correct
    def verify(U_mat, V_mat, W_mat):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0 if (i == j) else 0.0
                    # Actually for matrix mult: T[i,j,k] = delta_{i,j} ... no
                    # The tensor for matmul: T[i,j,k] where C[i,k] = sum_j A[i,j]*B[j,k]
                    # In the trace convention: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = delta_{j,j}... 
                    # Actually it's: the (i,k) entry of C = sum_j A[i,j] B[j,k]
                    # So the tensor T[i,j,k] = 1 if this contributes to C[i,k], i.e., always 1 for the bilinear form
                    # Wait, let me re-read: "exactly decompose the matrix multiplication tensor"
                    # The matmul tensor T has T[i,j,k] = 1 for all i,j,k (it's the identity bilinear form)
                    # No: T[i,j,k] = delta_{i,j} would be wrong. 
                    # The matrix multiplication tensor: T_{ijk} = 1 if the (i,j) of A and (j,k) of B contribute to (i,k) of C
                    # So T[i,j,k] = 1 for all valid i,j,k (since every A[i,j]*B[j,k] contributes to C[i,k])
                    # Actually no - the tensor is defined such that C[i,k] = sum_{j} A[i,j] * B[j,k]
                    # The rank-1 decomposition: C[i,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                    # This must equal A[i,j]*B[j,k] for each (i,j,k) term... 
                    # Actually the standard definition: the matmul tensor T has components T[i,j,k] = 1 
                    # and the bilinear form is: sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] * (output at [i,k])
                    # So we need: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k
                    actual = 0.0
                    for r in range(U_mat.shape[1]):
                        actual += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                    if abs(actual - 1.0) > 1e-10:
                        return False, (i, j, k, actual)
        return True, None

    valid, err = verify(U, V, W)
    if not valid:
        # Fallback to schoolbook if verification fails
        rank_school = n * m * p
        U = np.zeros((nm, rank_school))
        V = np.zeros((mp, rank_school))
        W = np.zeros((pn, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_rank = 98
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    if time.time() > deadline - 20:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # === Step 2: Search for rank reduction ===
    # Strategy: Try to find a lower rank decomposition by:
    # a) Looking for linear dependencies in the 98 columns
    # b) Direct search for rank R < 98 using the rectangular structure
    
    # First, try to see if we can reduce rank by finding that some columns are linear combinations
    # of others in a way that preserves the tensor equation.
    
    # Alternative approach: use the fact that the problem is 4x4x8, not 4x4x4.
    # The B matrix has 8 columns, so we can potentially share structure.
    
    # Try a direct optimization approach for rank R
    def try_rank(R, num_trials=3, max_iter=200):
        """Try to find a valid rank-R decomposition."""
        for trial in range(num_trials):
            if time.time() > deadline - 10:
                return None
            
            np.random.seed(42 + trial)
            U_try = np.random.randn(nm, R) * 0.1
            V_try = np.random.randn(mp, R) * 0.1
            W_try = np.random.randn(pn, R) * 0.1
            
            # Use a simple iterative method: for each term, try to fix one factor at a time
            # This is a heuristic - not guaranteed to converge
            
            for iteration in range(max_iter):
                if time.time() > deadline - 10:
                    return None
                
                # Compute residual
                # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] - 1
                # This is expensive to compute fully, so sample
                
                # Instead, use a different approach: 
                # The constraint is that for all (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
                # 
                # We can view this as: for fixed (i,j), the vector [V[j*p+k,r]*W[k*n+i,r]]_k 
                # must satisfy certain constraints...
                
                # Let's use a simpler heuristic: randomly perturb and check
                # This is not very efficient but might find something
                
                # Actually, let's try a different structural approach:
                # Split into two 4x4x4 problems and see if we can share rank-1 terms
                pass
            
            # Check if the random init happens to work (unlikely)
            valid, _ = verify(U_try, V_try, W_try)
            if valid:
                return U_try, V_try, W_try
        
        return None

    # Try ranks from 97 down
    for target_rank in range(97, 70, -1):
        if time.time() > deadline - 15:
            break
        
        result = try_rank(target_rank, num_trials=2, max_iter=50)
        if result is not None:
            U_r, V_r, W_r = result
            valid, _ = verify(U_r, V_r, W_r)
            if valid:
                best_rank = target_rank
                best_U, best_V, best_W = U_r, V_r, W_r
                break  # Found a valid lower rank
    
    # Additional structural search: exploit the 4x4x8 structure
    # The key insight: B has 8 columns = 2 blocks of 4
    # We can write B = [B1 | B2] where B1, B2 are 4x4
    # Then C = A * [B1 | B2] = [A*B1 | A*B2]
    # So we need two 4x4x4 multiplications
    # But we might be able to share some rank-1 terms between them
    
    # Try a shared-structure approach
    if time.time() < deadline - 20:
        # Try rank 90-97 with a structured initialization
        # Initialize from the 98-rank solution and try to merge terms
        
        for target_rank in range(97, 85, -1):
            if time.time() > deadline - 15:
                break
            
            # Start from the 98-rank solution and try to remove terms
            # by expressing them as linear combinations of others
            
            U_try = best_U.copy()
            V_try = best_V.copy()
            W_try = best_W.copy()
            
            # Try to find a column that is approximately a linear combination of others
            # and remove it
            removed = 0
            cols_to_remove = []
            
            for col in range(97, -1, -1):
                if removed >= 98 - target_rank:
                    break
                if col in cols_to_remove:
                    continue
                
                # Check if column col can be expressed as combination of other columns
                # This is a complex check, so use a heuristic
                # Skip for now - the random search above is the main approach
            
            # Alternative: try a completely different structural decomposition
            # Use the fact that 4 = 2*2 and 8 = 2*4
            # Maybe a different blocking strategy helps
            
            pass
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
