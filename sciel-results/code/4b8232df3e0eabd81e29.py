import numpy as np
import time
import random

DIMENSIONS = [2, 4, 4]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build the tensor T[i,j,k] = delta (should be 1 if i,j,k correspond to a valid product)
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # where i in [0,n), j in [0,m), k in [0,p)
    
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    # For each r, we have a rank-1 tensor: U[:,r] outer V[:,r] outer W[:,r]
    # But we need to check that the sum equals the matmul tensor.
    
    # The matmul tensor T[i,j,k] = 1 if the (i,j) entry of A times (j,k) entry of B
    # contributes to (i,k) entry of C.
    # Actually, the tensor for matmul is: T[i,j,k] = delta_{i,k} ... no.
    # The matrix multiplication tensor is defined as:
    # T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor has entries T[i,j,k] = 1 for all valid (i,j,k) tuples.
    # Wait, that's not right either.
    
    # The matrix multiplication tensor M_{n,m,p} has entries:
    # M[i,j,k] = 1 if the product A[i,j] * B[j,k] contributes to C[i,k]
    # Actually, M[i,j,k] = 1 for all i,j,k in their respective ranges.
    # The tensor is just the all-ones tensor of shape (n, m, p)? No.
    
    # Let me think again. The matrix multiplication tensor T has:
    # T[i,j,k] = 1 if and only if the (i,j) entry of A and (j,k) entry of B
    # multiply to contribute to the (i,k) entry of C.
    # So T[i,j,k] = 1 for all valid i, j, k.
    
    # The decomposition requires:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # So we need: for all i in [0,n), j in [0,m), k in [0,p):
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook_decomp(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()

def _strassen_2x2_decomp():
    """
    Return the Strassen rank-7 decomposition for 2x2 * 2x2 matrix multiplication.
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    """
    # Strassen's 7 multiplications for 2x2:
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M1-M4+M6
    # C21 = M2+M4-M3+M5
    # C22 = M1-M2+M3-M6
    
    # U indices: a11=0, a12=1, a21=2, a22=3
    # V indices: b11=0, b12=1, b21=2, b22=3
    # W indices: c11=0, c12=1, c21=2, c22=3
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (a11+a22)(b11+b22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1  # contributes to C11, C12, C21, C22? No.
    
    # Let me be more careful.
    # M1 contributes to: C11 (+1), C12 (+1), C21 (0), C22 (+1)
    # Wait: C11 = M1+M4-M5+M7, C12 = M3+M1-M4+M6, C21 = M2+M4-M3+M5, C22 = M1-M2+M3-M6
    # So M1: C11=+1, C12=+1, C21=0, C22=+1
    
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1; W[3, 0] = 1
    
    # M2 = (a21+a22)b11
    # M2: C11=0, C12=0, C21=+1, C22=-1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = a11(b12-b22)
    # M3: C11=0, C12=+1, C21=-1, C22=+1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[2, 2] = -1; W[3, 2] = 1
    
    # M4 = a22(b21-b11)
    # M4: C11=+1, C12=-1, C21=+1, C22=0
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = -1; W[2, 3] = 1
    
    # M5 = (a11+a12)b22
    # M5: C11=-1, C12=0, C21=+1, C22=0
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (a21-a11)(b11+b12)
    # M6: C11=0, C12=+1, C21=0, C22=-1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[1, 5] = 1; W[3, 5] = -1
    
    # M7 = (a12-a22)(b21+b22)
    # M7: C11=+1, C12=0, C21=0, C22=0
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U.tolist(), V.tolist(), W.tolist()

def _row_strassen_decomp():
    """
    Compute a single row of A (1x4) times B (4x4) using Strassen on 2x2 blocks.
    Returns (U_row, V_row, W_row) where:
      U_row shape: (4, 7)  - coefficients for the 4 entries of the A row
      V_row shape: (16, 7) - coefficients for the 16 entries of B
      W_row shape: (4, 7)  - coefficients for the 4 entries of the C row
    
    The A row is split into two 1x2 halves: a1 = A_row[0:2], a2 = A_row[2:4]
    B is split into 2x2 blocks: B11, B12, B21, B22
    C_row = a1 * [B11, B12] + a2 * [B21, B22]
           = [a1*B11 + a2*B21, a1*B12 + a2*B22]
    
    For each 1x2 * 2x2 product, we use a rank-3 decomposition:
      a1*B11: c1 = a1[0]*b11 + a1[1]*b12 (for column 1 of result)
              c2 = a1[0]*b21 + a1[1]*b22 (for column 2 of result)
      Actually, 1x2 * 2x2 = 1x2, which is just 2 scalar multiplications per output.
      So a1*B11 has rank 2 (two scalar products).
      Similarly for all other blocks.
    
    Total rank for one row: 4 blocks * 2 = 8? No wait.
    a1*B11: output is 1x2, each entry needs 2 mults, total 4 mults? No.
    a1 = [x, y], B11 = [[p, q], [r, s]]
    a1*B11 = [xp + yr, xq + ys]
    This is rank 2: we need x*p, y*r for first output, x*q, y*s for second.
    But we can share: actually it's just 4 scalar multiplications, rank 4.
    
    Wait, let me reconsider. The tensor for 1x2 * 2x2 -> 1x2:
    T[x,y,p,q,r,s; out1,out2] where out1 = xp+yr, out2 = xq+ys
    This has rank 4 (schoolbook) since each output entry is a dot product of length 2.
    
    Hmm, but we can do better. Let's use:
    M1 = x*(p+r), M2 = y*(q+s)? No, that doesn't work.
    
    Actually for 1x2 * 2x2, the minimal rank is 4 (same as schoolbook for this size).
    So 4 blocks * 4 = 16 per row, total 32. That's worse than Strassen.
    
    Let me think differently. Use the 2x2 Strassen approach properly.
    For row i of A, we compute C_row = A_row * B.
    Split A_row into [aL, aR] (each 1x2), B into [[B11,B12],[B21,B22]].
    C_row = [aL*B11 + aR*B21, aL*B12 + aR*B22]
    
    Each aL*B11 is 1x2 * 2x2 = 1x2, rank 4.
    So total per row: 4 * 4 = 16. Two rows: 32. Still schoolbook.
    
    The key insight from the directive: share Strassen-style combinations across rows.
    Since both rows of A multiply the same B, we can share the linear combinations
    on B. 
    
    Let's try a different approach: use the fact that we're computing 2x4 * 4x4.
    Think of it as: for each column pair of B, compute contributions.
    
    Actually, let me just try the rank-28 Strassen-on-blocks approach and verify it works,
    then attempt to reduce further.
    """
    pass

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Start with schoolbook rank 32
    U_school, V_school, W_school = _schoolbook_decomp(n, m, p)
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = 32
    
    # Rank-28 decomposition: Strassen on 2x2 blocks
    # A is 2x4, B is 4x4, C is 2x4
    # A = [A11, A12] where A11, A12 are 2x2
    # B = [[B11, B12], [B21, B22]] where each block is 2x2
    # C = [C11, C12] where C11, C12 are 2x2
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    
    U_strassen = np.zeros((n*m, 28))
    V_strassen = np.zeros((m*p, 28))
    W_strassen = np.zeros((p*n, 28))
    
    # Get Strassen decomposition for 2x2*2x2
    U_s, V_s, W_s = _strassen_2x2_decomp()
    U_s = np.array(U_s)
    V_s = np.array(V_s)
    W_s = np.array(W_s)
    
    # C11 = A11*B11 + A12*B21
    # A11*B11: A indices 0-3, B indices 0-3, C indices 0-3
    for r in range(7):
        for ai in range(4):
            U_strassen[ai, r] += U_s[ai, r]
        for bi in range(4):
            V_strassen[bi, r] += V_s[bi, r]
        for ci in range(4):
            W_strassen[ci, r] += W_s[ci, r]
    
    # A12*B21: A indices 4-7, B indices 8-11, C indices 0-3
    for r in range(7):
        for ai in range(4):
            U_strassen[4+ai, 7+r] += U_s[ai, r]
        for bi in range(4):
            V_strassen[8+bi, 7+r] += V_s[bi, r]
        for ci in range(4):
            W_strassen[ci, 7+r] += W_s[ci, r]
    
    # C12 = A11*B12 + A12*B22
    # A11*B12: A indices 0-3, B indices 4-7, C indices 4-7
    for r in range(7):
        for ai in range(4):
            U_strassen[ai, 14+r] += U_s[ai, r]
        for bi in range(4):
            V_strassen[4+bi, 14+r] += V_s[bi, r]
        for ci in range(4):
            W_strassen[4+ci, 14+r] += W_s[ci, r]
    
    # A12*B22: A indices 4-7, B indices 12-15, C indices 4-7
    for r in range(7):
        for ai in range(4):
            U_strassen[4+ai, 21+r] += U_s[ai, r]
        for bi in range(4):
            V_strassen[12+bi, 21+r] += V_s[bi, r]
        for ci in range(4):
            W_strassen[4+ci, 21+r] += W_s[ci, r]
    
    # Verify rank-28
    if _verify_decomposition(U_strassen.tolist(), V_strassen.tolist(), W_strassen.tolist(), n, m, p):
        best_U = U_strassen.tolist()
        best_V = V_strassen.tolist()
        best_W = W_strassen.tolist()
        best_rank = 28
    else:
        pass
    
    # Check time
    if time.time() > deadline:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Now attempt rank-21: use a custom bilinear algorithm
    # Key idea: C = A * B where A is 2x4, B is 4x4, C is 2x4
    # C[0,:] = A[0,:] * B  (row 0 of A times B)
    # C[1,:] = A[1,:] * B  (row 1 of A times B)
    # 
    # For each row of A, we compute a 1x4 * 4x4 = 1x4 product.
    # The 1x4 * 4x4 tensor has rank 16 (schoolbook).
    # But we can do better by exploiting structure.
    #
    # Alternative: think of C = A*B as computing 8 output entries, each a dot product
    # of length 4. That's rank 32 schoolbook.
    #
    # Let's try a different decomposition. Split B into columns:
    # B = [B_col0, B_col1, B_col2, B_col3] each 4x1
    # C[:,j] = A * B_col_j, which is 2x4 * 4x1 = 2x1
    # Each 2x4 * 4x1 product has rank 8 (schoolbook: 2*4*1)
    # Total: 4 * 8 = 32. Same as before.
    #
    # Let's try yet another approach: use the fact that we can share computations.
    # Define linear combinations of B's columns that are shared between the two rows.
    #
    # Actually, the known lower bound for 2x4x4 is 24 (from Coppersmith-Winograd type bounds
    # or direct analysis). Let me try to construct rank 24.
    #
    # Approach: Use 3 groups of 8 multiplications.
    # Group 1: Compute certain linear combinations of A rows with B blocks
    # Group 2: Same for another set
    # Group 3: Same for another set
    #
    # This is getting complex. Let me try a randomized optimization approach instead.
    
    # Randomized search for lower rank
    # Start from rank 28 and try to project down
    np.random.seed(42)
    
    for target_rank in [27, 26, 25, 24]:
        if time.time() > deadline - 10:
            break
        
        # Try to find a rank-target_rank decomposition by optimization
        # Initialize with random projection of the rank-28 solution
        U_init = U_strassen[:, :target_rank] if target_rank <= 28 else U_strassen
        V_init = V_strassen[:, :target_rank] if target_rank <= 28 else V_strassen
        W_init = W_strassen[:, :target_rank] if target_rank <= 28 else W_strassen
        
        # Use gradient-free optimization: try random perturbations
        best_obj = float('inf')
        best_U_opt = U_init.copy()
        best_V_opt = V_init.copy()
        best_W_opt = W_init.copy()
        
        # Objective: sum of squared errors
        def _objective(Uo, Vo, Wo):
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(target_rank):
                            val += Uo[i*m+j, r] * Vo[j*p+k, r] * Wo[k*n+i, r]
                        err += (val - 1.0)**2
            return err
        
        # Simple local search with random restarts
        n_restarts = 3
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break
            
            if restart == 0:
                Uo = U_init.copy()
                Vo = V_init.copy()
                Wo = W_init.copy()
            else:
                # Random initialization
                Uo = np.random.randn(n*m, target_rank) * 0.1
                Vo = np.random.randn(m*p, target_rank) * 0.1
                Wo = np.random.randn(p*n, target_rank) * 0.1
                # Add some structure: initialize from schoolbook projected
                if target_rank <= n*m*p:
                    for r in range(target_rank):
                        idx = r
                        i = idx // (m*p)
                        rem = idx % (m*p)
                        j = rem // p
                        k = rem % p
                        Uo[i*m+j, r] = 1.0
                        Vo[j*p+k, r] = 1.0
                        Wo[k*n+i, r] = 1.0
            
            obj = _objective(Uo, Vo, Wo)
            if obj < best_obj:
                best_obj = obj
                best_U_opt = Uo.copy()
                best_V_opt = Vo.copy()
                best_W_opt = Wo.copy()
            
            if obj < 1e-12:
                break
            
            # Coordinate descent / local search
            n_steps = 50
            step_size = 0.1
            for step in range(n_steps):
                if time.time() > deadline - 5:
                    break
                if obj < 1e-12:
                    break
                
                # Pick a random element to perturb
                mat_choice = random.choice([0, 1, 2])
                if mat_choice == 0:
                    row = random.randint(0, n*m-1)
                    col = random.randint(0, target_rank-1)
                    old_val = Uo[row, col]
                    Uo[row, col] = old_val + random.gauss(0, step_size)
                elif mat_choice == 1:
                    row = random.randint(0, m*p-1)
                    col = random.randint(0, target_rank-1)
                    old_val = Vo[row, col]
                    Vo[row, col] = old_val + random.gauss(0, step_size)
                else:
                    row = random.randint(0, p*n-1)
                    col = random.randint(0, target_rank-1)
                    old_val = Wo[row, col]
                    Wo[row, col] = old_val + random.gauss(0, step_size)
                
                new_obj = _objective(Uo, Vo, Wo)
                if new_obj < obj:
                    obj = new_obj
                    if obj < best_obj:
                        best_obj = obj
                        best_U_opt = Uo.copy()
                        best_V_opt = Vo.copy()
                        best_W_opt = Wo.copy()
                else:
                    # Revert
                    if mat_choice == 0:
                        Uo[row, col] = old_val
                    elif mat_choice == 1:
                        Vo[row, col] = old_val
                    else:
                        Wo[row, col] = old_val
                
                # Decay step size
                step_size *= 0.95
            
            if best_obj < 1e-12:
                break
        
        if best_obj < 1e-9:
            # Found a valid decomposition!
            # Round to nearest half-integer or integer for cleanliness
            best_U_rounded = np.round(best_U_opt * 2) / 2
            best_V_rounded = np.round(best_V_opt * 2) / 2
            best_W_rounded = np.round(best_W_opt * 2) / 2
            
            if _verify_decomposition(best_U_rounded.tolist(), best_V_rounded.tolist(), best_W_rounded.tolist(), n, m, p):
                best_U = best_U_rounded.tolist()
                best_V = best_V_rounded.tolist()
                best_W = best_W_rounded.tolist()
                best_rank = target_rank
                break
            else:
                # Try without rounding
                if _verify_decomposition(best_U_opt.tolist(), best_V_opt.tolist(), best_W_opt.tolist(), n, m, p):
                    best_U = best_U_opt.tolist()
                    best_V = best_V_opt.tolist()
                    best_W = best_W_opt.tolist()
                    best_rank = target_rank
                    break
    
    return {"U": best_U, "V": best_V, "W": best_W}
