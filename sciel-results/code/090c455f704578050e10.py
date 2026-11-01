"""Matrix multiplication 2x4 x 4x4 tensor decomposition - rank 27."""

import numpy as np
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s
    
    # Build rank-27 decomposition directly
    # Strategy: Use Strassen for each 2x2 block (7 each = 14 total)
    # but share one rank dimension between the two blocks
    # 
    # A is 2x4: A = [A0 | A1] where A0, A1 are 2x2
    # B is 4x4: B = [[B00, B01], [B10, B11]] where each is 2x2
    # C = A*B = [A0*B00 + A1*B10 | A0*B01 + A1*B11]
    # C = [C0 | C1] where C0, C1 are 2x2
    #
    # For each 2x2 output block, use Strassen (7 multiplications)
    # But we can share: if we use the same "diagonal" factor for both blocks
    #
    # Actually, let me try a different approach:
    # The tensor for 2x4 x 4x4 has dimension 8*16*8 = 1024
    # We need rank 27 decomposition
    #
    # Let me try to construct it explicitly using the structure:
    # Think of it as: for each of the 4 columns of A (j=0,1,2,3)
    # and 4 rows of B (j=0,1,2,3), we have a 2x2 outer product
    # 
    # Alternative: use the fact that 2x4 x 4x4 can be decomposed as:
    # (2x2 x 2x2) + (2x2 x 2x2) with shared structure
    #
    # Let me try a direct construction with rank 27:
    # 
    # Use 13 factors for the "left half" (columns 0,1 of A with rows 0,1 of B)
    # and 14 factors for the "right half" (columns 2,3 of A with rows 2,3 of B)
    # No, that's 27 but doesn't work cleanly.
    #
    # Better: Let me use a systematic approach.
    # The matrix multiplication tensor M_{2,4,4} can be seen as:
    # sum over j of A_{:,j} * B_{j,:}
    # Each term is (2x1)*(1x4) = 2x4 matrix
    #
    # For rank 27, I'll try to find it by solving the system.
    # But first, let me try a known construction.
    
    # Actually, let me try to use the following approach:
    # View 2x4 x 4x4 as two copies of 2x2 x 2x2 (for the two column pairs of A)
    # Block 0: A[:, 0:2] * B[0:2, :] gives 2x4 (using B rows 0,1)
    # Block 1: A[:, 2:4] * B[2:4, :] gives 2x4 (using B rows 2,3)
    # 
    # Wait, that's not right. Let me be more careful.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # j ranges 0..3, k ranges 0..3
    #
    # Split j into j_low (0,1) and j_high (2,3):
    # C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k] + sum_{j=2}^{3} A[i,j]*B[j,k]
    #
    # First sum: A_low (2x2) * B[:, 0:2]... no, B[j,k] for j in {0,1} and all k
    # So it's A_low (2x2) times B_top (2x4) = 2x4
    # Second sum: A_high (2x2) times B_bottom (2x4) = 2x4
    #
    # So C = A_low * B_top + A_high * B_bottom
    # where A_low = A[:, 0:2], A_high = A[:, 2:4]
    #       B_top = B[0:2, :], B_bottom = B[2:4, :]
    #
    # Now, A_low * B_top is 2x2 * 2x4, which has rank at most... 
    # We can use Strassen-like for 2x2 * 2x4:
    # This is equivalent to computing two 2x2 products:
    # A_low * B_top[:, 0:2] and A_low * B_top[:, 2:4]
    # Each 2x2*2x2 uses 7 factors, so 14 total for A_low*B_top
    # Similarly 14 for A_high*B_bottom
    # Total: 28 (the Strassen approach)
    #
    # To get 27, we need to share one factor.
    # 
    # Idea: The "diagonal" Strassen factor (the one that multiplies (A00+A11)*(B00+B11))
    # might be shareable if the structure allows.
    #
    # Let me try a different decomposition. What if I use:
    # For the 2x2 * 2x4 product, use a rank-6 or rank-7 algorithm
    # and share between the two halves?
    #
    # Actually, 2x2 * 2x4: the tensor is M_{2,2,4}
    # This can be seen as two M_{2,2,2} tensors
    # Rank of M_{2,2,4} <= 2*7 = 14, but maybe less?
    # Actually rank(M_{2,2,4}) = rank(M_{2,4,2}) by symmetry considerations
    # 
    # Let me just try to find rank 27 by solving the linear system
    # with the constraint that entries are in {0, 0.5, 1} or similar.
    
    # Direct approach: try to construct rank-27
    rank = 27
    U = np.zeros((n*m, rank))  # 8 x 27
    V = np.zeros((m*p, rank))  # 16 x 27  
    W = np.zeros((p*n, rank))  # 8 x 27
    
    # Let me try a specific construction:
    # Use 14 factors for the first half (j=0,1) and 13 for the second (j=2,3)
    # where one factor is shared
    
    # Strassen for 2x2:
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a00(b01-b11)
    # m4 = a11(b10-b00)
    # m5 = (a00+a10)b11
    # m6 = (a10-a11)(b00+b10)
    # m7 = (a01-a00)(b11+b01)
    # c00 = m1+m4-m5-m7
    # c01 = m3+m7
    # c10 = m2+m6
    # c11 = m1-m2+m3+m6
    
    # For 2x2 * 2x4 (B has 4 columns):
    # We can do it as two separate 2x2*2x2 products
    # Or use a combined approach
    
    # Let me try: for A_low (2x2) * B_top (2x4):
    # B_top = [B00 B01] where B00, B01 are 2x2
    # Result = [A_low*B00, A_low*B01]
    # Use Strassen for each: 7+7=14 factors
    
    # For A_high (2x2) * B_bottom (2x4):
    # B_bottom = [B10 B11] where B10, B11 are 2x2
    # Result = [A_high*B10, A_high*B11]
    # Use Strassen for each: 7+7=14 factors
    
    # Total 28. To get 27, share one factor.
    
    # Let me try a different splitting. What if I split differently?
    # C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k]
    # 
    # Group: (j=0,2) and (j=1,3)
    # C[i,k] = sum_{j in {0,2}} A[i,j]B[j,k] + sum_{j in {1,3}} A[i,j]B[j,k]
    #
    # First sum: involves A[i,0], A[i,2] and B[0,k], B[2,k]
    # This is like a 2x2 product if we view it as:
    # [A[i,0], A[i,2]] * [[B[0,k]], [B[2,k]]]
    # But that's just a dot product, not helpful.
    
    # Let me try yet another approach: use the fact that we can write
    # the 2x4 x 4x4 multiplication as a sum of 27 rank-1 tensors.
    
    # I'll use a randomized approach with good initialization.
    
    # Initialize from rank-28 Strassen, then try to merge two factors.
    
    def build_strassen_28():
        rank = 28
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        col = 0
        for b in range(2):  # which 2x2 block of B columns
            for a_blk in range(2):  # which 2x2 block of A columns
                for s in range(7):
                    c = col
                    j0 = 2 * a_blk
                    j1 = 2 * a_blk + 1
                    k0 = 2 * b
                    k1 = 2 * b + 1
                    
                    # U side: A[i, j] for i in {0,1}, j in {j0, j1}
                    U_contrib = {
                        0: [(0, j0, 1), (1, j1, 1)],
                        1: [(1, j0, 1), (1, j1, 1)],
                        2: [(0, j0, 1)],
                        3: [(1, j1, 1)],
                        4: [(0, j0, 1), (0, j1, 1)],
                        5: [(1, j0, 1), (0, j0, -1)],
                        6: [(0, j1, 1), (1, j1, -1)],
                    }
                    for (i, j, coeff) in U_contrib[s]:
                        U[i * 4 + j, c] += coeff
                    
                    # V side: B[j, k] for j in {j0, j1}, k in {k0, k1}
                    V_contrib = {
                        0: [(j0, k0, 1), (j1, k1, 1)],
                        1: [(j0, k0, 1)],
                        2: [(j0, k1, 1), (j1, k1, -1)],
                        3: [(j1, k0, 1), (j0, k0, -1)],
                        4: [(j1, k1, 1)],
                        5: [(j0, k0, 1), (j0, k1, 1)],
                        6: [(j1, k0, 1), (j1, k1, 1)],
                    }
                    for (j, k, coeff) in V_contrib[s]:
                        V[j * 4 + k, c] += coeff
                    
                    # W side: C[i, k] for i in {0,1}, k in {k0, k1}
                    W_contrib = {
                        0: {(0, k0): 1, (1, k1): 1},
                        1: {(1, k0): 1, (1, k1): -1},
                        2: {(0, k1): 1, (1, k1): 1},
                        3: {(0, k0): 1, (1, k0): 1},
                        4: {(0, k0): -1, (0, k1): 1},
                        5: {(1, k1): 1},
                        6: {(0, k0): 1},
                    }
                    for (i_loc, k_loc), coeff in W_contrib[s].items():
                        W[k_loc * 2 + i_loc, c] += coeff
                    
                    col += 1
        
        return U, V, W
    
    U28, V28, W28 = build_strassen_28()
    
    # Now try to find rank 27 by dropping one column and solving
    # For each column to drop, set up the linear system
    
    best_rank = 28
    best_result = {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
    
    # Try each column to remove
    for remove_col in range(28):
        if time.time() > deadline - 5:
            break
        
        keep = [c for c in range(28) if c != remove_col]
        U0 = U28[:, keep].copy()
        V0 = V28[:, keep].copy()
        W0 = W28[:, keep].copy()
        
        # Perturb and solve
        for trial in range(3):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(remove_col * 100 + trial)
            
            U = U0.copy() + np.random.randn(n*m, rank) * 0.5
            V = V0.copy() + np.random.randn(m*p, rank) * 0.5
            W = W0.copy() + np.random.randn(p*n, rank) * 0.5
            
            # Alternating least squares
            for step in range(15):
                if time.time() > deadline - 5:
                    break
                
                # Update U
                for a in range(n*m):
                    i = a // m
                    j = a % m
                    M = np.zeros((m*p*p*n, rank))
                    rhs = np.zeros(m*p*p*n)
                    for b_idx in range(m*p):
                        j2 = b_idx // p
                        k1 = b_idx % p
                        for c_idx in range(p*n):
                            i2 = c_idx // n
                            k2 = c_idx % n
                            M[b_idx*p*n + c_idx, :] = V[b_idx, :] * W[c_idx, :]
                            if i == i2 and j == j2 and k1 == k2:
                                rhs[b_idx*p*n + c_idx] = 1.0
                    try:
                        U[a, :] = np.linalg.lstsq(M, rhs, rcond=None)[0]
                    except:
                        pass
                
                # Update V
                for b_idx in range(m*p):
                    j2 = b_idx // p
                    k1 = b_idx % p
                    M = np.zeros((n*m*p*n, rank))
                    rhs = np.zeros(n*m*p*n)
                    for a in range(n*m):
                        i1 = a // m
                        j1 = a % m
                        for c_idx in range(p*n):
                            i2 = c_idx // n
                            k2 = c_idx % n
                            M[a*p*n + c_idx, :] = U[a, :] * W[c_idx, :]
                            if i1 == i2 and j1 == j2 and k1 == k2:
                                rhs[a*p*n + c_idx] = 1.0
                    try:
                        V[b_idx, :] = np.linalg.lstsq(M, rhs, rcond=None)[0]
                    except:
                        pass
                
                # Update W
                for c_idx in range(p*n):
                    i2 = c_idx // n
                    k2 = c_idx % n
                    M = np.zeros((n*m*m*p, rank))
                    rhs = np.zeros(n*m*m*p)
                    for a in range(n*m):
                        i1 = a // m
                        j1 = a % m
                        for b_idx in range(m*p):
                            j2 = b_idx // p
                            k1 = b_idx % p
                            M[a*m*p + b_idx, :] = U[a, :] * V[b_idx, :]
                            if i1 == i2 and j1 == j2 and k1 == k2:
                                rhs[a*m*p + b_idx] = 1.0
                    try:
                        W[c_idx, :] = np.linalg.lstsq(M, rhs, rcond=None)[0]
                    except:
                        pass
            
            # Validate
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                        if abs(s - 1.0) > 1e-8:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                # Check off-diagonals
                for a in range(n*m):
                    i1 = a // m
                    j1 = a % m
                    for b_idx in range(m*p):
                        j2 = b_idx // p
                        k1 = b_idx % p
                        if i1 == j1 and j1 == j2:
                            continue
                        for c_idx in range(p*n):
                            i2 = c_idx // n
                            k2 = c_idx % n
                            if i1 == i2 and j1 == j2 and k1 == k2:
                                continue
                            s = np.dot(U[a], V[b_idx] * W[c_idx])
                            if abs(s) > 1e-8:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if valid:
                    best_rank = rank
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    break
        
        if best_rank == rank:
            break
    
    return best_result

# EVOLVE-BLOCK-END
