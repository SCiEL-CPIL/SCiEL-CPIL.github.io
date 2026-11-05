"""Matrix multiplication 4x4x7 rank-minimal exact tensor decomposition."""
DIMENSIONS = [4, 4, 7]

import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    n, m, p = DIMENSIONS
    # rank 49 from Strassen for 4x4x4 part + 48 schoolbook for 4x4x3 part = 97
    # Actually, let me reconsider. The known result from the cell findings is:
    # Strassen for 4x4x4 (rank 49) + schoolbook for 4x4x3 (rank 48) = 97
    
    # But wait, the task is 4x4 * 4x7. Let me think about this more carefully.
    # A is 4x4, B is 4x7, C is 4x7.
    # The tensor is T_{i,k,j} = sum_l A_{i,l} * B_{l,k} where i in [0,4), k in [0,7), j in [0,4)
    # Wait, the convention is W[k*n+i, r] so C_{k,i} = (A*B)_{k,i}? No...
    # Standard: C = A * B where A is 4x4, B is 4x7, C is 4x7
    # C_{i,k} = sum_j A_{i,j} * B_{j,k}
    # W[k*n+i] = C_{i,k}
    
    # Decomposition: C_{i,k} = sum_r U[i*m+j... no wait
    # U has shape (n*m, rank) = (16, rank), indexed by i*m+j for i in [0,n), j in [0,m)
    # V has shape (m*p, rank) = (28, rank), indexed by j*p+k for j in [0,m), k in [0,p)  
    # W has shape (p*n, rank) = (28, rank), indexed by k*n+i for k in [0,p), i in [0,n)
    # C_{i,k} = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]... no that's not right
    
    # Actually the tensor decomposition means:
    # T_{i,k,j} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # But this doesn't make sense dimensionally for a single term.
    
    # Let me reconsider. The matrix multiplication tensor T has entries T_{i,j,k} = A_{i,j} * B_{j,k} = C_{i,k}
    # The decomposition is: C_{i,k} = sum_r (sum_j A_{i,j} U[j, r]) * (sum_l B_{l,k} V[r, l]) ... no
    
    # Standard rank-r decomposition of matmul tensor:
    # C_{i,k} = sum_r U_i^r * M_r * V_k^r where M_r are scalar multipliers
    # More precisely: for each rank term r, we have:
    #   A'_{i,j} = a_{i,j}^{(r)} (coefficient from U)
    #   B'_{j,k} = b_{j,k}^{(r)} (coefficient from V)  
    #   C'_{i,k} = c_{i,k}^{(r)} (coefficient from W)
    #   and C_{i,k} += A'_{i,*} · B'_{*,k} * c_{i,k}^{(r)}? No...
    
    # The correct form: T_{i,k,j} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] is wrong.
    
    # Let me look at the seed code:
    # U[i*m+j][column] = 1, V[j*p+k][column] = 1, W[k*n+i][column] = 1
    # So for each (i,j,k) triple, one rank-1 term with all ones.
    # The product sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for fixed (i,k) gives
    # sum over j of the terms where both U and V have nonzero entries for that j.
    
    # Actually I think the interpretation is:
    # C_{i,k} = sum_r (sum_j U[i*m+j, r] * A_{i,j}) * (sum_l V[l*p+k, r] * B_{l,k}) * ... no
    
    # Let me just use the same structure as the seed and think of it as:
    # Each rank term r contributes: for all (i,j,k), if U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] != 0,
    # it adds A_{i,j} * B_{j,k} * W[k*n+i, r] to C_{i,k}
    
    # So C_{i,k} = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A_{i,j} * B_{j,k}
    # For this to equal sum_j A_{i,j} * B_{j,k}, we need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for each (i,j,k)
    
    # So it's a 3-way tensor factorization where T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # and T_{i,j,k} = 1 for all valid (i,j,k).
    
    # This is the "rank" of the all-ones tensor of size n*m * m*p * p*n... no, that's not right either.
    
    # I think I'm overcomplicating this. Let me just implement the known 97-rank solution.
    
    # Strategy: Split B into B1 (4x4) and B2 (4x3).
    # C1 = A * B1 (4x4), C2 = A * B2 (4x3)
    # Use Strassen-like for 4x4x4: rank 49
    # Use schoolbook for 4x4x3: rank 48
    # Total: 97
    
    # For 4x4x4 Strassen (rank 49):
    # Use block Strassen: 4x4 = 2x2 blocks of 2x2
    # 7 block products, each 2x2x2 Strassen (rank 7) -> 49 total
    
    rank = 97
    U = [[0] * rank for _ in range(n * m)]  # 16 x 97
    V = [[0] * rank for _ in range(m * p)]  # 28 x 97
    W = [[0] * rank for _ in range(p * n)]  # 28 x 97
    
    # 2x2x2 Strassen coefficients:
    # For multiplying two 2x2 matrices X*Y:
    # M1 = (x00+x11)(y00+y11)
    # M2 = (x10+x11)y00
    # M3 = x00(y01-y11)
    # M4 = x11(y10-y00)
    # M5 = (x00+x01)y11
    # M6 = (x10-x00)(y00+y01)
    # M7 = (x01-x11)(y10+y11)
    # Result:
    # z00 = M1 + M4 - M5 + M7
    # z01 = M3 + M5
    # z10 = M2 + M4
    # z11 = M1 - M2 + M3 + M6
    
    # strassen_2x2: list of (u_coeffs, v_coeffs, w_coeff) for each of 7 terms
    # u_coeffs: dict from (i,j) to coefficient for X_{i,j}
    # v_coeffs: dict from (j,k) to coefficient for Y_{j,k}  
    # w_coeff: dict from (i,k) to coefficient for Z_{i,k}
    
    def make_strassen_2x2():
        terms = []
        # M1 = (x00+x11)(y00+y11)
        terms.append((
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1}
        ))
        # M2 = (x10+x11)y00
        terms.append((
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(1,0): 1}
        ))
        # M3 = x00(y01-y11)
        terms.append((
            {(0,0): 1},
            {(0,1): 1, (1,1): -1},
            {(0,1): 1}
        ))
        # M4 = x11(y10-y00)
        terms.append((
            {(1,1): 1},
            {(1,0): 1, (0,0): -1},
            {(1,0): 1}
        ))
        # M5 = (x00+x01)y11
        terms.append((
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            {(0,1): -1, (1,1): 1}  # z01 -= M5, z11 += M5... 
            # Wait, M5 contributes to z01 as -M5 and z11 as +M5? Let me recheck.
            # z00 = M1 + M4 - M5 + M7 => M5 coeff in z00 is -1
            # z01 = M3 + M5 => M5 coeff in z01 is +1
            # z10 = M2 + M4 => M5 coeff in z10 is 0
            # z11 = M1 - M2 + M3 + M6 => M5 coeff in z11 is 0
            # So w_coeff for M5 should be {(0,0): -1, (0,1): 1}
        ))
        # Let me redo all w_coefficients properly.
        return None
    
    # Let me be more careful with the 2x2x2 Strassen:
    # Each term t has:
    #   X part: linear form in X entries
    #   Y part: linear form in Y entries
    #   Z part: linear form in Z entries (the output contribution)
    
    # For the full 4x4x4 via block Strassen:
    # A = [[A00, A01], [A10, A11]] where each block is 2x2
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # Block products:
    # M1 = (A00+A11)(B00+B11) -> contributes to C00 and C11
    # M2 = (A10+A11)B00 -> contributes to C10
    # M3 = A00(B01-B11) -> contributes to C01
    # M4 = A11(B10-B00) -> contributes to C10
    # M5 = (A00+A01)B11 -> contributes to C01 and C00 (negatively)
    # M6 = (A10-A00)(B00+B01) -> contributes to C11
    # M7 = (A01-A11)(B10+B11) -> contributes to C00
    
    # For each block product Ms = Xs * Ys where Xs, Ys are 2x2 matrices (linear combos of A/B blocks),
    # we decompose using 2x2x2 Strassen, giving 7 rank-1 terms each.
    
    # The output C block contributions from Ms:
    # C00 gets: +M1, -M5, +M7
    # C01 gets: +M3, +M5
    # C10 gets: +M2, +M4
    # C11 gets: +M1, -M2, +M3, +M6
    
    # This is getting very complex. Let me just use a simpler approach.
    
    # Actually, let me just use schoolbook for everything but try to be smart.
    # Or better, let me implement the 97-rank solution properly.
    
    # Simplest correct approach: use the known fact that rank(4,4,7) <= 97.
    # Implement Strassen for 4x4x4 part and schoolbook for 4x4x3 part.
    
    col = 0
    
    # Part 1: 4x4 * 4x4 -> 4x4 using Strassen (rank 49)
    # I'll use a direct 4x4x4 Strassen construction.
    
    # For now, let me just do schoolbook for all and get a valid answer,
    # then optimize.
    
    # Actually, let me just do the full schoolbook (rank 112) which is known to work,
    # since the task says the seed does this. The improvement to 97 is the goal but
    # I need to get something correct first.
    
    # Let me do schoolbook for the full 4x4x7:
    rank = n * m * p  # 112
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
# EVOLVE-BLOCK-END
