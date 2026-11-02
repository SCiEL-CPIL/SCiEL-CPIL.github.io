"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    def get_trivial():
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
    
    # Attempt to find a low-rank decomposition for 3x3x5 using a constructive approach.
    # We split the multiplication into 3x3x3 (first 3 cols of B) and 3x3x2 (last 2 cols of B).
    # Target: Rank 23 for 3x3x3 (Laderman) + Rank 17 for 3x3x2 = Total 40.
    # If we can't find 17 for 3x3x2, we fall back to 18 (Trivial), Total 41.
    # If we can't find Laderman, we fall back to 27 (Trivial for 3x3x3), Total 45.
    
    # --- Part 1: 3x3x3 Decomposition ---
    # We will use a known valid rank-23 decomposition for 3x3x3.
    # The standard Laderman decomposition is complex to hardcode correctly without verification.
    # Instead, we can try to construct a rank-23 decomposition by solving a linear system
    # or using a known sparse pattern.
    # However, a safer and faster approach for this specific task is to use the fact that
    # R(3,3,3) = 23 is proven. We can hardcode a verified set of coefficients.
    
    # Here is a verified set of coefficients for Laderman's 3x3x3 algorithm.
    # Source: Standard references on fast matrix multiplication.
    # Indices: A is 3x3 (flattened 0..8), B is 3x3 (flattened 0..8), C is 3x3 (flattened 0..8).
    # U[r, a], V[r, b], W[r, c] such that C = sum_r U_r * A * V_r * W_r (conceptually).
    # In tensor notation: T = sum_r u_r v_r w_r where u_r is a vector in A-space, etc.
    # The condition is: for all i,j,k: delta_{ik} * delta_{jj'} ... wait.
    # The tensor T_mult satisfies: <T, A \otimes B \otimes C> = tr(A B C^T)? No.
    # The bilinear map is M(A,B) = AB.
    # The tensor is defined such that (M(A,B))_{ik} = sum_j A_{ij} B_{jk}.
    # This can be written as: sum_{i,j,k} (AB)_{ik} A^*_{ij} B^*_{jk} = tr(AB (AB)^T)? No.
    # Let's stick to the index definition:
    # T_{i,j,k} = 1 if the term A_{ij} B_{jk} contributes to C_{ik}.
    # In the flattened vector space:
    # U has shape (9, r), V has shape (9, r), W has shape (9, r).
    # The decomposition is valid if for all i,j,k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1  (Note: W index is k*n+i for C_{ik})
    # Wait, the W factor corresponds to the output C.
    # C_{ik} = sum_j A_{ij} B_{jk}.
    # So the tensor component T_{i,j,k} is 1.
    # The factorization is T = sum_r u_r \otimes v_r \otimes w_r.
    # u_r is a vector in R^9 (A-space), v_r in R^9 (B-space), w_r in R^9 (C-space).
    # T_{i,j,k} = sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i].
    # We require this sum to be 1 for all i,j,k.
    
    # Let's use a known valid rank-23 decomposition.
    # I will hardcode the Laderman coefficients.
    
    def get_laderman_3x3():
        rank = 23
        U = np.zeros((9, rank))
        V = np.zeros((9, rank))
        W = np.zeros((9, rank))
        
        # Laderman's 23 terms
        # A11=0, A12=1, A13=2
        # A21=3, A22=4, A23=5
        # A31=6, A32=7, A33=8
        
        # B11=0, B12=1, B13=2
        # B21=3, B22=4, B23=5
        # B31=6, B32=7, B33=8
        
        # C11=0, C12=1, C13=2
        # C21=3, C22=4, C23=5
        # C31=6, C32=7, C33=8
        
        # Term 1: A11*B11 -> C11
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        # Term 2: A12*B21 -> C11
        U[1, 1] = 1; V[3, 1] = 1; W[0, 1] = 1
        # Term 3: A13*B31 -> C11
        U[2, 2] = 1; V[6, 2] = 1; W[0, 2] = 1
        # Term 4: A21*B12 -> C12
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
        # Term 5: A22*B22 -> C12
        U[4, 4] = 1; V[4, 4] = 1; W[1, 4] = 1
        # Term 6: A23*B32 -> C12
        U[5, 5] = 1; V[7, 5] = 1; W[1, 5] = 1
        # Term 7: A31*B13 -> C13
        U[6, 6] = 1; V[2, 6] = 1; W[2, 6] = 1
        # Term 8: A32*B23 -> C13
        U[7, 7] = 1; V[5, 7] = 1; W[2, 7] = 1
        # Term 9: A33*B33 -> C13
        U[8, 8] = 1; V[8, 8] = 1; W[2, 8] = 1
        # Term 10: A11*B22 -> C12
        U[0, 9] = 1; V[4, 9] = 1; W[1, 9] = 1
        # Term 11: A12*B11 -> C11
        U[1, 10] = 1; V[0, 10] = 1; W[0, 10] = 1
        # Term 12: A13*B12 -> C12
        U[2, 11] = 1; V[1, 11] = 1; W[1, 11] = 1
        # Term 13: A21*B11 -> C11
        U[3, 12] = 1; V[0, 12] = 1; W[0, 12] = 1
        # Term 14: A22*B12 -> C12
        U[4, 13] = 1; V[1, 13] = 1; W[1, 13] = 1
        # Term 15: A23*B13 -> C13
        U[5, 14] = 1; V[2, 14] = 1; W[2, 14] = 1
        # Term 16: A31*B11 -> C11
        U[6, 15] = 1; V[0, 15] = 1; W[0, 15] = 1
        # Term 17: A32*B12 -> C12
        U[7, 16] = 1; V[1, 16] = 1; W[1, 16] = 1
        # Term 18: A33*B13 -> C13
        U[8, 17] = 1; V[2, 17] = 1; W[2, 17] = 1
        # Term 19: A11*B33 -> C13
        U[0, 18] = 1; V[8, 18] = 1; W[2, 18] = 1
        # Term 20: A12*B32 -> C12
        U[1, 19] = 1; V[7, 19] = 1; W[1, 19] = 1
        # Term 21: A13*B31 -> C11
        U[2, 20] = 1; V[6, 20] = 1; W[0, 20] = 1
        # Term 22: A21*B33 -> C13
        U[3, 21] = 1; V[8, 21] = 1; W[2, 21] = 1
        # Term 23: A22*B32 -> C12
        U[4, 22] = 1; V[7, 22] = 1; W[1, 22] = 1
        
        # Verify this specific set.
        # It looks like a subset of trivial terms, which is suspicious for rank 23.
        # Let's check the sum for C11 (index 0 in W):
        # Terms contributing to W[0, r]: r=0, 1, 2, 10, 12, 15, 20.
        # Sum = U[0,0]V[0,0]W[0,0] + U[1,1]V[3,1]W[0,1] + U[2,2]V[6,2]W[0,2] +
        #       U[1,10]V[0,10]W[0,10] + U[3,12]V[0,12]W[0,12] + U[6,15]V[0,15]W[0,15] + U[2,20]V[6,20]W[0,20]
        # = A11*B11 + A12*B21 + A13*B31 + A12*B11 + A21*B11 + A31*B11 + A13*B31
        # This is NOT equal to C11 = A11*B11 + A12*B21 + A13*B31.
        # There are extra terms and duplicates. This hardcoded set is WRONG.
        
        # I will abandon the hardcoded Laderman and use a safer approach:
        # Use a rank-27 decomposition for 3x3x3 (which is easy to construct: 3x trivial 3x3x3? No, 27 is 3*3*3).
        # Actually, trivial is 27.
        # So 3x3x3 trivial is 27.
        # 3x3x2 trivial is 18.
        # Total 45. No improvement.
        
        # Let's try to find a rank-20 decomposition for 3x3x3? No, min is 19, best known 23.
        # If I can't get 23 right, I should stick to 45 or find a different block split.
        
        # Alternative: Use Strassen for 2x2x2 blocks?
        # 3x3 can be padded to 4x4? No, that increases rank.
        
        # Let's try to construct a valid rank-23 decomposition using a solver.
        # We have 9*9*9 = 729 equations.
        # We have 3 * 23 * 9 = 621 variables.
        # Underdetermined. We can set some variables to 0 to make it solvable.
        # This is a non-linear problem (products of variables).
        # Too slow for 150s.
        
        return None, None, None # Signal failure
    
    U_l, V_l, W_l = get_laderman_3x3()
    
    if U_l is not None:
        # Verify Laderman
        valid = True
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    s = 0
                    for r in range(23):
                        s += U_l[i*3+j, r] * V_l[j*3+k, r] * W_l[k*3+i, r]
                    if abs(s - 1) > 1e-9:
                        valid = False
                        break
                if not valid: break
            if not valid: break
        
        if valid:
            # Use Laderman for first 3 cols of B (3x3x3)
            # Use Trivial for last 2 cols of B (3x3x2)
            rank_total = 23 + 18
            U = np.zeros((9, rank_total))
            V = np.zeros((15, rank_total)) # 3*5 = 15
            W = np.zeros((15, rank_total)) # 5*3 = 15
            
            # Copy Laderman part
            U[:, :23] = U_l
            V[:9, :23] = V_l # B rows 0..8 (first 3 cols)
            W[:9, :23] = W_l # C rows 0..8 (first 3 cols)
            
            # Trivial part for last 2 cols
            col_offset = 23
            for i in range(3):
                for j in range(3):
                    for k in range(2):
                        # B index: j*5 + (3+k)
                        # C index: (3+k)*3 + i
                        U[i*3+j, col_offset] = 1
                        V[j*5 + 3 + k, col_offset] = 1
                        W[(3+k)*3 + i, col_offset] = 1
                        col_offset += 1
            
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: Trivial
    return get_trivial()
# EVOLVE-BLOCK-END
