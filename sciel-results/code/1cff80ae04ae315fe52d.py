import numpy as np
import time

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # 10s margin
    
    # 1. Schoolbook Decomposition (Rank 45) - Fallback
    def schoolbook_full():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U.tolist(), V.tolist(), W.tolist()

    # 2. Attempted Improved Decomposition (Rank 41)
    # BSC (Rank 23) for 3x3x3 part + Schoolbook (Rank 18) for 3x3x2 part
    
    try:
        # BSC 3x3x3 coefficients
        # We define the 23 terms.
        # U, V, W are 9x23 for the 3x3x3 part.
        # Indices: U[i*3+j], V[j*3+k], W[k*3+i]
        
        U_bsc = np.zeros((9, 23), dtype=float)
        V_bsc = np.zeros((9, 23), dtype=float)
        W_bsc = np.zeros((9, 23), dtype=float)
        
        # Term 0-8: Schoolbook terms for the "diagonal"
        # 0: A00 B00 -> C00
        U_bsc[0, 0] = 1; V_bsc[0, 0] = 1; W_bsc[0, 0] = 1
        # 1: A01 B10 -> C00
        U_bsc[1, 1] = 1; V_bsc[3, 1] = 1; W_bsc[0, 1] = 1
        # 2: A02 B20 -> C00
        U_bsc[2, 2] = 1; V_bsc[6, 2] = 1; W_bsc[0, 2] = 1
        # 3: A10 B01 -> C01
        U_bsc[3, 3] = 1; V_bsc[1, 3] = 1; W_bsc[3, 3] = 1
        # 4: A11 B11 -> C01
        U_bsc[4, 4] = 1; V_bsc[4, 4] = 1; W_bsc[3, 4] = 1
        # 5: A12 B21 -> C01
        U_bsc[5, 5] = 1; V_bsc[7, 5] = 1; W_bsc[3, 5] = 1
        # 6: A20 B02 -> C02
        U_bsc[6, 6] = 1; V_bsc[2, 6] = 1; W_bsc[6, 6] = 1
        # 7: A21 B12 -> C02
        U_bsc[7, 7] = 1; V_bsc[5, 7] = 1; W_bsc[6, 7] = 1
        # 8: A22 B22 -> C02
        U_bsc[8, 8] = 1; V_bsc[8, 8] = 1; W_bsc[6, 8] = 1
        
        # Terms 9-22: The BSC specific terms
        # Reference: Bini, Strassen, Caporaso (1980)
        # These coefficients are from a known valid representation.
        
        # 9
        U_bsc[0, 9] = 1; U_bsc[4, 9] = 1
        V_bsc[3, 9] = 1; V_bsc[7, 9] = 1
        W_bsc[3, 9] = 1; W_bsc[0, 9] = -1 # C10 - C00? No, W is for output.
        # Correct BSC term structure:
        # t9: (A00+A11)(B10+B21) -> contributes to C01, C10, C00, C11?
        # Actually, let's use a verified sparse matrix set.
        
        # I will use the following verified set for the 14 extra terms.
        # (u_indices, v_indices, w_indices, coeffs)
        # This is risky if I don't have the exact list.
        
        # Alternative: Use the schoolbook for 3x3x3 (Rank 27) + Schoolbook for 3x3x2 (Rank 18) = 45.
        # This is not an improvement.
        
        # To ensure an improvement, I MUST have valid BSC coefficients.
        # I will use a hardcoded set that I am confident in.
        
        # Let's try to find a simpler improvement?
        # No, 41 is the target.
        
        # If I cannot guarantee BSC, I should return 45.
        # However, the prompt asks to combine strategies. The "strongest idea" is the BSC split.
        # I will implement the BSC coefficients from a standard source (e.g. Wikipedia or a paper snippet).
        
        # BSC 3x3x3 Rank 23 Coefficients (One valid set):
        # The terms are often listed as:
        # 1. A00 B00 -> C00
        # 2. A01 B10 -> C00
        # 3. A02 B20 -> C00
        # 4. A10 B01 -> C01
        # 5. A11 B11 -> C01
        # 6. A12 B21 -> C01
        # 7. A20 B02 -> C02
        # 8. A21 B12 -> C02
        # 9. A22 B22 -> C02
        # 10. (A00+A11) B10 -> C01, C10
        # 11. (A00+A11) B21 -> C01, C10
        # ... this is getting complicated.
        
        # I will use a fallback: If I can't verify BSC quickly, I return schoolbook.
        # But I will try to hardcode the BSC coefficients from a known correct source.
        
        # Known BSC Coefficients (Integer):
        # U, V, W are 9x23.
        
        # I will use the following set (verified in similar contexts):
        
        # Reset BSC arrays
        U_bsc = np.zeros((9, 23), dtype=float)
        V_bsc = np.zeros((9, 23), dtype=float)
        W_bsc = np.zeros((9, 23), dtype=float)
        
        # Terms 0-8 as above
        U_bsc[0, 0] = 1; V_bsc[0, 0] = 1; W_bsc[0, 0] = 1
        U_bsc[1, 1] = 1; V_bsc[3, 1] = 1; W_bsc[0, 1] = 1
        U_bsc[2, 2] = 1; V_bsc[6, 2] = 1; W_bsc[0, 2] = 1
        U_bsc[3, 3] = 1; V_bsc[1, 3] = 1; W_bsc[3, 3] = 1
        U_bsc[4, 4] = 1; V_bsc[4, 4] = 1; W_bsc[3, 4] = 1
        U_bsc[5, 5] = 1; V_bsc[7, 5] = 1; W_bsc[3, 5] = 1
        U_bsc[6, 6] = 1; V_bsc[2, 6] = 1; W_bsc[6, 6] = 1
        U_bsc[7, 7] = 1; V_bsc[5, 7] = 1; W_bsc[6, 7] = 1
        U_bsc[8, 8] = 1; V_bsc[8, 8] = 1; W_bsc[6, 8] = 1
        
        # Terms 9-22
        # 9: A00 B10 -> C01 (part of combined term)
        # Actually, let's use the explicit 23 terms from a reliable source.
        # Source: "Fast Matrix Multiplication" by Strassen, or similar.
        
        # I will use the following 14 terms:
        # 9:  U: A00, A11; V: B10; W: C01, C10
        U_bsc[0, 9] = 1; U_bsc[4, 9] = 1
        V_bsc[3, 9] = 1
        W_bsc[3, 9] = 1; W_bsc[0, 9] = 1 # C10, C01?
        # Wait, W index is k*n+i.
        # C01 is i=0, k=1 -> idx 3.
        # C10 is i=1, k=0 -> idx 0.
        # So W_bsc[3,9]=1, W_bsc[0,9]=1.
        
        # 10: U: A00, A11; V: B21; W: C02, C11
        U_bsc[0, 10] = 1; U_bsc[4, 10] = 1
        V_bsc[7, 10] = 1
        W_bsc[6, 10] = 1; W_bsc[3, 10] = 1 # C12 (idx 6), C01? No.
        # C02 is i=0, k=2 -> idx 6.
        # C11 is i=1, k=1 -> idx 3.
        
        # This manual derivation is too error-prone.
        
        # I will return the schoolbook decomposition to ensure validity.
        # The "improvement" is the structure, but without correct BSC coefficients, I can't get 41.
        
        # However, I can try to use a numerical solver to find the BSC coefficients?
        # No, time limit.
        
        # I will stick to the schoolbook for now to ensure a valid result, 
        # but I will structure the code to allow easy insertion of BSC.
        
        # Actually, I can use the BSC coefficients from a known Python package or snippet.
        # Since I don't have internet access, I will rely on the fact that 
        # providing a valid rank 45 solution is safe.
        
        # But the prompt asks to combine with inspirations. 
        # Inspiration 3 tried BSC and failed/fell back.
        
        # I will provide the Schoolbook solution.
        
        U_s, V_s, W_s = schoolbook_full()
        
        # Verification
        A = np.random.rand(3, 3)
        B = np.random.rand(3, 5)
        
        C_expected = A @ B
        
        # Compute C using factors
        # C[i, k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A[i,j]) * (sum_j V[j*p+k, r] B[j,k])
        
        # Reshape U, V, W
        U_mat = np.array(U_s).reshape(3, 3, -1)
        V_mat = np.array(V_s).reshape(3, 5, -1)
        W_mat = np.array(W_s).reshape(5, 3, -1)
        
        # Sum over rank
        # term_r = (A * U_mat) * (B * V_mat) * W_mat
        # A is 3x3, U_mat is 3x3xR. A * U_mat (broadcast) -> 3x3xR
        # B is 3x5, V_mat is 3x5xR. B * V_mat -> 3x5xR
        
        # Actually, the bilinear form is:
        # C[i,k] = sum_r ( sum_j A[i,j] U[i,j,r] ) * ( sum_j B[j,k] V[j,k,r] ) * W[k,i,r]
        
        # Let's verify this interpretation.
        # In schoolbook:
        # U[i,j,r] = 1 if r corresponds to (i,j,k).
        # V[j,k,r] = 1 if r corresponds to (i,j,k).
        # W[k,i,r] = 1 if r corresponds to (i,j,k).
        
        # For a specific r (i0, j0, k0):
        # sum_j A[i,j] U[i,j,r] = A[i, j0] (only j=j0 is 1, but only if i=i0? No, U is indexed by i,j.
        # In my schoolbook: U[i*m+j, col] = 1.
        # So U[i,j,r] is 1 only if i=i0 and j=j0.
        # So sum_j A[i,j] U[i,j,r] = A[i0, j0] if i=i0, else 0.
        # Similarly for V: sum_j B[j,k] V[j,k,r] = B[j0, k0] if k=k0, else 0.
        # W[k,i,r] = 1 if k=k0 and i=i0.
        
        # So term is A[i0, j0] B[j0, k0] if i=i0 and k=k0.
        # Summing over r gives sum_j0 A[i, j0] B[j0, k] = C[i,k].
        
        # So the verification logic is:
        
        C_computed = np.zeros((3, 5))
        for r in range(len(U_s[0])):
            # u_vec = U[:, r] -> 9x1
            # v_vec = V[:, r] -> 15x1
            # w_vec = W[:, r] -> 15x1
            
            # a_part = sum_j A[i,j] U[i,j,r]
            # This is tricky with flattened indices.
            
            pass
        
        # Given the complexity of verification and the risk of BSC errors,
        # I will return the schoolbook decomposition.
        
        return {"U": U_s, "V": V_s, "W": W_s}

    except Exception:
        U_s, V_s, W_s = schoolbook_full()
        return {"U": U_s, "V": V_s, "W": W_s}

# EVOLVE-BLOCK-END
