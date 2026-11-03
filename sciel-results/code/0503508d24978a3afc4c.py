import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 40 second margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # Strassen decomposition for 2x2x2
    # Local indices: 0=A00/B00/C00, 1=A01/B01/C01, 2=A10/B10/C10, 3=A11/B11/C11
    #
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # Reconstruction:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each term t, we store (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: which A elements contribute to the first factor
    # V_coeffs: which B elements contribute to the second factor
    # W_coeffs: which C elements this term contributes to (with sign)
    
    strassen_decomp = [
        # M1: (A00+A11)(B00+B11) -> C00=+1, C11=+1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2: (A10+A11)B00 -> C10=+1, C11=-1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3: A00(B01-B11) -> C01=+1, C11=+1
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4: A11(B10-B00) -> C00=+1, C10=+1
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5: (A00+A01)B11 -> C01=+1, C00=-1
        ({0: 1, 1: 1}, {3: 1}, {1: 1, 0: -1}),
        # M6: (A10-A00)(B00+B01) -> C11=+1
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7: (A01-A11)(B10+B11) -> C00=+1
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    for j1 in range(2):  # 2 blocks of 2 in m dimension (j = 2*j1, 2*j1+1)
        # Top part: 2x2x8 (rows i=0,1 of A)
        for l in range(4):  # 4 blocks of 2 in p dimension (k = 2*l, 2*l+1)
            for u_coeffs, v_coeffs, w_coeffs in strassen_decomp:
                # U (A): local (li_i, li_j) -> global (i, j)
                for li, coeff in u_coeffs.items():
                    li_i, li_j = li // 2, li % 2
                    global_i = li_i  # 0 or 1
                    global_j = 2 * j1 + li_j  # 2*j1 or 2*j1+1
                    U[global_i * m + global_j][col] += coeff
                
                # V (B): local (li_i, li_j) -> global (j, k)
                # B[j, k]: j is row index (maps to local row), k is col index (maps to local col)
                for li, coeff in v_coeffs.items():
                    li_i, li_j = li // 2, li % 2
                    global_j = 2 * j1 + li_i  # B row
                    global_k = 2 * l + li_j    # B col
                    V[global_j * p + global_k][col] += coeff
                
                # W (C): local (li_i, li_j) -> global (k, i)
                # C[k, i]: k is col of C (maps to local col), i is row of C (maps to local row)
                for li, coeff in w_coeffs.items():
                    li_i, li_j = li // 2, li % 2
                    global_i = li_i    # C row (output row)
                    global_k = 2 * l + li_j  # C col (output col)
                    W[global_k * n + global_i][col] += coeff
                
                col += 1
        
        # Bot part: 1x2x8 (row i=2 of A), schoolbook
        # C[2, k] = sum_j A[2,j] * B[j,k] for j in {2*j1, 2*j1+1}
        for k in range(8):
            for j2 in range(2):
                global_i = 2
                global_j = 2 * j1 + j2
                U[global_i * m + global_j][col] = 1
                V[global_j * p + k][col] = 1
                W[k * n + global_i][col] = 1
                col += 1
    
    # Verify the decomposition
    if time.time() < deadline - 5:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        verified = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Sum over all rank terms: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    s = np.dot(U_arr[i * m + j] * V_arr[j * p + k], W_arr[k * n + i])
                    if abs(s - 1.0) > 1e-9:
                        verified = False
                        break
                if not verified:
                    break
            if not verified:
                break
        
        if not verified:
            # Fallback: schoolbook rank 96
            rank_fb = n * m * p
            U_fb = [[0] * rank_fb for _ in range(n * m)]
            V_fb = [[0] * rank_fb for _ in range(m * p)]
            W_fb = [[0] * rank_fb for _ in range(p * n)]
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i * m + j][idx] = 1
                        V_fb[j * p + k][idx] = 1
                        W_fb[k * n + i][idx] = 1
                        idx += 1
            return {"U": U_fb, "V": V_fb, "W": W_fb}
        
        # Attempt linear algebra pruning
        # Construct matrix M where M[entry, term] = U[i,j,r] * V[j,k,r] * W[k,i,r]
        # entry index = i*m*p + j*p + k
        # term index = r
        num_entries = n * m * p
        M = np.zeros((num_entries, rank), dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    entry_idx = i * m * p + j * p + k
                    # Vector of products for all terms
                    prod = U_arr[i * m + j] * V_arr[j * p + k] * W_arr[k * n + i]
                    M[entry_idx, :] = prod
        
        # Check rank of M
        # If rank < 88, there is linear dependence
        # We try to remove terms that are linearly dependent
        # Use SVD to find singular values
        try:
            U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
            # s is sorted descending
            # If smallest s is close to 0, rank is deficient
            tol = 1e-9
            rank_M = np.sum(s > tol)
            
            if rank_M < rank:
                # Find null space vectors (right singular vectors corresponding to small s)
                # These vectors z satisfy M z = 0
                # We can use these to eliminate terms
                # For each null vector, find the term with largest coefficient and eliminate it
                # Then re-solve for the remaining terms?
                # This is complex. For now, just report if we found dependence.
                # If we found dependence, we might try to remove the term with max |z_r|
                # and set its coefficients to 0, then adjust others?
                # This requires re-solving the non-linear system, which is hard.
                # Given the time constraint, we might just stick to 88 if no easy prune found.
                pass
        except:
            pass

    return {"U": U, "V": V, "W": W}
