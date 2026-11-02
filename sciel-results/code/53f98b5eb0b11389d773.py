"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Define the rank-7 Strassen decomposition for 2x2x2
    # For 2x2x2: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # U_2[i*2+j, r]: coefficient of A[i,j] in term r
    # V_2[j*2+k, r]: coefficient of B[j,k] in term r  
    # W_2[k*2+i, r]: coefficient of C[k,i] in term r (note: C = AB, so C[k,i] = sum_j A[i,j]*B[j,k])
    
    U_2 = np.array([
        [1, 0, 1, 0, 1, 0, 0],  # A00
        [0, 0, 0, 0, 0, 1, 1],  # A01
        [1, 1, 0, 0, 0, 0, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1], # A11
    ], dtype=float)
    
    V_2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],  # B00
        [0, 0, 1, 0, 1, 1, 1],   # B01
        [0, 0, 0, 1, 0, 0, 1],   # B10
        [1, 0, -1, 0, 1, 0, 1],  # B11
    ], dtype=float)
    
    # W_2[k*2+i, r]: For C[k,i] = sum_j A[i,j]*B[j,k]
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    W_2 = np.array([
        [1, 0, 0, 0, 0, 0, 0],   # C00: k=0, i=0
        [0, 0, 1, 0, 0, 0, 0],   # C01: k=0, i=1
        [0, 1, 0, 1, 0, 0, 0],   # C10: k=1, i=0
        [0, 0, 0, 0, 1, 0, 0],   # C11: k=1, i=1
    ], dtype=float)
    
    # Verify 2x2x2 Strassen
    def verify_2x2(U, V, W):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = sum(U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r] for r in range(U.shape[1]))
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True
    
    strassen_ok = verify_2x2(U_2, V_2, W_2)
    
    if not strassen_ok:
        # Fallback to schoolbook
        schoolbook_rank = n * m * p
        U_school = np.zeros((n * m, schoolbook_rank))
        V_school = np.zeros((m * p, schoolbook_rank))
        W_school = np.zeros((p * n, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j, col] = 1.0
                    V_school[j * p + k, col] = 1.0
                    W_school[k * n + i, col] = 1.0
                    col += 1
        
        def to_list_with_ints(arr):
            result = []
            for i in range(arr.shape[0]):
                row = []
                for j in range(arr.shape[1]):
                    val = arr[i,j]
                    if abs(val - round(val)) < 1e-8:
                        row.append(int(round(val)))
                    elif abs(val - (round(val*2)/2)) < 1e-8:
                        row.append(round(val*2)/2)
                    else:
                        row.append(float(val))
                result.append(row)
            return result
        
        return {"U": to_list_with_ints(U_school), "V": to_list_with_ints(V_school), "W": to_list_with_ints(W_school)}
    
    # Build 4x4x4 decomposition using Strassen
    # 4x4 matrix as 2x2 blocks of 2x2 matrices
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # Using Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each Mi is a 2x2 matrix product, decomposed with rank 7
    # Total rank = 7 * 7 = 49
    
    # For the 4x4x4 part (k=0,1,2,3), we need rank 49
    # For k=4 (5th column), we need additional terms
    
    # Let's construct the full 4x4x5 decomposition
    # U has shape (16, R), V has shape (20, R), W has shape (20, R)
    
    # For k=0,1,2,3: use the 49-term Strassen decomposition
    # For k=4: we need to handle the 5th column separately
    
    # First, build the 49-term decomposition for 4x4x4
    U_44_4 = np.zeros((16, 49))
    V_44_4 = np.zeros((20, 49))
    W_44_4 = np.zeros((20, 49))
    
    # For each of the 7 Strassen terms (a = 0..6), and each of the 7 inner terms (b = 0..6)
    # term index r = a * 7 + b
    
    # Strassen coefficients for the outer 2x2 block structure
    # For term a, we need:
    # - Which A-blocks contribute and with what coefficients
    # - Which B-blocks contribute and with what coefficients
    # - Which C-blocks the result contributes to and with what coefficients
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # A-block contributions: for each Strassen term a, which (i_block, j_block) pairs
    # i_block, j_block in {0,1}
    A_block_coeffs = [
        # a=0: M1 = (A00+A11)(...)
        [(0,0,1), (1,1,1)],
        # a=1: M2 = (A10+A11)B00
        [(1,0,1), (1,1,1)],
        # a=2: M3 = A00(...)
        [(0,0,1)],
        # a=3: M4 = A11(...)
        [(1,1,1)],
        # a=4: M5 = (A00+A01)(...)
        [(0,0,1), (0,1,1)],
        # a=5: M6 = (A10-A00)(...)
        [(1,0,1), (0,0,-1)],
        # a=6: M7 = (A01-A11)(...)
        [(0,1,1), (1,1,-1)],
    ]
    
    # B-block contributions: for each Strassen term a, which (j_block, k_block) pairs
    B_block_coeffs = [
        # a=0: M1 = (...)(B00+B11)
        [(0,0,1), (1,1,1)],
        # a=1: M2 = (...)B00
        [(0,0,1)],
        # a=2: M3 = (...)(B01-B11)
        [(0,1,1), (1,1,-1)],
        # a=3: M4 = (...)(B10-B00)
        [(1,0,1), (0,0,-1)],
        # a=4: M5 = (...)B11
        [(1,1,1)],
        # a=5: M6 = (...)(B00+B01)
        [(0,0,1), (0,1,1)],
        # a=6: M7 = (...)(B10+B11)
        [(1,0,1), (1,1,1)],
    ]
    
    # C-block contributions: for each Strassen term a, which (k_block, i_block) pairs
    # C = AB, so C[k_block, i_block]
    C_block_coeffs = [
        # a=0: M1 contributes to C00 (+1) and C11 (+1)
        [(0,0,1), (1,1,1)],
        # a=1: M2 contributes to C10 (+1) and C11 (-1)
        [(1,0,1), (1,1,-1)],
        # a=2: M3 contributes to C01 (+1) and C11 (+1)
        [(0,1,1), (1,1,1)],
        # a=3: M4 contributes to C00 (+1) and C10 (+1)
        [(0,0,1), (1,0,1)],
        # a=4: M5 contributes to C00 (-1) and C01 (+1)
        [(0,0,-1), (0,1,1)],
        # a=5: M6 contributes to C11 (+1)
        [(1,1,1)],
        # a=6: M7 contributes to C00 (+1) and C11 (+1)
        [(0,0,1), (1,1,1)],
    ]
    
    for a in range(7):
        for b in range(7):
            r = a * 7 + b
            
            # For U: index is i*4 + j, where i is row of A, j is col of A
            # A is 4x4, viewed as 2x2 blocks of 2x2
            # A[i,j] where i = i_block*2 + i_local, j = j_block*2 + j_local
            for i_block, j_block, coeff_a in A_block_coeffs[a]:
                # For each local position in the 2x2 block
                for i_local in range(2):
                    for j_local in range(2):
                        i = i_block * 2 + i_local
                        j = j_block * 2 + j_local
                        # U_2[i_local*2 + j_local, b] gives the coefficient for A[i_local,j_local] in inner term b
                        U_44_4[i * 4 + j, r] = coeff_a * U_2[i_local * 2 + j_local, b]
            
            # For V: index is j*5 + k, where j is row of B, k is col of B
            # B is 4x5, but for the 4x4x4 part we only use k=0,1,2,3
            # B[j,k] where j = j_block*2 + j_local, k = k_block*2 + k_local
            for j_block, k_block, coeff_b in B_block_coeffs[a]:
                for j_local in range(2):
                    for k_local in range(2):
                        j = j_block * 2 + j_local
                        k = k_block * 2 + k_local
                        # V_2[j_local*2 + k_local, b] gives the coefficient for B[j_local,k_local] in inner term b
                        V_44_4[j * 5 + k, r] = coeff_b * V_2[j_local * 2 + k_local, b]
            
            # For W: index is k*4 + i, where k is row of C, i is col of C
            # C is 4x4, C[k,i] where k = k_block*2 + k_local, i = i_block*2 + i_local
            for k_block, i_block, coeff_c in C_block_coeffs[a]:
                for k_local in range(2):
                    for i_local in range(2):
                        k = k_block * 2 + k_local
                        i = i_block * 2 + i_local
                        # W_2[k_local*2 + i_local, b] gives the coefficient for C[k_local,i_local] in inner term b
                        W_44_4[k * 4 + i, r] = coeff_c * W_2[k_local * 2 + i_local, b]
    
    # Verify 4x4x4 decomposition
    def verify_44_4(U, V, W):
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    s = sum(U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r] for r in range(U.shape[1]))
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True
    
    # Note: V_44_4 has shape (20, 49) but we only filled the first 16 rows (j*5+k for k=0..3)
    # For verification, we need to adjust
    V_44_4_verify = V_44_4.copy()
    # Actually, for the 4x4x4 part, V should have shape (16, 49) for j=0..3, k=0..3
    # But our V_44_4 has shape (20, 49) with j*5+k indexing
    # Let's create a temporary V for verification
    V_temp = np.zeros((16, 49))
    for j in range(4):
        for k in range(4):
            V_temp[j*4+k, :] = V_44_4[j*5+k, :]
    
    verify_ok = verify_44_4(U_44_4, V_temp, W_44_4)
    
    if not verify_ok:
        # Fallback to schoolbook
        schoolbook_rank = n * m * p
        U_school = np.zeros((n * m, schoolbook_rank))
        V_school = np.zeros((m * p, schoolbook_rank))
        W_school = np.zeros((p * n, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j, col] = 1.0
                    V_school[j * p + k, col] = 1.0
                    W_school[k * n + i, col] = 1.0
                    col += 1
        
        def to_list_with_ints(arr):
            result = []
            for i in range(arr.shape[0]):
                row = []
                for j in range(arr.shape[1]):
                    val = arr[i,j]
                    if abs(val - round(val)) < 1e-8:
                        row.append(int(round(val)))
                    elif abs(val - (round(val*2)/2)) < 1e-8:
                        row.append(round(val*2)/2)
                    else:
                        row.append(float(val))
                result.append(row)
            return result
        
        return {"U": to_list_with_ints(U_school), "V": to_list_with_ints(V_school), "W": to_list_with_ints(W_school)}
    
    # Now add the 5th column (k=4)
    # For k=4, we need T[i,j,4] = 1 for all i,j
    # This means sum_r U[i*4+j, r] * V[j*5+4, r] * W[4*4+i, r] = 1
    
    # We can use a simple rank-16 decomposition for this part:
    # For each (i,j) pair, add a term with U[i*4+j, r] = 1, V[j*5+4, r] = 1, W[4*4+i, r] = 1
    # But this gives rank 16, total rank 49 + 16 = 65
    
    # Actually, we can be smarter. Let's use the fact that we already have some structure.
    # For now, let's just add 16 more terms for the 5th column.
    
    total_rank = 49 + 16
    U_full = np.zeros((16, total_rank))
    V_full = np.zeros((20, total_rank))
    W_full = np.zeros((20, total_rank))
    
    # Copy the 4x4x4 part
    U_full[:, :49] = U_44_4
    V_full[:, :49] = V_44_4
    W_full[:, :49] = W_44_4
    
    # Add terms for k=4
    for idx, (i, j) in enumerate([(i, j) for i in range(4) for j in range(4)]):
        r = 49 + idx
        U_full[i * 4 + j, r] = 1.0
        V_full[j * 5 + 4, r] = 1.0
        W_full[4 * 4 + i, r] = 1.0
    
    # Verify the full 4x4x5 decomposition
    def verify_full(U, V, W):
        for i in range(4):
            for j in range(4):
                for k in range(5):
                    s = sum(U[i*4+j, r] * V[j*5+k, r] * W[k*4+i, r] for r in range(U.shape[1]))
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True
    
    full_ok = verify_full(U_full, V_full, W_full)
    
    def to_list_with_ints(arr):
        result = []
        for i in range(arr.shape[0]):
            row = []
            for j in range(arr.shape[1]):
                val = arr[i,j]
                if abs(val - round(val)) < 1e-8:
                    row.append(int(round(val)))
                elif abs(val - (round(val*2)/2)) < 1e-8:
                    row.append(round(val*2)/2)
                else:
                    row.append(float(val))
            result.append(row)
        return result
    
    if full_ok:
        return {"U": to_list_with_ints(U_full), "V": to_list_with_ints(V_full), "W": to_list_with_ints(W_full)}
    else:
        # Fallback to schoolbook
        schoolbook_rank = n * m * p
        U_school = np.zeros((n * m, schoolbook_rank))
        V_school = np.zeros((m * p, schoolbook_rank))
        W_school = np.zeros((p * n, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j, col] = 1.0
                    V_school[j * p + k, col] = 1.0
                    W_school[k * n + i, col] = 1.0
                    col += 1
        
        return {"U": to_list_with_ints(U_school), "V": to_list_with_ints(V_school), "W": to_list_with_ints(W_school)}
# EVOLVE-BLOCK-END
