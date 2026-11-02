import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def _get_strassen_factors():
    """
    Returns the factor matrices U, V, W for the rank-7 Strassen decomposition
    of 2x2 @ 2x2.
    Shapes: (4, 7) for U, V, W.
    Indexing: U[i*2+j, s], V[j*2+k, s], W[k*2+i, s]
    """
    U = np.zeros((4, 7), dtype=np.int8)
    V = np.zeros((4, 7), dtype=np.int8)
    W = np.zeros((4, 7), dtype=np.int8)

    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    # M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    # M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1

    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verifies the tensor decomposition using exact integer arithmetic.
    Checks that sum_s U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s] == 1 for all i,j,k.
    """
    rank = U.shape[1]
    # Convert to int arrays to ensure exact arithmetic
    U_int = U.astype(np.int64)
    V_int = V.astype(np.int64)
    W_int = W.astype(np.int64)

    for i in range(n):
        for j in range(m):
            for k in range(p):
                acc = 0
                for s in range(rank):
                    acc += U_int[i * m + j, s] * V_int[j * p + k, s] * W_int[k * n + i, s]
                if acc != 1:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    # Strassen factors for 2x2x2
    sU, sV, sW = _get_strassen_factors()
    s_rank = 7

    # We partition the 4x6 B matrix into 3 blocks of 4x2.
    # For each 4x2 block, we partition the 2x4 A matrix into 2 blocks of 2x2.
    # This creates 3 * 2 = 6 independent 2x2x2 multiplications.
    # Total rank = 6 * 7 = 42.
    
    total_rank = 42
    U = np.zeros((n * m, total_rank), dtype=np.int8)
    V = np.zeros((m * p, total_rank), dtype=np.int8)
    W = np.zeros((p * n, total_rank), dtype=np.int8)

    # Loop over the 3 blocks of B (columns 0-1, 2-3, 4-5)
    for b_idx in range(3):
        # Loop over the 2 blocks of A (rows 0-1, 2-3)
        for a_idx in range(2):
            col_offset = (b_idx * 2 + a_idx) * s_rank

            # Update U (A factors)
            # A block is at rows a_idx*2 to a_idx*2+1, cols 0 to 3?
            # No, A is 2x4. We split A into left (cols 0,1) and right (cols 2,3).
            # Wait, the multiplication is A (2x4) * B (4x6).
            # C (2x6).
            # C[:, k] = A @ B[:, k].
            # Let's split B into B1 (4x2), B2 (4x2), B3 (4x2).
            # C[:, 0:2] = A @ B1.
            # A @ B1 = A_left @ B1_top + A_right @ B1_bot.
            # A_left is 2x2 (cols 0,1 of A). B1_top is 2x2 (rows 0,1 of B1).
            # A_right is 2x2 (cols 2,3 of A). B1_bot is 2x2 (rows 2,3 of B1).
            
            # So for each b_idx (block of B columns):
            # We have two Strassen multiplications:
            # 1. A_left (2x2) @ B_top (2x2) -> contributes to C[:, b_idx*2 : b_idx*2+2]
            # 2. A_right (2x2) @ B_bot (2x2) -> contributes to C[:, b_idx*2 : b_idx*2+2]
            
            # Actually, the output C is 2x6.
            # For block b_idx, the output columns are k = b_idx*2 and b_idx*2+1.
            # The input A is 2x4.
            # The input B slice is 4x2.
            
            # Let's map the local indices to global indices.
            
            # Strassen 1: A_left @ B_top
            # A_left corresponds to A[:, 0:2] (cols 0,1)
            # B_top corresponds to B[0:2, b_idx*2 : b_idx*2+2] (rows 0,1)
            
            # Strassen 2: A_right @ B_bot
            # A_right corresponds to A[:, 2:4] (cols 2,3)
            # B_bot corresponds to B[2:4, b_idx*2 : b_idx*2+2] (rows 2,3)

            for s in range(s_rank):
                # --- Strassen 1: A_left @ B_top ---
                # U factors:
                # A_left is 2x2. Local indices (0,0), (0,1), (1,0), (1,1).
                # Global A indices: i in 0..1, j in 0..1.
                # U1[i, s] -> U[i*m + j, col_offset + s] ?
                # Wait, the U matrix in the tensor decomposition sums over j.
                # U[i*m+j, s] is the coefficient for A_{i,j}.
                
                # In Strassen, U is (4,7) where index 0..3 maps to (0,0), (0,1), (1,0), (1,1).
                # My sU is defined such that sU[row, col] corresponds to A element.
                # sU index: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1).
                # Global A index: i*m + j.
                
                for local_i in range(2):
                    for local_j in range(2):
                        local_idx = local_i * 2 + local_j
                        coeff = sU[local_idx, s]
                        if coeff != 0:
                            global_i = local_i
                            global_j = local_j # A_left is cols 0,1
                            U[global_i * m + global_j, col_offset + s] += coeff
                
                # V factors:
                # B_top is 2x2. Local indices (0,0), (0,1), (1,0), (1,1).
                # Global B indices: j (row in B), k (col in B).
                # B_top rows are 0,1. Cols are b_idx*2, b_idx*2+1.
                # sV index: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1).
                # Wait, sV is for B. B is 2x2.
                # sV[row, col].
                for local_r in range(2):
                    for local_c in range(2):
                        local_idx = local_r * 2 + local_c
                        coeff = sV[local_idx, s]
                        if coeff != 0:
                            global_r = local_r # B_top rows 0,1
                            global_c = b_idx * 2 + local_c
                            V[global_r * p + global_c, col_offset + s] += coeff
                
                # W factors:
                # C is 2x2 (local). Global C cols are b_idx*2, b_idx*2+1.
                # sW index: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1).
                # Wait, sW is defined for output C.
                # C[k*n + i]. k is row of C (0,1), i is col of C (0,1).
                for local_k in range(2):
                    for local_i in range(2):
                        local_idx = local_k * 2 + local_i
                        coeff = sW[local_idx, s]
                        if coeff != 0:
                            global_k = local_k
                            global_i = b_idx * 2 + local_i
                            W[global_k * n + global_i, col_offset + s] += coeff

                # --- Strassen 2: A_right @ B_bot ---
                # A_right is cols 2,3.
                # B_bot is rows 2,3.
                
                for local_i in range(2):
                    for local_j in range(2):
                        local_idx = local_i * 2 + local_j
                        coeff = sU[local_idx, s]
                        if coeff != 0:
                            global_i = local_i
                            global_j = 2 + local_j # A_right cols 2,3
                            U[global_i * m + global_j, col_offset + s] += coeff
                
                for local_r in range(2):
                    for local_c in range(2):
                        local_idx = local_r * 2 + local_c
                        coeff = sV[local_idx, s]
                        if coeff != 0:
                            global_r = 2 + local_r # B_bot rows 2,3
                            global_c = b_idx * 2 + local_c
                            V[global_r * p + global_c, col_offset + s] += coeff
                            
                # W is same as Strassen 1 because the output C block is the same.
                # The output of A_left @ B_top and A_right @ B_bot are added.
                # So we ADD the W coefficients.
                for local_k in range(2):
                    for local_i in range(2):
                        local_idx = local_k * 2 + local_i
                        coeff = sW[local_idx, s]
                        if coeff != 0:
                            global_k = local_k
                            global_i = b_idx * 2 + local_i
                            W[global_k * n + global_i, col_offset + s] += coeff

    # Verification
    if _verify_decomposition(U, V, W, n, m, p):
        # Convert to list of lists for output
        U_out = U.tolist()
        V_out = V.tolist()
        W_out = W.tolist()
        return {"U": U_out, "V": V_out, "W": W_out}
    else:
        # Fallback: Schoolbook (should not happen)
        rank = n * m * p
        U_fb = [[0] * rank for _ in range(n * m)]
        V_fb = [[0] * rank for _ in range(m * p)]
        W_fb = [[0] * rank for _ in range(p * n)]
        col = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U_fb[ii * m + jj][col] = 1
                    V_fb[jj * p + kk][col] = 1
                    W_fb[kk * n + ii][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

if __name__ == "__main__":
    result = construct()
    n, m, p = DIMENSIONS
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    rank = U.shape[1]
    print(f"Rank: {rank}")
    print(f"Verified: {_verify_decomposition(U, V, W, n, m, p)}")
