"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    timeout = 140.0  # Safety margin

    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # We use a block decomposition:
    # Split m=3 into m1=1 and m2=2.
    # Part 1: A[:, 0] * B[0, :] -> 2x1x8 multiplication (Rank 16)
    # Part 2: A[:, 1:3] * B[1:3, :] -> 2x2x8 multiplication
    #         Split p=8 into 4 blocks of 2.
    #         Each block is 2x2x2 multiplication (Rank 7 via Strassen).
    #         Total Rank for Part 2: 4 * 7 = 28.
    # Total Rank: 16 + 28 = 44.

    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank for _ in range(p * n)]  # 16 x 44

    col = 0

    # --- Part 1: 2x1x8 (j=0) ---
    # A is 2x1, B is 1x8.
    # C[i, k] = A[i, 0] * B[0, k]
    # Rank 16.
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    # Check time
    if time.time() - start_time > timeout:
        return {"U": U, "V": V, "W": W}

    # --- Part 2: 2x2x8 (j=1,2) ---
    # A is 2x2 (cols 1,2), B is 2x8 (rows 1,2).
    # Split B into 4 blocks of 2 columns.
    for block_idx in range(4):
        k_start = block_idx * 2
        
        # Strassen 2x2x2
        # Local A:
        # A00 -> Global A[0, 1]
        # A01 -> Global A[0, 2]
        # A10 -> Global A[1, 1]
        # A11 -> Global A[1, 2]
        
        # Local B (for this block):
        # B00 -> Global B[1, k_start]
        # B01 -> Global B[1, k_start + 1]
        # B10 -> Global B[2, k_start]
        # B11 -> Global B[2, k_start + 1]
        
        # Local C (for this block):
        # C00 -> Global C[0, k_start]
        # C01 -> Global C[0, k_start + 1]
        # C10 -> Global C[1, k_start]
        # C11 -> Global C[1, k_start + 1]

        # Define Strassen steps:
        # Each step is (U_coeffs, V_coeffs, W_coeffs)
        # U_coeffs: dict {global_u_idx: coeff}
        # V_coeffs: dict {global_v_idx: coeff}
        # W_coeffs: dict {global_w_idx: coeff}

        steps = []

        # M1 = (A00 + A11) * (B00 + B11)
        # C00 += M1, C11 += M1
        steps.append({
            'U': {0*m + 1: 1, 1*m + 2: 1},
            'V': {1*p + k_start: 1, 2*p + (k_start+1): 1},
            'W': {k_start*n + 0: 1, (k_start+1)*n + 1: 1}
        })

        # M2 = (A10 + A11) * B00
        # C10 += M2, C11 -= M2
        steps.append({
            'U': {1*m + 1: 1, 1*m + 2: 1},
            'V': {1*p + k_start: 1},
            'W': {k_start*n + 1: 1, (k_start+1)*n + 1: -1}
        })

        # M3 = A00 * (B01 - B11)
        # C01 += M3, C11 += M3
        steps.append({
            'U': {0*m + 1: 1},
            'V': {1*p + (k_start+1): 1, 2*p + (k_start+1): -1},
            'W': {(k_start+1)*n + 0: 1, (k_start+1)*n + 1: 1}
        })

        # M4 = A11 * (B10 - B00)
        # C00 += M4, C10 += M4
        steps.append({
            'U': {1*m + 2: 1},
            'V': {2*p + k_start: 1, 1*p + k_start: -1},
            'W': {k_start*n + 0: 1, k_start*n + 1: 1}
        })

        # M5 = (A00 + A01) * B11
        # C00 -= M5, C01 += M5
        steps.append({
            'U': {0*m + 1: 1, 0*m + 2: 1},
            'V': {2*p + (k_start+1): 1},
            'W': {k_start*n + 0: -1, (k_start+1)*n + 0: 1}
        })

        # M6 = (A10 - A00) * (B00 + B01)
        # C11 += M6
        steps.append({
            'U': {1*m + 1: 1, 0*m + 1: -1},
            'V': {1*p + k_start: 1, 1*p + (k_start+1): 1},
            'W': {(k_start+1)*n + 1: 1}
        })

        # M7 = (A01 - A11) * (B10 + B11)
        # C00 += M7
        steps.append({
            'U': {0*m + 2: 1, 1*m + 2: -1},
            'V': {2*p + k_start: 1, 2*p + (k_start+1): 1},
            'W': {k_start*n + 0: 1}
        })

        for step in steps:
            for u_idx, val in step['U'].items():
                U[u_idx][col] += val
            for v_idx, val in step['V'].items():
                V[v_idx][col] += val
            for w_idx, val in step['W'].items():
                W[w_idx][col] += val
            col += 1

    # Check time
    if time.time() - start_time > timeout:
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
