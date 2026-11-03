"""Matrix multiplication 2x3x8 tensor decomposition.
Strategy: Split m=3 into m1=1 and m2=2.
Part 1: 2x1x8 multiplication (Rank 16, schoolbook).
Part 2: 2x2x8 multiplication, split into 4 blocks of 2x2x2 (Strassen, Rank 7 each).
Total Rank: 16 + 4*7 = 44.
"""
import time

DIMENSIONS = [2, 3, 8]

def construct():
    """Return a rank-44 exact tensor decomposition for 2x3x8 matrix multiplication."""
    start_time = time.time()
    timeout = 140.0  # Safety margin before 150s limit

    n, m, p = DIMENSIONS  # 2, 3, 8
    
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
            
    if time.time() - start_time > timeout:
        return {"U": U, "V": V, "W": W}

    # --- Part 2: 2x2x8 (j=1,2) ---
    # A is 2x2 (cols 1,2), B is 2x8 (rows 1,2).
    # Split B into 4 blocks of 2 columns.
    for block_idx in range(4):
        k_start = block_idx * 2
        
        # Strassen 2x2x2
        # Local A indices:
        # A00 -> Global A[0, 1] -> U[0*3+1] = U[1]
        # A01 -> Global A[0, 2] -> U[0*3+2] = U[2]
        # A10 -> Global A[1, 1] -> U[1*3+1] = U[4]
        # A11 -> Global A[1, 2] -> U[1*3+2] = U[5]
        
        # Local B indices:
        # B00 -> Global B[1, k_start]     -> V[1*8+k_start] = V[8+k_start]
        # B01 -> Global B[1, k_start+1]   -> V[1*8+k_start+1] = V[9+k_start]
        # B10 -> Global B[2, k_start]     -> V[2*8+k_start] = V[16+k_start]
        # B11 -> Global B[2, k_start+1]   -> V[2*8+k_start+1] = V[17+k_start]
        
        # Local C indices:
        # C00 -> Global C[0, k_start]     -> W[k_start*2+0] = W[2*k_start]
        # C01 -> Global C[0, k_start+1]   -> W[(k_start+1)*2+0] = W[2*k_start+2]
        # C10 -> Global C[1, k_start]     -> W[k_start*2+1] = W[2*k_start+1]
        # C11 -> Global C[1, k_start+1]   -> W[(k_start+1)*2+1] = W[2*k_start+3]

        steps = []

        # M1 = (A00 + A11) * (B00 + B11)
        # C00 += M1, C11 += M1
        steps.append({
            'U': {1: 1, 5: 1},
            'V': {8+k_start: 1, 17+k_start: 1},
            'W': {2*k_start: 1, 2*k_start+3: 1}
        })

        # M2 = (A10 + A11) * B00
        # C10 += M2, C11 -= M2
        steps.append({
            'U': {4: 1, 5: 1},
            'V': {8+k_start: 1},
            'W': {2*k_start+1: 1, 2*k_start+3: -1}
        })

        # M3 = A00 * (B01 - B11)
        # C01 += M3, C11 += M3
        steps.append({
            'U': {1: 1},
            'V': {9+k_start: 1, 17+k_start: -1},
            'W': {2*k_start+2: 1, 2*k_start+3: 1}
        })

        # M4 = A11 * (B10 - B00)
        # C00 += M4, C10 += M4
        steps.append({
            'U': {5: 1},
            'V': {16+k_start: 1, 8+k_start: -1},
            'W': {2*k_start: 1, 2*k_start+1: 1}
        })

        # M5 = (A00 + A01) * B11
        # C00 -= M5, C01 += M5
        steps.append({
            'U': {1: 1, 2: 1},
            'V': {17+k_start: 1},
            'W': {2*k_start: -1, 2*k_start+2: 1}
        })

        # M6 = (A10 - A00) * (B00 + B01)
        # C11 += M6
        steps.append({
            'U': {4: 1, 1: -1},
            'V': {8+k_start: 1, 9+k_start: 1},
            'W': {2*k_start+3: 1}
        })

        # M7 = (A01 - A11) * (B10 + B11)
        # C00 += M7
        steps.append({
            'U': {2: 1, 5: -1},
            'V': {16+k_start: 1, 17+k_start: 1},
            'W': {2*k_start: 1}
        })

        for step in steps:
            for u_idx, val in step['U'].items():
                U[u_idx][col] += val
            for v_idx, val in step['V'].items():
                V[v_idx][col] += val
            for w_idx, val in step['W'].items():
                W[w_idx][col] += val
            col += 1

    if time.time() - start_time > timeout:
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
