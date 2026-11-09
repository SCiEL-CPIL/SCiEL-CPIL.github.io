"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    n, m, p = 2, 4, 4
    
    # Rank-28 Strassen decomposition for [2,4,4]
    # Split j into 2 blocks of 2, k into 2 blocks of 2
    # For each (j_block, k_block) pair, compute 2x2*2x2 via Strassen (7 products)
    # Total: 4 * 7 = 28 products
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # (8, 28)
    V = [[0] * rank for _ in range(m * p)]  # (16, 28)
    W = [[0] * rank for _ in range(p * n)]  # (8, 28)
    
    # Strassen for 2x2 * 2x2:
    # A = [[a00, a01],[a10, a11]], B = [[b00, b01],[b10, b11]]
    # M1 = (a00+a11)(b00+b11) -> C00 += M1, C11 += M1
    # M2 = (a10+a11)(b00)     -> C10 += M2, C11 -= M2
    # M3 = (a00)(b01-b11)     -> C01 += M3, C11 += M3
    # M4 = (a11)(b10-b00)     -> C10 += M4, C00 += M4
    # M5 = (a00+a01)(b11)     -> C01 += M5, C00 -= M5
    # M6 = (a10-a00)(b00+b01) -> C00 += M6, C10 += M6
    # M7 = (a01-a11)(b10+b11) -> C01 += M7, C10 -= M7
    
    # U factors (A-side): [pi, pj] coefficient for each M
    u_coeffs = [
        [[1, 0], [0, 1]],   # M1: a00+a11
        [[0, 0], [1, 1]],   # M2: a10+a11
        [[1, 0], [0, 0]],   # M3: a00
        [[0, 0], [0, 1]],   # M4: a11
        [[1, 1], [0, 0]],   # M5: a00+a01
        [[-1, 0], [1, 0]],  # M6: a10-a00
        [[0, 1], [0, -1]],  # M7: a01-a11
    ]
    # V factors (B-side): [pj, qk] coefficient for each M
    v_coeffs = [
        [[1, 0], [0, 1]],   # M1: b00+b11
        [[1, 0], [0, 0]],   # M2: b00
        [[0, 1], [0, -1]],  # M3: b01-b11
        [[-1, 0], [1, 0]],  # M4: b10-b00
        [[0, 0], [0, 1]],   # M5: b11
        [[1, 1], [0, 0]],   # M6: b00+b01
        [[0, 0], [1, 1]],   # M7: b10+b11
    ]
    # W factors (C-side): [qk, pi] coefficient for each M
    w_coeffs = [
        [[1, 0], [0, 1]],   # M1: C00+=1, C11+=1
        [[0, 1], [0, -1]],  # M2: C10+=1, C11-=1
        [[0, 0], [1, 1]],   # M3: C01+=1, C11+=1
        [[1, 1], [0, 0]],   # M4: C10+=1, C00+=1
        [[-1, 1], [0, 0]],  # M5: C01+=1, C00-=1
        [[0, 0], [0, 1]],   # M6: C00+=1, C10+=1
        [[1, 0], [0, 0]],   # M7: C01+=1, C10-=1
    ]
    
    for s in range(4):
        j_block = s // 2
        k_block = s % 2
        for f in range(7):
            col = s * 7 + f
            for pi in range(2):
                for pj in range(2):
                    val = u_coeffs[f][pi][pj]
                    if val != 0:
                        j = 2 * j_block + pj
                        U[pi * m + j][col] += val
            for pj in range(2):
                for qk in range(2):
                    val = v_coeffs[f][pj][qk]
                    if val != 0:
                        j = 2 * j_block + pj
                        k = 2 * k_block + qk
                        V[j * p + k][col] += val
            for qk in range(2):
                for pi in range(2):
                    val = w_coeffs[f][qk][pi]
                    if val != 0:
                        k = 2 * k_block + qk
                        W[k * n + pi][col] += val
    
    # Verify correctness
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(total - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to schoolbook rank 32
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
