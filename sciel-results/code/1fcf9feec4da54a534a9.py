"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 140.0
    n, m, p = 2, 4, 4
    
    # Rank-28 decomposition using 4 independent Strassen 2x2x2 blocks
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # (8, 28)
    V = [[0] * rank for _ in range(m * p)]  # (16, 28)
    W = [[0] * rank for _ in range(p * n)]  # (8, 28)
    
    # Strassen 2x2x2 coefficients
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)(b00)
    # M3 = (a00)(b01-b11)
    # M4 = (a11)(b10-b00)
    # M5 = (a00+a01)(b11)
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # U factors: coefficient for A[i,j] in each M
    u_strassen = [
        [1, 0, 0, 0],  # M1: a00+a11 -> a00:1, a01:0, a10:0, a11:1
        [0, 0, 1, 1],  # M2: a10+a11
        [1, 0, 0, 0],  # M3: a00
        [0, 0, 0, 1],  # M4: a11
        [1, 1, 0, 0],  # M5: a00+a01
        [-1, 0, 1, 0], # M6: a10-a00
        [0, 1, 0, -1], # M7: a01-a11
    ]
    
    # V factors: coefficient for B[j,k] in each M
    v_strassen = [
        [1, 0, 0, 1],  # M1: b00+b11
        [1, 0, 0, 0],  # M2: b00
        [0, 1, 0, -1], # M3: b01-b11
        [-1, 0, 1, 0], # M4: b10-b00
        [0, 0, 0, 1],  # M5: b11
        [1, 1, 0, 0],  # M6: b00+b01
        [0, 0, 1, 1],  # M7: b10+b11
    ]
    
    # W factors: coefficient for C[i,k] in each M
    # M1: C00+=1, C11+=1
    # M2: C10+=1, C11-=1
    # M3: C01+=1, C11+=1
    # M4: C10+=1, C00+=1
    # M5: C01+=1, C00-=1
    # M6: C00+=1, C10+=1
    # M7: C01+=1, C10-=1
    w_strassen = [
        [1, 0, 0, 1],  # M1: C00:1, C01:0, C10:0, C11:1
        [0, 0, 1, -1], # M2: C00:0, C01:0, C10:1, C11:-1
        [0, 1, 0, 1],  # M3: C00:0, C01:1, C10:0, C11:1
        [1, 0, 1, 0],  # M4: C00:1, C01:0, C10:1, C11:0
        [-1, 1, 0, 0], # M5: C00:-1, C01:1, C10:0, C11:0
        [1, 0, 1, 0],  # M6: C00:1, C01:0, C10:1, C11:0
        [0, 1, -1, 0], # M7: C00:0, C01:1, C10:-1, C11:0
    ]
    
    # For each of the 4 blocks (j_block, k_block):
    for jb in range(2):
        for kb in range(2):
            base = (jb * 2 + kb) * 7
            for f in range(7):
                col = base + f
                # U: A[i, 2*jb + pi]
                for i in range(2):
                    for pi in range(2):
                        val = u_strassen[f][i * 2 + pi]
                        if val != 0:
                            j = 2 * jb + pi
                            U[i * m + j][col] += val
                # V: B[2*jb + pi, 2*kb + qk]
                for pi in range(2):
                    for qk in range(2):
                        val = v_strassen[f][pi * 2 + qk]
                        if val != 0:
                            j = 2 * jb + pi
                            k = 2 * kb + qk
                            V[j * p + k][col] += val
                # W: C[2*kb + qk, i]
                for qk in range(2):
                    for i in range(2):
                        val = w_strassen[f][qk * 2 + i]
                        if val != 0:
                            k = 2 * kb + qk
                            W[k * n + i][col] += val
    
    # Verify
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank))
                if abs(total - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to schoolbook rank 32
        rank = 32
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
