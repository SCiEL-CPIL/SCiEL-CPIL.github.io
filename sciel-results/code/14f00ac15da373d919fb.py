"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2 coefficients
    # Local indices: 0:00, 1:01, 2:10, 3:11
    U_in = [
        [1, 0, 0, 1], # r=0
        [0, 0, 1, 1], # r=1
        [1, 0, 0, 0], # r=2
        [0, 0, 0, 1], # r=3
        [1, 1, 0, 0], # r=4
        [-1, 0, 1, 0], # r=5
        [0, 1, 0, -1]  # r=6
    ]
    
    V_in = [
        [1, 0, 0, 1], # r=0
        [1, 0, 0, 0], # r=1
        [0, 1, 0, -1], # r=2
        [-1, 0, 1, 0], # r=3
        [0, 0, 0, 1], # r=4
        [1, 1, 0, 0], # r=5
        [0, 0, 1, 1]  # r=6
    ]
    
    W_in = [
        [1, 0, 0, 1], # r=0: C00, C11
        [0, 1, 0, -1], # r=1: C10, C11 (Note: W index is k*n+i. C10 is k=0,i=1 -> idx 1. C11 is k=1,i=1 -> idx 3)
        [0, 0, 1, 1], # r=2: C01, C11 (C01 is k=1,i=0 -> idx 2. C11 is idx 3)
        [1, 1, 0, 0], # r=3: C00, C10 (C00 is idx 0. C10 is idx 1)
        [-1, 0, 1, 0], # r=4: C00, C01 (C00 is idx 0. C01 is idx 2)
        [0, 0, 0, 1], # r=5: C11 (idx 3)
        [1, 0, 0, 0]  # r=6: C00 (idx 0)
    ]
    
    # Outer Strassen coefficients for 4x4 (block level)
    # Block indices: 0:TL, 1:TR, 2:BL, 3:BR
    U_out = [
        [1, 0, 0, 1], # r=0
        [0, 0, 1, 1], # r=1
        [1, 0, 0, 0], # r=2
        [0, 0, 0, 1], # r=3
        [1, 1, 0, 0], # r=4
        [-1, 0, 1, 0], # r=5
        [0, 1, 0, -1]  # r=6
    ]
    
    V_out = [
        [1, 0, 0, 1], # r=0
        [1, 0, 0, 0], # r=1
        [0, 1, 0, -1], # r=2
        [-1, 0, 1, 0], # r=3
        [0, 0, 0, 1], # r=4
        [1, 1, 0, 0], # r=5
        [0, 0, 1, 1]  # r=6
    ]
    
    W_out = [
        [1, 0, 0, 1], # r=0: C0, C3
        [0, 0, 1, -1], # r=1: C2, C3
        [0, 1, 0, 1], # r=2: C1, C3
        [1, 0, 1, 0], # r=3: C0, C2
        [-1, 1, 0, 0], # r=4: C0, C1
        [0, 0, 0, 1], # r=5: C3
        [1, 0, 0, 0]  # r=6: C0
    ]
    
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    for ro in range(7):
        for ri in range(7):
            r = ro * 7 + ri
            for i in range(4):
                for j in range(4):
                    # Map global i,j to block I and local i',j'
                    I = (i // 2) * 2 + (j // 2)
                    ip = i % 2
                    jp = j % 2
                    local_idx = ip * 2 + jp
                    
                    U[i * 4 + j][r] = U_out[ro][I] * U_in[ri][local_idx]
                    
            for j in range(4):
                for k in range(4):
                    I = (j // 2) * 2 + (k // 2)
                    jp = j % 2
                    kp = k % 2
                    local_idx = jp * 2 + kp
                    
                    V[j * 4 + k][r] = V_out[ro][I] * V_in[ri][local_idx]
                    
            for k in range(4):
                for i in range(4):
                    I = (k // 2) * 2 + (i // 2)
                    kp = k % 2
                    ip = i % 2
                    local_idx = kp * 2 + ip
                    
                    W[k * 4 + i][r] = W_out[ro][I] * W_in[ri][local_idx]
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
