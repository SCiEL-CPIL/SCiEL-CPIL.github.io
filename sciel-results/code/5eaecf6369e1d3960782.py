import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strassen for 2x2 (rank 7)
    # U2, V2, W2 shapes: (4, 7)
    U2 = [
        [1, 0, 1, 0, 1, -1, 0], # i=0, j=0
        [0, 0, 0, 0, 1, 0, 1],  # i=0, j=1
        [0, 1, 0, 0, 0, 1, -1], # i=1, j=0
        [1, 1, 0, 1, 0, 0, 0]   # i=1, j=1
    ]
    V2 = [
        [1, 1, 0, 0, 0, 1, 0], # j=0, k=0
        [0, 0, 1, 0, 1, 0, 1], # j=0, k=1
        [1, 0, -1, 1, 0, 0, 0],# j=1, k=0
        [1, 0, 0, -1, 1, 0, 0] # j=1, k=1
    ]
    W2 = [
        [1, 0, 0, 1, -1, 0, 1], # k=0, i=0
        [0, 0, 1, 0, 1, 0, 0],  # k=0, i=1
        [0, 1, 0, 0, 0, 1, 0],  # k=1, i=0
        [1, -1, 1, 0, 0, 1, 0]  # k=1, i=1
    ]
    
    # Schoolbook for 3x3 (rank 27)
    # U3, V3, W3 shapes: (9, 27)
    U3 = [[0]*27 for _ in range(9)]
    V3 = [[0]*27 for _ in range(9)]
    W3 = [[0]*27 for _ in range(9)]
    col = 0
    for i3 in range(3):
        for j3 in range(3):
            for k3 in range(3):
                U3[i3*3 + j3][col] = 1
                V3[j3*3 + k3][col] = 1
                W3[k3*3 + i3][col] = 1
                col += 1

    rank = 7 * 27
    U = [[0] * rank for _ in range(36)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]
    
    # Fill U, V, W using Kronecker product structure
    for i2 in range(2):
        for j2 in range(2):
            for k2 in range(2):
                for l2 in range(7):
                    u2_val = U2[i2*2 + j2][l2]
                    v2_val = V2[j2*2 + k2][l2]
                    w2_val = W2[k2*2 + i2][l2]
                    
                    if u2_val == 0 and v2_val == 0 and w2_val == 0:
                        continue

                    for i3 in range(3):
                        for j3 in range(3):
                            for k3 in range(3):
                                for l3 in range(27):
                                    u3_val = U3[i3*3 + j3][l3]
                                    v3_val = V3[j3*3 + k3][l3]
                                    w3_val = W3[k3*3 + i3][l3]
                                    
                                    if u3_val == 0 and v3_val == 0 and w3_val == 0:
                                        continue
                                    
                                    l = l2 * 27 + l3
                                    
                                    i = i2 * 3 + i3
                                    j = j2 * 3 + j3
                                    k = k2 * 3 + k3
                                    
                                    u_row = i * 6 + j
                                    v_row = j * 6 + k
                                    w_row = k * 6 + i
                                    
                                    U[u_row][l] += u2_val * u3_val
                                    V[v_row][l] += v2_val * v3_val
                                    W[w_row][l] += w2_val * w3_val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
