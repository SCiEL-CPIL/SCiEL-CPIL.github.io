import time
import numpy as np

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 56  # 8 independent 2x2x2 blocks, each with rank 7 (Strassen)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7-term algorithm for 2x2 matrix multiplication
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = A*B
    # M1=(a+d)(e+h), M2=(c+d)e, M3=a(f-h), M4=d(g-e), M5=(a+b)h, M6=(c-d)(e+f), M7=(b-d)(g+h)
    # C11=M1-M3+M5-M7, C12=M4+M5, C21=M2+M3, C22=M1-M2+M4+M6
    
    # U vectors: A flat indices [a,b,c,d] = [0,1,2,3]
    u_vecs = [
        [1, 0, 0, 1],    # M1: a+d
        [0, 0, 1, 1],    # M2: c+d
        [1, 0, 0, 0],    # M3: a
        [0, 0, 0, 1],    # M4: d
        [1, 1, 0, 0],    # M5: a+b
        [-1, 0, 1, 0],   # M6: c-d
        [0, 1, 0, -1]    # M7: b-d
    ]
    
    # V vectors: B local 2x2 flat [e,f,g,h] = [0,1,2,3]
    v_vecs = [
        [1, 0, 0, 1],    # M1: e+h
        [1, 0, 0, 0],    # M2: e
        [0, 1, 0, -1],   # M3: f-h
        [-1, 0, 1, 0],   # M4: g-e
        [0, 0, 0, 1],    # M5: h
        [1, 1, 0, 0],    # M6: e+f
        [0, 0, 1, 1]     # M7: g+h
    ]
    
    # W vectors: C local 2x2 flat [C11,C12,C21,C22] = [0,1,2,3]
    # C11=M1-M3+M5-M7, C12=M4+M5, C21=M2+M3, C22=M1-M2+M4+M6
    w_vecs = [
        [1, 0, 0, 1],    # M1: C11 + C22
        [0, 0, 1, -1],   # M2: C21 - C22
        [-1, 0, 1, 0],   # M3: -C11 + C21
        [0, 1, 0, 1],    # M4: C12 + C22
        [1, 1, 0, 0],    # M5: C11 + C12
        [0, 0, 0, 1],    # M6: C22
        [-1, 0, 0, 0]    # M7: -C11
    ]
    
    for r in range(8):
        base_col = 7 * r
        for s in range(7):
            col = base_col + s
            
            # U: A indices [0,1,2,3] map directly to U rows
            for li in range(4):
                if u_vecs[s][li] != 0:
                    U[li][col] = u_vecs[s][li]
                    
            # V: B local [e,f,g,h] -> global V indices
            # e=B[0,2r]=V[2r], f=B[0,2r+1]=V[2r+1], g=B[1,2r]=V[16+2r], h=B[1,2r+1]=V[16+2r+1]
            v_idx = [2*r, 2*r+1, 16+2*r, 16+2*r+1]
            for li in range(4):
                if v_vecs[s][li] != 0:
                    V[v_idx[li]][col] = v_vecs[s][li]
                    
            # W: C local [C11,C12,C21,C22] -> global W indices
            # C11=C[0,2r]=W[4r], C12=C[0,2r+1]=W[4r+2], C21=C[1,2r]=W[4r+1], C22=C[1,2r+1]=W[4r+3]
            w_idx = [4*r, 4*r+2, 4*r+1, 4*r+3]
            for li in range(4):
                if w_vecs[s][li] != 0:
                    W[w_idx[li]][col] = w_vecs[s][li]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
