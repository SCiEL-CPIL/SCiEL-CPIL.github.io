"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time

def construct():
    """Rank-49 Strassen-based decomposition for 4x4 matrix multiplication.
    
    Uses Kronecker product of two 2x2 Strassen rank-7 decompositions.
    U: (16, 49), V: (16, 49), W: (16, 49)
    """
    n, m, p = DIMENSIONS
    
    # 2x2 Strassen decomposition: U2, V2, W2 are 4x7 matrices
    # U2 indexed by [i*2+j], V2 by [j*2+k], W2 by [k*2+i]
    
    U2 = [
        [1, 0, 1, 0, 1, -1, 0],  # i=0,j=0: A[0,0]
        [0, 0, 0, 0, 1,  0, 1],  # i=0,j=1: A[0,1]
        [0, 1, 0, 0, 0,  1, 0],  # i=1,j=0: A[1,0]
        [1, 1, 0, 1, 0,  0,-1],  # i=1,j=1: A[1,1]
    ]
    
    V2 = [
        [1, 1, 0, -1, 0, 1, 0],  # j=0,k=0: B[0,0]
        [0, 0, 1,  0, 0, 1, 0],  # j=0,k=1: B[0,1]
        [1, 0, 0,  1, 0, 0, 1],  # j=1,k=0: B[1,0]
        [1, 0,-1,  0, 1, 0, 1],  # j=1,k=1: B[1,1]
    ]
    
    W2 = [
        [1, 0, 0, 1, 0, 0, 1],  # k=0,i=0: C[0,0]
        [0, 1, 0, 1, 0, 0, 0],  # k=0,i=1: C[1,0]
        [0, 0, 1, 0, 1, 0, 0],  # k=1,i=0: C[0,1]
        [1, 0, 0, 0, 1, 1, 1],  # k=1,i=1: C[1,1]
    ]
    
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    for c1 in range(7):
        for c0 in range(7):
            c = c1 * 7 + c0
            for i1 in range(2):
                for j1 in range(2):
                    u_hi = U2[i1 * 2 + j1][c1]
                    if u_hi == 0:
                        continue
                    for i0 in range(2):
                        for j0 in range(2):
                            u_lo = U2[i0 * 2 + j0][c0]
                            if u_lo == 0:
                                continue
                            U[i1 * 8 + i0 * 4 + j1 * 2 + j0][c] = u_hi * u_lo
    
    for c1 in range(7):
        for c0 in range(7):
            c = c1 * 7 + c0
            for j1 in range(2):
                for k1 in range(2):
                    v_hi = V2[j1 * 2 + k1][c1]
                    if v_hi == 0:
                        continue
                    for j0 in range(2):
                        for k0 in range(2):
                            v_lo = V2[j0 * 2 + k0][c0]
                            if v_lo == 0:
                                continue
                            V[j1 * 8 + j0 * 4 + k1 * 2 + k0][c] = v_hi * v_lo
    
    for c1 in range(7):
        for c0 in range(7):
            c = c1 * 7 + c0
            for k1 in range(2):
                for i1 in range(2):
                    w_hi = W2[k1 * 2 + i1][c1]
                    if w_hi == 0:
                        continue
                    for k0 in range(2):
                        for i0 in range(2):
                            w_lo = W2[k0 * 2 + i0][c0]
                            if w_lo == 0:
                                continue
                            W[k1 * 8 + k0 * 4 + i1 * 2 + i0][c] = w_hi * w_lo
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
