"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Kronecker product decomposition for 3x4 x 4x8 matrix multiplication.
    
    Factor: n=3=1*3, m=4=2*2, p=8=2*4
    Sub-problem 1: 1x2 x 2x2 (schoolbook rank 4)
    Sub-problem 2: 3x2 x 2x4 (schoolbook rank 12)
    Total rank = 4 * 12 = 48
    """
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # Factor: n1=1, m1=2, p1=2 and n2=3, m2=2, p2=4
    n1, m1, p1 = 1, 2, 2
    n2, m2, p2 = 3, 2, 4
    
    # Sub-problem 1: 1x2 x 2x2, schoolbook rank 4
    r1 = n1 * m1 * p1  # 4
    U1 = np.zeros((n1 * m1, r1))
    V1 = np.zeros((m1 * p1, r1))
    W1 = np.zeros((p1 * n1, r1))
    col = 0
    for i in range(n1):
        for j in range(m1):
            for k in range(p1):
                U1[i * m1 + j, col] = 1
                V1[j * p1 + k, col] = 1
                W1[k * n1 + i, col] = 1
                col += 1
    
    # Sub-problem 2: 3x2 x 2x4, schoolbook rank 12
    r2 = n2 * m2 * p2  # 12
    U2 = np.zeros((n2 * m2, r2))
    V2 = np.zeros((m2 * p2, r2))
    W2 = np.zeros((p2 * n2, r2))
    col = 0
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                U2[i * m2 + j, col] = 1
                V2[j * p2 + k, col] = 1
                W2[k * n2 + i, col] = 1
                col += 1
    
    # Kronecker product combination
    total_rank = r1 * r2
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    for a in range(r1):
        for b in range(r2):
            idx = a * r2 + b
            
            # U: combined index (i1*n2+i2)*(m1*m2) + j1*m2 + j2
            for i1 in range(n1):
                for j1 in range(m1):
                    u1_val = U1[i1 * m1 + j1, a]
                    if u1_val == 0:
                        continue
                    for i2 in range(n2):
                        for j2 in range(m2):
                            u2_val = U2[i2 * m2 + j2, b]
                            if u2_val == 0:
                                continue
                            row = (i1 * n2 + i2) * (m1 * m2) + j1 * m2 + j2
                            U[row, idx] = u1_val * u2_val
            
            # V: combined index (j1*m2+j2)*(p1*p2) + k1*p2 + k2
            for j1 in range(m1):
                for k1 in range(p1):
                    v1_val = V1[j1 * p1 + k1, a]
                    if v1_val == 0:
                        continue
                    for j2 in range(m2):
                        for k2 in range(p2):
                            v2_val = V2[j2 * p2 + k2, b]
                            if v2_val == 0:
                                continue
                            row = (j1 * m2 + j2) * (p1 * p2) + k1 * p2 + k2
                            V[row, idx] = v1_val * v2_val
            
            # W: combined index (k1*p2+k2)*(n1*n2) + i1*n2 + i2
            for k1 in range(p1):
                for i1 in range(n1):
                    w1_val = W1[k1 * n1 + i1, a]
                    if w1_val == 0:
                        continue
                    for k2 in range(p2):
                        for i2 in range(n2):
                            w2_val = W2[k2 * n2 + i2, b]
                            if w2_val == 0:
                                continue
                            row = (k1 * p2 + k2) * (n1 * n2) + i1 * n2 + i2
                            W[row, idx] = w1_val * w2_val
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
