"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-49 tensor decomposition for 2x2 * 2x14 matrix multiplication
    using Strassen's algorithm on 7 blocks of 2x2.
    
    Tensor convention: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    We need T[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k in {0,...,13}.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=14
    nm = n * m  # 4
    mp = m * p  # 28
    pn = p * n  # 28
    num_blocks = p // 2  # 7
    rank = num_blocks * 7  # 49

    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))

    # Strassen's 7 terms for 2x2 * 2x2
    for t in range(num_blocks):
        base = t * 7
        ve = 2 * t
        vf = 2 * t + 1
        vg = 14 + 2 * t
        vh = 14 + 2 * t + 1
        
        w_i = 4 * t
        w_j = 4 * t + 2
        w_k = 4 * t + 1
        w_l = 4 * t + 3

        # M1 = (a+d)(e+h)
        U[0, base] = 1; U[3, base] = 1
        V[ve, base] = 1; V[vh, base] = 1
        W[w_i, base] = 1; W[w_l, base] = 1

        # M2 = (c+d)e
        U[2, base+1] = 1; U[3, base+1] = 1
        V[ve, base+1] = 1
        W[w_k, base+1] = 1; W[w_l, base+1] = -1

        # M3 = a(f-h)
        U[0, base+2] = 1
        V[vf, base+2] = 1; V[vh, base+2] = -1
        W[w_j, base+2] = 1; W[w_l, base+2] = 1

        # M4 = d(g-e)
        U[3, base+3] = 1
        V[vg, base+3] = 1; V[ve, base+3] = -1
        W[w_i, base+3] = 1; W[w_k, base+3] = 1

        # M5 = (a+b)h
        U[0, base+4] = 1; U[1, base+4] = 1
        V[vh, base+4] = 1
        W[w_i, base+4] = -1; W[w_j, base+4] = 1

        # M6 = (c-a)(e+f)
        U[2, base+5] = 1; U[0, base+5] = -1
        V[ve, base+5] = 1; V[vf, base+5] = 1
        W[w_l, base+5] = 1

        # M7 = (b-d)(g+h)
        U[1, base+6] = 1; U[3, base+6] = -1
        V[vg, base+6] = 1; V[vh, base+6] = 1
        W[w_i, base+6] = 1

    # Convert to lists
    def to_int_list(arr):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in arr.tolist()]

    return {"U": to_int_list(U), "V": to_int_list(V), "W": to_int_list(W)}
# EVOLVE-BLOCK-END
