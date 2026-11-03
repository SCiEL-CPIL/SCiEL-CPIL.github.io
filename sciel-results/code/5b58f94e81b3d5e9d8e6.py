import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = start_time + 130  # Safety margin before 150s limit
    n, m, p = 3, 4, 7

    U_vecs = []
    V_vecs = []
    W_vecs = []

    def add_rank1_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        u = [Fraction(0)] * (n * m)
        v = [Fraction(0)] * (m * p)
        w = [Fraction(0)] * (p * n)

        for idx, val in zip(u_indices, u_vals):
            u[idx] += Fraction(val)
        for idx, val in zip(v_indices, v_vals):
            v[idx] += Fraction(val)
        for idx, val in zip(w_indices, w_vals):
            w[idx] += Fraction(val)

        U_vecs.append(u)
        V_vecs.append(v)
        W_vecs.append(w)

    # Strassen 2x2x2 decomposition
    # Indices: i in {0,1}, j in {0,1}, k in {0,1}
    # U indices: i*m + j
    # V indices: j*p + k
    # W indices: k*n + i
    
    def add_strassen_2x2x2(i0, j0, k0):
        # Term 1: (A11+A22)(B11+B22) -> C11, C22
        add_rank1_term(
            [i0*m + j0, (i0+1)*m + (j0+1)], [1, 1],
            [j0*p + k0, (j0+1)*p + (k0+1)], [1, 1],
            [k0*n + i0, (k0+1)*n + (i0+1)], [1, 1]
        )
        # Term 2: (A21+A22)B11 -> C21, C22
        add_rank1_term(
            [(i0+1)*m + j0, (i0+1)*m + (j0+1)], [1, 1],
            [j0*p + k0], [1],
            [k0*n + (i0+1), (k0+1)*n + (i0+1)], [1, -1]
        )
        # Term 3: A11(B21-B11) -> C11, C12
        add_rank1_term(
            [i0*m + j0], [1],
            [(j0+1)*p + (k0+1), j0*p + (k0+1)], [1, -1],
            [(k0+1)*n + i0, (k0+1)*n + (i0+1)], [1, 1]
        )
        # Term 4: A22(B12-B21) -> C12, C21
        add_rank1_term(
            [(i0+1)*m + (j0+1)], [1],
            [(j0+1)*p + k0, j0*p + k0], [1, -1],
            [k0*n + i0, k0*n + (i0+1)], [1, 1]
        )
        # Term 5: (A11+A12)B22 -> C11, C12
        add_rank1_term(
            [i0*m + j0, i0*m + (j0+1)], [1, 1],
            [(j0+1)*p + (k0+1)], [1],
            [k0*n + i0, (k0+1)*n + i0], [-1, 1]
        )
        # Term 6: (A21-A11)(B11+B12) -> C21, C22
        add_rank1_term(
            [(i0+1)*m + j0, i0*m + j0], [1, -1],
            [j0*p + k0, j0*p + (k0+1)], [1, 1],
            [(k0+1)*n + (i0+1)], [1]
        )
        # Term 7: (A12-A22)(B21+B22) -> C12, C21
        add_rank1_term(
            [i0*m + (j0+1), (i0+1)*m + (j0+1)], [1, -1],
            [(j0+1)*p + k0, (j0+1)*p + (k0+1)], [1, 1],
            [k0*n + i0], [1]
        )

    def add_schoolbook_2x2x1(i0, j0, k0):
        # 4 terms
        add_rank1_term([i0*m + j0], [1], [j0*p + k0], [1], [k0*n + i0], [1])
        add_rank1_term([i0*m + (j0+1)], [1], [(j0+1)*p + k0], [1], [k0*n + i0], [1])
        add_rank1_term([(i0+1)*m + j0], [1], [j0*p + k0], [1], [k0*n + (i0+1)], [1])
        add_rank1_term([(i0+1)*m + (j0+1)], [1], [(j0+1)*p + k0], [1], [k0*n + (i0+1)], [1])

    def add_schoolbook_1x4x7(i0, j0, k0):
        # 28 terms
        for i in range(1):
            for j in range(4):
                for k in range(7):
                    ii = i0 + i
                    jj = j0 + j
                    kk = k0 + k
                    add_rank1_term([ii*m + jj], [1], [jj*p + kk], [1], [kk*n + ii], [1])

    # Construct 2x2x7 decomposition
    # 3 blocks of 2x2x2 and 1 block of 2x2x1
    # k ranges from 0 to 6
    # Block 1: k=0,1 (2x2x2)
    # Block 2: k=2,3 (2x2x2)
    # Block 3: k=4,5 (2x2x2)
    # Block 4: k=6 (2x2x1)
    
    def add_2x2x7(i0, j0, k0):
        add_strassen_2x2x2(i0, j0, k0)
        add_strassen_2x2x2(i0, j0, k0+2)
        add_strassen_2x2x2(i0, j0, k0+4)
        add_schoolbook_2x2x1(i0, j0, k0+6)

    # Construct 2x4x7 decomposition
    # 2 blocks of 2x2x7
    # j ranges from 0 to 3
    # Block 1: j=0,1
    # Block 2: j=2,3
    
    def add_2x4x7(i0, j0, k0):
        add_2x2x7(i0, j0, k0)
        add_2x2x7(i0, j0+2, k0)

    # Main construction
    # 3x4x7
    # Split into 2x4x7 (rows 0,1) and 1x4x7 (row 2)
    
    if time.time() < DEADLINE:
        add_2x4x7(0, 0, 0)
        add_schoolbook_1x4x7(2, 0, 0)
    else:
        # Fallback to schoolbook if deadline exceeded (unlikely)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    add_rank1_term([i*m+j], [1], [j*p+k], [1], [k*n+i], [1])

    # Convert to float for output
    U = np.array([[float(x) for x in row] for row in U_vecs]).T
    V = np.array([[float(x) for x in row] for row in V_vecs]).T
    W = np.array([[float(x) for x in row] for row in W_vecs]).T

    # Verify with Fraction
    valid = True
    for ci in range(n):
        for cj in range(m):
            for ck in range(p):
                prod_sum = sum(U_vecs[t][ci*m + cj] * V_vecs[t][cj*p + ck] * W_vecs[t][ck*n + ci] for t in range(len(U_vecs)))
                if prod_sum != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if not valid:
        # Fallback to schoolbook if verification fails
        U_vecs.clear()
        V_vecs.clear()
        W_vecs.clear()
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    add_rank1_term([ii*m+jj], [1], [jj*p+kk], [1], [kk*n+ii], [1])
        U = np.array([[float(x) for x in row] for row in U_vecs]).T
        V = np.array([[float(x) for x in row] for row in V_vecs]).T
        W = np.array([[float(x) for x in row] for row in W_vecs]).T

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
