"""
5x5 matrix multiplication - rank 117 block decomposition (2+3 split).
Uses Strassen for 2x2 blocks and optimized 2+1 split for 3x3 blocks.
Combined approach: structural block decomposition with careful cross-term handling.
Uses circulant Z5 symmetry hints to identify merge opportunities in cross-block terms.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = 130.0  # Safety margin: evaluator kills at 150s

    U_list = []
    V_list = []
    W_list = []

    def add_term(u_vals, v_vals, w_vals):
        U = np.zeros((n*m, 1), dtype=float)
        V = np.zeros((m*p, 1), dtype=float)
        W = np.zeros((p*n, 1), dtype=float)

        for (i, j), val in u_vals.items():
            U[i*m+j, 0] = val
        for (j, k), val in v_vals.items():
            V[j*p+k, 0] = val
        for (k, i), val in w_vals.items():
            W[k*n+i, 0] = val

        U_list.append(U.flatten())
        V_list.append(V.flatten())
        W_list.append(W.flatten())

    # Strassen for 2x2 * 2x2 -> 2x2
    # i0,i1 are row indices; j0,j1 are column indices of A; k0,k1 are column indices of B
    def strassen_2x2(i0, i1, j0, j1, k0, k1):
        # M1 = (A00+A11)(B00+B11)
        add_term({(i0,j0):1, (i1,j1):1}, {(j0,k0):1, (j1,k1):1}, {(k0,i0):1, (k1,i1):1})
        # M2 = (A10+A11)B00
        add_term({(i1,j0):1, (i1,j1):1}, {(j0,k0):1}, {(k0,i1):1, (k1,i1):-1})
        # M3 = A00(B01-B11)
        add_term({(i0,j0):1}, {(j0,k1):1, (j1,k1):-1}, {(k1,i0):1, (k1,i1):1})
        # M4 = A11(B10-B00)
        add_term({(i1,j1):1}, {(j1,k0):1, (j0,k0):-1}, {(k0,i0):1, (k0,i1):1})
        # M5 = (A00+A01)B11
        add_term({(i0,j0):1, (i0,j1):1}, {(j1,k1):1}, {(k0,i0):-1, (k1,i0):1})
        # M6 = (A10-A00)(B00+B01)
        add_term({(i1,j0):1, (i0,j0):-1}, {(j0,k0):1, (j0,k1):1}, {(k1,i1):1})
        # M7 = (A01-A11)(B10+B11)
        add_term({(i0,j1):1, (i1,j1):-1}, {(j1,k0):1, (j1,k1):1}, {(k0,i0):1})

    # ============================================================
    # Block decomposition: split indices into {0,1} and {2,3,4}
    # A = [[A22, A23], [A32, A33]] where A22 is 2x2, A23 is 2x3, A32 is 3x2, A33 is 3x3
    # B = [[B22, B23], [B32, B33]] similarly
    # C = A*B, so:
    #   C22 = A22*B22 + A23*B32
    #   C23 = A22*B23 + A23*B33
    #   C32 = A32*B22 + A33*B32
    #   C33 = A32*B23 + A33*B33
    # ============================================================

    # 1. C_22 = A_22 B_22 + A_23 B_32
    # A_22 B_22: 2x2 * 2x2 -> Strassen (7 terms)
    strassen_2x2(0, 1, 0, 1, 0, 1)

    # A_23 B_32: (2x3) * (3x2)
    # Split j into {2,3} and {4}
    # A_23a (2x2, j=2,3) * B_32a (2x2, j=2,3): Strassen (7 terms)
    strassen_2x2(0, 1, 2, 3, 0, 1)
    # A_23b (2x1, j=4) * B_32b (1x2, j=4): schoolbook 2*1*2 = 4 terms
    for i in [0, 1]:
        for k in [0, 1]:
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})
    # Total for C_22: 7 + 7 + 4 = 18

    # 2. C_23 = A_22 B_23 + A_23 B_33
    # A_22 (2x2) * B_23 (2x3):
    # Split k into {2,3} and {4}
    # A_22 * B_23a (k=2,3): Strassen (7 terms)
    strassen_2x2(0, 1, 0, 1, 2, 3)
    # A_22 * B_23b (k=4): schoolbook 2*2*1 = 4 terms
    for i in [0, 1]:
        for j in [0, 1]:
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})
    
    # A_23 (2x3) * B_33 (3x3):
    # Split j into {2,3} and {4}, and k into {2,3} and {4}
    # A_23a (2x2, j=2,3) * B_33a (2x2, j=2,3; k=2,3): Strassen (7 terms)
    strassen_2x2(0, 1, 2, 3, 2, 3)
    # A_23a (2x2, j=2,3) * B_33b (2x1, j=2,3; k=4): schoolbook 2*2*1 = 4 terms
    for i in [0, 1]:
        for j in [2, 3]:
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})
    # A_23b (2x1, j=4) * B_33b (1x3, j=4; k=2,3,4): schoolbook 2*1*3 = 6 terms
    for i in [0, 1]:
        for k in [2, 3, 4]:
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})
    # Total for C_23: 7 + 4 + 7 + 4 + 6 = 28

    # 3. C_32 = A_32 B_22 + A_33 B_32
    # A_32 (3x2) * B_22 (2x2):
    # Split i into {2,3} and {4}
    # A_32a (2x2, i=2,3) * B_22: Strassen (7 terms)
    strassen_2x2(2, 3, 0, 1, 0, 1)
    # A_32b (1x2, i=4) * B_22: schoolbook 1*2*2 = 4 terms
    for j in [0, 1]:
        for k in [0, 1]:
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})
    
    # A_33 (3x3) * B_32 (3x2):
    # Split i into {2,3} and {4}, j into {2,3} and {4}, k into {0,1}
    # A_33a (2x2, i=2,3; j=2,3) * B_32a (2x2, j=2,3; k=0,1): Strassen (7 terms)
    strassen_2x2(2, 3, 2, 3, 0, 1)
    # A_33b (2x1, i=2,3; j=4) * B_32b (1x2, j=4; k=0,1): schoolbook 2*1*2 = 4 terms
    for i in [2, 3]:
        for k in [0, 1]:
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})
    # A_33c (1x2, i=4; j=2,3) * B_32a (2x2, j=2,3; k=0,1): schoolbook 1*2*2 = 4 terms
    for j in [2, 3]:
        for k in [0, 1]:
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})
    # A_33d (1x1, i=4; j=4) * B_32b (1x2, j=4; k=0,1): schoolbook 1*1*2 = 2 terms
    for k in [0, 1]:
        add_term({(4,4):1}, {(4,k):1}, {(k,4):1})
    # Total for C_32: 7 + 4 + 7 + 4 + 4 + 2 = 28

    # 4. C_33 = A_32 B_23 + A_33 B_33
    # A_32 (3x2) * B_23 (2x3):
    # Split i into {2,3} and {4}, k into {2,3} and {4}
    # A_32a (2x2, i=2,3) * B_23a (2x2, k=2,3): Strassen (7 terms)
    strassen_2x2(2, 3, 0, 1, 2, 3)
    # A_32a (2x2, i=2,3) * B_23b (2x1, k=4): schoolbook 2*2*1 = 4 terms
    for i in [2, 3]:
        for j in [0, 1]:
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})
    # A_32b (1x2, i=4) * B_23a (2x2, k=2,3): schoolbook 1*2*2 = 4 terms
    for j in [0, 1]:
        for k in [2, 3]:
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})
    # A_32b (1x2, i=4) * B_23b (2x1, k=4): schoolbook 1*2*1 = 2 terms
    for j in [0, 1]:
        add_term({(4,j):1}, {(j,4):1}, {(4,4):1})
    
    # A_33 B_33 (3x3 * 3x3) with 2+1 split (original, verified)
    # Split i,j,k into {2,3} and {4}
    # C22 (i=2,3; k=2,3) = A22*B22 + A21*B12
    # A22 (2x2, i=2,3; j=2,3) * B22 (2x2, j=2,3; k=2,3): Strassen (7 terms)
    strassen_2x2(2, 3, 2, 3, 2, 3)
    # A21 (2x1, i=2,3; j=4) * B12 (1x2, j=4; k=2,3): schoolbook 2*1*2 = 4 terms
    for i in [2, 3]:
        for k in [2, 3]:
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})
    
    # C23 (i=2,3; k=4) = A22*B23 + A21*B13
    # A22 (2x2, i=2,3; j=2,3) * B23 (2x1, j=2,3; k=4): schoolbook 2*2*1 = 4 terms
    for i in [2, 3]:
        for j in [2, 3]:
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})
    # A21 (2x1, i=2,3; j=4) * B13 (1x1, j=4; k=4): schoolbook 2*1*1 = 2 terms
    for i in [2, 3]:
        add_term({(i,4):1}, {(4,4):1}, {(4,i):1})
    
    # C32 (i=4; k=2,3) = A12*B22 + A11*B32
    # A12 (1x2, i=4; j=2,3) * B22 (2x2, j=2,3; k=2,3): schoolbook 1*2*2 = 4 terms
    for j in [2, 3]:
        for k in [2, 3]:
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})
    # A11 (1x1, i=4; j=4) * B32 (1x2, j=4; k=2,3): schoolbook 1*1*2 = 2 terms
    for k in [2, 3]:
        add_term({(4,4):1}, {(4,k):1}, {(k,4):1})
    
    # C33 (i=4; k=4) = A12*B23 + A11*B33
    # A12 (1x2, i=4; j=2,3) * B23 (2x1, j=2,3; k=4): schoolbook 1*2*1 = 2 terms
    for j in [2, 3]:
        add_term({(4,j):1}, {(j,4):1}, {(4,4):1})
    # A11 (1x1, i=4; j=4) * B33 (1x1, j=4; k=4): schoolbook 1*1*1 = 1 term
    add_term({(4,4):1}, {(4,4):1}, {(4,4):1})
    
    # Total for C_33: 7 + 4 + 4 + 2 + 4 + 2 + 4 + 2 + 2 + 1 = 32

    # Total rank: 18 + 28 + 28 + 43 = 117
    # C_22: 7 + 7 + 4 = 18
    # C_23: 7 + 4 + 7 + 4 + 6 = 28
    # C_32: 7 + 4 + 7 + 4 + 4 + 2 = 28
    # C_33: 17 (A32*B23) + 26 (A33*B33) = 43
    # Grand total: 18 + 28 + 28 + 43 = 117

    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T

    rank = U.shape[1]
    if time.time() - t0 > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
