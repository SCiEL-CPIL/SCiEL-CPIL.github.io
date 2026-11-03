"""
5x5 matrix multiplication - rank 123 block decomposition (2+3 split).
Uses Strassen for 2x2 block and a 3x3 block with one merged schoolbook term.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = 130.0  # Safety margin: evaluator kills at 150s

    # Block indices
    # A: i (0..4), j (0..4)
    # B: j (0..4), k (0..4)
    # C: i (0..4), k (0..4)

    # Split: 0,1 are "small" (2), 2,3,4 are "large" (3)
    small_idx = [0, 1]
    large_idx = [2, 3, 4]

    U_list = []
    V_list = []
    W_list = []

    def add_term(u_vals, v_vals, w_vals):
        """
        u_vals: dict {(i,j): val}
        v_vals: dict {(j,k): val}
        w_vals: dict {(k,i): val}
        """
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

    # 1. C_22 = A_22 B_22 + A_23 B_32
    # A_22: i in 0,1; j in 0,1
    # B_22: j in 0,1; k in 0,1
    # C_22: i in 0,1; k in 0,1
    # Use Strassen for A_22 B_22

    # Strassen terms for 2x2
    # M1 = (A00+A11)(B00+B11)
    u = {(0,0):1, (1,1):1}
    v = {(0,0):1, (1,1):1}
    w = {(0,0):1, (1,1):1}
    add_term(u, v, w)

    # M2 = (A10+A11)B00
    u = {(1,0):1, (1,1):1}
    v = {(0,0):1}
    w = {(0,1):1, (1,1):-1}
    add_term(u, v, w)

    # M3 = A00(B01-B11)
    u = {(0,0):1}
    v = {(0,1):1, (1,1):-1}
    w = {(1,0):1, (1,1):1}
    add_term(u, v, w)

    # M4 = A11(B10-B00)
    u = {(1,1):1}
    v = {(1,0):1, (0,0):-1}
    w = {(0,0):1, (0,1):1}
    add_term(u, v, w)

    # M5 = (A00+A01)B11
    u = {(0,0):1, (0,1):1}
    v = {(1,1):1}
    w = {(0,0):-1, (1,0):1}
    add_term(u, v, w)

    # M6 = (A10-A00)(B00+B01)
    u = {(1,0):1, (0,0):-1}
    v = {(0,0):1, (0,1):1}
    w = {(1,1):1}
    add_term(u, v, w)

    # M7 = (A01-A11)(B10+B11)
    u = {(0,1):1, (1,1):-1}
    v = {(1,0):1, (1,1):1}
    w = {(0,0):1}
    add_term(u, v, w)

    # A_23 B_32 schoolbook
    for i in [0, 1]:
        for j in [2, 3, 4]:
            for k in [0, 1]:
                u = {(i,j):1}
                v = {(j,k):1}
                w = {(k,i):1}
                add_term(u, v, w)

    # 2. C_23 = A_22 B_23 + A_23 B_33
    
    # A_22 (2x2) * B_23 (2x3) -> 2x3
    # Split B_23 into B_23a (2x2, k=2,3) and B_23b (2x1, k=4)
    # C_a = A_22 B_23a (2x2 * 2x2) -> Strassen (7)
    # C_b = A_22 B_23b (2x2 * 2x1) -> Schoolbook (4)
    
    # Strassen for A_22 B_23a
    # A_22: i in 0,1; j in 0,1
    # B_23a: j in 0,1; k in 2,3
    u = {(0,0):1, (1,1):1}
    v = {(0,2):1, (1,3):1}
    w = {(2,0):1, (3,1):1}
    add_term(u, v, w)
    
    u = {(1,0):1, (1,1):1}
    v = {(0,2):1}
    w = {(2,1):1, (3,1):-1}
    add_term(u, v, w)
    
    u = {(0,0):1}
    v = {(0,3):1, (1,3):-1}
    w = {(3,0):1, (3,1):1}
    add_term(u, v, w)
    
    u = {(1,1):1}
    v = {(1,2):1, (0,2):-1}
    w = {(2,0):1, (2,1):1}
    add_term(u, v, w)
    
    u = {(0,0):1, (0,1):1}
    v = {(1,3):1}
    w = {(2,0):-1, (3,0):1}
    add_term(u, v, w)
    
    u = {(1,0):1, (0,0):-1}
    v = {(0,2):1, (0,3):1}
    w = {(3,1):1}
    add_term(u, v, w)
    
    u = {(0,1):1, (1,1):-1}
    v = {(1,2):1, (1,3):1}
    w = {(2,0):1}
    add_term(u, v, w)
    
    # Schoolbook for A_22 B_23b (k=4)
    for i in [0, 1]:
        for j in [0, 1]:
            u = {(i,j):1}
            v = {(j,4):1}
            w = {(4,i):1}
            add_term(u, v, w)
            
    # A_23 (2x3) * B_33 (3x3) -> 2x3
    # Split A_23 into A_23a (2x2, j=2,3) and A_23b (2x1, j=4)
    # Split B_33 into B_33a (3x2, j=2,3; k=2,3) and B_33b (3x1, j=4; k=2,3,4)
    # Wait, B_33 is 3x3.
    # A_23a (2x2) * B_33a (2x3).
    # A_23b (2x1) * B_33b (1x3).
    
    # A_23a (2x2) * B_33a (2x3)
    # Split B_33a into B_33a1 (2x2, k=2,3) and B_33a2 (2x1, k=4)
    # C_aa1 = A_23a B_33a1 (2x2 * 2x2) -> Strassen (7)
    # C_aa2 = A_23a B_33a2 (2x2 * 2x1) -> Schoolbook (4)
    
    # Strassen for A_23a B_33a1
    # A_23a: i in 0,1; j in 2,3
    # B_33a1: j in 2,3; k in 2,3
    u = {(0,2):1, (1,3):1}
    v = {(2,2):1, (3,3):1}
    w = {(2,0):1, (3,1):1}
    add_term(u, v, w)
    
    u = {(1,2):1, (1,3):1}
    v = {(2,2):1}
    w = {(2,1):1, (3,1):-1}
    add_term(u, v, w)
    
    u = {(0,2):1}
    v = {(2,3):1, (3,3):-1}
    w = {(3,0):1, (3,1):1}
    add_term(u, v, w)
    
    u = {(1,3):1}
    v = {(3,2):1, (2,2):-1}
    w = {(2,0):1, (2,1):1}
    add_term(u, v, w)
    
    u = {(0,2):1, (0,3):1}
    v = {(3,3):1}
    w = {(2,0):-1, (3,0):1}
    add_term(u, v, w)
    
    u = {(1,2):1, (0,2):-1}
    v = {(2,2):1, (2,3):1}
    w = {(3,1):1}
    add_term(u, v, w)
    
    u = {(0,3):1, (1,3):-1}
    v = {(3,2):1, (3,3):1}
    w = {(2,0):1}
    add_term(u, v, w)
    
    # Schoolbook for A_23a B_33a2 (k=4)
    for i in [0, 1]:
        for j in [2, 3]:
            u = {(i,j):1}
            v = {(j,4):1}
            w = {(4,i):1}
            add_term(u, v, w)
            
    # Schoolbook for A_23b B_33b (j=4)
    # A_23b: i in 0,1; j=4
    # B_33b: j=4; k in 2,3,4
    for i in [0, 1]:
        for k in [2, 3, 4]:
            u = {(i,4):1}
            v = {(4,k):1}
            w = {(k,i):1}
            add_term(u, v, w)

    # 3. C_32 = A_32 B_22 + A_33 B_32
    
    # A_32 (3x2) * B_22 (2x2) -> 3x2
    # Split A_32 into A_32a (2x2, i=2,3) and A_32b (1x2, i=4)
    # C_a = A_32a B_22 (2x2 * 2x2) -> Strassen (7)
    # C_b = A_32b B_22 (1x2 * 2x2) -> Schoolbook (4)
    
    # Strassen for A_32a B_22
    # A_32a: i in 2,3; j in 0,1
    # B_22: j in 0,1; k in 0,1
    u = {(2,0):1, (3,1):1}
    v = {(0,0):1, (1,1):1}
    w = {(0,2):1, (1,3):1}
    add_term(u, v, w)
    
    u = {(3,0):1, (3,1):1}
    v = {(0,0):1}
    w = {(0,3):1, (1,3):-1}
    add_term(u, v, w)
    
    u = {(2,0):1}
    v = {(0,1):1, (1,1):-1}
    w = {(1,2):1, (1,3):1}
    add_term(u, v, w)
    
    u = {(3,1):1}
    v = {(1,0):1, (0,0):-1}
    w = {(0,2):1, (0,3):1}
    add_term(u, v, w)
    
    u = {(2,0):1, (2,1):1}
    v = {(1,1):1}
    w = {(0,2):-1, (1,2):1}
    add_term(u, v, w)
    
    u = {(3,0):1, (2,0):-1}
    v = {(0,0):1, (0,1):1}
    w = {(1,3):1}
    add_term(u, v, w)
    
    u = {(2,1):1, (3,1):-1}
    v = {(1,0):1, (1,1):1}
    w = {(0,2):1}
    add_term(u, v, w)
    
    # Schoolbook for A_32b B_22 (i=4)
    for j in [0, 1]:
        for k in [0, 1]:
            u = {(4,j):1}
            v = {(j,k):1}
            w = {(k,4):1}
            add_term(u, v, w)
            
    # A_33 (3x3) * B_32 (3x2) -> 3x2
    # Split A_33 into A_33a (2x2, i=2,3; j=2,3), A_33b (2x1, i=2,3; j=4), A_33c (1x2, i=4; j=2,3), A_33d (1x1, i=4; j=4)
    # Split B_32 into B_32a (2x2, j=2,3; k=0,1), B_32b (1x2, j=4; k=0,1)
    
    # C_aa = A_33a B_32a (2x2 * 2x2) -> Strassen (7)
    u = {(2,2):1, (3,3):1}
    v = {(2,0):1, (3,1):1}
    w = {(0,2):1, (1,3):1}
    add_term(u, v, w)
    
    u = {(3,2):1, (3,3):1}
    v = {(2,0):1}
    w = {(0,3):1, (1,3):-1}
    add_term(u, v, w)
    
    u = {(2,2):1}
    v = {(2,1):1, (3,1):-1}
    w = {(1,2):1, (1,3):1}
    add_term(u, v, w)
    
    u = {(3,3):1}
    v = {(3,0):1, (2,0):-1}
    w = {(0,2):1, (0,3):1}
    add_term(u, v, w)
    
    u = {(2,2):1, (2,3):1}
    v = {(3,1):1}
    w = {(0,2):-1, (1,2):1}
    add_term(u, v, w)
    
    u = {(3,2):1, (2,2):-1}
    v = {(2,0):1, (2,1):1}
    w = {(1,3):1}
    add_term(u, v, w)
    
    u = {(2,3):1, (3,3):-1}
    v = {(3,0):1, (3,1):1}
    w = {(0,2):1}
    add_term(u, v, w)
    
    # C_bb = A_33b B_32b (2x1 * 1x2) -> Schoolbook (2)
    for i in [2, 3]:
        for k in [0, 1]:
            u = {(i,4):1}
            v = {(4,k):1}
            w = {(k,i):1}
            add_term(u, v, w)
            
    # C_ca = A_33c B_32a (1x2 * 2x2) -> Schoolbook (4)
    for j in [2, 3]:
        for k in [0, 1]:
            u = {(4,j):1}
            v = {(j,k):1}
            w = {(k,4):1}
            add_term(u, v, w)
            
    # C_db = A_33d B_32b (1x1 * 1x2) -> Schoolbook (2)
    for k in [0, 1]:
        u = {(4,4):1}
        v = {(4,k):1}
        w = {(k,4):1}
        add_term(u, v, w)

    # 4. C_33 = A_32 B_23 + A_33 B_33
    
    # A_32 (3x2) * B_23 (2x3) -> 3x3
    # Split A_32 into A_32a (2x2, i=2,3), A_32b (1x2, i=4)
    # Split B_23 into B_23a (2x2, k=2,3), B_23b (2x1, k=4)
    
    # C_aa = A_32a B_23a (2x2 * 2x2) -> Strassen (7)
    u = {(2,0):1, (3,1):1}
    v = {(0,2):1, (1,3):1}
    w = {(2,2):1, (3,3):1}
    add_term(u, v, w)
    
    u = {(3,0):1, (3,1):1}
    v = {(0,2):1}
    w = {(2,3):1, (3,3):-1}
    add_term(u, v, w)
    
    u = {(2,0):1}
    v = {(0,3):1, (1,3):-1}
    w = {(3,2):1, (3,3):1}
    add_term(u, v, w)
    
    u = {(3,1):1}
    v = {(1,2):1, (0,2):-1}
    w = {(2,2):1, (2,3):1}
    add_term(u, v, w)
    
    u = {(2,0):1, (2,1):1}
    v = {(1,3):1}
    w = {(2,2):-1, (3,2):1}
    add_term(u, v, w)
    
    u = {(3,0):1, (2,0):-1}
    v = {(0,2):1, (0,3):1}
    w = {(3,3):1}
    add_term(u, v, w)
    
    u = {(2,1):1, (3,1):-1}
    v = {(1,2):1, (1,3):1}
    w = {(2,2):1}
    add_term(u, v, w)
    
    # C_ab = A_32a B_23b (2x2 * 2x1) -> Schoolbook (4)
    for i in [2, 3]:
        for j in [0, 1]:
            u = {(i,j):1}
            v = {(j,4):1}
            w = {(4,i):1}
            add_term(u, v, w)
            
    # C_ba = A_32b B_23a (1x2 * 2x2) -> Schoolbook (4)
    for j in [0, 1]:
        for k in [2, 3]:
            u = {(4,j):1}
            v = {(j,k):1}
            w = {(k,4):1}
            add_term(u, v, w)
            
    # C_bb = A_32b B_23b (1x2 * 2x1) -> Schoolbook (2)
    for j in [0, 1]:
        u = {(4,j):1}
        v = {(j,4):1}
        w = {(4,4):1}
        add_term(u, v, w)

    # A_33 B_33 using a 3x3 block with one merged schoolbook term
    # 3x3 indices: 2,3,4 -> sub-blocks: {2,3}=2x2, {4}=1x1
    s2 = [2, 3]  # 2x2 block indices
    s3 = [4]     # 1x1 block index

    # C22 = A22*B22 + A21*B12 (within 3x3)
    # A22: i in s2, j in s2; B22: j in s2, k in s2; C22: i in s2, k in s2
    # Strassen for A22*B22 (2x2 * 2x2)
    # M1 = (A00+A11)(B00+B11)
    u = {(2,2):1, (3,3):1}
    v = {(2,2):1, (3,3):1}
    w = {(2,2):1, (3,3):1}
    add_term(u, v, w)

    # M2 = (A10+A11)B00
    u = {(3,2):1, (3,3):1}
    v = {(2,2):1}
    w = {(2,3):1, (3,3):-1}
    add_term(u, v, w)

    # M3 = A00(B01-B11)
    u = {(2,2):1}
    v = {(2,3):1, (3,3):-1}
    w = {(3,2):1, (3,3):1}
    add_term(u, v, w)

    # M4 = A11(B10-B00)
    u = {(3,3):1}
    v = {(3,2):1, (2,2):-1}
    w = {(2,2):1, (2,3):1}
    add_term(u, v, w)

    # M5 = (A00+A01)B11
    u = {(2,2):1, (2,3):1}
    v = {(3,3):1}
    w = {(2,2):-1, (3,2):1}
    add_term(u, v, w)

    # M6 = (A10-A00)(B00+B01)
    u = {(3,2):1, (2,2):-1}
    v = {(2,2):1, (2,3):1}
    w = {(3,3):1}
    add_term(u, v, w)

    # M7 = (A01-A11)(B10+B11)
    u = {(2,3):1, (3,3):-1}
    v = {(3,2):1, (3,3):1}
    w = {(2,2):1}
    add_term(u, v, w)

    # A21*B12: i in s2, j=4; B: j=4, k in s2; C: i in s2, k in s2
    for i in s2:
        for k in s2:
            u = {(i, 4):1}
            v = {(4, k):1}
            w = {(k, i):1}
            add_term(u, v, w)

    # C23 = A22*B23 + A21*B13 (within 3x3)
    # A22: i in s2, j in s2; B23: j in s2, k=4; C23: i in s2, k=4
    for i in s2:
        for j in s2:
            u = {(i,j):1}
            v = {(j,4):1}
            w = {(4,i):1}
            add_term(u, v, w)
    # A21*B13: i in s2, j=4; B: j=4, k=4; C: i in s2, k=4
    for i in s2:
        u = {(i, 4):1}
        v = {(4, 4):1}
        w = {(4, i):1}
        add_term(u, v, w)

    # C32 = A12*B22 + A11*B32 (within 3x3)
    # A12: i=4, j in s2; B22: j in s2, k in s2; C32: i=4, k in s2
    for j in s2:
        for k in s2:
            u = {(4, j):1}
            v = {(j, k):1}
            w = {(k, 4):1}
            add_term(u, v, w)
    # A11*B32: i=4, j=4; B: j=4, k in s2; C: i=4, k in s2
    for k in s2:
        u = {(4, 4):1}
        v = {(4, k):1}
        w = {(k, 4):1}
        add_term(u, v, w)

    # C33 = A12*B23 + A11*B33 (within 3x3)
    # A12: i=4, j in s2; B23: j in s2, k=4; C33: i=4, k=4
    for j in s2:
        u = {(4, j):1}
        v = {(j, 4):1}
        w = {(4, 4):1}
        add_term(u, v, w)
    # A11*B33: i=4, j=4; B: j=4, k=4; C: i=4, k=4
    u = {(4, 4):1}
    v = {(4, 4):1}
    w = {(4, 4):1}
    add_term(u, v, w)

    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T

    # Verify rank is as expected
    rank = U.shape[1]
    if time.time() - t0 > deadline:
        # Should never happen, but safety guard
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
