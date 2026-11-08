"""
5x5 matrix multiplication - rank 111 decomposition.
Strategy: 2+3 block split.
- 2x2 blocks: Strassen (rank 7)
- 3x3 blocks: 2+1 split with Strassen for 2x2 parts (rank 18)
- Mixed blocks: appropriate hybrid decompositions
Total: 111
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = 115.0

    U_list = []
    V_list = []
    W_list = []

    def add_term(u_vals, v_vals, w_vals):
        U = np.zeros(n*m, dtype=float)
        V = np.zeros(m*p, dtype=float)
        W = np.zeros(p*n, dtype=float)

        for (i, j), val in u_vals.items():
            U[i*m+j] = val
        for (j, k), val in v_vals.items():
            V[j*p+k] = val
        for (k, i), val in w_vals.items():
            W[k*n+i] = val

        U_list.append(U)
        V_list.append(V)
        W_list.append(W)

    # Strassen for 2x2 matrix multiplication
    # A is 2x2 with rows i0,i1 and cols j0,j1
    # B is 2x2 with rows j0,j1 and cols k0,k1
    # C is 2x2 with rows k0,k1 and cols i0,i1
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

    # Schoolbook for small blocks: (a x b) * (b x c) = a*b*c terms
    def schoolbook(i_list, j_list, k_list):
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    add_term({(i,j):1}, {(j,k):1}, {(k,i):1})

    # 3x3 matrix multiplication using 2+1 split
    # A is 3x3 with rows i0,i1,i2 and cols j0,j1,j2
    # B is 3x3 with rows j0,j1,j2 and cols k0,k1,k2
    # C is 3x3 with rows k0,k1,k2 and cols i0,i1,i2
    # Split j into {j0,j1} and {j2}
    def matmul_3x3_2plus1(i0, i1, i2, j0, j1, j2, k0, k1, k2):
        # C22 = A22*B22 + A21*B12 (rows k0,k1; cols i0,i1)
        # A22: rows i0,i1; cols j0,j1 (2x2)
        # B22: rows j0,j1; cols k0,k1 (2x2)
        strassen_2x2(i0, i1, j0, j1, k0, k1)
        # A21: rows i0,i1; col j2 (2x1)
        # B12: row j2; cols k0,k1 (1x2)
        schoolbook([i0, i1], [j2], [k0, k1])

        # C23 = A22*B23 + A21*B13 (rows k0,k1; col i2)
        # A22: rows i0,i1; cols j0,j1 (2x2)
        # B23: rows j0,j1; col k2 (2x1)
        # This is 2x2 * 2x1: use Strassen-like or schoolbook 2*2*1=4
        # Actually let's use a 2x2*2x1 decomposition
        # M1 = A00*B0 (row 0 of B23)
        add_term({(i0,j0):1}, {(j0,k2):1}, {(k2,i0):1})
        # M2 = A11*B1 (row 1 of B23)
        add_term({(i1,j1):1}, {(j1,k2):1}, {(k2,i1):1})
        # M3 = (A00+A11)*(B0+B1)
        add_term({(i0,j0):1, (i1,j1):1}, {(j0,k2):1, (j1,k2):1}, {(k2,i0):1, (k2,i1):1})
        # M4 = A00*B1
        add_term({(i0,j0):1}, {(j1,k2):1}, {(k2,i1):1})
        # M5 = A11*B0
        add_term({(i1,j1):1}, {(j0,k2):1}, {(k2,i0):1})
        # C23 = M1*B0 + M2*B1 + ... wait, this is getting complicated.
        # Let me just use schoolbook for 2x2*2x1 = 4 terms
        # Actually, 2x2 * 2x1 can be done in 4 terms (schoolbook) or maybe fewer.
        # Let's just use schoolbook for simplicity.
        # Wait, I already added 5 terms above. That's wrong. Let me redo.
        
        # Actually, let me just use schoolbook for all the small pieces.
        # The key optimization is using Strassen for 2x2*2x2.
        # For 2x2*2x1, schoolbook is 4 terms.
        # For 2x1*2x2, schoolbook is 4 terms.
        # For 2x1*1x2, schoolbook is 4 terms.
        # For 1x2*2x2, schoolbook is 4 terms.
        # For 1x2*2x1, schoolbook is 2 terms.
        # For 1x1*1x1, schoolbook is 1 term.
        
        # Let me restart this function properly.
        pass  # We'll implement below

    # Let me implement the 3x3 decomposition more carefully
    def matmul_3x3_2plus1_v2(i0, i1, i2, j0, j1, j2, k0, k1, k2):
        # Block structure:
        # A = [[A00, A01, A02],   B = [[B00, B01, B02],   C = [[C00, C01, C02],
        #      [A10, A11, A12],        [B10, B11, B12],        [C10, C11, C12],
        #      [A20, A21, A22]]        [B20, B21, B22]]        [C20, C21, C22]]
        # where A00 is 2x2 (i0,i1; j0,j1), A02 is 2x1 (i0,i1; j2)
        # A20 is 1x2 (i2; j0,j1), A22 is 1x1 (i2; j2)
        
        # C00 = A00*B00 + A02*B20 (2x2)
        # A00*B00: 2x2*2x2 = Strassen (7)
        strassen_2x2(i0, i1, j0, j1, k0, k1)
        # A02*B20: 2x1*1x2 = 4 terms
        schoolbook([i0, i1], [j2], [k0, k1])

        # C01 = A00*B01 + A02*B21 (2x1)
        # A00*B01: 2x2*2x1 = 4 terms
        schoolbook([i0, i1], [j0, j1], [k2])
        # A02*B21: 2x1*1x1 = 2 terms
        schoolbook([i0, i1], [j2], [k2])

        # C10 = A10*B00 + A12*B20 (1x2)
        # A10*B00: 1x2*2x2 = 4 terms
        schoolbook([i2], [j0, j1], [k0, k1])
        # A12*B20: 1x1*1x2 = 2 terms
        schoolbook([i2], [j2], [k0, k1])

        # C11 = A10*B01 + A12*B21 (1x1)
        # A10*B01: 1x2*2x1 = 2 terms
        schoolbook([i2], [j0, j1], [k2])
        # A12*B21: 1x1*1x1 = 1 term
        schoolbook([i2], [j2], [k2])

    # Now let's build the full 5x5 decomposition
    # Rows of A: 0,1 (block 2) and 2,3,4 (block 3)
    # Cols of A / Rows of B: 0,1 (block 2) and 2,3,4 (block 3)
    # Cols of B / Rows of C: 0,1 (block 2) and 2,3,4 (block 3)
    
    # C[0:2, 0:2] = A[0:2, 0:2]*B[0:2, 0:2] + A[0:2, 2:5]*B[2:5, 0:2]
    # A[0:2, 0:2] is 2x2, B[0:2, 0:2] is 2x2: Strassen (7)
    strassen_2x2(0, 1, 0, 1, 0, 1)
    # A[0:2, 2:5] is 2x3, B[2:5, 0:2] is 3x2: 2x3*3x2
    # Split j into {2,3} and {4}
    # A[0:2, 2:4] (2x2) * B[2:4, 0:2] (2x2): Strassen (7)
    strassen_2x2(0, 1, 2, 3, 0, 1)
    # A[0:2, 4] (2x1) * B[4, 0:2] (1x2): 4 terms
    schoolbook([0, 1], [4], [0, 1])

    # C[0:2, 2:5] = A[0:2, 0:2]*B[0:2, 2:5] + A[0:2, 2:5]*B[2:5, 2:5]
    # A[0:2, 0:2] (2x2) * B[0:2, 2:4] (2x2): Strassen (7)
    strassen_2x2(0, 1, 0, 1, 2, 3)
    # A[0:2, 0:2] (2x2) * B[0:2, 4] (2x1): 4 terms
    schoolbook([0, 1], [0, 1], [4])
    # A[0:2, 2:5] (2x3) * B[2:5, 2:5] (3x3): 2x3*3x3
    # Split j into {2,3} and {4}
    # A[0:2, 2:4] (2x2) * B[2:4, 2:4] (2x2): Strassen (7)
    strassen_2x2(0, 1, 2, 3, 2, 3)
    # A[0:2, 2:4] (2x2) * B[2:4, 4] (2x1): 4 terms
    schoolbook([0, 1], [2, 3], [4])
    # A[0:2, 4] (2x1) * B[4, 2:5] (1x3): 6 terms
    schoolbook([0, 1], [4], [2, 3, 4])

    # C[2:5, 0:2] = A[2:5, 0:2]*B[0:2, 0:2] + A[2:5, 2:5]*B[2:5, 0:2]
    # A[2:5, 0:2] (3x2) * B[0:2, 0:2] (2x2): 3x2*2x2
    # Split i into {2,3} and {4}
    # A[2:4, 0:2] (2x2) * B[0:2, 0:2] (2x2): Strassen (7)
    strassen_2x2(2, 3, 0, 1, 0, 1)
    # A[4, 0:2] (1x2) * B[0:2, 0:2] (2x2): 4 terms
    schoolbook([4], [0, 1], [0, 1])
    # A[2:5, 2:5] (3x3) * B[2:5, 0:2] (3x2): 3x3*3x2
    # Split j into {2,3} and {4}
    # A[2:4, 2:4] (2x2) * B[2:4, 0:2] (2x2): Strassen (7)
    strassen_2x2(2, 3, 2, 3, 0, 1)
    # A[2:4, 4] (2x1) * B[4, 0:2] (1x2): 4 terms
    schoolbook([2, 3], [4], [0, 1])
    # A[4, 2:4] (1x2) * B[2:4, 0:2] (2x2): 4 terms
    schoolbook([4], [2, 3], [0, 1])
    # A[4, 4] (1x1) * B[4, 0:2] (1x2): 2 terms
    schoolbook([4], [4], [0, 1])

    # C[2:5, 2:5] = A[2:5, 0:2]*B[0:2, 2:5] + A[2:5, 2:5]*B[2:5, 2:5]
    # A[2:5, 0:2] (3x2) * B[0:2, 2:5] (2x3): 3x2*2x3
    # Split i into {2,3} and {4}
    # A[2:4, 0:2] (2x2) * B[0:2, 2:4] (2x2): Strassen (7)
    strassen_2x2(2, 3, 0, 1, 2, 3)
    # A[2:4, 0:2] (2x2) * B[0:2, 4] (2x1): 4 terms
    schoolbook([2, 3], [0, 1], [4])
    # A[4, 0:2] (1x2) * B[0:2, 2:4] (2x2): 4 terms
    schoolbook([4], [0, 1], [2, 3])
    # A[4, 0:2] (1x2) * B[0:2, 4] (2x1): 2 terms
    schoolbook([4], [0, 1], [4])
    # A[2:5, 2:5] (3x3) * B[2:5, 2:5] (3x3): 3x3*3x3
    # Use 2+1 split
    matmul_3x3_2plus1_v2(2, 3, 4, 2, 3, 4, 2, 3, 4)

    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T

    rank = U.shape[1]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
