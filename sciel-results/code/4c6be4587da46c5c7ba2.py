import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # Total rank is 44
    # Decomposition:
    # Split p=8 into two p=4 blocks.
    # For each p=4 block:
    #   Split m=3 into m=2 and m=1.
    #   m=2 part (2x2x4): Use two Strassen 2x2x2 blocks. Rank 14.
    #   m=1 part (2x1x4): Use schoolbook. Rank 8.
    #   Total per p=4 block: 22.
    # Total rank: 22 + 22 = 44.
    
    total_rank = 44
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    
    # Helper to add Strassen 2x2x2 block
    # i_off: offset in n (0 or 1)
    # j_off: offset in m (0 or 1)
    # k_off: offset in p (0..7)
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # U indices: (i, j) -> i*m + j
        # V indices: (j, k) -> j*p + k
        # W indices: (k, i) -> k*n + i
        
        # m1: (a+d)(e+h)
        U[i_off*m + j_off, col] += 1
        U[(i_off+1)*m + (j_off+1), col] += 1
        V[j_off*p + k_off, col] += 1
        V[(j_off+1)*p + (k_off+1), col] += 1
        W[k_off*n + i_off, col] += 1
        W[(k_off+1)*n + (i_off+1), col] += 1
        col += 1

        # m2: (c+d)e
        U[(i_off+1)*m + j_off, col] += 1
        U[(i_off+1)*m + (j_off+1), col] += 1
        V[j_off*p + k_off, col] += 1
        W[k_off*n + (i_off+1), col] += 1
        W[(k_off+1)*n + (i_off+1), col] += -1
        col += 1

        # m3: a(f-h)
        U[i_off*m + j_off, col] += 1
        V[j_off*p + (k_off+1), col] += 1
        V[(j_off+1)*p + (k_off+1), col] += -1
        W[(k_off+1)*n + i_off, col] += 1
        W[(k_off+1)*n + (i_off+1), col] += 1
        col += 1

        # m4: d(g-e)
        U[(i_off+1)*m + (j_off+1), col] += 1
        V[(j_off+1)*p + k_off, col] += 1
        V[j_off*p + k_off, col] += -1
        W[k_off*n + i_off, col] += 1
        W[k_off*n + (i_off+1), col] += 1
        col += 1

        # m5: (a+b)h
        U[i_off*m + j_off, col] += 1
        U[i_off*m + (j_off+1), col] += 1
        V[(j_off+1)*p + (k_off+1), col] += 1
        W[k_off*n + i_off, col] += -1
        W[(k_off+1)*n + i_off, col] += 1
        col += 1

        # m6: (c-a)(e+f)
        U[(i_off+1)*m + j_off, col] += 1
        U[i_off*m + j_off, col] += -1
        V[j_off*p + k_off, col] += 1
        V[j_off*p + (k_off+1), col] += 1
        W[(k_off+1)*n + (i_off+1), col] += 1
        col += 1

        # m7: (b-d)(g+h)
        U[i_off*m + (j_off+1), col] += 1
        U[(i_off+1)*m + (j_off+1), col] += -1
        V[(j_off+1)*p + k_off, col] += 1
        V[(j_off+1)*p + (k_off+1), col] += 1
        W[k_off*n + i_off, col] += 1
        col += 1

    # Helper to add schoolbook 2x1x4 block
    # i_off: offset in n
    # j_off: offset in m (must be 2, the third column)
    # k_off: offset in p
    def add_schoolbook_2x1x4(i_off, j_off, k_off):
        nonlocal col
        for i in range(2):
            for k in range(4):
                U[(i_off+i)*m + (j_off), col] = 1
                V[(j_off)*p + (k_off+k), col] = 1
                W[(k_off+k)*n + (i_off+i), col] = 1
                col += 1

    # Block 1: k in 0..3
    # m=2 part (j=0,1)
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    # m=1 part (j=2)
    add_schoolbook_2x1x4(0, 2, 0)

    # Block 2: k in 4..7
    # m=2 part (j=0,1)
    add_strassen(0, 0, 4)
    add_strassen(0, 0, 6)
    # m=1 part (j=2)
    add_schoolbook_2x1x4(0, 2, 4)

    return {"U": U, "V": V, "W": W}
