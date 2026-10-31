import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose 4x4 matrix multiplication using recursive Strassen.
    
    This implementation constructs the factor matrices U, V, W of rank 49.
    The decomposition is based on viewing the 4x4 matrices as 2x2 blocks of 2x2 matrices.
    The outer structure is a Strassen decomposition of the 2x2 block matrix multiplication,
    and the inner structure is a Strassen decomposition of the 2x2 scalar block multiplication.
    
    Rank = 7 (outer) * 7 (inner) = 49.
    """
    n, m, p = DIMENSIONS  # all 4
    
    # Define the Strassen factors for 2x2 matrix multiplication.
    # These are lists of tuples (coeff, row, col) representing the non-zero entries
    # in the U, V, W matrices for the 7 multiplications in Strassen's algorithm.
    
    # U factors: Coefficients for the left matrix X
    # m1 = (X00 + X11) * ...
    # m2 = (X10 + X11) * ...
    # m3 = X00 * ...
    # m4 = X11 * ...
    # m5 = (X00 + X01) * ...
    # m6 = (X10 - X00) * ...
    # m7 = (X01 - X11) * ...
    strassen_U = [
        [(1, 0, 0), (1, 1, 1)],       # m1
        [(1, 1, 0), (1, 1, 1)],       # m2
        [(1, 0, 0)],                  # m3
        [(1, 1, 1)],                  # m4
        [(1, 0, 0), (1, 0, 1)],       # m5
        [(1, 1, 0), (-1, 0, 0)],      # m6
        [(1, 0, 1), (-1, 1, 1)],      # m7
    ]
    
    # V factors: Coefficients for the right matrix Y
    # m1 = ... * (Y00 + Y11)
    # m2 = ... * Y00
    # m3 = ... * (Y01 - Y11)
    # m4 = ... * (Y10 - Y00)
    # m5 = ... * Y11
    # m6 = ... * (Y00 + Y01)
    # m7 = ... * (Y10 + Y11)
    strassen_V = [
        [(1, 0, 0), (1, 1, 1)],       # m1
        [(1, 0, 0)],                  # m2
        [(1, 0, 1), (-1, 1, 1)],      # m3
        [(1, 1, 0), (-1, 0, 0)],      # m4
        [(1, 1, 1)],                  # m5
        [(1, 0, 0), (1, 0, 1)],       # m6
        [(1, 1, 0), (1, 1, 1)],       # m7
    ]
    
    # W factors: Coefficients for the result matrix Z
    # Z00 = m1 + m4 - m5 + m7
    # Z01 = m3 + m5
    # Z10 = m2 + m4
    # Z11 = m1 - m2 + m3 + m6
    strassen_W = [
        [(1, 0, 0), (1, 1, 1)],       # m1: Z00 += 1, Z11 += 1
        [(1, 1, 0), (-1, 1, 1)],      # m2: Z10 += 1, Z11 += -1
        [(1, 0, 1), (1, 1, 1)],       # m3: Z01 += 1, Z11 += 1
        [(1, 0, 0), (1, 1, 0)],       # m4: Z00 += 1, Z10 += 1
        [(-1, 0, 0), (1, 0, 1)],      # m5: Z00 += -1, Z01 += 1
        [(1, 1, 1)],                  # m6: Z11 += 1
        [(1, 0, 0)],                  # m7: Z00 += 1
    ]
    
    rank = 49
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for outer_s in range(7):
        for inner_s in range(7):
            # Process U
            for (oc, obr, obc) in strassen_U[outer_s]:
                for (ic, ibr, ibc) in strassen_U[inner_s]:
                    gr = 2 * obr + ibr
                    gc = 2 * obc + ibc
                    U[gr * m + gc, col] += oc * ic
            
            # Process V
            for (oc, obr, obc) in strassen_V[outer_s]:
                for (ic, ibr, ibc) in strassen_V[inner_s]:
                    gr = 2 * obr + ibr
                    gc = 2 * obc + ibc
                    V[gr * p + gc, col] += oc * ic
            
            # Process W
            # Note: W[k*n+i] where k is column index of result, i is row index.
            # In our loop, gr is the row of the result block, gc is the column of the result block.
            # So global row index in W matrix (which corresponds to the output element) is gc * n + gr.
            for (oc, obr, obc) in strassen_W[outer_s]:
                for (ic, ibr, ibc) in strassen_W[inner_s]:
                    gr = 2 * obr + ibr
                    gc = 2 * obc + ibc
                    W[gc * n + gr, col] += oc * ic
            
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
