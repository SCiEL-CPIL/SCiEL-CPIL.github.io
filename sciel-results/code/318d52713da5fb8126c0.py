"""4x4 matrix multiplication tensor decomposition using recursive Strassen."""

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Decompose 4x4 matrix multiplication using an explicit enumeration of the
    49 terms from the recursive Strassen algorithm.
    Uses numpy for construction to ensure robustness and speed, then converts
    to the required list-of-lists format.
    """
    n, m, p = 4, 4, 4
    rank = 49
    
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    # Base Strassen 2x2 factors
    # Format: list of 7 entries, each a list of (coeff, row, col)
    # Indices (row, col) refer to the 2x2 block structure
    sU = [
        [(1, 0, 0), (1, 1, 1)],  # M1: A00 + A11
        [(1, 1, 0), (1, 1, 1)],  # M2: A10 + A11
        [(1, 0, 0)],             # M3: A00
        [(1, 1, 1)],             # M4: A11
        [(1, 0, 0), (1, 0, 1)],  # M5: A00 + A01
        [(1, 1, 0), (-1, 0, 0)], # M6: A10 - A00
        [(1, 0, 1), (-1, 1, 1)], # M7: A01 - A11
    ]
    sV = [
        [(1, 0, 0), (1, 1, 1)],  # M1: B00 + B11
        [(1, 0, 0)],             # M2: B00
        [(1, 0, 1), (-1, 1, 1)], # M3: B01 - B11
        [(1, 1, 0), (-1, 0, 0)], # M4: B10 - B00
        [(1, 1, 1)],             # M5: B11
        [(1, 0, 0), (1, 0, 1)],  # M6: B00 + B01
        [(1, 1, 0), (1, 1, 1)],  # M7: B10 + B11
    ]
    sW = [
        [(1, 0, 0), (1, 1, 1)],  # M1: C00 + C11
        [(1, 1, 0), (-1, 1, 1)], # M2: C10 - C11
        [(1, 0, 1), (1, 1, 1)],  # M3: C01 + C11
        [(1, 0, 0), (1, 1, 0)],  # M4: C00 + C10
        [(-1, 0, 0), (1, 0, 1)], # M5: -C00 + C01
        [(1, 1, 1)],             # M6: C11
        [(1, 0, 0)],             # M7: C00
    ]
    
    # Outer Strassen factors (identical structure to inner for 4x4 built from 2x2s)
    oU = sU
    oV = sV
    oW = sW
    
    col = 0
    for o in range(7):
        for i in range(7):
            # Fill U
            # Global index for U: row = 2*block_row + local_row, col = 2*block_col + local_col
            # U is (n*m, rank) = (16, 49). Index is row*m + col? 
            # Wait, standard flattening for matrix A (n x m) is usually row-major: idx = row*m + col.
            # The task says U is (n*m, rank).
            for (oc, obr, obc) in oU[o]:
                for (ic, ir, icl) in sU[i]:
                    gr = 2 * obr + ir
                    gc = 2 * obc + icl
                    U[gr * m + gc, col] += oc * ic
            
            # Fill V
            # V is (m*p, rank) = (16, 49).
            # B is (m x p). Flattening: row*p + col.
            for (oc, obr, obc) in oV[o]:
                for (ic, ir, icl) in sV[i]:
                    gr = 2 * obr + ir
                    gc = 2 * obc + icl
                    V[gr * p + gc, col] += oc * ic
            
            # Fill W
            # W is (p*n, rank) = (16, 49).
            # Task says: "output factor follows the official trace convention W[k*n+i,r]"
            # This implies W is indexed by [k*n + i, r].
            # In matrix multiplication C = A B, C[i, k] = sum_j A[i, j] B[j, k].
            # The tensor decomposition usually maps to C[i, k].
            # If W[k*n + i, r] corresponds to C[i, k], then:
            # The row index of W is k*n + i.
            # The local indices from Strassen are (row, col) of the C block.
            # Let's say local row is 'i_local' and local col is 'k_local'.
            # Then global i = 2*block_row + i_local, global k = 2*block_col + k_local.
            # So W index = k_global * n + i_global.
            
            # In sW, the entries are (coeff, row, col) where row/col are local C indices.
            # So ir -> i_local, icl -> k_local? 
            # Let's check sW definition.
            # sW[0] = [(1, 0, 0), (1, 1, 1)] for M1.
            # M1 contributes to C00 and C11.
            # C00 is local (0,0), C11 is local (1,1).
            # So (0,0) means row=0, col=0. (1,1) means row=1, col=1.
            # So ir is row index, icl is col index.
            # Global i = 2*obr + ir
            # Global k = 2*obc + icl
            # W index = k * n + i
            
            for (oc, obr, obc) in oW[o]:
                for (ic, ir, icl) in sW[i]:
                    gi = 2 * obr + ir
                    gk = 2 * obc + icl
                    W[gk * n + gi, col] += oc * ic
            
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
