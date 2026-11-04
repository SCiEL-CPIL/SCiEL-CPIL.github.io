"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]
_START_TIME = time.time()
_DEADLINE = 120.0  # seconds; evaluator kills at 150s, keep safety margin

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Rank 189 decomposition:
    # Partition 6x6 into 3x3 blocks of 2x2 matrices.
    # Use schoolbook for 3x3 block multiplication (27 block products).
    # Use Strassen (Rank 7) for each 2x2 block multiplication.
    # Total Rank = 27 * 7 = 189.
    
    rank = 189
    
    # Use NumPy arrays for faster construction
    U = np.zeros((n * m, rank), dtype=np.int8)  # (36, 189)
    V = np.zeros((m * p, rank), dtype=np.int8)  # (36, 189)
    W = np.zeros((p * n, rank), dtype=np.int8)  # (36, 189)
    
    # Strassen 7-term decomposition for 2x2 matrix multiplication
    # Offsets: 0=A11, 1=A12, 2=A21, 3=A22
    strassen_terms = [
        (([0, 3], [1, 1]), ([0, 3], [1, 1])),
        (([2, 3], [1, 1]), ([0], [1])),
        (([0], [1]), ([1, 3], [1, -1])),
        (([3], [1]), ([2, 0], [1, -1])),
        (([0, 1], [1, 1]), ([3], [1])),
        (([2, 0], [1, -1]), ([0, 1], [1, 1])),
        (([1, 3], [1, -1]), ([2, 3], [1, 1])),
    ]
    
    # C output contributions: list of (c_offset, coeff) for each term
    # C offsets: 0=C11, 1=C12, 2=C21, 3=C22
    c_contributions = [
        [(0, 1), (3, 1)],   # M1
        [(2, 1), (3, -1)],  # M2
        [(1, 1), (3, 1)],   # M3
        [(0, 1), (2, 1)],   # M4
        [(0, -1), (1, 1)],  # M5
        [(3, 1)],           # M6
        [(0, 1)],           # M7
    ]
    
    col = 0
    # bi: block row of A (0..2)
    # bk: block col of A / block row of B (0..2)
    # bj: block col of B (0..2)
    for bi in range(3):
        for bk in range(3):
            for bj in range(3):
                for t in range(7):
                    u_offsets, u_coeffs = strassen_terms[t][0]
                    v_offsets, v_coeffs = strassen_terms[t][1]
                    
                    # Fill U: A[i,k] -> U[i*6+k, col]
                    for off, coeff in zip(u_offsets, u_coeffs):
                        r, c = off // 2, off % 2
                        gr = 2 * bi + r
                        gc = 2 * bk + c
                        U[gr * 6 + gc, col] += coeff
                    
                    # Fill V: B[k,j] -> V[k*6+j, col]
                    for off, coeff in zip(v_offsets, v_coeffs):
                        r, c = off // 2, off % 2
                        gr = 2 * bk + r
                        gc = 2 * bj + c
                        V[gr * 6 + gc, col] += coeff
                    
                    # Fill W: C[i,j] -> W[j*6+i, col] (trace convention)
                    for c_off, coeff in c_contributions[t]:
                        r, c = c_off // 2, c_off % 2
                        gr = 2 * bi + r
                        gc = 2 * bj + c
                        W[gc * 6 + gr, col] += coeff
                    
                    col += 1
    
    # Convert to lists of lists as required by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
