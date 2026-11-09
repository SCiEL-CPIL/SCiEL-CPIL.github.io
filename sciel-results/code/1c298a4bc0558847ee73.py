"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Use 3x3 block decomposition with Strassen for each 2x2 sub-block.
    # This gives 9 block positions, each needing rank 7 for 2x2 Strassen.
    # Total rank = 9 * 7 = 63.
    
    block_size = 2
    blocks_per_dim = 3  # 6 / 2 = 3
    
    # Strassen patterns for 2x2 matrices
    # Each pattern: (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict mapping (row, col) in A-block to value
    # v_coeffs: dict mapping (row, col) in B-block to value
    # w_coeffs: dict mapping (row, col) in C-block to value
    strassen_patterns = [
        # M0 = (A00+A11)(B00+B11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M1 = (A10+A11)B00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M2 = A00(B01-B11)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M3 = A11(B10-B00)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M4 = (A00+A01)B11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M5 = (A10-A00)(B00+B01)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M6 = (A01-A11)(B10+B11)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    # Build U, V, W matrices
    # U shape: (n*m, rank) = (36, rank)
    # V shape: (m*p, rank) = (36, rank)
    # W shape: (p*n, rank) = (36, rank)
    
    # For each pair of block positions (I, J) for A, (J, K) for B, (I, K) for C
    # We need to compute C[I,K] += A[I,J] * B[J,K]
    # Using Strassen on the 2x2 blocks
    
    rank = 0
    U_rows = []
    V_rows = []
    W_rows = []
    
    for I in range(blocks_per_dim):
        for J in range(blocks_per_dim):
            for K in range(blocks_per_dim):
                for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
                    # Create sparse row entries for U, V, W
                    u_entry = {}
                    v_entry = {}
                    w_entry = {}
                    
                    # U: row index = (I*block_size + pi) * m + (J*block_size + pj)
                    for (pi, pj), val in u_coeffs.items():
                        r = I * block_size + pi
                        c = J * block_size + pj
                        idx = r * m + c
                        u_entry[idx] = u_entry.get(idx, 0) + val
                    
                    # V: row index = (J*block_size + pi) * p + (K*block_size + pj)
                    for (pi, pj), val in v_coeffs.items():
                        r = J * block_size + pi
                        c = K * block_size + pj
                        idx = r * p + c
                        v_entry[idx] = v_entry.get(idx, 0) + val
                    
                    # W: row index = (K*block_size + pj) * n + (I*block_size + pi)
                    # Note: W[k*n+i, r] where k is col of C, i is row of C
                    for (pi, pj), val in w_coeffs.items():
                        k = K * block_size + pj
                        i = I * block_size + pi
                        idx = k * n + i
                        w_entry[idx] = w_entry.get(idx, 0) + val
                    
                    U_rows.append(u_entry)
                    V_rows.append(v_entry)
                    W_rows.append(w_entry)
                    rank += 1
    
    # Convert to dense arrays
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    for col in range(rank):
        for idx, val in U_rows[col].items():
            U[idx, col] = val
        for idx, val in V_rows[col].items():
            V[idx, col] = val
        for idx, val in W_rows[col].items():
            W[idx, col] = val
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
