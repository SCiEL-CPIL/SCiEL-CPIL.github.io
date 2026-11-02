"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition (rank = n*m*p = 54)
    # This is the standard, known-valid decomposition.
    # U has shape (n*m, rank)
    # V has shape (m*p, rank)
    # W has shape (p*n, rank)
    # The tensor T_{i,j,k} = \sum_{c=1}^{rank} U_{i,j,c} V_{j,k,c} W_{k,i,c}
    # For matrix multiplication C = A * B, C_{i,k} = \sum_j A_{i,j} B_{j,k}.
    # So we need U_{i,j,c} V_{j,k,c} W_{k,i,c} to sum to A_{i,j} B_{j,k} for each (i,k).
    # The schoolbook term for (i,j,k) is:
    # U_{i,j,c} = 1 if c corresponds to (i,j,k), else 0
    # V_{j,k,c} = 1 if c corresponds to (i,j,k), else 0
    # W_{k,i,c} = 1 if c corresponds to (i,j,k), else 0
    # Wait, the standard decomposition is:
    # T = \sum_{i,j,k} e_{i,j} \otimes e_{j,k} \otimes e_{k,i}
    # Here U maps A indices (i,j), V maps B indices (j,k), W maps C indices (k,i).
    # Note: W index is usually (output_row, output_col) or similar.
    # The task specifies W[k*n+i, r]. This means W row index is k*n + i.
    # This corresponds to output C[i,k] being stored at row k*n + i?
    # Let's check the convention: "output factor follows the official trace convention W[k*n+i,r]"
    # Usually C is n x p. Indices are (i, k) where i in 0..n-1, k in 0..p-1.
    # If W row is k*n + i, then for a fixed k, i varies 0..n-1.
    # This implies W is indexed by [col, row] of C? Or [k, i]?
    # Let's assume the standard flattening where the index is linearized.
    # If W[k*n+i], then for k=0, indices are 0..n-1. For k=1, n..2n-1.
    # This looks like column-major order for an n x p matrix? No, n is rows, p is cols.
    # C is 2x9. n=2, p=9.
    # W size is p*n = 18.
    # If W[k*n+i], max index is 8*2 + 1 = 17. Correct.
    # So W_{k,i} corresponds to C_{i,k}.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
