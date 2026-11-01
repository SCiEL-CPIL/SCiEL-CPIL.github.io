import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=12
    # We use Strassen's algorithm for 2x2 matrix multiplication.
    # We split the p=12 columns into 6 blocks of 2 columns each.
    # Each block multiplication uses 7 scalar multiplications.
    # Total rank = 6 * 7 = 42.
    
    rank = 42
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Indices for A (2x2)
    # A[0,0] -> U[0]
    # A[0,1] -> U[1]
    # A[1,0] -> U[2]
    # A[1,1] -> U[3]
    
    # We iterate over 6 blocks
    for b in range(6):
        k_start = 2 * b
        k_end = k_start + 1
        base_rank = 7 * b
        
        # V indices for B block (2x2)
        # B[0, k_start] -> V[0*12 + k_start]
        # B[0, k_end]   -> V[0*12 + k_end]
        # B[1, k_start] -> V[1*12 + k_start]
        # B[1, k_end]   -> V[1*12 + k_end]
        
        v_00 = 0 * p + k_start
        v_01 = 0 * p + k_end
        v_10 = 1 * p + k_start
        v_11 = 1 * p + k_end
        
        # W indices for C block (2x2)
        # C[0, k_start] -> W[k_start*2 + 0]
        # C[0, k_end]   -> W[k_end*2 + 0]
        # C[1, k_start] -> W[k_start*2 + 1]
        # C[1, k_end]   -> W[k_end*2 + 1]
        
        w_00 = k_start * n + 0
        w_01 = k_end * n + 0
        w_10 = k_start * n + 1
        w_11 = k_end * n + 1
        
        # Term 1: M1 = (A00 + A11) * (B00 + B11)
        # C00 += M1, C11 += M1
        r = base_rank + 0
        U[0, r] += 1
        U[3, r] += 1
        V[v_00, r] += 1
        V[v_11, r] += 1
        W[w_00, r] += 1
        W[w_11, r] += 1
        
        # Term 2: M2 = (A10 + A11) * B00
        # C10 += M2, C11 -= M2
        r = base_rank + 1
        U[2, r] += 1
        U[3, r] += 1
        V[v_00, r] += 1
        W[w_10, r] += 1
        W[w_11, r] -= 1
        
        # Term 3: M3 = A00 * (B01 - B11)
        # C01 += M3, C11 += M3
        r = base_rank + 2
        U[0, r] += 1
        V[v_01, r] += 1
        V[v_11, r] -= 1
        W[w_01, r] += 1
        W[w_11, r] += 1
        
        # Term 4: M4 = A11 * (B10 - B00)
        # C00 += M4, C10 += M4
        r = base_rank + 3
        U[3, r] += 1
        V[v_10, r] += 1
        V[v_00, r] -= 1
        W[w_00, r] += 1
        W[w_10, r] += 1
        
        # Term 5: M5 = (A00 + A01) * B11
        # C00 -= M5, C01 += M5
        r = base_rank + 4
        U[0, r] += 1
        U[1, r] += 1
        V[v_11, r] += 1
        W[w_00, r] -= 1
        W[w_01, r] += 1
        
        # Term 6: M6 = (A10 - A00) * (B00 + B01)
        # C11 += M6
        r = base_rank + 5
        U[2, r] += 1
        U[0, r] -= 1
        V[v_00, r] += 1
        V[v_01, r] += 1
        W[w_11, r] += 1
        
        # Term 7: M7 = (A01 - A11) * (B10 + B11)
        # C00 += M7
        r = base_rank + 6
        U[1, r] += 1
        U[3, r] -= 1
        V[v_10, r] += 1
        V[v_11, r] += 1
        W[w_00, r] += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
