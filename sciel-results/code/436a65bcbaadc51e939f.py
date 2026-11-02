import time
import numpy as np

DIMENSIONS = [2, 2, 8]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # The tensor M_{2,2,8} is the direct sum of 4 M_{2,2,2} tensors.
    # The minimum rank of M_{2,2,2} is 7 (Strassen).
    # Therefore, the minimum rank of M_{2,2,8} is 4 * 7 = 28.
    
    rank = 28
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen terms for 2x2 matrix multiplication
    # Indices: A is 2x2, B is 2x2, C is 2x2
    # U indices: 0:A00, 1:A01, 2:A10, 3:A11
    # V indices: 0:B00, 1:B01, 2:B10, 3:B11
    # W indices: 0:C00, 1:C01, 2:C10, 3:C11
    
    # The 7 Strassen products:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    strassen_terms = [
        # (U_vec, V_vec, W_vec)
        # M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 2: 1}),
        # M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    col = 0
    for b in range(4):
        # Block b corresponds to columns 2b and 2b+1 of B, and rows of C
        # B indices in V: row 0 is B[0, :], row 1 is B[1, :]
        # B[0, 2b] is V[0 * p + 2b]
        # B[0, 2b+1] is V[0 * p + 2b+1]
        # B[1, 2b] is V[1 * p + 2b]
        # B[1, 2b+1] is V[1 * p + 2b+1]
        
        # C indices in W: row k is C[k, :], col i is C[i, k]?
        # Convention: W[k*n+i, r]
        # C is 2x8. C[i, j] where i in 0..1, j in 0..7.
        # W index = j * n + i = j * 2 + i.
        # For block b, j is 2b and 2b+1.
        # C[0, 2b] -> W[2b * 2 + 0] = W[4b]
        # C[0, 2b+1] -> W[(2b+1) * 2 + 0] = W[4b+2]
        # C[1, 2b] -> W[2b * 2 + 1] = W[4b+1]
        # C[1, 2b+1] -> W[(2b+1) * 2 + 1] = W[4b+3]
        
        # Let's map the local 2x2 C indices (0,1,2,3) to global W indices
        # Local C00 -> Global C[0, 2b] -> W[4b]
        # Local C01 -> Global C[0, 2b+1] -> W[4b+2]
        # Local C10 -> Global C[1, 2b] -> W[4b+1]
        # Local C11 -> Global C[1, 2b+1] -> W[4b+3]
        
        # Local B00 -> Global B[0, 2b] -> V[2b]
        # Local B01 -> Global B[0, 2b+1] -> V[2b+1]
        # Local B10 -> Global B[1, 2b] -> V[p + 2b]
        # Local B11 -> Global B[1, 2b+1] -> V[p + 2b+1]
        
        for u_dict, v_dict, w_dict in strassen_terms:
            for u_idx, val in u_dict.items():
                U[u_idx, col] += val
            for v_idx, val in v_dict.items():
                # v_idx is 0,1,2,3 for local B00, B01, B10, B11
                if v_idx == 0:
                    v_global = 2 * b
                elif v_idx == 1:
                    v_global = 2 * b + 1
                elif v_idx == 2:
                    v_global = p + 2 * b
                elif v_idx == 3:
                    v_global = p + 2 * b + 1
                V[v_global, col] += val
            for w_idx, val in w_dict.items():
                # w_idx is 0,1,2,3 for local C00, C01, C10, C11
                if w_idx == 0:
                    w_global = 4 * b
                elif w_idx == 1:
                    w_global = 4 * b + 1
                elif w_idx == 2:
                    w_global = 4 * b + 2
                elif w_idx == 3:
                    w_global = 4 * b + 3
                W[w_global, col] += val
            col += 1
            
    # Check time
    if time.time() - start_time > 110:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
