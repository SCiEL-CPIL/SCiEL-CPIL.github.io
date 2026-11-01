import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 140.0  # Safety margin before 150s kill

def _check_deadline():
    return (time.time() - _START_TIME) >= _DEADLINE

def construct():
    n, m, p = 2, 2, 8
    
    # Strassen factors for 2x2 matrix multiplication
    # U: shape (4, 7) - rows correspond to A indices (0:00, 1:01, 2:10, 3:11)
    # V: shape (4, 7) - rows correspond to B indices (0:00, 1:01, 2:10, 3:11)
    # W: shape (4, 7) - rows correspond to C indices (0:00, 1:01, 2:10, 3:11)
    
    # Standard Strassen decomposition
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # U_s: rows are A indices, cols are P terms
    U_s = np.array([
        [1, 0, 1, 0, 1, -1, 0],  # A00: P1, P3, P5, P6
        [0, 0, 0, 0, 1, 0, 1],   # A01: P5, P7
        [0, 1, 0, 0, 0, 1, 0],   # A10: P2, P6
        [1, 1, 0, 1, 0, 0, -1]   # A11: P1, P2, P4, P7
    ], dtype=int)
    
    # V_s: rows are B indices, cols are P terms
    V_s = np.array([
        [1, 1, 0, -1, 0, 1, 0],  # B00: P1, P2, P4, P6
        [0, 0, 1, 0, 0, 1, 0],   # B01: P3, P6
        [0, 0, 0, 1, 0, 0, 1],   # B10: P4, P7
        [1, 0, -1, 0, 1, 0, 1]   # B11: P1, P3, P5, P7
    ], dtype=int)
    
    # W_s: rows are C indices, cols are P terms
    W_s = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # C00: P1, P4, P5, P7
        [0, 0, 1, 0, 1, 0, 0],   # C01: P3, P5
        [0, 1, 0, -1, 0, 0, 0],  # C10: P2, P4
        [1, -1, 0, 0, 0, 1, 0]   # C11: P1, P2, P6
    ], dtype=int)
    
    num_blocks = p // 2
    rank = num_blocks * 7
    
    # Initialize matrices
    # U: (n*m, rank) = (4, 28)
    # V: (m*p, rank) = (16, 28)
    # W: (p*n, rank) = (16, 28)
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col_offset = 0
    for t in range(num_blocks):
        # U is the same for all blocks
        U[:, col_offset:col_offset+7] = U_s.T
        
        # V mapping for block t:
        # Local B indices 0,1,2,3 map to global B indices:
        # B is (m, p) = (2, 8). Row-major? 
        # Interface says V is (m*p, rank). 
        # Usually indices for B are j*p + k? Or k*m + j?
        # Let's check the standard convention for these tasks.
        # "V: (m*p, rank)".
        # If B is stored row-major, index is j*p + k.
        # If B is stored col-major, index is k*m + j.
        # The task description says "output factor follows ... W[k*n+i,r]".
        # This implies C is stored with k (col) as major index? 
        # W index = k*n + i. Since n=2, index = 2k + i.
        # This is column-major for C (i varies fastest).
        # By symmetry/consistency, B (m x p) likely uses column-major too?
        # Or does it follow the same "row * cols + col" pattern?
        # For W: index = col * rows + row.
        # For V: index = ? 
        # Let's look at the previous successful code's logic for V.
        # Previous code: v_indices = [2*t, 2*t+1, 8+2*t, 8+2*t+1]
        # m=2, p=8.
        # If index = col * m + row:
        # Col 2t, Row 0 -> 2t*2 + 0 = 4t.
        # Col 2t, Row 1 -> 2t*2 + 1 = 4t+1.
        # This doesn't match the previous code's [2t, 2t+1, ...].
        
        # Let's re-read the previous code's V mapping.
        # v_indices = [2*t, 2*t+1, 8+2*t, 8+2*t+1]
        # This looks like index = row * p + col.
        # Row 0, Col 2t -> 0*8 + 2t = 2t.
        # Row 0, Col 2t+1 -> 0*8 + 2t+1 = 2t+1.
        # Row 1, Col 2t -> 1*8 + 2t = 8+2t.
        # Row 1, Col 2t+1 -> 1*8 + 2t+1 = 8+2t+1.
        
        # So V uses row-major (row * p + col).
        # And W uses col-major (col * n + row) based on "W[k*n+i,r]".
        # This is a bit inconsistent but we must follow the contract.
        # Contract: "output factor follows the official trace convention W[k*n+i,r]".
        # It does NOT specify the convention for V explicitly, but the seed/previous 
        # successful program (score 1.1429) used row-major for V.
        # I will stick to the convention that worked: V is row-major (row * p + col).
        
        v_indices = [0*p + 2*t, 0*p + 2*t+1, 1*p + 2*t, 1*p + 2*t+1]
        # v_indices = [2t, 2t+1, 8+2t, 8+2t+1]
        
        # W mapping for block t:
        # Convention W[k*n + i, r] where k is output col, i is output row.
        # Local C indices 0,1,2,3 (00, 01, 10, 11 in 2x2 block)
        # 0: c[0][2t] -> i=0, k=2t -> global idx 2t*2 + 0 = 4t
        # 1: c[0][2t+1] -> i=0, k=2t+1 -> global idx (2t+1)*2 + 0 = 4t+2
        # 2: c[1][2t] -> i=1, k=2t -> global idx 2t*2 + 1 = 4t+1
        # 3: c[1][2t+1] -> i=1, k=2t+1 -> global idx (2t+1)*2 + 1 = 4t+3
        
        w_indices = [4*t, 4*t+2, 4*t+1, 4*t+3]
        
        for r in range(7):
            for l in range(4):
                V[v_indices[l], col_offset + r] = V_s[l, r]
                W[w_indices[l], col_offset + r] = W_s[l, r]
                
        col_offset += 7
        
    if _check_deadline():
        # This shouldn't happen as construction is O(1), but guard just in case
        pass

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
