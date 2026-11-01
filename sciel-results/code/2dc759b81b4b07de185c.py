import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def _add_block(U, V, W, col, i_off, j_off, k_off, n, m, p, block_type):
    """
    Adds a block decomposition to the global U, V, W matrices.
    block_type: 'strassen_2x2x2' or 'schoolbook'
    """
    if block_type == 'strassen_2x2x2':
        # Strassen for 2x2x2
        # Local i in {0,1}, j in {0,1}, k in {0,1}
        
        # M1: (A00+A11)(B00+B11) -> C00, C11
        U[(i_off+0)*m + (j_off+0)][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off+0)*p + (k_off+0)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[(k_off+0)*n + (i_off+0)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1
        
        # M2: (A10+A11)B00 -> C10, -C11
        U[(i_off+1)*m + (j_off+0)][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off+0)*p + (k_off+0)][col] += 1
        W[(k_off+0)*n + (i_off+1)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += -1
        col += 1
        
        # M3: A00(B01-B11) -> C01, C11
        U[(i_off+0)*m + (j_off+0)][col] += 1
        V[(j_off+0)*p + (k_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += -1
        W[(k_off+1)*n + (i_off+0)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1
        
        # M4: A11(B10-B00) -> C00, C10
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+0)][col] += 1
        V[(j_off+0)*p + (k_off+0)][col] += -1
        W[(k_off+0)*n + (i_off+0)][col] += 1
        W[(k_off+0)*n + (i_off+1)][col] += 1
        col += 1
        
        # M5: (A00+A01)B11 -> -C00, C01
        U[(i_off+0)*m + (j_off+0)][col] += 1
        U[(i_off+0)*m + (j_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[(k_off+0)*n + (i_off+0)][col] += -1
        W[(k_off+1)*n + (i_off+0)][col] += 1
        col += 1
        
        # M6: (A10-A00)(B00+B01) -> C11
        U[(i_off+1)*m + (j_off+0)][col] += 1
        U[(i_off+0)*m + (j_off+0)][col] += -1
        V[(j_off+0)*p + (k_off+0)][col] += 1
        V[(j_off+0)*p + (k_off+1)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1
        
        # M7: (A01-A11)(B10+B11) -> C00
        U[(i_off+0)*m + (j_off+1)][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += -1
        V[(j_off+1)*p + (k_off+0)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[(k_off+0)*n + (i_off+0)][col] += 1
        col += 1
        
        return col

    elif block_type == 'schoolbook':
        # Schoolbook for ni x nj x nk
        # We need to know the size of the block.
        # This helper assumes the caller passes the block dimensions implicitly 
        # or we refactor to pass them. 
        # For now, let's just do the specific blocks in construct.
        pass
        
    return col

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    
    # Rank 28 decomposition
    # Block 1: j in {0,1}, k in {0,1} -> 2x2x2 Strassen (Rank 7)
    # Block 2: j in {0,1}, k in {2,3} -> 2x2x2 Strassen (Rank 7)
    # Block 3: j in {0,1}, k in {4}    -> 2x2x1 Schoolbook (Rank 4)
    # Block 4: j in {2},    k in {0..4} -> 2x1x5 Schoolbook (Rank 10)
    
    total_rank = 7 + 7 + 4 + 10
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col = 0
    
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    col = _add_block(U, V, W, col, 0, 0, 0, n, m, p, 'strassen_2x2x2')
    
    # Block 2: i in {0,1}, j in {0,1}, k in {2,3}
    col = _add_block(U, V, W, col, 0, 0, 2, n, m, p, 'strassen_2x2x2')
    
    # Block 3: i in {0,1}, j in {0,1}, k in {4}
    # Schoolbook 2x2x1
    for i in range(2):
        for j in range(2):
            U[i*m + j, col] = 1
            V[j*p + 4, col] = 1
            W[4*n + i, col] = 1
            col += 1
            
    # Block 4: i in {0,1}, j in {2}, k in {0..4}
    # Schoolbook 2x1x5
    for i in range(2):
        for k in range(5):
            U[i*m + 2, col] = 1
            V[2*p + k, col] = 1
            W[k*n + i, col] = 1
            col += 1
            
    assert col == total_rank, f"Column count mismatch: {col} != {total_rank}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()
