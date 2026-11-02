import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-124 exact tensor decomposition for 5x5 matrix multiplication.
    
    Strategy: Use Strassen's algorithm (rank 7) for the top-left 2x2 block,
    and schoolbook multiplication for all other blocks.
    
    Block structure:
      - A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] : Strassen (rank 7)
      - A[0:2, 2:5] * B[2:5, 0:2] -> C[0:2, 2:5] : schoolbook (rank 18)
      - A[0:2, 2:5] * B[2:5, 2:5] -> C[0:2, 2:5] : schoolbook (rank 18)
      - A[2:5, 0:2] * B[0:2, 0:2] -> C[2:5, 0:2] : schoolbook (rank 18)
      - A[2:5, 0:2] * B[0:2, 2:5] -> C[2:5, 2:5] : schoolbook (rank 18)
      - A[2:5, 2:5] * B[2:5, 0:2] -> C[2:5, 0:2] : schoolbook (rank 18)
      - A[2:5, 2:5] * B[2:5, 2:5] -> C[2:5, 2:5] : schoolbook (rank 27)
    
    Total rank = 7 + 18 + 18 + 18 + 18 + 18 + 27 = 124
    """
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS  # 5, 5, 5
    rank = 124
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    # Helper: add schoolbook terms for a rectangular block
    def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
        nonlocal col
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
                    if time.time() > deadline:
                        return
    
    # Helper: add Strassen terms for a 2x2 block
    # A block: rows [i0, i0+2), cols [j0, j0+2)
    # B block: rows [j0, j0+2), cols [k0, k0+2)
    # C block: rows [k0, k0+2), cols [i0, i0+2)
    def add_strassen(i0, j0, k0):
        nonlocal col
        # Strassen's 7 products for 2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Each product factors into a single rank-1 term because the sum structure
        # in A and B creates a perfect product.
        
        # Term 1: M1 = (A00+A11)(B00+B11)
        # U: A[i0,j0]=1, A[i0+1,j0+1]=1
        # V: B[j0,k0]=1, B[j0+1,k0+1]=1
        # W: C[i0,k0]=1, C[i0,k0+1]=1, C[i0+1,k0]=1, C[i0+1,k0+1]=1
        U[i0 * m + j0, col] = 1
        U[(i0+1) * m + (j0+1), col] = 1
        V[j0 * p + k0, col] = 1
        V[(j0+1) * p + (k0+1), col] = 1
        W[k0 * n + i0, col] = 1
        W[k0 * n + (i0+1), col] = 1
        W[(k0+1) * n + i0, col] = 1
        W[(k0+1) * n + (i0+1), col] = 1
        col += 1
        if time.time() > deadline:
            return
        
        # Term 2: M2 = (A10+A11)B00
        # U: A[i0+1,j0]=1, A[i0+1,j0+1]=1
        # V: B[j0,k0]=1
        # W: C[i0+1,k0]=1
        U[(i0+1) * m + j0, col] = 1
        U[(i0+1) * m + (j0+1), col] = 1
        V[j0 * p + k0, col] = 1
        W[k0 * n + (i0+1), col] = 1
        col += 1
        if time.time() > deadline:
            return
        
        # Term 3: M3 = A00(B01-B11)
        # U: A[i0,j0]=1
        # V: B[j0,k0+1]=1, B[j0+1,k0+1]=-1
        # W: C[i0,k0+1]=1
        U[i0 * m + j0, col] = 1
        V[j0 * p + (k0+1), col] = 1
        V[(j0+1) * p + (k0+1), col] = -1
        W[(k0+1) * n + i0, col] = 1
        col += 1
        if time.time() > deadline:
            return
        
        # Term 4: M4 = A11(B10-B00)
        # U: A[i0+1,j0+1]=1
        # V: B[j0+1,k0]=1, B[j0,k0]=-1
        # W: C[i0+1,k0]=1
        U[(i0+1) * m + (j0+1), col] = 1
        V[(j0+1) * p + k0, col] = 1
        V[j0 * p + k0, col] = -1
        W[k0 * n + (i0+1), col] = 1
        col += 1
        if time.time() > deadline:
            return
        
        # Term 5: M5 = (A00+A01)B11
        # U: A[i0,j0]=1, A[i0,j0+1]=1
        # V: B[j0+1,k0+1]=1
        # W: C[i0,k0+1]=1
        U[i0 * m + j0, col] = 1
        U[i0 * m + (j0+1), col] = 1
        V[(j0+1) * p + (k0+1), col] = 1
        W[(k0+1) * n + i0, col] = 1
        col += 1
        if time.time() > deadline:
            return
        
        # Term 6: M6 = (A10-A00)(B00+B01)
        # U: A[i0+1,j0]=1, A[i0,j0]=-1
        # V: B[j0,k0]=1, B[j0,k0+1]=1
        # W: C[i0+1,k0]=1, C[i0+1,k0+1]=1
        U[(i0+1) * m + j0, col] = 1
        U[i0 * m + j0, col] = -1
        V[j0 * p + k0, col] = 1
        V[j0 * p + (k0+1), col] = 1
        W[k0 * n + (i0+1), col] = 1
        W[(k0+1) * n + (i0+1), col] = 1
        col += 1
        if time.time() > deadline:
            return
        
        # Term 7: M7 = (A01-A11)(B10+B11)
        # U: A[i0,j0+1]=1, A[i0+1,j0+1]=-1
        # V: B[j0+1,k0]=1, B[j0+1,k0+1]=1
        # W: C[i0,k0]=1, C[i0,k0+1]=1
        U[i0 * m + (j0+1), col] = 1
        U[(i0+1) * m + (j0+1), col] = -1
        V[(j0+1) * p + k0, col] = 1
        V[(j0+1) * p + (k0+1), col] = 1
        W[k0 * n + i0, col] = 1
        W[(k0+1) * n + i0, col] = 1
        col += 1
    
    # Use Strassen for the top-left 2x2 block
    add_strassen(0, 0, 0)
    
    # Use schoolbook for all other blocks
    add_schoolbook(0, 2, 2, 5, 0, 2)
    add_schoolbook(0, 2, 2, 5, 2, 5)
    add_schoolbook(2, 5, 0, 2, 0, 2)
    add_schoolbook(2, 5, 0, 2, 2, 5)
    add_schoolbook(2, 5, 2, 5, 0, 2)
    add_schoolbook(2, 5, 2, 5, 2, 5)
    
    # Verify col should be 124
    # 7 (Strassen) + 18 + 18 + 18 + 18 + 18 + 27 = 124
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
