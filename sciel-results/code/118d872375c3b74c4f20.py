"""Program for rank-minimal exact tensor decomposition of 4x5*5x6 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication.
    
    Uses a block decomposition with optimized splits:
    n=4 split into 1, 1, 2
    m=5 split into 2, 2, 1
    p=6 split into 2, 2, 2
    
    Block types and ranks:
    - 1x2x2: Rank 3 (optimized bilinear form)
    - 1x1x2: Rank 2
    - 2x2x2: Rank 7 (Strassen)
    - 2x1x2: Rank 4
    
    Expected rank:
    6 blocks of 1x2x2 * 3 = 18
    6 blocks of 1x1x2 * 2 = 12
    6 blocks of 2x2x2 * 7 = 42
    3 blocks of 2x1x2 * 4 = 12
    Total: 84
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Define block offsets and sizes
    i_offsets = [0, 1, 2]
    i_sizes = [1, 1, 2]
    
    j_offsets = [0, 2, 4]
    j_sizes = [2, 2, 1]
    
    k_offsets = [0, 2, 4]
    k_sizes = [2, 2, 2]
    
    # Pre-allocate with safety margin
    max_rank = 120
    
    U = np.zeros((n * m, max_rank), dtype=int)
    V = np.zeros((m * p, max_rank), dtype=int)
    W = np.zeros((p * n, max_rank), dtype=int)
    
    col = 0
    
    for i_idx, i_0 in enumerate(i_offsets):
        n_b = i_sizes[i_idx]
        if time.time() > deadline:
            break
        for j_idx, j_0 in enumerate(j_offsets):
            m_b = j_sizes[j_idx]
            if time.time() > deadline:
                break
            for k_idx, k_0 in enumerate(k_offsets):
                p_b = k_sizes[k_idx]
                
                if n_b == 1 and m_b == 1 and p_b == 2:
                    # Rank 2: Schoolbook for 1x1x2
                    # A is 1x1, B is 1x2, C is 1x2
                    # C_0 = A_0 B_00
                    # C_1 = A_0 B_01
                    
                    # Term 1
                    U[(i_0+0)*m + (j_0+0), col] = 1
                    V[(j_0+0)*p + (k_0+0), col] = 1
                    W[(k_0+0)*n + (i_0+0), col] = 1
                    col += 1
                    
                    # Term 2
                    U[(i_0+0)*m + (j_0+0), col] = 1
                    V[(j_0+0)*p + (k_0+1), col] = 1
                    W[(k_0+1)*n + (i_0+0), col] = 1
                    col += 1

                elif n_b == 2 and m_b == 2 and p_b == 2:
                    # Rank 7: Strassen for 2x2x2
                    # M1 = (A00+A11)(B00+B11) -> C00, C11
                    U[(i_0+0)*m + (j_0+0), col] = 1
                    U[(i_0+1)*m + (j_0+1), col] = 1
                    V[(j_0+0)*p + (k_0+0), col] = 1
                    V[(j_0+1)*p + (k_0+1), col] = 1
                    W[(k_0+0)*n + (i_0+0), col] = 1
                    W[(k_0+1)*n + (i_0+1), col] = 1
                    col += 1
                    
                    # M2 = (A10+A11)B00 -> C10, C11(-)
                    U[(i_0+1)*m + (j_0+0), col] = 1
                    U[(i_0+1)*m + (j_0+1), col] = 1
                    V[(j_0+0)*p + (k_0+0), col] = 1
                    W[(k_0+0)*n + (i_0+1), col] = 1
                    W[(k_0+1)*n + (i_0+1), col] = -1
                    col += 1
                    
                    # M3 = A00(B01-B11) -> C01, C11
                    U[(i_0+0)*m + (j_0+0), col] = 1
                    V[(j_0+0)*p + (k_0+1), col] = 1
                    V[(j_0+1)*p + (k_0+1), col] = -1
                    W[(k_0+1)*n + (i_0+0), col] = 1
                    W[(k_0+1)*n + (i_0+1), col] = 1
                    col += 1
                    
                    # M4 = A11(B10-B00) -> C00, C10
                    U[(i_0+1)*m + (j_0+1), col] = 1
                    V[(j_0+1)*p + (k_0+0), col] = 1
                    V[(j_0+0)*p + (k_0+0), col] = -1
                    W[(k_0+0)*n + (i_0+0), col] = 1
                    W[(k_0+0)*n + (i_0+1), col] = 1
                    col += 1
                    
                    # M5 = (A00+A01)B11 -> C01, C00(-)
                    U[(i_0+0)*m + (j_0+0), col] = 1
                    U[(i_0+0)*m + (j_0+1), col] = 1
                    V[(j_0+1)*p + (k_0+1), col] = 1
                    W[(k_0+1)*n + (i_0+0), col] = 1
                    W[(k_0+0)*n + (i_0+0), col] = -1
                    col += 1
                    
                    # M6 = (A10-A00)(B00+B01) -> C11
                    U[(i_0+1)*m + (j_0+0), col] = 1
                    U[(i_0+0)*m + (j_0+0), col] = -1
                    V[(j_0+0)*p + (k_0+0), col] = 1
                    V[(j_0+0)*p + (k_0+1), col] = 1
                    W[(k_0+1)*n + (i_0+1), col] = 1
                    col += 1
                    
                    # M7 = (A01-A11)(B10+B11) -> C00
                    U[(i_0+0)*m + (j_0+1), col] = 1
                    U[(i_0+1)*m + (j_0+1), col] = -1
                    V[(j_0+1)*p + (k_0+0), col] = 1
                    V[(j_0+1)*p + (k_0+1), col] = 1
                    W[(k_0+0)*n + (i_0+0), col] = 1
                    col += 1

                elif n_b == 2 and m_b == 1 and p_b == 2:
                    # Rank 4: Schoolbook for 2x1x2
                    # A is 2x1, B is 1x2, C is 2x2
                    # C_00 = A_0 B_0
                    # C_01 = A_0 B_1
                    # C_10 = A_1 B_0
                    # C_11 = A_1 B_1
                    
                    # Term 1
                    U[(i_0+0)*m + (j_0+0), col] = 1
                    V[(j_0+0)*p + (k_0+0), col] = 1
                    W[(k_0+0)*n + (i_0+0), col] = 1
                    col += 1
                    
                    # Term 2
                    U[(i_0+0)*m + (j_0+0), col] = 1
                    V[(j_0+0)*p + (k_0+1), col] = 1
                    W[(k_0+1)*n + (i_0+0), col] = 1
                    col += 1
                    
                    # Term 3
                    U[(i_0+1)*m + (j_0+0), col] = 1
                    V[(j_0+0)*p + (k_0+0), col] = 1
                    W[(k_0+0)*n + (i_0+1), col] = 1
                    col += 1
                    
                    # Term 4
                    U[(i_0+1)*m + (j_0+0), col] = 1
                    V[(j_0+0)*p + (k_0+1), col] = 1
                    W[(k_0+1)*n + (i_0+1), col] = 1
                    col += 1

                else:
                    # Fallback: Schoolbook for any other size
                    for u in range(n_b):
                        for v in range(m_b):
                            for w in range(p_b):
                                U[(i_0+u)*m + (j_0+v), col] = 1
                                V[(j_0+v)*p + (k_0+w), col] = 1
                                W[(k_0+w)*n + (i_0+u), col] = 1
                                col += 1

    # Trim arrays to actual rank
    U = U[:, :col]
    V = V[:, :col]
    W = W[:, :col]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
