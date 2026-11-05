"""Program for rank-minimal exact tensor decomposition of 4x5*5x6 matrix multiplication."""
import numpy as np
import time
import itertools

DIMENSIONS = [4, 5, 6]
TIME_LIMIT = 120.0

def _strassen_2x2x2():
    """Return Strassen's 7-term decomposition for 2x2x2 matrix multiplication."""
    # Returns list of (U_b, V_b, W_b) triples where:
    # U_b is 2x2 (coefficients for A), V_b is 2x2 (coefficients for B), W_b is 2x2 (coefficients for C)
    # W_b[k, i] is the coefficient of M_r in C[i, k]
    
    # M1 = (a11+a22)*(b11+b22)
    U1 = np.array([[1, 0], [0, 1]], dtype=float)
    V1 = np.array([[1, 1], [0, 0]], dtype=float)
    # C11 += M1, C22 += M1 -> W1[0,0]=1, W1[1,1]=1
    W1 = np.array([[1, 0], [0, 1]], dtype=float)
    
    # M2 = (a21+a22)*b11
    U2 = np.array([[0, 0], [1, 1]], dtype=float)
    V2 = np.array([[1, 0], [0, 0]], dtype=float)
    # C21 += M2, C22 -= M2 -> W2[0,1]=1, W2[1,1]=-1
    W2 = np.array([[0, 1], [-1, 0]], dtype=float)
    
    # M3 = a11*(b12-b22)
    U3 = np.array([[1, 0], [0, 0]], dtype=float)
    V3 = np.array([[0, 1], [0, -1]], dtype=float)
    # C12 += M3, C22 += M3 -> W3[1,0]=1, W3[1,1]=1
    W3 = np.array([[0, 0], [1, 1]], dtype=float)
    
    # M4 = a22*(b21-b11)
    U4 = np.array([[0, 0], [0, 1]], dtype=float)
    V4 = np.array([[-1, 0], [1, 0]], dtype=float)
    # C11 += M4, C21 += M4 -> W4[0,0]=1, W4[0,1]=1
    W4 = np.array([[1, 1], [0, 0]], dtype=float)
    
    # M5 = (a11+a12)*b22
    U5 = np.array([[1, 1], [0, 0]], dtype=float)
    V5 = np.array([[0, 0], [0, 1]], dtype=float)
    # C11 -= M5, C12 += M5 -> W5[0,0]=-1, W5[1,0]=1
    W5 = np.array([[-1, 0], [1, 0]], dtype=float)
    
    # M6 = (a21-a11)*(b11+b12)
    U6 = np.array([[-1, 0], [1, 0]], dtype=float)
    V6 = np.array([[1, 1], [0, 0]], dtype=float)
    # C22 += M6 -> W6[1,1]=1
    W6 = np.array([[0, 0], [0, 1]], dtype=float)
    
    # M7 = (a12-a22)*(b21+b22)
    U7 = np.array([[0, 1], [0, -1]], dtype=float)
    V7 = np.array([[0, 0], [1, 1]], dtype=float)
    # C11 += M7 -> W7[0,0]=1
    W7 = np.array([[1, 0], [0, 0]], dtype=float)
    
    return [(U1, V1, W1), (U2, V2, W2), (U3, V3, W3), 
            (U4, V4, W4), (U5, V5, W5), (U6, V6, W6), (U7, V7, W7)]

def _combine_blocks(block_decomp, n, m, p, block_n, block_m, block_p):
    """
    Combine block decomposition into full decomposition.
    
    block_decomp: list of (U_block, V_block, W_block) where each is a 2D array
                  representing a rank-1 contribution to the block multiplication
    n, m, p: full dimensions
    block_n, block_m, block_p: block dimensions
    
    Returns: (U, V, W) full matrices
    """
    # Number of blocks in each dimension
    blocks_n = n // block_n
    blocks_m = m // block_m
    blocks_p = p // block_p
    
    # Check if dimensions divide evenly
    if n % block_n != 0 or m % block_m != 0 or p % block_p != 0:
        return None
    
    block_rank = len(block_decomp)
    total_rank = block_rank * blocks_n * blocks_m * blocks_p
    
    # Initialize full matrices
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Fill in the block structure
    for bn in range(blocks_n):
        for bm in range(blocks_m):
            for bp in range(blocks_p):
                # Base column index for this specific block instance
                block_instance_idx = (bn * blocks_m + bm) * blocks_p + bp
                base_col = block_instance_idx * block_rank
                
                for r, (U_b, V_b, W_b) in enumerate(block_decomp):
                    global_col = base_col + r
                    
                    # A block (bn, bm)
                    for i in range(block_n):
                        for j in range(block_m):
                            a_idx = (bn * block_n + i) * m + (bm * block_m + j)
                            U[a_idx, global_col] = U_b[i, j]
                    
                    # B block (bm, bp)
                    for j in range(block_m):
                        for k in range(block_p):
                            b_idx = (bm * block_m + j) * p + (bp * block_p + k)
                            V[b_idx, global_col] = V_b[j, k]
                    
                    # C block (bp, bn)
                    for k in range(block_p):
                        for i in range(block_n):
                            c_idx = (bp * block_p + k) * n + (bn * block_n + i)
                            W[c_idx, global_col] = W_b[k, i]
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact using vectorized operations."""
    # The tensor T is defined such that T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
    # We want T to be 1 at indices corresponding to valid matrix multiplications and 0 elsewhere.
    # Valid indices:
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    # for i in [0, n), j in [0, m), k in [0, p)
    
    # We can compute the full tensor T efficiently?
    # T = np.einsum('ar,br,cr->abc', U, V, W)
    # This might be memory heavy for large dimensions, but 4x5x6 is small.
    # Dimensions: (20, 30, 24). 20*30*24 = 14400 entries. Very small.
    
    T = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Create a mask of valid positions
    # valid[a, b, c] is True if there exist i, j, k such that:
    # a = i*m + j, b = j*p + k, c = k*n + i
    
    # Instead of building a mask, we can just check the specific entries.
    # There are n*m*p = 120 valid entries.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if abs(T[a, b, c] - 1.0) > 1e-9:
                    return False
                
    # Check that all other entries are 0
    # We can zero out the valid entries and check if the rest is zero
    T_check = T.copy()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_check[a, b, c] = 0.0
                
    if np.any(np.abs(T_check) > 1e-9):
        return False
        
    return True

def construct():
    """
    Construct an exact tensor decomposition for 4x5 * 5x6 matrix multiplication.
    
    Uses block decomposition with Strassen's algorithm for sub-blocks.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try different block configurations
    best_result = None
    best_rank = n * m * p  # Schoolbook rank as baseline
    
    # First, try schoolbook as fallback
    U_school = np.zeros((n * m, best_rank))
    V_school = np.zeros((m * p, best_rank))
    W_school = np.zeros((p * n, best_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    if _verify_decomposition(U_school, V_school, W_school, n, m, p):
        best_result = (U_school, V_school, W_school)
    
    # Try block decompositions
    # Possible block sizes that divide 4, 5, 6
    block_sizes = [
        (2, 2, 2),  # 2x2x2 Strassen
        (2, 1, 2),  # 2x1x2
        (1, 2, 2),  # 1x2x2
        (2, 2, 1),  # 2x2x1
        (1, 1, 2),  # 1x1x2
        (1, 2, 1),  # 1x2x1
        (2, 1, 1),  # 2x1x1
        (1, 1, 1),  # 1x1x1 (schoolbook)
    ]
    
    for bn, bm, bp in block_sizes:
        # Time check
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        
        if n % bn != 0 or m % bm != 0 or p % bp != 0:
            continue
        
        # Get the block decomposition
        block_decomp = []
        
        if bn == 2 and bm == 2 and bp == 2:
            # Use Strassen's 7-term decomposition
            block_decomp = _strassen_2x2x2()
        else:
            # For smaller blocks, use schoolbook within each block
            for i in range(bn):
                for j in range(bm):
                    for k in range(bp):
                        U_b = np.zeros((bn, bm))
                        V_b = np.zeros((bm, bp))
                        W_b = np.zeros((bp, bn))
                        U_b[i, j] = 1
                        V_b[j, k] = 1
                        W_b[k, i] = 1
                        block_decomp.append((U_b, V_b, W_b))
        
        result = _combine_blocks(block_decomp, n, m, p, bn, bm, bp)
        if result is None:
            continue
        
        U, V, W = result
        if _verify_decomposition(U, V, W, n, m, p):
            current_rank = U.shape[1]
            if current_rank < best_rank:
                best_rank = current_rank
                best_result = (U, V, W)
    
    if best_result is None:
        # Fallback to schoolbook
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
