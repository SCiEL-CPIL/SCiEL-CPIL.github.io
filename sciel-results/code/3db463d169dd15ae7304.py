"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, rank):
    """Check if the decomposition is exact."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    if U_arr.shape != (n*m, rank) or V_arr.shape != (m*p, rank) or W_arr.shape != (p*n, rank):
        return False
    
    # Build target tensor
    target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Reconstruct
    recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr, optimize=True)
    
    if not np.allclose(recon, target, atol=1e-9, rtol=1e-9):
        return False
    
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W, rank

def _try_block_decomposition(n, m, p):
    """
    Try to decompose n×m × m×p using block structure.
    For 4×5 × 5×6, try splitting and using sub-decompositions.
    """
    # Try splitting 4=2+2, 5=2+3, 6=3+3
    # This gives 4 block products, each of which is a sum of 2 small matmuls
    # Use schoolbook for each sub-matmul
    
    n1, n2 = 2, 2
    m1, m2 = 2, 3
    p1, p2 = 3, 3
    
    # Sub-problems:
    # C11 = A11*B11 + A12*B21: (2x2)*(2x3) + (2x3)*(3x3)
    # C12 = A11*B12 + A12*B22: (2x2)*(2x3) + (2x3)*(3x3)
    # C21 = A21*B11 + A22*B21: (2x2)*(2x3) + (2x3)*(3x3)
    # C22 = A21*B12 + A22*B22: (2x2)*(2x3) + (2x3)*(3x3)
    
    # Each block product has rank: schoolbook for each sub-matmul
    # (2x2)*(2x3): rank 12
    # (2x3)*(3x3): rank 18
    # Total per block: 30, 4 blocks: 120. No improvement.
    
    # Try a different split: 4=2+2, 5=3+2, 6=2+4
    # This also gives 120 with schoolbook.
    
    # Without Strassen on equal-sized blocks, we can't improve with simple blocking.
    # Return None to indicate no improvement found.
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # Safety margin
    
    # Start with schoolbook
    U_sb, V_sb, W_sb, rank_sb = _schoolbook(n, m, p)
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, rank_sb
    
    # Try block decomposition
    block_result = _try_block_decomposition(n, m, p)
    if block_result is not None:
        U_b, V_b, W_b, rank_b = block_result
        if _verify(U_b, V_b, W_b, n, m, p, rank_b):
            if rank_b < best_rank:
                best_U, best_V, best_W, best_rank = U_b, V_b, W_b, rank_b
    
    # Return the best decomposition found
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
