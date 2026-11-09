"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]
import time
import random
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START

def _validate(U, V, W, n, m, p):
    """Check that U, V, W form a valid tensor decomposition."""
    rank = len(U[0])
    rng = np.random.RandomState(42)
    A = rng.randn(n, m)
    B = rng.randn(m, p)
    AB = A @ B
    
    result = np.zeros((n, p))
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for c in range(rank):
        U_c = U_arr[:, c].reshape(n, m)
        V_c = V_arr[:, c].reshape(m, p)
        W_c = W_arr[:, c].reshape(p, n)
        scalar_U = np.sum(A * U_c)
        scalar_V = np.sum(B * V_c)
        result += scalar_U * scalar_V * W_c.T
    
    if not np.allclose(result, AB, atol=1e-8):
        return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2_block(A11, A12, A21, A22, B11, B12, B21, B22):
    """
    Compute C = A*B where A and B are 2x2 block matrices.
    Returns 8 Strassen products.
    Each product is a block matrix product.
    """
    M1 = (A11 + A22) @ (B11 + B22)
    M2 = (A21 + A22) @ B11
    M3 = A11 @ (B12 - B22)
    M4 = A22 @ (B21 - B11)
    M5 = (A11 + A12) @ B22
    M6 = (A21 - A11) @ (B11 + B12)
    M7 = (A12 - A22) @ (B21 + B22)
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    return C11, C12, C21, C22

def _build_strassen_decomposition(n, m, p):
    """
    Build a Strassen-based decomposition for n×m × m×p matrix multiplication.
    Strategy: Partition into 2×2 blocks where possible, use Strassen for 2×2 subproblems.
    For 3×4×6: Pad to 4×4×6 or use mixed approach.
    
    Actually, let's use a more direct approach:
    - Split A (3×4) into blocks: top 2 rows and bottom 1 row
    - Split B (4×6) into blocks: left 2 cols and right 2 cols (wait, 4 cols total)
    - This gets complicated. Let me try a different approach.
    
    Alternative: Use the fact that we can decompose the problem into smaller multiplications.
    For 3×4×6, we can write it as a combination of smaller matrix products.
    
    Let's try: 
    A = [[A11, A12], [A21, A22]] where A11 is 2×2, A12 is 2×2, A21 is 1×2, A22 is 1×2
    B = [[B11, B12], [B21, B22]] where B11 is 2×3, B12 is 2×3, B21 is 2×3, B22 is 2×3
    
    Then C = AB has blocks:
    C11 = A11@B11 + A12@B21  (2×3)
    C12 = A11@B12 + A12@B22  (2×3)
    C21 = A21@B11 + A22@B21  (1×3)
    C22 = A21@B12 + A22@B22  (1×3)
    
    This doesn't directly give us Strassen savings. Let me think differently.
    
    Actually, the cleanest approach: pad 3×4×6 to 4×4×8 and use recursive Strassen,
    but that increases the problem size. 
    
    Better: Use a hybrid approach where we handle the 3×4×6 directly by finding
    a decomposition that exploits the structure.
    
    Let me try a simpler approach: just return schoolbook for now and add a note
    that a proper Strassen implementation would be needed.
    """
    # For now, return schoolbook as a baseline
    return _schoolbook(n, m, p)

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix by a 4 by 6 matrix.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    sb = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_decomp = sb
    
    # Try Strassen-based approach
    try:
        strassen_decomp = _build_strassen_decomposition(n, m, p)
        strassen_rank = len(strassen_decomp['U'][0])
        if strassen_rank < best_rank and _validate(strassen_decomp['U'], strassen_decomp['V'], strassen_decomp['W'], n, m, p):
            best_rank = strassen_rank
            best_decomp = strassen_decomp
    except Exception:
        pass
    
    return best_decomp

# EVOLVE-BLOCK-END
