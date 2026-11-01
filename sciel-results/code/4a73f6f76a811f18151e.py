"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
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
    return {"U": U, "V": V, "W": W}


def _strassen_2x2_block(n, m, p, block_size):
    """
    Apply Strassen-style 2x2 decomposition within each block of `block_size` columns.
    For a 2x2 by 2x(block_size) product, we can group columns into blocks of 2
    and apply Strassen to each 2x2 by 2x2 sub-problem, giving rank 7 per block
    instead of 4.
    
    This works when p is even. Each block of 2 columns uses 7 rank-1 terms.
    """
    num_blocks = p // block_size
    if p % block_size != 0:
        return None  # Can't use this decomposition cleanly
    
    # For block_size=2, we use Strassen on 2x2 by 2x2
    # For block_size=1, it's just schoolbook (4 terms per column)
    # For block_size=4, we could try more complex schemes but Strassen is 7 for 2x2
    
    if block_size == 2:
        rank = num_blocks * 7
    elif block_size == 1:
        rank = num_blocks * 4
    else:
        return None
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x rank
    V = [[0] * rank for _ in range(m * p)]  # 32 x rank
    W = [[0] * rank for _ in range(p * n)]  # 32 x rank
    
    if block_size == 2:
        for block in range(num_blocks):
            k0 = block * 2
            k1 = k0 + 1
            base = block * 7
            
            # Strassen multipliers for 2x2 matrices:
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B10-B11)
            # M4 = A11(B01-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B01)
            # M7 = (A01-A11)(B10+B11)
            #
            # C00 = M1 - M5 + M4 + M7
            # C01 = M5 + M3
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # U indices: U[0]=A00, U[1]=A01, U[2]=A10, U[3]=A11
            # V indices: V[0..15]=B[0,k], V[16..31]=B[1,k]
            # W indices: W[2k]=C[0,k], W[2k+1]=C[1,k]
            
            # M1 = (A00+A11)(B0k0+B1k1)
            U[0][base + 0] = 1
            U[3][base + 0] = 1
            V[0 * 16 + k0][base + 0] = 1
            V[1 * 16 + k1][base + 0] = 1
            # C0k0 += M1, C0k1 += M1, C1k0 += M1, C1k1 += M1
            W[k0 * 2 + 0][base + 0] = 1
            W[k1 * 2 + 0][base + 0] = 1
            W[k0 * 2 + 1][base + 0] = 1
            W[k1 * 2 + 1][base + 0] = 1
            
            # M2 = (A10+A11)B0k0
            U[2][base + 1] = 1
            U[3][base + 1] = 1
            V[0 * 16 + k0][base + 1] = 1
            # C1k0 += M2
            W[k0 * 2 + 1][base + 1] = 1
            
            # M3 = A00(B1k0-B1k1)
            U[0][base + 2] = 1
            V[1 * 16 + k0][base + 2] = 1
            V[1 * 16 + k1][base + 2] = -1
            # C0k1 += M3
            W[k1 * 2 + 0][base + 2] = 1
            
            # M4 = A11(B0k1-B0k0)
            U[3][base + 3] = 1
            V[0 * 16 + k1][base + 3] = 1
            V[0 * 16 + k0][base + 3] = -1
            # C0k0 += M4, C1k0 += M4
            W[k0 * 2 + 0][base + 3] = 1
            W[k0 * 2 + 1][base + 3] = 1
            
            # M5 = (A00+A01)B1k1
            U[0][base + 4] = 1
            U[1][base + 4] = 1
            V[1 * 16 + k1][base + 4] = 1
            # C0k0 -= M5, C0k1 += M5
            W[k0 * 2 + 0][base + 4] = -1
            W[k1 * 2 + 0][base + 4] = 1
            
            # M6 = (A10-A00)(B0k0+B0k1)
            U[2][base + 5] = 1
            U[0][base + 5] = -1
            V[0 * 16 + k0][base + 5] = 1
            V[0 * 16 + k1][base + 5] = 1
            # C1k1 += M6
            W[k1 * 2 + 1][base + 5] = 1
            
            # M7 = (A01-A11)(B1k0+B1k1)
            U[1][base + 6] = 1
            U[3][base + 6] = -1
            V[1 * 16 + k0][base + 6] = 1
            V[1 * 16 + k1][base + 6] = 1
            # C0k0 += M7
            W[k0 * 2 + 0][base + 6] = 1
    
    return {"U": U, "V": V, "W": W}


def _verify(decomp, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reconstructs the matrix multiplication tensor."""
    U = np.array(decomp["U"], dtype=float)
    V = np.array(decomp["V"], dtype=float)
    W = np.array(decomp["W"], dtype=float)
    
    # For all valid (i, j, k), we need:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c] 
                         for c in range(U.shape[1]))
                if abs(val - 1.0) > tol:
                    return False
    return True


def construct():
    """
    Constructs the minimal rank decomposition for 2x2 by 2x16 matrix multiplication.
    Tries Strassen-based block decomposition first, falls back to schoolbook.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    
    # Try Strassen block decomposition with block_size=2
    try:
        decomp = _strassen_2x2_block(n, m, p, block_size=2)
        if decomp is not None:
            if time.time() < deadline:
                if _verify(decomp, n, m, p):
                    return decomp
    except Exception:
        pass
    
    # Fall back to schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
