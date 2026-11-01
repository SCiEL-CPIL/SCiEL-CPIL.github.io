"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """
    Return the Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U, V, W matrices of shape (4,7) for a single 2x2x2 block.
    U indexed by [i*m+j], V by [j*p+k], W by [k*n+i]
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # M1 = (a00 + a11) * (b00 + b11)
    # c00 += M1, c11 += M1
    U[0, 0] = 1.0  # a00
    U[3, 0] = 1.0  # a11
    V[0, 0] = 1.0  # b00
    V[3, 0] = 1.0  # b11
    W[0, 0] = 1.0  # c00 (k=0, i=0)
    W[3, 0] = 1.0  # c11 (k=1, i=1)
    
    # M2 = (a10 + a11) * b00
    # c10 += M2, c11 -= M2
    U[2, 1] = 1.0  # a10
    U[3, 1] = 1.0  # a11
    V[0, 1] = 1.0  # b00
    W[2, 1] = 1.0  # c10 (k=0, i=1)
    W[3, 1] = -1.0 # c11 (k=1, i=1)
    
    # M3 = a00 * (b10 - b11)
    # c01 += M3, c11 += M3
    U[0, 2] = 1.0  # a00
    V[2, 2] = 1.0  # b10
    V[3, 2] = -1.0 # b11
    W[1, 2] = 1.0  # c01 (k=1, i=0)
    W[3, 2] = 1.0  # c11 (k=1, i=1)
    
    # M4 = a11 * (b01 - b00)
    # c00 -= M4, c10 += M4
    U[3, 3] = 1.0  # a11
    V[1, 3] = 1.0  # b01
    V[0, 3] = -1.0 # b00
    W[0, 3] = -1.0 # c00 (k=0, i=0)
    W[2, 3] = 1.0  # c10 (k=0, i=1)
    
    # M5 = (a00 + a01) * b11
    # c00 += M5, c01 += M5
    U[0, 4] = 1.0  # a00
    U[1, 4] = 1.0  # a01
    V[3, 4] = 1.0  # b11
    W[0, 4] = 1.0  # c00 (k=0, i=0)
    W[1, 4] = 1.0  # c01 (k=1, i=0)
    
    # M6 = (a10 - a00) * (b00 + b10)
    # c11 += M6
    U[2, 5] = 1.0  # a10
    U[0, 5] = -1.0 # a00
    V[0, 5] = 1.0  # b00
    V[2, 5] = 1.0  # b10
    W[3, 5] = 1.0  # c11 (k=1, i=1)
    
    # M7 = (a01 - a11) * (b01 + b11)
    # c00 -= M7
    U[1, 6] = 1.0  # a01
    U[3, 6] = -1.0 # a11
    V[1, 6] = 1.0  # b01
    V[3, 6] = 1.0  # b11
    W[0, 6] = -1.0 # c00 (k=0, i=0)
    
    return U, V, W

def _schoolbook_2x2x1():
    """
    Return schoolbook decomposition for 2x2x1 matrix multiplication.
    U shape (4, 4), V shape (2, 4), W shape (2, 4)
    U indexed by [i*m+j], V by [j*p+k] where p=1 so just [j], W by [k*n+i] where k=0 so just [i]
    """
    U = np.zeros((4, 4), dtype=np.float64)
    V = np.zeros((2, 4), dtype=np.float64)
    W = np.zeros((2, 4), dtype=np.float64)
    col = 0
    for i in range(2):
        for j in range(2):
            U[i * 2 + j, col] = 1.0
            V[j, col] = 1.0
            W[i, col] = 1.0
            col += 1
    return U, V, W

def _block_decomposition(n, m, p):
    """
    Construct a block decomposition for M_{2,2,p}.
    Uses Strassen for 2x2 blocks and Schoolbook for 1x1 blocks.
    p=15: 7 blocks of 2, 1 block of 1. Total rank = 7*7 + 4 = 53.
    """
    if n != 2 or m != 2:
        return None
        
    # Determine block sizes
    block_sizes = []
    remaining = p
    while remaining > 0:
        if remaining >= 2:
            block_sizes.append(2)
            remaining -= 2
        else:
            block_sizes.append(1)
            remaining -= 1
            
    num_blocks = len(block_sizes)
    
    # Precompute Strassen and Schoolbook 1x1 decompositions
    U_s, V_s, W_s = _strassen_2x2x2()
    U_1, V_1, W_1 = _schoolbook_2x2x1()
    
    # Calculate total rank
    total_rank = 0
    for bs in block_sizes:
        if bs == 2:
            total_rank += 7
        elif bs == 1:
            total_rank += 4
            
    U = np.zeros((n * m, total_rank), dtype=np.float64)
    V = np.zeros((m * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)
    
    col_offset = 0
    k_offset = 0
    
    for bs in block_sizes:
        if bs == 2:
            for r in range(7):
                # U: same for all k in this block
                U[0, col_offset] = U_s[0, r]  # a00
                U[1, col_offset] = U_s[1, r]  # a01
                U[2, col_offset] = U_s[2, r]  # a10
                U[3, col_offset] = U_s[3, r]  # a11
                
                # V: for j=0,1 and k=k_offset, k_offset+1
                # V index: j * p + k
                V[0 * p + k_offset, col_offset] = V_s[0, r]   # b0, k0
                V[0 * p + k_offset + 1, col_offset] = V_s[1, r] # b0, k1
                V[1 * p + k_offset, col_offset] = V_s[2, r]   # b1, k0
                V[1 * p + k_offset + 1, col_offset] = V_s[3, r] # b1, k1
                
                # W: for k=k_offset, k_offset+1 and i=0,1
                # W index: k * n + i
                W[k_offset * n + 0, col_offset] = W_s[0, r]   # c00 (k=k0, i=0)
                W[k_offset * n + 1, col_offset] = W_s[1, r]   # c01 (k=k0, i=1)
                W[(k_offset + 1) * n + 0, col_offset] = W_s[2, r] # c10 (k=k1, i=0)
                W[(k_offset + 1) * n + 1, col_offset] = W_s[3, r] # c11 (k=k1, i=1)
                
                col_offset += 1
            k_offset += 2
        elif bs == 1:
            for r in range(4):
                # U: a00, a01, a10, a11
                U[0, col_offset] = U_1[0, r]  # a00
                U[1, col_offset] = U_1[1, r]  # a01
                U[2, col_offset] = U_1[2, r]  # a10
                U[3, col_offset] = U_1[3, r]  # a11
                
                # V: b0, b1
                V[0 * p + k_offset, col_offset] = V_1[0, r]  # b0
                V[1 * p + k_offset, col_offset] = V_1[1, r]  # b1
                
                # W: c0, c1
                W[k_offset * n + 0, col_offset] = W_1[0, r]  # c0
                W[k_offset * n + 1, col_offset] = W_1[1, r]  # c1
                
                col_offset += 1
            k_offset += 1
            
    return U, V, W

def _is_valid(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition exactly decomposes the matrix multiplication tensor."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(U.shape[1]):
                    u_val = U[i * m + j, c]
                    v_val = V[j * p + k, c]
                    w_val = W[k * n + i, c]
                    if u_val != 0 and v_val != 0 and w_val != 0:
                        val += u_val * v_val * w_val
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the standard rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Try block decomposition first
    U, V, W = _block_decomposition(n, m, p)
    
    if U is not None and _is_valid(U, V, W, n, m, p):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
        
    # Fallback to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
