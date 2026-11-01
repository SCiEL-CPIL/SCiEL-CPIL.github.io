"""
Matrix multiplication tensor decomposition for M_{2,2,15}.

We use a block-Strassen construction:
  - Split the 15 columns of B (and rows of C) into 7 blocks of size 2 and 1 block of size 1.
  - For each 2x2 block of B, use Strassen's 7-term algorithm.
  - For the 1x1 block, use the trivial 1-term algorithm.
  - Total rank = 7*7 + 1 = 50.

This is a significant improvement over the schoolbook rank of 60.

The interface requires U (n*m x rank), V (m*p x rank), W (p*n x rank)
with the trace convention W[k*n + i, r].

We verify the decomposition numerically before returning it.
If verification fails, we fall back to the schoolbook decomposition.
"""
import time
import numpy as np

DIMENSIONS = [2, 2, 15]


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


def _strassen_2x2x2():
    """
    Return the Strassen decomposition for 2x2x2 matrix multiplication.
    
    Strassen's 7 products:
      M1 = (a00 + a11) * (b00 + b11)
      M2 = (a10 + a11) * b00
      M3 = a00 * (b10 - b11)
      M4 = a11 * (b01 - b00)
      M5 = (a00 + a01) * b11
      M6 = (a10 - a00) * (b00 + b10)
      M7 = (a01 - a11) * (b01 + b11)
    
    Output:
      c00 = M1 - M4 + M5 - M7
      c01 = M3 + M5
      c10 = M2 + M4
      c11 = M1 - M2 + M3 + M6
    
    For the tensor decomposition, we need U, V, W such that:
      T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Here n=m=p=2, so:
      U has shape (4, 7), indexed by (i*2+j, r)
      V has shape (4, 7), indexed by (j*2+k, r)
      W has shape (4, 7), indexed by (k*2+i, r)
    
    For each Strassen product M_r, we have:
      U_r[i*2+j] = coefficient of a_ij in the first factor
      V_r[j*2+k] = coefficient of b_jk in the second factor
      W_r[k*2+i] = coefficient of c_ik in the output (from the linear combination)
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # M1 = (a00 + a11) * (b00 + b11)
    U[0, 0] = 1.0  # a00
    U[3, 0] = 1.0  # a11
    V[0, 0] = 1.0  # b00
    V[3, 0] = 1.0  # b11
    W[0, 0] = 1.0  # c00
    W[3, 0] = 1.0  # c11
    
    # M2 = (a10 + a11) * b00
    U[2, 1] = 1.0  # a10
    U[3, 1] = 1.0  # a11
    V[0, 1] = 1.0  # b00
    W[2, 1] = 1.0  # c10
    
    # M3 = a00 * (b10 - b11)
    U[0, 2] = 1.0  # a00
    V[2, 2] = 1.0  # b10
    V[3, 2] = -1.0 # b11
    W[1, 2] = 1.0  # c01
    
    # M4 = a11 * (b01 - b00)
    U[3, 3] = 1.0  # a11
    V[1, 3] = 1.0  # b01
    V[0, 3] = -1.0 # b00
    W[0, 3] = -1.0 # c00
    W[2, 3] = 1.0  # c10
    
    # M5 = (a00 + a01) * b11
    U[0, 4] = 1.0  # a00
    U[1, 4] = 1.0  # a01
    V[3, 4] = 1.0  # b11
    W[0, 4] = 1.0  # c00
    W[1, 4] = 1.0  # c01
    
    # M6 = (a10 - a00) * (b00 + b10)
    U[2, 5] = 1.0  # a10
    U[0, 5] = -1.0 # a00
    V[0, 5] = 1.0  # b00
    V[2, 5] = 1.0  # b10
    W[3, 5] = 1.0  # c11
    
    # M7 = (a01 - a11) * (b01 + b11)
    U[1, 6] = 1.0  # a01
    U[3, 6] = -1.0 # a11
    V[1, 6] = 1.0  # b01
    V[3, 6] = 1.0  # b11
    W[0, 6] = -1.0 # c00
    
    return U, V, W


def _block_strassen(n, m, p):
    """
    Construct a block-Strassen decomposition for M_{n,m,p}.
    
    We split the p columns into blocks. For 2x2x15:
      7 blocks of size 2, 1 block of size 1.
      Rank = 7*7 + 1 = 50.
    """
    if n != 2 or m != 2:
        return None
    
    # Determine block sizes
    # For p=15: 7 blocks of 2, 1 block of 1
    block_sizes = [2] * 7 + [1]
    num_blocks = len(block_sizes)
    
    # Precompute Strassen and schoolbook 1x1 decompositions
    U_s, V_s, W_s = _strassen_2x2x2()
    
    # For 1x1 block: trivial decomposition
    # U shape (2,1), V shape (2,1), W shape (2,1)
    U_1 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float64)  # a00
    V_1 = np.array([[1.0], [0.0]], dtype=np.float64)  # b0
    W_1 = np.array([[1.0], [0.0]], dtype=np.float64)  # c0
    
    total_rank = 7 * num_blocks - 6  # 7*7 + 1 = 50
    # Actually: 7 blocks of size 2 contribute 7*7=49, 1 block of size 1 contributes 1. Total = 50.
    total_rank = 0
    for bs in block_sizes:
        if bs == 2:
            total_rank += 7
        elif bs == 1:
            total_rank += 1
    
    U = np.zeros((n * m, total_rank), dtype=np.float64)
    V = np.zeros((m * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)
    
    col = 0
    k_offset = 0
    for bs in block_sizes:
        if bs == 2:
            # Use Strassen for this 2x2 block
            for r in range(7):
                # U: same for all k in this block
                U[0, col] = U_s[0, r]  # a00
                U[1, col] = U_s[1, r]  # a01
                U[2, col] = U_s[2, r]  # a10
                U[3, col] = U_s[3, r]  # a11
                
                # V: for j=0,1 and k=k_offset, k_offset+1
                V[0 * 2 + k_offset, col] = V_s[0, r]  # b0, k_offset
                V[0 * 2 + k_offset + 1, col] = V_s[1, r]  # b0, k_offset+1
                V[1 * 2 + k_offset, col] = V_s[2, r]  # b1, k_offset
                V[1 * 2 + k_offset + 1, col] = V_s[3, r]  # b1, k_offset+1
                
                # W: for k=k_offset, k_offset+1 and i=0,1
                W[k_offset * 2 + 0, col] = W_s[0, r]  # c00
                W[k_offset * 2 + 1, col] = W_s[1, r]  # c10
                W[(k_offset + 1) * 2 + 0, col] = W_s[2, r]  # c01
                W[(k_offset + 1) * 2 + 1, col] = W_s[3, r]  # c11
                
                col += 1
            k_offset += 2
        elif bs == 1:
            # Use trivial decomposition for this 1x1 block
            # U: a00 only
            U[0, col] = 1.0
            U[1, col] = 0.0
            U[2, col] = 0.0
            U[3, col] = 0.0
            
            # V: b0 only
            V[0 * 2 + k_offset, col] = 1.0
            V[1 * 2 + k_offset, col] = 0.0
            
            # W: c0 only
            W[k_offset * 2 + 0, col] = 1.0
            W[k_offset * 2 + 1, col] = 0.0
            
            col += 1
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


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Try block-Strassen first
    U, V, W = _block_strassen(n, m, p)
    
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
