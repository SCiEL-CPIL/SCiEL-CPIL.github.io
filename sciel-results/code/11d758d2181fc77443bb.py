"""Combined program using Strassen blocks for 2x2x15 matrix multiplication."""
import time
import numpy as np
from scipy.linalg import qr

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def _build_strassen_2x2x2():
    """
    Build Strassen's rank-7 decomposition for 2x2 x 2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) following the trace convention W[k*n+i, r].
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Index mapping: U[i*m + j, r], V[j*p + k, r], W[k*n + i, r]
    
    # Term 1: M1 = (A00 + A11) * (B00 + B11)
    U[0*2+0, 0] = 1  # A00
    U[1*2+1, 0] = 1  # A11
    V[0*2+0, 0] = 1  # B00
    V[1*2+1, 0] = 1  # B11
    W[0*2+0, 0] = 1  # C00 (k=0, i=0)
    W[1*2+1, 0] = 1  # C11 (k=1, i=1)
    
    # Term 2: M2 = (A10 + A11) * B00
    U[1*2+0, 1] = 1  # A10
    U[1*2+1, 1] = 1  # A11
    V[0*2+0, 1] = 1  # B00
    W[0*2+1, 1] = 1  # C10 (k=0, i=1)
    W[1*2+1, 1] = -1 # C11 (k=1, i=1)
    
    # Term 3: M3 = A00 * (B01 - B11)
    U[0*2+0, 2] = 1  # A00
    V[0*2+1, 2] = 1  # B01 (j=0, k=1)
    V[1*2+1, 2] = -1 # B11 (j=1, k=1)
    W[1*2+0, 2] = 1  # C01 (k=1, i=0)
    W[1*2+1, 2] = 1  # C11 (k=1, i=1)
    
    # Term 4: M4 = A11 * (B10 - B00)
    U[1*2+1, 3] = 1  # A11
    V[1*2+0, 3] = 1  # B10 (j=1, k=0)
    V[0*2+0, 3] = -1 # B00 (j=0, k=0)
    W[0*2+0, 3] = 1  # C00 (k=0, i=0)
    W[0*2+1, 3] = 1  # C10 (k=0, i=1)
    
    # Term 5: M5 = (A00 + A01) * B11
    U[0*2+0, 4] = 1  # A00
    U[0*2+1, 4] = 1  # A01 (i=0, j=1)
    V[1*2+1, 4] = 1  # B11 (j=1, k=1)
    W[0*2+0, 4] = -1 # C00 (k=0, i=0)
    W[1*2+0, 4] = 1  # C01 (k=1, i=0)
    
    # Term 6: M6 = (A10 - A00) * (B00 + B01)
    U[1*2+0, 5] = 1  # A10
    U[0*2+0, 5] = -1 # A00
    V[0*2+0, 5] = 1  # B00
    V[0*2+1, 5] = 1  # B01
    W[1*2+1, 5] = 1  # C11 (k=1, i=1)
    
    # Term 7: M7 = (A01 - A11) * (B10 + B11)
    U[0*2+1, 6] = 1  # A01
    U[1*2+1, 6] = -1 # A11
    V[1*2+0, 6] = 1  # B10
    V[1*2+1, 6] = 1  # B11
    W[0*2+0, 6] = 1  # C00 (k=0, i=0)
    
    return U, V, W

def _build_schoolbook_2x2x1():
    """
    Build schoolbook rank-4 decomposition for 2x2 x 2x1 matrix multiplication.
    Returns U (4x4), V (2x4), W (2x4).
    """
    n, m, p = 2, 2, 1
    rank = 4
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            U[i * m + j, col] = 1
            V[j * p, col] = 1
            W[0 * n + i, col] = 1
            col += 1
            
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # First, build the baseline block decomposition
    block_sizes = [2, 2, 2, 2, 2, 2, 2, 1]
    num_blocks = len(block_sizes)
    
    U_str, V_str, W_str = _build_strassen_2x2x2()
    U_sb, V_sb, W_sb = _build_schoolbook_2x2x1()
    
    baseline_rank = 7 * 7 + 4  # 53
    
    U = np.zeros((n * m, baseline_rank))
    V = np.zeros((m * p, baseline_rank))
    W = np.zeros((p * n, baseline_rank))
    
    col_offset = 0
    for b in range(num_blocks):
        block_size = block_sizes[b]
        start_k = sum(block_sizes[:b])
        
        if block_size == 2:
            U[:, col_offset:col_offset+7] = U_str
            for j in range(m):
                for k_local in range(2):
                    k_global = start_k + k_local
                    V[j * p + k_global, col_offset:col_offset+7] = V_str[j * 2 + k_local, :]
            for k_local in range(2):
                k_global = start_k + k_local
                for i in range(n):
                    W[k_global * n + i, col_offset:col_offset+7] = W_str[k_local * 2 + i, :]
            col_offset += 7
        else:
            U[:, col_offset:col_offset+4] = U_sb
            for j in range(m):
                V[j * p + start_k, col_offset:col_offset+4] = V_sb[j * 1, :]
            for i in range(n):
                W[start_k * n + i, col_offset:col_offset+4] = W_sb[0 * 2 + i, :]
            col_offset += 4
    
    # Now attempt to reduce rank using QR-based column selection
    # The tensor identity requires that for all (i,j,k):
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,k} (for matrix mult)
    # Actually for 2x2 * 2x15: C[i,k] = sum_j A[i,j]*B[j,k]
    # So the constraint is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if this is a valid term, 0 otherwise
    # But actually we need the full tensor decomposition.
    
    # Let's construct the constraint matrix differently.
    # We have rank r. For each output position (i, k) where i in [0,n), k in [0,p):
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    # In tensor form: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal the multiplication tensor.
    
    # Actually, let me think about this more carefully.
    # The matrix multiplication tensor T[i,j,k] = 1 (it's just the identity in the appropriate sense)
    # Wait, no. For matrix multiplication (n x m) * (m x p) = (n x p):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has entries T[i,j,k] which when contracted gives C.
    # Actually T is defined such that <T, A ⊗ B> = C, where A is n x m, B is m x p, C is n x p.
    # So T[i,j,k] = 1 for all valid i,j,k? No, that's not right either.
    
    # Let me reconsider. The multiplication tensor M_{n,m,p} has entries:
    # M[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # Actually, the tensor is defined as: for each triple (i,j,k), the coefficient is 1
    # because C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor has 1s at positions corresponding to valid (i,j,k) triples.
    
    # Wait, I think I'm overcomplicating this. Let me just verify the current solution works
    # and then try to reduce rank.
    
    # For now, return the baseline
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
# EVOLVE-BLOCK-END
