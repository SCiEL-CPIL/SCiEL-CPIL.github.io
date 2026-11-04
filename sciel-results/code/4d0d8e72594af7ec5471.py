"""Matrix multiplication 3,5,5 — rank-minimal exact tensor decomposition."""
import numpy as np
from itertools import product
import time

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def _strassen_222():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
    # Strassen's algorithm: 7 multiplications for 2x2 matrix mult
    # C = A * B where A, B are 2x2
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Strassen's 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Row indices for U: i*2+j where i in {0,1}, j in {0,1}
    # Row indices for V: j*2+k where j in {0,1}, k in {0,1}
    # Row indices for W: k*2+i where k in {0,1}, i in {0,1}
    
    # Column 0: M1 = (A00+A11)(B00+B11)
    U[0][0] = 1  # A00
    U[3][0] = 1  # A11
    V[0][0] = 1  # B00
    V[3][0] = 1  # B11
    W[0][0] = 1  # C00 (k=0,i=0)
    W[3][0] = 1  # C11 (k=1,i=1)
    
    # Column 1: M2 = (A10+A11)B00
    U[2][1] = 1  # A10
    U[3][1] = 1  # A11
    V[0][1] = 1  # B00
    W[2][1] = 1  # C10 (k=0,i=1)
    
    # Column 2: M3 = A00(B10-B11)
    U[0][2] = 1  # A00
    V[2][2] = 1  # B10
    V[3][2] = -1  # B11
    W[2][2] = 1  # C01 (k=1,i=0)
    
    # Column 3: M4 = A11(B01-B10)
    U[3][3] = 1  # A11
    V[1][3] = 1  # B01
    V[2][3] = -1  # B10
    W[0][3] = 1  # C00 (k=0,i=0)
    W[3][3] = 1  # C11 (k=1,i=1)
    
    # Column 4: M5 = (A00+A01)B11
    U[0][4] = 1  # A00
    U[1][4] = 1  # A01
    V[3][4] = 1  # B11
    W[0][4] = -1  # C00 (k=0,i=0)
    W[1][4] = 1  # C01 (k=1,i=0)
    
    # Column 5: M6 = (A10-A00)(B00+B10)
    U[2][5] = 1  # A10
    U[0][5] = -1  # A00
    V[0][5] = 1  # B00
    V[2][5] = 1  # B10
    W[3][5] = 1  # C11 (k=1,i=1)
    
    # Column 6: M7 = (A01-A11)(B01+B11)
    U[1][6] = 1  # A01
    U[3][6] = -1  # A11
    V[1][6] = 1  # B01
    V[3][6] = 1  # B11
    W[0][6] = 1  # C00 (k=0,i=0)
    W[1][6] = 1  # C01 (k=1,i=0)
    
    return U, V, W

def _schoolbook_block(n_block, m_block, p_block):
    """Return U, V, W for schoolbook multiplication of n_block x m_block by m_block x p_block."""
    rank = n_block * m_block * p_block
    U = [[0] * rank for _ in range(n_block * m_block)]
    V = [[0] * rank for _ in range(m_block * p_block)]
    W = [[0] * rank for _ in range(p_block * n_block)]
    column = 0
    for i in range(n_block):
        for j in range(m_block):
            for k in range(p_block):
                U[i * m_block + j][column] = 1
                V[j * p_block + k][column] = 1
                W[k * n_block + i][column] = 1
                column += 1
    return U, V, W

def _get_splits(dim):
    """Generate all ways to split dim into 2+ parts (ordered)."""
    def gen(d, remaining):
        if d == 1:
            if remaining > 0:
                yield [remaining]
            return
        for first in range(1, remaining + 1):
            for rest in gen(d - 1, remaining - first):
                yield [first] + rest
    return list(gen(dim, dim))

def _compute_rank(n_splits, m_splits, p_splits):
    """Compute total rank for given splits."""
    total_rank = 0
    for ns in n_splits:
        for ms in m_splits:
            for ps in p_splits:
                if ns == 2 and ms == 2 and ps == 2:
                    total_rank += 7  # Strassen
                else:
                    total_rank += ns * ms * ps
    return total_rank

def _build_decomposition(n_splits, m_splits, p_splits):
    """Build the full decomposition from block splits."""
    n, m, p = DIMENSIONS
    total_rank = _compute_rank(n_splits, m_splits, p_splits)
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    i_offset = 0
    for i_block_size in n_splits:
        j_offset = 0
        for j_block_size in m_splits:
            k_offset = 0
            for k_block_size in p_splits:
                if i_block_size == 2 and j_block_size == 2 and k_block_size == 2:
                    U_block, V_block, W_block = _strassen_222()
                    block_rank = 7
                else:
                    U_block, V_block, W_block = _schoolbook_block(i_block_size, j_block_size, k_block_size)
                    block_rank = i_block_size * j_block_size * k_block_size
                
                # Place block into global matrices
                for r in range(len(U_block)):
                    for c in range(block_rank):
                        U[i_offset + r][col_offset + c] = U_block[r][c]
                for r in range(len(V_block)):
                    for c in range(block_rank):
                        V[j_offset + r][col_offset + c] = V_block[r][c]
                for r in range(len(W_block)):
                    for c in range(block_rank):
                        W[k_offset + r][col_offset + c] = W_block[r][c]
                
                col_offset += block_rank
                k_offset += k_block_size
            j_offset += j_block_size
        i_offset += i_block_size
    
    return U, V, W, total_rank

def _verify_decomposition(U, V, W):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    
    # Build the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # This should equal 1 for all (i,j,k) and 0 otherwise... wait, no.
    # The matrix multiplication tensor has T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor is: for each (i,j,k), the coefficient is 1 (it's a rank-1 tensor for each triple)
    # Actually, the matrix multiplication tensor T has T[i,j,k] = 1 for all i,j,k, and the
    # decomposition satisfies: for all i,k: sum_j A[i,j]*B[j,k] = sum_c (sum_j U[i*m+j,c]*V[j*p+k,c]) * W[k*n+i,c]
    # Wait, I need to be more careful. The tensor for matrix multiplication is:
    # T[i,j,k] = 1 for all i in [n], j in [m], k in [p]
    # And the decomposition is: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # So we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c] for c in range(rank))
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < 1e-10

def _try_split(n_splits, m_splits, p_splits, deadline):
    """Try a specific split and return the result if valid."""
    if time.time() > deadline:
        return None
    U, V, W, rank = _build_decomposition(n_splits, m_splits, p_splits)
    if _verify_decomposition(U, V, W):
        return {"U": U, "V": V, "W": W, "rank": rank}
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for evaluation overhead
    
    # Generate all possible splits
    n_splits_list = _get_splits(n)
    m_splits_list = _get_splits(m)
    p_splits_list = _get_splits(p)
    
    # Find the split with minimum rank
    best_result = None
    best_rank = n * m * p  # Schoolbook as fallback
    
    # First, try the schoolbook decomposition as a guaranteed valid baseline
    U_school, V_school, W_school = _schoolbook_block(n, m, p)
    best_result = {"U": U_school, "V": V_school, "W": W_school, "rank": best_rank}
    
    # Search over all combinations of splits
    for ns in n_splits_list:
        if time.time() > deadline:
            break
        for ms in m_splits_list:
            if time.time() > deadline:
                break
            for ps in p_splits_list:
                if time.time() > deadline:
                    break
                rank = _compute_rank(ns, ms, ps)
                if rank >= best_rank:
                    continue
                result = _try_split(ns, ms, ps, deadline)
                if result is not None and result["rank"] < best_rank:
                    best_rank = result["rank"]
                    best_result = result
    
    return {"U": best_result["U"], "V": best_result["V"], "W": best_result["W"]}
# EVOLVE-BLOCK-END
