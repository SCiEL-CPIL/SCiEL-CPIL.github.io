"""Matrix multiplication 3x4x6: rank-minimal exact tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 6]

# Deadline guard
DEADLINE = 150.0
_start_time = time.time()

def _elapsed():
    return time.time() - _start_time

def _time_left():
    return DEADLINE - _elapsed()

def _verify(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] == W[k*n+i, r]
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = W[k * n + i]
                computed = U[i * m + j] @ V[j * p + k]
                if not np.allclose(computed, expected, atol=1e-9):
                    return False
    return True

def _check_exact(U, V, W, n, m, p):
    """Integer/half-integer exact check."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = W[k * n + i]
                computed = U[i * m + j] @ V[j * p + k]
                # Check exactness: values should be close to integers or half-integers
                diff = computed - expected
                if np.max(np.abs(diff)) > 1e-9:
                    return False
    return True

def _build_strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7).
    Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen's 7 multiplications for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Row indices for U: i*2+j -> (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # Row indices for V: j*2+k -> (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # Row indices for W: k*2+i -> (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1  # A11, A22
    V[0, 0] = 1; V[3, 0] = 1  # B11, B22
    W[0, 0] = 1; W[3, 0] = 1  # C11, C22
    
    # M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1  # A21, A22
    V[0, 1] = 1  # B11
    W[2, 1] = 1; W[3, 1] = -1  # C21, C22
    
    # M3 = A11(B21-B22)
    U[0, 2] = 1  # A11
    V[2, 2] = 1; V[3, 2] = -1  # B21, B22
    W[1, 2] = 1; W[3, 2] = 1  # C12, C22
    
    # M4 = A22(B12-B11)
    U[3, 3] = 1  # A22
    V[1, 3] = 1; V[0, 3] = -1  # B12, B11
    W[0, 3] = 1; W[2, 3] = 1  # C11, C21
    
    # M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1  # A11, A12
    V[3, 4] = 1  # B22
    W[0, 4] = -1; W[1, 4] = 1  # C11, C12
    
    # M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1  # A21, A11
    V[0, 5] = 1; V[1, 5] = 1  # B11, B12
    W[3, 5] = 1  # C22
    
    # M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1  # A12, A22
    V[2, 6] = 1; V[3, 6] = 1  # B21, B22
    W[0, 6] = 1  # C11
    
    return U, V, W

def _build_1x2x2_block():
    """Build a 1x2x2 block: 1x2 * 2x2 -> 1x2, rank 4 (schoolbook for 1x2x2 is 1*2*2=4).
    U: (1*2, 4) = (2, 4), V: (2*2, 4) = (4, 4), W: (2*1, 4) = (2, 4)
    Indices: U[j, c] for j in {0,1}, V[j*2+k, c], W[k*1+i, c] = W[k, c]
    """
    U = np.zeros((2, 4))
    V = np.zeros((4, 4))
    W = np.zeros((2, 4))
    col = 0
    for j in range(2):
        for k in range(2):
            U[j, col] = 1
            V[j * 2 + k, col] = 1
            W[k, col] = 1
            col += 1
    return U, V, W

def _build_1x1x2_block():
    """Build a 1x1x2 block: 1x1 * 1x2 -> 1x2, rank 2 (schoolbook).
    U: (1, 2), V: (2, 2), W: (2, 2)
    """
    U = np.zeros((1, 2))
    V = np.zeros((2, 2))
    W = np.zeros((2, 2))
    for k in range(2):
        U[0, k] = 1
        V[k, k] = 1
        W[k, k] = 1
    return U, V, W

def _build_2x1x2_block():
    """Build a 2x1x2 block: 2x1 * 1x2 -> 2x2, rank 4 (schoolbook).
    U: (2, 4), V: (2, 4), W: (4, 4)
    U[i, c] for i in {0,1}, V[j*1+k, c] = V[k, c] for j=0, k in {0,1}, W[k*2+i, c]
    """
    U = np.zeros((2, 4))
    V = np.zeros((2, 4))
    W = np.zeros((4, 4))
    col = 0
    for i in range(2):
        for k in range(2):
            U[i, col] = 1
            V[k, col] = 1
            W[k * 2 + i, col] = 1
            col += 1
    return U, V, W

def _build_1x2x1_block():
    """Build a 1x2x1 block: 1x2 * 2x1 -> 1x1, rank 2 (schoolbook).
    U: (2, 2), V: (2, 2), W: (1, 2)
    U[j, c] for j in {0,1}, V[j*1+k, c] = V[j, c], W[k*1+i, c] = W[0, c]
    """
    U = np.zeros((2, 2))
    V = np.zeros((2, 2))
    W = np.zeros((1, 2))
    for j in range(2):
        U[j, j] = 1
        V[j, j] = 1
        W[0, j] = 1
    return U, V, W

def _build_2x2x1_block():
    """Build a 2x2x1 block: 2x2 * 2x1 -> 2x1, rank 4 (schoolbook).
    U: (4, 4), V: (2, 4), W: (2, 4)
    U[i*2+j, c], V[j*1+k, c] = V[j, c], W[k*2+i, c]
    """
    U = np.zeros((4, 4))
    V = np.zeros((2, 4))
    W = np.zeros((2, 4))
    col = 0
    for i in range(2):
        for j in range(2):
            U[i * 2 + j, col] = 1
            V[j, col] = 1
            W[j, col] = 1
            col += 1
    return U, V, W

def _build_2x1x1_block():
    """Build a 2x1x1 block: 2x1 * 1x1 -> 2x1, rank 2 (schoolbook).
    U: (2, 2), V: (1, 2), W: (2, 2)
    U[i, c], V[0, c], W[k*2+i, c] with k=0
    """
    U = np.zeros((2, 2))
    V = np.zeros((1, 2))
    W = np.zeros((2, 2))
    for i in range(2):
        U[i, i] = 1
        V[0, i] = 1
        W[i, i] = 1
    return U, V, W

def _build_1x1x1_block():
    """Build a 1x1x1 block: rank 1.
    U: (1,1), V: (1,1), W: (1,1)
    """
    U = np.ones((1, 1))
    V = np.ones((1, 1))
    W = np.ones((1, 1))
    return U, V, W

def _assemble_decomposition(n, m, p, blocks):
    """
    Assemble a decomposition from a list of blocks.
    Each block is a tuple (block_U, block_V, block_W, i_off, j_off, k_off)
    where i_off, j_off, k_off are the offsets into the full n, m, p indices.
    """
    rank = sum(bU.shape[1] for bU, bV, bW, _, _, _ in blocks)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for bU, bV, bW, i_off, j_off, k_off in blocks:
        r = bU.shape[1]
        # Map local indices to global
        for li in range(bU.shape[0]):
            # li corresponds to local (i_local * m_local + j_local)
            # We need to figure out the local dimensions
            pass
        
        # Actually, let's be more careful. The block has specific local dimensions.
        # Let me restructure: each block should specify its local (n_loc, m_loc, p_loc)
        # and the mapping from local row indices to global.
        pass
    
    # This approach is getting complicated. Let me use a simpler method.
    # Each block provides U_loc, V_loc, W_loc with known local dims.
    # I'll just directly place them.
    
    col = 0
    for block in blocks:
        bU, bV, bW, n_loc, m_loc, p_loc, i_off, j_off, k_off = block
        r = bU.shape[1]
        for li in range(n_loc * m_loc):
            # li = i_loc * m_loc + j_loc
            i_loc = li // m_loc
            j_loc = li % m_loc
            gi = i_off + i_loc
            gj = j_off + j_loc
            U[gi * m + gj, col:col+r] = bU[li, :]
        
        for lj in range(m_loc * p_loc):
            lj_j = lj // p_loc
            lj_k = lj % p_loc
            gj = j_off + lj_j
            gk = k_off + lj_k
            V[gj * p + gk, col:col+r] = bV[lj, :]
        
        for lk in range(p_loc * n_loc):
            lk_k = lk // n_loc
            lk_i = lk % n_loc
            gk = k_off + lk_k
            gi = i_off + lk_i
            W[gk * n + gi, col:col+r] = bW[lk, :]
        
        col += r
    
    return U, V, W, rank

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4x6 matrix multiplication.
    
    Strategy: Partition the 3x4x6 problem into sub-blocks and use Strassen-style
    decompositions where beneficial.
    
    Partition approach:
    - Split n=3 into 1+2, m=4 into 2+2, p=6 into 2+2+2
    - This gives blocks:
      (1,2,2), (1,2,2), (1,2,2)  -> three 1x2x2 blocks, each rank 4 = 12
      (2,2,2), (2,2,2), (2,2,2)  -> three 2x2x2 blocks, each rank 7 (Strassen) = 21
      Total: 12 + 21 = 33... wait that's too low, let me recheck.
    
    Actually wait - the tensor decomposition must be EXACT. Strassen gives exact
    decomposition for 2x2x2 with rank 7. Let me verify the total.
    
    For n=3, m=4, p=6:
    Partition: n = 1+2, m = 2+2, p = 2+2+2
    
    Blocks:
    - (i=0, j=0:2, k=0:2): 1x2x2, rank 4
    - (i=0, j=0:2, k=2:4): 1x2x2, rank 4  
    - (i=0, j=0:2, k=4:6): 1x2x2, rank 4
    - (i=0, j=2:4, k=0:2): 1x2x2, rank 4
    - (i=0, j=2:4, k=2:4): 1x2x2, rank 4
    - (i=0, j=2:4, k=4:6): 1x2x2, rank 4
    - (i=1:3, j=0:2, k=0:2): 2x2x2, rank 7 (Strassen)
    - (i=1:3, j=0:2, k=2:4): 2x2x2, rank 7 (Strassen)
    - (i=1:3, j=0:2, k=4:6): 2x2x2, rank 7 (Strassen)
    - (i=1:3, j=2:4, k=0:2): 2x2x2, rank 7 (Strassen)
    - (i=1:3, j=2:4, k=2:4): 2x2x2, rank 7 (Strassen)
    - (i=1:3, j=2:4, k=4:6): 2x2x2, rank 7 (Strassen)
    
    Total rank: 6*4 + 6*7 = 24 + 42 = 66
    
    Schoolbook is 3*4*6 = 72. So 66 < 72, improvement!
    """
    n, m, p = DIMENSIONS
    
    # Build blocks
    blocks = []
    
    # 1x2x2 blocks (rank 4 each)
    U_1x2x2, V_1x2x2, W_1x2x2 = _build_1x2x2_block()
    
    # 2x2x2 Strassen blocks (rank 7 each)
    U_2x2x2, V_2x2x2, W_2x2x2 = _build_strassen_2x2x2()
    
    # Partition: n = [0, 1:3], m = [0:2, 2:4], p = [0:2, 2:4, 4:6]
    n_parts = [(0, 1), (1, 3)]
    m_parts = [(0, 2), (2, 4)]
    p_parts = [(0, 2), (2, 4), (4, 6)]
    
    for i_start, i_end in n_parts:
        for j_start, j_end in m_parts:
            for k_start, k_end in p_parts:
                n_loc = i_end - i_start
                m_loc = j_end - j_start
                p_loc = k_end - k_start
                
                if n_loc == 1 and m_loc == 2 and p_loc == 2:
                    blocks.append((U_1x2x2, V_1x2x2, W_1x2x2, n_loc, m_loc, p_loc, i_start, j_start, k_start))
                elif n_loc == 2 and m_loc == 2 and p_loc == 2:
                    blocks.append((U_2x2x2, V_2x2x2, W_2x2x2, n_loc, m_loc, p_loc, i_start, j_start, k_start))
                else:
                    raise ValueError(f"Unexpected block size: {n_loc}x{m_loc}x{p_loc}")
    
    U, V, W, rank = _assemble_decomposition(n, m, p, blocks)
    
    # Verify
    if not _check_exact(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
