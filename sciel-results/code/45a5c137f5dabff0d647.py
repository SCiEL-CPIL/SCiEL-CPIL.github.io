"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 10]
DEADLINE = 140.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # The matrix multiplication tensor: T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need this to equal 1.0 for all (i,j,k).
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i * m + j] * V_arr[j * p + k] * W_arr[k * n + i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _strassen_block_2x2x2():
    """
    Strassen's 7-rank decomposition of 2x2x2 matrix multiplication.
    Returns (U, V, W) each of shape (4, 7) for the 2x2x2 case.
    U indexed by [i*2+j, r], V by [j*2+k, r], W by [k*2+i, r]
    """
    # Strassen's 7 products for 2x2x2:
    # P1 = (A11)(B11 - B22)
    # P2 = (A11 + A22)(B22)
    # P3 = (A11 + A12)(B11)
    # P4 = (A22 - A11)(B21 + B22)
    # P5 = (A12 + A22)(B21)
    # P6 = (A11 - A12)(B11 + B12)
    # P7 = (A21 + A12)(B22 - B21)
    #
    # C11 = P1 + P2 - P3 + P7
    # C12 = P3 + P5
    # C21 = P4 + P6
    # C22 = P1 - P2 + P4 + P5
    
    n2, m2, p2 = 2, 2, 2
    rank = 7
    U = [[0.0] * rank for _ in range(n2 * m2)]  # 4x7
    V = [[0.0] * rank for _ in range(m2 * p2)]  # 4x7
    W = [[0.0] * rank for _ in range(p2 * n2)]  # 4x7
    
    # U[i*m2 + j, r], V[j*p2 + k, r], W[k*n2 + i, r]
    
    # P1 = A11 * (B11 - B22)
    # U[0*2+0, 0] = 1 (A11)
    # V[0*2+0, 0] = 1 (B11), V[1*2+1, 0] = -1 (B22)
    # W[0*2+0, 0] = 1 (C11), W[1*2+0, 0] = -1 (C22)
    U[0][0] = 1; V[0][0] = 1; V[3][0] = -1
    W[0][0] = 1; W[2][0] = -1  # W[k*n2+i]: C11 is k=0,i=0 -> idx 0; C22 is k=1,i=1 -> idx 3? 
    # Wait, W[k*n2 + i]: for 2x2, k in {0,1}, i in {0,1}
    # C11: k=0, i=0 -> W[0]
    # C12: k=1, i=0 -> W[1]  
    # C21: k=0, i=1 -> W[2]
    # C22: k=1, i=1 -> W[3]
    # So W[0]=C11, W[1]=C12, W[2]=C21, W[3]=C22
    
    # Let me redo:
    U = [[0.0] * rank for _ in range(n2 * m2)]
    V = [[0.0] * rank for _ in range(m2 * p2)]
    W = [[0.0] * rank for _ in range(p2 * n2)]
    
    # Indexing: U[i*m2+j], V[j*p2+k], W[k*n2+i]
    # A indices: i in {0,1}, j in {0,1} -> A00=U[0], A01=U[1], A10=U[2], A11=U[3]
    # B indices: j in {0,1}, k in {0,1} -> B00=V[0], B01=V[1], B10=V[2], B11=V[3]
    # C indices: k in {0,1}, i in {0,1} -> C00=W[0], C01=W[1], C10=W[2], C11=W[3]
    
    # P1 = A00 * (B00 - B11)
    U[0][0] = 1
    V[0][0] = 1; V[3][0] = -1
    W[0][0] = 1; W[3][0] = 1  # C00 += P1, C11 += P1
    
    # P2 = (A00 + A11) * B11
    U[0][1] = 1; U[3][1] = 1
    V[3][1] = 1
    W[0][1] = 1; W[3][1] = -1  # C00 += P2, C11 -= P2
    
    # P3 = (A00 + A01) * B00
    U[0][2] = 1; U[1][2] = 1
    V[0][2] = 1
    W[0][2] = -1; W[1][2] = 1  # C00 -= P3, C01 += P3
    
    # P4 = (A11 - A00) * (B10 + B11)
    U[3][3] = 1; U[0][3] = -1
    V[2][3] = 1; V[3][3] = 1
    W[2][3] = 1; W[3][3] = 1  # C10 += P4, C11 += P4
    
    # P5 = (A01 + A11) * B10
    U[1][4] = 1; U[3][4] = 1
    V[2][4] = 1
    W[1][4] = 1; W[3][4] = 1  # C01 += P5, C11 += P5
    
    # P6 = (A00 - A01) * (B00 + B01)
    U[0][5] = 1; U[1][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[2][5] = 1  # C10 += P6
    
    # P7 = (A10 + A01) * (B11 - B10)
    U[2][6] = 1; U[1][6] = 1
    V[3][6] = 1; V[2][6] = -1
    W[0][6] = 1  # C00 += P7
    
    return U, V, W

def _build_strassen_based_decomposition(n, m, p):
    """
    Build a decomposition using Strassen on 2x2x2 blocks.
    For n=2, m=3, p=10:
    - Split the m=3 dimension: use one 2x2x2 Strassen block for j in {0,1}, 
      and schoolbook for j=2.
    - Actually, let's think more carefully.
    
    The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    We can decompose this by splitting dimensions into blocks.
    For n=2: split into 1 block of size 2.
    For m=3: split into 1 block of size 2 + 1 block of size 1.
    For p=10: split into 5 blocks of size 2.
    
    For each combination of (n_block, m_block, p_block):
    - If n_block=2, m_block=2, p_block=2: use Strassen (rank 7)
    - Otherwise: use schoolbook (rank = n_block * m_block * p_block)
    
    Total rank = sum over all block combinations.
    """
    # Split m=3 into blocks: [2, 1]
    # Split p=10 into blocks: [2, 2, 2, 2, 2]
    # n=2 is one block: [2]
    
    m_blocks = [2, 1]
    p_blocks = [2, 2, 2, 2, 2]
    n_blocks = [2]
    
    # Build the full decomposition
    all_U = []
    all_V = []
    all_W = []
    
    offset_m = 0
    offset_p = 0
    
    for nb in n_blocks:
        offset_m = 0
        for mb in m_blocks:
            offset_p = 0
            for pb in p_blocks:
                if nb == 2 and mb == 2 and pb == 2:
                    # Use Strassen
                    U_blk, V_blk, W_blk = _strassen_block_2x2x2()
                    # Remap indices
                    U_remapped = [[0.0] * 7 for _ in range(n * m)]
                    V_remapped = [[0.0] * 7 for _ in range(m * p)]
                    W_remapped = [[0.0] * 7 for _ in range(p * n)]
                    
                    for ri in range(nb):
                        for rj in range(mb):
                            for r in range(7):
                                U_remapped[(offset_m + rj) * 1 + ri][r] = U_blk[ri * 2 + rj][r]
                        # Wait, U is indexed by [i*m + j] in the full problem,
                        # but in the block it's [i*mb + j]. We need to map.
                    
                    # Actually this is getting complex. Let me use a different approach.
                    # Just build the full U, V, W directly.
                    pass
                else:
                    # Schoolbook for this block
                    for ri in range(nb):
                        for rj in range(mb):
                            for rk in range(pb):
                                col = len(all_U[0]) if all_U else 0
                                # Add a new column
                                for row in range(n * m):
                                    all_U[row].append(0.0)
                                for row in range(m * p):
                                    all_V[row].append(0.0)
                                for row in range(p * n):
                                    all_W[row].append(0.0)
                                all_U[(offset_m + rj) * 1 + ri][col] = 1.0  # Hmm, indexing is wrong
                                all_V[(offset_m + rj) * p + (offset_p + rk)][col] = 1.0
                                all_W[(offset_p + rk) * n + ri][col] = 1.0
                        offset_p += pb
            offset_m += mb
    
    # This approach is getting messy. Let me use a cleaner method.
    return None

def _build_strassen_decomposition_v2(n, m, p):
    """
    Cleaner Strassen-based decomposition.
    Split m into blocks of size 2 (plus remainder 1).
    Split p into blocks of size 2.
    For each (m_block, p_block) pair where both are size 2, use Strassen.
    For others, use schoolbook.
    """
    # Determine block structure
    m_blocks = []
    rem = m
    while rem >= 2:
        m_blocks.append(2)
        rem -= 2
    if rem > 0:
        m_blocks.append(rem)
    
    p_blocks = []
    rem = p
    while rem >= 2:
        p_blocks.append(2)
        rem -= 2
    if rem > 0:
        p_blocks.append(rem)
    
    # For n=2, it's always one block of size 2
    n_blocks = [n]
    
    # Build decomposition
    U = []
    V = []
    W = []
    
    def add_column(u_vals, v_vals, w_vals):
        """Add a new column to U, V, W."""
        for i in range(len(U)):
            U[i].append(0.0)
        for i in range(len(V)):
            V[i].append(0.0)
        for i in range(len(W)):
            W[i].append(0.0)
        col = len(U[0]) - 1
        for idx, val in u_vals:
            U[idx][col] = val
        for idx, val in v_vals:
            V[idx][col] = val
        for idx, val in w_vals:
            W[idx][col] = val
    
    # Initialize
    for i in range(n * m):
        U.append([])
    for i in range(m * p):
        V.append([])
    for i in range(p * n):
        W.append([])
    
    m_offset = 0
    for mb in m_blocks:
        p_offset = 0
        for pb in p_blocks:
            if mb == 2 and pb == 2:
                # Use Strassen for this 2x2x2 block (n=2, m=2, p=2)
                U_blk, V_blk, W_blk = _strassen_block_2x2x2()
                for r in range(7):
                    u_vals = []
                    v_vals = []
                    w_vals = []
                    for ri in range(2):
                        for rj in range(2):
                            u_vals.append(((ri) * m + (m_offset + rj), U_blk[ri * 2 + rj][r]))
                    for rj in range(2):
                        for rk in range(2):
                            v_vals.append(((m_offset + rj) * p + (p_offset + rk), V_blk[rj * 2 + rk][r]))
                    for rk in range(2):
                        for ri in range(2):
                            w_vals.append(((p_offset + rk) * n + ri, W_blk[rk * 2 + ri][r]))
                    # Filter out zeros
                    u_vals = [(idx, val) for idx, val in u_vals if val != 0]
                    v_vals = [(idx, val) for idx, val in v_vals if val != 0]
                    w_vals = [(idx, val) for idx, val in w_vals if val != 0]
                    add_column(u_vals, v_vals, w_vals)
            else:
                # Schoolbook for this block
                for ri in range(n):
                    for rj in range(mb):
                        for rk in range(pb):
                            add_column(
                                [(ri * m + (m_offset + rj), 1.0)],
                                [((m_offset + rj) * p + (p_offset + rk), 1.0)],
                                [((p_offset + rk) * n + ri, 1.0)]
                            )
            p_offset += pb
        m_offset += mb
    
    return U, V, W

def _verify_fast(U_arr, V_arr, W_arr, n, m, p, tol=1e-9):
    """Fast verification using numpy broadcasting."""
    # U_arr: (n*m, rank), V_arr: (m*p, rank), W_arr: (p*n, rank)
    # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1.0
    # Build the tensor efficiently
    rank = U_arr.shape[1]
    # Reshape for broadcasting
    # U: (n, m, rank), V: (m, p, rank), W: (p, n, rank)
    U_reshaped = U_arr.reshape(n, m, rank)
    V_reshaped = V_arr.reshape(m, p, rank)
    W_reshaped = W_arr.reshape(p, n, rank)
    
    # Compute T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Use einsum for efficiency
    T = np.einsum('ijr,jkr,kin->ijk', U_reshaped, V_reshaped, W_reshaped)
    return np.allclose(T, 1.0, atol=tol)

def _try_reduce_rank(U, V, W, n, m, p, time_limit=30.0):
    """
    Try to reduce the rank by finding linear dependencies among columns.
    Uses fast verification with numpy einsum.
    """
    start = time.time()
    rank = len(U[0])
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # First pass: try removing single columns
    for col_to_remove in range(rank):
        if time.time() - start > time_limit:
            break
        
        U_test = np.delete(U_arr, col_to_remove, axis=1)
        V_test = np.delete(V_arr, col_to_remove, axis=1)
        W_test = np.delete(W_arr, col_to_remove, axis=1)
        
        if _verify_fast(U_test, V_test, W_test, n, m, p):
            best_rank = rank - 1
            best_U = U_test.tolist()
            best_V = V_test.tolist()
            best_W = W_test.tolist()
            U_arr = U_test
            V_arr = V_test
            W_arr = W_test
            rank = best_rank
    
    # Second pass: try removing pairs of columns (if we found a single removal)
    if best_rank < len(U[0]) and time.time() - start < time_limit:
        for col1 in range(rank):
            if time.time() - start > time_limit:
                break
            for col2 in range(col1 + 1, rank):
                if time.time() - start > time_limit:
                    break
                idx = np.arange(rank)
                mask = (idx != col1) & (idx != col2)
                U_test = U_arr[:, mask]
                V_test = V_arr[:, mask]
                W_test = W_arr[:, mask]
                
                if _verify_fast(U_test, V_test, W_test, n, m, p):
                    best_rank = rank - 2
                    best_U = U_test.tolist()
                    best_V = V_test.tolist()
                    best_W = W_test.tolist()
                    U_arr = U_test
                    V_arr = V_test
                    W_arr = W_test
                    rank = best_rank
                    break
    
    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Build the Strassen-based decomposition (rank 55)
    U, V, W = _build_strassen_decomposition_v2(n, m, p)
    rank = len(U[0])
    
    # Verify it's correct
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    if not _verify_fast(U_arr, V_arr, W_arr, n, m, p):
        # Fall back to schoolbook if Strassen construction is wrong
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
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
    
    # Try to reduce rank further if time permits
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 10:
        remaining = min(DEADLINE - 10 - elapsed, 60.0)
        U, V, W, rank = _try_reduce_rank(U, V, W, n, m, p, time_limit=remaining)
    
    # Final safety check
    if time.time() > DEADLINE - 5:
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
