"""
Program to find a rank-minimal exact tensor decomposition for matrix multiplication 2x5 by 5x6.
The output factor follows the official trace convention W[k*n+i,r].
"""

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    
    # For all i in [n], j in [m], k in [p]:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _build_schoolbook(n, m, p):
    """Build the schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    return U, V, W

def _build_strassen_222():
    """Return Strassen decomposition for 2x2x2 matrix multiplication."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (a00+a11)*(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (a10+a11)*b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1
    
    # M3: a00*(b01-b11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[0, 2] = 1
    
    # M4: a11*(b10-b00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[3, 3] = 1
    
    # M5: (a00+a01)*b11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[2, 4] = 1
    
    # M6: (a10-a00)*(b00+b01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[1, 5] = 1
    
    # M7: (a01-a11)*(b10+b11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[2, 6] = 1
    
    return U, V, W

def _embed_strassen_block(U_full, V_full, W_full, i0, j0, k0, n, m, p, block_size=2):
    """
    Embed a Strassen 2x2x2 block into the full decomposition.
    i0, j0, k0 are the starting indices for the block.
    Returns the column offset where this block's columns start.
    """
    U_s, V_s, W_s = _build_strassen_222()
    
    # Map local indices to global
    # U local (i,j) -> global (i0+i, j0+j)
    # V local (j,k) -> global (j0+j, k0+k)
    # W local (k,i) -> global (k0+k, i0+i)
    
    # Find available columns - we'll just append
    current_rank = U_full.shape[1]
    
    # Extend matrices if needed
    new_rank = current_rank + 7
    U_new = np.zeros((n * m, new_rank))
    V_new = np.zeros((m * p, new_rank))
    W_new = np.zeros((p * n, new_rank))
    
    U_new[:, :current_rank] = U_full
    V_new[:, :current_rank] = V_full
    W_new[:, :current_rank] = W_full
    
    for local_i in range(block_size):
        for local_j in range(block_size):
            for local_k in range(block_size):
                local_idx = local_i * 4 + local_j * 2 + local_k  # This is wrong, need proper mapping
                
                # Actually, let's use the Strassen matrices directly
                # U_s has rows indexed by (local_i, local_j) -> row = local_i*2 + local_j
                # V_s has rows indexed by (local_j, local_k) -> row = local_j*2 + local_k
                # W_s has rows indexed by (local_k, local_i) -> row = local_k*2 + local_i
                
                u_row = (i0 + local_i) * m + (j0 + local_j)
                v_row = (j0 + local_j) * p + (k0 + local_k)
                w_row = (k0 + local_k) * n + (i0 + local_i)
                
                for r in range(7):
                    U_new[u_row, current_rank + r] = U_s[local_i * 2 + local_j, r]
                    V_new[v_row, current_rank + r] = V_s[local_j * 2 + local_k, r]
                    W_new[w_row, current_rank + r] = W_s[local_k * 2 + local_i, r]
    
    return U_new, V_new, W_new, current_rank

def _build_strassen_based(n, m, p):
    """
    Build a Strassen-based decomposition.
    For 2x5x6, we can decompose into blocks.
    
    Strategy: 
    - The 2x5 matrix has rows 0,1 and columns 0-4
    - The 5x6 matrix has rows 0-4 and columns 0-5
    
    We can group columns of A into pairs and rows of B into pairs.
    A is 2x5: group columns as (0,1), (2,3), (4) - last one is single
    B is 5x6: group rows as (0,1), (2,3), (4) - last one is single
    
    For each (col_group, row_group) pair, if both are size 2, use Strassen (rank 7)
    Otherwise use schoolbook for that sub-block.
    
    Actually, let me think more carefully.
    
    C = A * B where A is 2x5, B is 5x6, C is 2x6
    
    C[i,k] = sum_{j=0}^{4} A[i,j] * B[j,k]
    
    We can split the sum over j into groups:
    j in {0,1}, j in {2,3}, j in {4}
    
    For j in {0,1}: A[:, 0:2] is 2x2, B[0:2, :] is 2x6
    For j in {2,3}: A[:, 2:4] is 2x2, B[2:4, :] is 2x6
    For j in {4}: A[:, 4:5] is 2x1, B[4:5, :] is 1x6
    
    For the 2x2 * 2x6 case, we can further split columns of B.
    B[0:2, 0:6] -> split into B[0:2, 0:2], B[0:2, 2:4], B[0:2, 4:6]
    
    So for each 2x2 block of A and 2x2 block of B, we get a 2x2 output block.
    Use Strassen for each 2x2*2x2 -> 2x2 multiplication.
    
    Number of 2x2 A blocks: 2 (columns 0-1, 2-3) + 1 single (column 4)
    Number of 2x2 B blocks: 3 (columns 0-1, 2-3, 4-5) + but we need row groups of B
    
    Wait, I'm confusing myself. Let me re-approach.
    
    A is 2x5. Split A's columns into: A1 = A[:,0:2] (2x2), A2 = A[:,2:4] (2x2), A3 = A[:,4:5] (2x1)
    B is 5x6. Split B's rows into: B1 = B[0:2,:] (2x6), B2 = B[2:4,:] (2x6), B3 = B[4:5,:] (1x6)
    
    C = A1*B1 + A2*B2 + A3*B3
    
    Now A1*B1 is 2x2 * 2x6 = 2x6. Split B1's columns: B1a = B1[:,0:2], B1b = B1[:,2:4], B1c = B1[:,4:6]
    A1*B1 = A1*B1a + A1*B1b + A1*B1c, each 2x2*2x2=2x2. Use Strassen for each: 3*7 = 21 rank
    
    Similarly A2*B2: 3*7 = 21 rank
    
    A3*B3: 2x1 * 1x6 = 2x6. Schoolbook: 2*1*6 = 12 rank. But we can do better.
    A3*B3[i,k] = A3[i,0]*B3[0,k]. This is outer product. Rank 2*6 = 12 with schoolbook.
    Actually for 2x1 * 1x6, the rank is 2*6 = 12 (each output element needs one multiplication).
    
    Total: 21 + 21 + 12 = 54 rank.
    
    Let's implement this.
    """
    n, m, p = 2, 5, 6
    
    # We'll build the decomposition by accumulating contributions
    # Each "multiplication" adds columns to U, V, W
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_block(u_vals, v_vals, w_vals):
        """Add a block of multiplications.
        u_vals: dict mapping (i,j) -> value
        v_vals: dict mapping (j,k) -> value  
        w_vals: dict mapping (k,i) -> value
        """
        # For each unique combination, add a column
        # Actually, we need to be more careful.
        # Each "rank-1" component has specific U, V, W entries.
        # Let's just append columns.
        pass
    
    # Let's just build it directly with explicit Strassen formulas
    
    U = np.zeros((n * m, 0))
    V = np.zeros((m * p, 0))
    W = np.zeros((p * n, 0))
    
    def append_column(u_entries, v_entries, w_entries):
        """Append a single rank-1 component.
        u_entries: dict (row_idx) -> value
        v_entries: dict (row_idx) -> value
        w_entries: dict (row_idx) -> value
        """
        nonlocal U, V, W
        current_rank = U.shape[1]
        new_rank = current_rank + 1
        
        U_new = np.zeros((n * m, new_rank))
        V_new = np.zeros((m * p, new_rank))
        W_new = np.zeros((p * n, new_rank))
        
        U_new[:, :current_rank] = U
        V_new[:, :current_rank] = V
        W_new[:, :current_rank] = W
        
        for row, val in u_entries.items():
            U_new[row, current_rank] = val
        for row, val in v_entries.items():
            V_new[row, current_rank] = val
        for row, val in w_entries.items():
            W_new[row, current_rank] = val
        
        U, V, W = U_new, V_new, W_new
    
    def strassen_2x2x2(i0, j0, k0):
        """Add Strassen decomposition for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2] -> C[i0:i0+2, k0:k0+2]"""
        U_s, V_s, W_s = _build_strassen_222()
        
        for r in range(7):
            u_entries = {}
            v_entries = {}
            w_entries = {}
            
            for li in range(2):
                for lj in range(2):
                    val = U_s[li*2 + lj, r]
                    if val != 0:
                        u_entries[(i0 + li) * m + (j0 + lj)] = val
            
            for lj in range(2):
                for lk in range(2):
                    val = V_s[lj*2 + lk, r]
                    if val != 0:
                        v_entries[(j0 + lj) * p + (k0 + lk)] = val
            
            for lk in range(2):
                for li in range(2):
                    val = W_s[lk*2 + li, r]
                    if val != 0:
                        w_entries[(k0 + lk) * n + (i0 + li)] = val
            
            append_column(u_entries, v_entries, w_entries)
    
    def schoolbook_2x1x6(i0, j0, k0):
        """Add schoolbook for A[i0:i0+2, j0:j0+1] * B[j0:j0+1, k0:k0+6] -> C[i0:i0+2, k0:k0+6]"""
        for i in range(2):
            for k in range(6):
                # C[i0+i, k0+k] += A[i0+i, j0] * B[j0, k0+k]
                u_entries = {(i0 + i) * m + j0: 1}
                v_entries = {j0 * p + (k0 + k): 1}
                w_entries = {(k0 + k) * n + (i0 + i): 1}
                append_column(u_entries, v_entries, w_entries)
    
    # A1 = A[:, 0:2], B1 = B[0:2, :]
    # A1*B1: split B1 columns into (0:2), (2:4), (4:6)
    strassen_2x2x2(0, 0, 0)  # A[:,0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    strassen_2x2x2(0, 0, 2)  # A[:,0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    strassen_2x2x2(0, 0, 4)  # A[:,0:2] * B[0:2, 4:6] -> C[0:2, 4:6]
    
    # A2 = A[:, 2:4], B2 = B[2:4, :]
    strassen_2x2x2(0, 2, 0)  # A[:,2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    strassen_2x2x2(0, 2, 2)  # A[:,2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    strassen_2x2x2(0, 2, 4)  # A[:,2:4] * B[2:4, 4:6] -> C[0:2, 4:6]
    
    # A3 = A[:, 4:5], B3 = B[4:5, :]
    schoolbook_2x1x6(0, 4, 0)  # A[:,4:5] * B[4:5, 0:6] -> C[0:2, 0:6]
    
    return U, V, W

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    U: shape (n*m, rank)
    V: shape (m*p, rank)
    W: shape (p*n, rank)
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Build Strassen-based decomposition (rank 54)
    U_strassen, V_strassen, W_strassen = _build_strassen_based(n, m, p)
    
    # Verify
    if _verify_decomposition(U_strassen.tolist(), V_strassen.tolist(), W_strassen.tolist(), n, m, p):
        return {"U": U_strassen.tolist(), "V": V_strassen.tolist(), "W": W_strassen.tolist()}
    
    # Fallback to schoolbook
    U_school, V_school, W_school = _build_schoolbook(n, m, p)
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
