import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 6]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition is valid."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    
    # Convert to arrays
    U_arr = np.array(U, dtype=object)
    V_arr = np.array(V, dtype=object)
    W_arr = np.array(W, dtype=object)
    
    # Check shapes
    if U_arr.shape != (n * m, rank) or V_arr.shape != (m * p, rank) or W_arr.shape != (p * n, rank):
        return False
    
    # Check all (i,j,k) entries equal 1
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    # Compute tensor exactly using fractions
    T = np.zeros((n, m, p), dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(rank):
                    s += U_3d[i, j, r] * V_3d[j, k, r] * W_3d[k, i, r]
                T[i, j, k] = s
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if T[i, j, k] != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()

def _strassen_2x2x2():
    """Return Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U, V, W as lists of lists with shape (4, 7), (4, 7), (4, 7).
    Convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    # Strassen's 7 multiplications
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U: A indices (i,j) -> row i*2+j
    # V: B indices (j,k) -> row j*2+k  
    # W: C indices (k,i) -> row k*2+i
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (A11+A22)(B11+B22)
    # A11 is (0,0) -> row 0, A22 is (1,1) -> row 3
    # B11 is (0,0) -> row 0, B22 is (1,1) -> row 3
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    # C11 += M1, C22 += M1
    # C11 is (i=0,j=0) -> W index k*2+i, for C11: k=0,i=0 -> row 0
    # C22 is (i=1,j=1) -> W index k*2+i, for C22: k=1,i=1 -> row 3
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (A21+A22)B11
    # A21 is (1,0) -> row 2, A22 is (1,1) -> row 3
    # B11 is (0,0) -> row 0
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    # C21 += M2, C22 -= M2
    # C21 is (i=1,j=0) -> W index k*2+i, for C21: k=0,i=1 -> row 1
    # C22 is (i=1,j=1) -> W index k*2+i, for C22: k=1,i=1 -> row 3
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3: A11(B12-B22)
    # A11 is (0,0) -> row 0
    # B12 is (0,1) -> row 1, B22 is (1,1) -> row 3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    # C12 += M3, C22 += M3
    # C12 is (i=0,j=1) -> W index k*2+i, for C12: k=1,i=0 -> row 2
    # C22 is (i=1,j=1) -> W index k*2+i, for C22: k=1,i=1 -> row 3
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4: A22(B21-B11)
    # A22 is (1,1) -> row 3
    # B21 is (1,0) -> row 2, B11 is (0,0) -> row 0
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    # C11 += M4, C21 += M4
    # C11 is (i=0,j=0) -> W index k*2+i, for C11: k=0,i=0 -> row 0
    # C21 is (i=1,j=0) -> W index k*2+i, for C21: k=0,i=1 -> row 1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5: (A11+A12)B22
    # A11 is (0,0) -> row 0, A12 is (0,1) -> row 1
    # B22 is (1,1) -> row 3
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    # C11 -= M5, C12 += M5
    # C11 is (i=0,j=0) -> W index k*2+i, for C11: k=0,i=0 -> row 0
    # C12 is (i=0,j=1) -> W index k*2+i, for C12: k=1,i=0 -> row 2
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6: (A21-A11)(B11+B12)
    # A21 is (1,0) -> row 2, A11 is (0,0) -> row 0
    # B11 is (0,0) -> row 0, B12 is (0,1) -> row 1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    # C22 += M6
    # C22 is (i=1,j=1) -> W index k*2+i, for C22: k=1,i=1 -> row 3
    W[3, 5] = 1
    
    # M7: (A12-A22)(B21+B22)
    # A12 is (0,1) -> row 1, A22 is (1,1) -> row 3
    # B21 is (1,0) -> row 2, B22 is (1,1) -> row 3
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    # C11 += M7
    # C11 is (i=0,j=0) -> W index k*2+i, for C11: k=0,i=0 -> row 0
    W[0, 6] = 1
    
    return U.tolist(), V.tolist(), W.tolist()

def _block_decompose(n, m, p):
    """
    Try to decompose n×m×p using block structure.
    Split m into blocks of size 2, p into blocks of size 2.
    For each block, use Strassen if possible.
    """
    if m % 2 != 0 or p % 2 != 0:
        return None
    
    num_m_blocks = m // 2
    num_p_blocks = p // 2
    
    # Each block is n×2 × 2×2
    # We can use Strassen for 2×2×2 part, but we have n rows
    # Actually, for n×2 × 2×2, we need to multiply an n×2 matrix by a 2×2 matrix
    # This is just n independent 2×2 multiplications... no, it's one n×2 times 2×2 = n×2
    # Rank for n×2 × 2×2: schoolbook is n*2*2 = 4n
    # Can we do better?
    
    # Let's just use schoolbook for each block for now
    total_rank = 0
    U_blocks = []
    V_blocks = []
    W_blocks = []
    
    for mb in range(num_m_blocks):
        for pb in range(num_p_blocks):
            # Block: rows i=0..n-1, cols j=2*mb..2*mb+1, output k=2*pb..2*pb+1
            # This is n×2 × 2×2
            block_rank = n * 2 * 2  # schoolbook
            U_block = np.zeros((n * 2, block_rank))
            V_block = np.zeros((2 * 2, block_rank))
            W_block = np.zeros((2 * n, block_rank))
            
            col = 0
            for i in range(n):
                for jj in range(2):
                    for kk in range(2):
                        U_block[i*2+jj, col] = 1
                        V_block[jj*2+kk, col] = 1
                        W_block[kk*n+i, col] = 1
                        col += 1
            
            U_blocks.append(U_block)
            V_blocks.append(V_block)
            W_blocks.append(W_block)
            total_rank += block_rank
    
    # Combine blocks
    U_full = np.zeros((n * m, total_rank))
    V_full = np.zeros((m * p, total_rank))
    W_full = np.zeros((p * n, total_rank))
    
    col_offset = 0
    for mb in range(num_m_blocks):
        for pb in range(num_p_blocks):
            block_idx = mb * num_p_blocks + pb
            block_rank = n * 2 * 2
            
            # U: row i*m + j where j = 2*mb + jj
            for i in range(n):
                for jj in range(2):
                    U_full[i*m + 2*mb + jj, col_offset:col_offset+block_rank] = U_blocks[block_idx][i*2+jj, :]
            
            # V: row j*p + k where j = 2*mb + jj, k = 2*pb + kk
            for jj in range(2):
                for kk in range(2):
                    V_full[(2*mb+jj)*p + 2*pb + kk, col_offset:col_offset+block_rank] = V_blocks[block_idx][jj*2+kk, :]
            
            # W: row k*n + i where k = 2*pb + kk
            for kk in range(2):
                for i in range(n):
                    W_full[(2*pb+kk)*n + i, col_offset:col_offset+block_rank] = W_blocks[block_idx][kk*n+i, :]
            
            col_offset += block_rank
    
    return U_full.tolist(), V_full.tolist(), W_full.tolist(), total_rank

def _local_search(n, m, p, rank, seed, deadline):
    """
    Local search using block coordinate descent with decoupled analytic updates.
    """
    rng = np.random.RandomState(seed)
    
    # Initialize U, V, W with small random values
    U = rng.randn(n * m, rank) * 0.1
    V = rng.randn(m * p, rank) * 0.1
    W = rng.randn(p * n, rank) * 0.1
    
    # Target tensor T is all ones
    T = np.ones((n, m, p))
    
    # Function to update U given V, W
    def update_U(V, W):
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                u, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                U_new[i*m+j, :] = u
        return U_new

    # Function to update V given U, W
    def update_V(U, W):
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                B = np.zeros((n, rank))
                for i in range(n):
                    B[i, :] = U[i*m+j, :] * W[k*n+i, :]
                v, _, _, _ = np.linalg.lstsq(B, np.ones(n), rcond=None)
                V_new[j*p+k, :] = v
        return V_new

    # Function to update W given U, V
    def update_W(U, V):
        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                C = np.zeros((m, rank))
                for j in range(m):
                    C[j, :] = U[i*m+j, :] * V[j*p+k, :]
                w, _, _, _ = np.linalg.lstsq(C, np.ones(m), rcond=None)
                W_new[k*n+i, :] = w
        return W_new

    # Iterate
    for _ in range(100):
        if time.time() > deadline - 10:
            break
        
        U = update_U(V, W)
        V = update_V(U, W)
        W = update_W(U, V)
        
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        T_calc = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        err = np.linalg.norm(T_calc - T)
        
        if err < 1e-6:
            break
    
    # Try rounding to half-integers
    U_rounded = np.round(U * 2) / 2
    V_rounded = np.round(V * 2) / 2
    W_rounded = np.round(W * 2) / 2
    
    if _verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
        return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
    
    # Try rounding to integers
    U_int = np.round(U)
    V_int = np.round(V)
    W_int = np.round(W)
    if _verify(U_int.tolist(), V_int.tolist(), W_int.tolist(), n, m, p):
        return U_int.tolist(), V_int.tolist(), W_int.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U, "V": V, "W": W}
    
    if not _verify(U, V, W, n, m, p):
        raise ValueError("Schoolbook verification failed")
    
    # Try block decomposition
    block_result = _block_decompose(n, m, p)
    if block_result is not None:
        U_block, V_block, W_block, block_rank = block_result
        if _verify(U_block, V_block, W_block, n, m, p):
            if block_rank < best_rank:
                best_rank = block_rank
                best_result = {"U": U_block, "V": V_block, "W": W_block}
    
    # Try local search from best_rank - 1 down
    seeds = list(range(20))
    
    for r in range(best_rank - 1, 24, -1):
        if time.time() > deadline - 20:
            break
        
        found = False
        for seed in seeds:
            if time.time() > deadline - 10:
                break
            
            result = _local_search(n, m, p, r, seed, deadline)
            if result is not None:
                U_new, V_new, W_new = result
                if _verify(U_new, V_new, W_new, n, m, p):
                    best_result = {"U": U_new, "V": V_new, "W": W_new}
                    best_rank = r
                    found = True
                    break
        
        if found:
            pass
    
    return best_result
