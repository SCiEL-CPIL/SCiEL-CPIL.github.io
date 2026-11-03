import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140s deadline, 10s safety margin
    
    # Start with schoolbook (rank 72)
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best = {"U": U, "V": V, "W": W}
    best_rank = rank_schoolbook
    
    # Block-structured decomposition using Strassen for 2x2x2 blocks
    # Split n=3 as 2+1, m=4 as 2+2, p=6 as 2+2+2
    # 2x2x2 blocks: 1 * 2 * 3 = 6 blocks, each rank 7 -> 42
    # 1x2x2 blocks: 1 * 2 * 3 = 6 blocks, each rank 4 -> 24
    # Total: 66
    
    # Strassen's algorithm for 2x2x2
    def strassen_block(U_rows, V_rows, W_rows, base_idx, n_b, m_b, p_b):
        """
        Add Strassen decomposition for a 2x2x2 block.
        U_rows: list of 2 row indices in the full U matrix (offset by base_idx)
        V_rows: list of 2 row indices in the full V matrix
        W_rows: list of 2 row indices in the full W matrix
        Returns list of (U_vals, V_vals, W_vals) for the 7 Strassen terms
        """
        # Strassen's 7 products for 2x2 matrices:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B12 - B22)
        # M4 = A22(B21 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # The tensor entries for C[i,j] = sum_k A[i,k]B[k,j]
        # For 2x2: C11 = M1 - M4 + M7 - M5
        #          C12 = M3 + M5
        #          C21 = M2 + M4
        #          C22 = M1 - M2 + M3 + M6
        
        # U encodes A, V encodes B, W encodes the coefficient for C
        # U[row, col] = coefficient for A[row%2, col%2] in that product
        # V[row, col] = coefficient for B[row%2, col%2] in that product  
        # W[row, col] = coefficient for C[row%2, col%2] in the final sum
        
        terms = []
        
        # M1 = (A11+A22)(B11+B22): U has A11=1,A22=1; V has B11=1,B22=1
        # C11: +1, C12: 0, C21: 0, C22: +1
        terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
        
        # M2 = (A21+A22)B11: U has A21=1,A22=1; V has B11=1
        # C21: +1
        terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]))
        
        # M3 = A11(B12-B22): U has A11=1; V has B12=1,B22=-1
        # C12: +1
        terms.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]))
        
        # M4 = A22(B21-B11): U has A22=1; V has B21=1,B11=-1
        # C11: -1, C21: +1
        terms.append(([0, 0, 0, 1], [-1, 0, 1, 0], [-1, 0, 1, 0]))
        
        # M5 = (A11+A12)B22: U has A11=1,A12=1; V has B22=1
        # C11: -1, C12: +1
        terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
        
        # M6 = (A21-A11)(B11+B12): U has A21=1,A11=-1; V has B11=1,B12=1
        # C22: +1
        terms.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
        
        # M7 = (A12-A22)(B21+B22): U has A12=1,A22=-1; V has B21=1,B22=1
        # C11: +1
        terms.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
        
        return terms
    
    # Direct decomposition for 1x2x2 (rank 4)
    def direct_1x2x2(U_row, V_rows, W_rows, base_idx):
        """
        Direct decomposition for 1x2x2 block (just dot products).
        U has 1 row, V has 2 rows, W has 2 rows.
        Rank is 2*2=4 (each k,j pair needs its own term)
        """
        terms = []
        for k_idx in range(2):
            for j_idx in range(2):
                u_val = [1]  # single U row
                v_val = [0, 0, 0, 0]
                v_val[k_idx * 2 + j_idx] = 1  # B[k_idx, j_idx]
                w_val = [0, 0, 0, 0]
                w_val[j_idx] = 1  # C[j_idx]
                terms.append((u_val, v_val, w_val))
        return terms
    
    # Build the block decomposition
    U_cols = []
    V_cols = []
    W_cols = []
    
    # n=3: split into [0,1] and [2]
    # m=4: split into [0,1] and [2,3]
    # p=6: split into [0,1], [2,3], [4,5]
    
    n_blocks = [[0, 1], [2]]
    m_blocks = [[0, 1], [2, 3]]
    p_blocks = [[0, 1], [2, 3], [4, 5]]
    
    for ni, n_blk in enumerate(n_blocks):
        for mi, m_blk in enumerate(m_blocks):
            for pi, p_blk in enumerate(p_blocks):
                if len(n_blk) == 2 and len(m_blk) == 2 and len(p_blk) == 2:
                    # 2x2x2 block: use Strassen (rank 7)
                    # U rows: n_blk[0]*m + m_blk[0], n_blk[0]*m + m_blk[1],
                    #          n_blk[1]*m + m_blk[0], n_blk[1]*m + m_blk[1]
                    # V rows: m_blk[0]*p + p_blk[0], m_blk[0]*p + p_blk[1],
                    #          m_blk[1]*p + p_blk[0], m_blk[1]*p + p_blk[1]
                    # W rows: p_blk[0]*n + n_blk[0], p_blk[0]*n + n_blk[1],
                    #          p_blk[1]*n + n_blk[0], p_blk[1]*n + n_blk[1]
                    
                    u_full = [n_blk[0]*m + m_blk[0], n_blk[0]*m + m_blk[1],
                              n_blk[1]*m + m_blk[0], n_blk[1]*m + m_blk[1]]
                    v_full = [m_blk[0]*p + p_blk[0], m_blk[0]*p + p_blk[1],
                              m_blk[1]*p + p_blk[0], m_blk[1]*p + p_blk[1]]
                    w_full = [p_blk[0]*n + n_blk[0], p_blk[0]*n + n_blk[1],
                              p_blk[1]*n + n_blk[0], p_blk[1]*n + n_blk[1]]
                    
                    terms = strassen_block(u_full, v_full, w_full, 0, 2, 2, 2)
                    for u_vals, v_vals, w_vals in terms:
                        U_cols.append(u_vals)
                        V_cols.append(v_vals)
                        W_cols.append(w_vals)
                
                elif len(n_blk) == 1 and len(m_blk) == 2 and len(p_blk) == 2:
                    # 1x2x2 block: direct (rank 4)
                    u_row = n_blk[0]*m + 0  # only one row in n, but m has 2 cols
                    # U row index: n_blk[0]*m + m_blk[0] and n_blk[0]*m + m_blk[1]
                    # Actually for 1x2x2, U has shape (1*2, rank) = (2, rank)
                    # V has shape (2*2, rank) = (4, rank)
                    # W has shape (2*1, rank) = (2, rank)
                    
                    # U rows: n_blk[0]*m + m_blk[0], n_blk[0]*m + m_blk[1]
                    # V rows: m_blk[0]*p + p_blk[0], m_blk[0]*p + p_blk[1],
                    #          m_blk[1]*p + p_blk[0], m_blk[1]*p + p_blk[1]
                    # W rows: p_blk[0]*n + n_blk[0], p_blk[1]*n + n_blk[0]
                    
                    u_rows = [n_blk[0]*m + m_blk[0], n_blk[0]*m + m_blk[1]]
                    v_full = [m_blk[0]*p + p_blk[0], m_blk[0]*p + p_blk[1],
                              m_blk[1]*p + p_blk[0], m_blk[1]*p + p_blk[1]]
                    w_full = [p_blk[0]*n + n_blk[0], p_blk[1]*n + n_blk[0]]
                    
                    # Direct: for each (k, j) in 2x2, one term
                    # C[j] = sum_k A[j] * B[k, j]... wait, let me think again
                    # A is 1x2: A[0, j] for j=0,1
                    # B is 2x2: B[k, j] for k=0,1; j=0,1
                    # C is 2x1: C[j] for j=0,1
                    # C[j] = sum_k A[0, k] * B[k, j]
                    
                    # So U encodes A, V encodes B, W encodes C
                    # U has 2 entries (for A[0,0], A[0,1])
                    # V has 4 entries (for B[0,0], B[0,1], B[1,0], B[1,1])
                    # W has 2 entries (for C[0], C[1])
                    
                    for k_idx in range(2):
                        for j_idx in range(2):
                            # Term: A[0,k_idx] * B[k_idx, j_idx] contributes to C[j_idx]
                            u_vals = [0, 0]
                            u_vals[k_idx] = 1
                            v_vals = [0, 0, 0, 0]
                            v_vals[k_idx * 2 + j_idx] = 1
                            w_vals = [0, 0]
                            w_vals[j_idx] = 1
                            U_cols.append(u_vals)
                            V_cols.append(v_vals)
                            W_cols.append(w_vals)
    
    rank_found = len(U_cols)
    
    # Build matrices
    U_mat = np.zeros((n * m, rank_found))
    V_mat = np.zeros((m * p, rank_found))
    W_mat = np.zeros((p * n, rank_found))
    
    for c in range(rank_found):
        for idx, val in enumerate(U_cols[c]):
            if val != 0:
                U_mat[idx, c] = val
        for idx, val in enumerate(V_cols[c]):
            if val != 0:
                V_mat[idx, c] = val
        for idx, val in enumerate(W_cols[c]):
            if val != 0:
                W_mat[idx, c] = val
    
    # Convert to lists
    U_res = [list(map(int, U_mat[i])) for i in range(n * m)]
    V_res = [list(map(int, V_mat[i])) for i in range(m * p)]
    W_res = [list(map(int, W_mat[i])) for i in range(p * n)]
    
    if verify(U_res, V_res, W_res, n, m, p):
        if rank_found < best_rank:
            best = {"U": U_res, "V": V_res, "W": W_res}
            best_rank = rank_found
            print(f"Block decomposition rank {rank_found} verified")
    else:
        print(f"Block decomposition rank {rank_found} failed verification")
    
    # Try to find even better with random search from lower rank
    import random
    random.seed(42)
    
    for target_rank in range(best_rank - 1, 40, -1):
        if time.time() > deadline:
            break
        
        remaining = deadline - time.time()
        if remaining < 2:
            break
        
        max_attempts = min(2000, int(remaining * 30))
        if max_attempts < 5:
            break
        
        attempts = 0
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            
            vals = [0, 0, 0, 1, 1, -1, 2, -2]
            U_cand = [[random.choice(vals) for _ in range(target_rank)] for _ in range(n * m)]
            V_cand = [[random.choice(vals) for _ in range(target_rank)] for _ in range(m * p)]
            W_cand = [[random.choice(vals) for _ in range(target_rank)] for _ in range(p * n)]
            
            if verify(U_cand, V_cand, W_cand, n, m, p):
                best = {"U": U_cand, "V": V_cand, "W": W_cand}
                best_rank = target_rank
                print(f"Random search found rank {target_rank}")
                break
    
    return best

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140s deadline, 10s safety margin
    
    # Start with schoolbook (rank 72)
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best = {"U": U, "V": V, "W": W}
    best_rank = rank_schoolbook
    
    # Block-structured decomposition using Strassen for 2x2x2 blocks
    # n=3 as 2+1, m=4 as 2+2, p=6 as 2+2+2
    # 2x2x2 blocks: 1*2*3 = 6 blocks x rank 7 = 42
    # 1x2x2 blocks: 1*2*3 = 6 blocks x rank 4 = 24
    # Total: 66
    
    def strassen_2x2x2():
        """Return the 7 Strassen terms as (U_4, V_4, W_4) tuples for 2x2x2."""
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        #
        # C11 = M1 - M4 + M7 - M5
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        terms = []
        # M1
        terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
        # M2
        terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]))
        # M3
        terms.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]))
        # M4
        terms.append(([0, 0, 0, 1], [-1, 0, 1, 0], [-1, 0, 1, 0]))
        # M5
        terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
        # M6
        terms.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
        # M7
        terms.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
        
        return terms
    
    # Build block decomposition
    U_cols = []
    V_cols = []
    W_cols = []
    
    n_blocks = [[0, 1], [2]]
    m_blocks = [[0, 1], [2, 3]]
    p_blocks = [[0, 1], [2, 3], [4, 5]]
    
    strassen_terms = strassen_2x2x2()
    
    for ni, n_blk in enumerate(n_blocks):
        for mi, m_blk in enumerate(m_blocks):
            for pi, p_blk in enumerate(p_blocks):
                if len(n_blk) == 2 and len(m_blk) == 2 and len(p_blk) == 2:
                    # 2x2x2: use Strassen
                    for u_vals, v_vals, w_vals in strassen_terms:
                        U_cols.append(u_vals)
                        V_cols.append(v_vals)
                        W_cols.append(w_vals)
                else:
                    # 1x2x2: direct rank 4
                    # A is 1x2, B is 2x2, C is 2x1
                    # C[j] = sum_k A[0,k] * B[k,j]
                    for k_idx in range(2):
                        for j_idx in range(2):
                            u_vals = [0, 0]
                            u_vals[k_idx] = 1
                            v_vals = [0, 0, 0, 0]
                            v_vals[k_idx * 2 + j_idx] = 1
                            w_vals = [0, 0]
                            w_vals[j_idx] = 1
                            U_cols.append(u_vals)
                            V_cols.append(v_vals)
                            W_cols.append(w_vals)
    
    rank_found = len(U_cols)
    
    # Build full matrices
    U_mat = np.zeros((n * m, rank_found))
    V_mat = np.zeros((m * p, rank_found))
    W_mat = np.zeros((p * n, rank_found))
    
    for c in range(rank_found):
        for idx, val in enumerate(U_cols[c]):
            if val != 0:
                U_mat[idx, c] = val
        for idx, val in enumerate(V_cols[c]):
            if val != 0:
                V_mat[idx, c] = val
        for idx, val in enumerate(W_cols[c]):
            if val != 0:
                W_mat[idx, c] = val
    
    U_res = [list(map(int, U_mat[i])) for i in range(n * m)]
    V_res = [list(map(int, V_mat[i])) for i in range(m * p)]
    W_res = [list(map(int, W_mat[i])) for i in range(p * n)]
    
    if verify(U_res, V_res, W_res, n, m, p):
        if rank_found < best_rank:
            best = {"U": U_res, "V": V_res, "W": W_res}
            best_rank = rank_found
    else:
        # Fallback: verify schoolbook
        pass
    
    # Random search for potentially better rank
    import random
    random.seed(42)
    
    for target_rank in range(best_rank - 1, 40, -1):
        if time.time() > deadline:
            break
        remaining = deadline - time.time()
        if remaining < 2:
            break
        
        max_attempts = min(1000, int(remaining * 20))
        if max_attempts < 5:
            break
        
        attempts = 0
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            vals = [0, 0, 0, 1, 1, -1, 2, -2]
            U_cand = [[random.choice(vals) for _ in range(target_rank)] for _ in range(n * m)]
            V_cand = [[random.choice(vals) for _ in range(target_rank)] for _ in range(m * p)]
            W_cand = [[random.choice(vals) for _ in range(target_rank)] for _ in range(p * n)]
            
            if verify(U_cand, V_cand, W_cand, n, m, p):
                best = {"U": U_cand, "V": V_cand, "W": W_cand}
                best_rank = target_rank
                break
    
    return best
# EVOLVE-BLOCK-END
