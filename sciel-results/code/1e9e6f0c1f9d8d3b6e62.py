"""Program for rank-minimal 5x5x5 matrix multiplication tensor decomposition.
Strategy: 1+2+2 block split with structured cross-term sharing.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Verify tensor decomposition exactly using integer scaling by 2."""
    def scale2(mat):
        arr = np.array(mat, dtype=np.float64) * 2
        if np.max(np.abs(arr - np.round(arr))) > 1e-9:
            return None
        return np.round(arr).astype(np.int64)
    
    U2 = scale2(U)
    V2 = scale2(V)
    W2 = scale2(W)
    if U2 is None or V2 is None or W2 is None:
        return False
    
    target = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for a in range(n):
        for b in range(m):
            for c in range(p):
                target[a*m + b, b*p + c, c*n + a] = 1
    
    T_computed = np.einsum('ic,jc,kc->ijk', U2, V2, W2, optimize=True)
    
    if np.max(np.abs(T_computed - target * 8)) > 1e-6:
        return False
    return True


def _build_strassen_2x2():
    """Return U, V, W factors for 2x2x2 Strassen (rank 7)."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1: (A00+A11)*(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[1][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2: (A10+A11)*B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3: A00*(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4: A11*(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5: (A00+A01)*B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6: (A10-A00)*(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7: (A01-A11)*(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W


def _build_1x2_decomposition():
    """
    Build decomposition for 1x2 x 2x1 multiplication (scalar * vector = vector).
    Actually this is 1x2 matrix times 2x1 matrix = 1x1 scalar.
    Rank: just 1 term? No, it's a*1 + b*1 = a+b, that's rank 1.
    Wait: A is 1x2, B is 2x1, C is 1x1.
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    That's 2 rank-1 terms. Rank = 2.
    """
    # U: shape (1*2, rank) = (2, 2)
    # V: shape (2*1, rank) = (2, 2)
    # W: shape (1*1, rank) = (1, 2)
    U = [[1, 0],
         [0, 1]]
    V = [[1, 0],
         [0, 1]]
    W = [[1, 1]]
    return U, V, W


def _build_2x1_decomposition():
    """
    Build decomposition for 2x1 x 1x2 multiplication = 2x2 matrix.
    A is 2x1, B is 1x2, C is 2x2.
    C[i,j] = A[i,0] * B[0,j]
    That's 4 rank-1 terms (outer product of column and row).
    Rank = 4.
    """
    # U: shape (2*1, 4) = (2, 4)
    # V: shape (1*2, 4) = (2, 4)
    # W: shape (2*2, 4) = (4, 4)
    U = [[1, 0, 0, 0],
         [0, 1, 0, 0]]
    V = [[1, 1, 0, 0],
         [0, 0, 1, 1]]
    W = [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]]
    return U, V, W


def _build_1x1_decomposition():
    """1x1 x 1x1 = 1x1. Rank 1."""
    U = [[1]]
    V = [[1]]
    W = [[1]]
    return U, V, W


def _build_5x5_block_decomposition():
    """
    Build 5x5x5 decomposition using 1+2+2 block split.
    
    Partition indices 0..4 into blocks: [0], [1,2], [3,4]
    Block sizes: s = [1, 2, 2]
    
    For each pair of block rows (i) and block cols (j):
    - If both are size 2: use Strassen (rank 7)
    - If one is size 1 and other is size 2: use structured decomposition
    - If both are size 1: rank 1
    
    The key insight: cross-block terms where one dimension is 1 can be
    handled more efficiently by sharing structure.
    
    Block layout for 5x5:
    A = [[A00, A01, A02],
         [A10, A11, A12],
         [A20, A21, A22]]
    where A00 is 1x1, A01 is 1x2, A02 is 1x2,
          A10 is 2x1, A11 is 2x2, A12 is 2x2,
          A20 is 2x1, A21 is 2x2, A22 is 2x2
    
    Similarly for B (5x5) and C (5x5).
    
    C_ij = sum_k A_ik * B_kj
    
    We need to compute all C_ij blocks.
    
    Let's enumerate all (i,k,j) triples and determine the rank needed:
    
    (i,k,j) -> (size_i, size_k, size_j) -> sub-problem -> rank
    (0,0,0) -> (1,1,1) -> 1x1*1x1=1x1 -> rank 1
    (0,0,1) -> (1,1,2) -> 1x1*1x2=1x2 -> rank 2
    (0,0,2) -> (1,1,2) -> 1x1*1x2=1x2 -> rank 2
    (0,1,0) -> (1,2,1) -> 1x2*2x1=1x1 -> rank 2
    (0,1,1) -> (1,2,2) -> 1x2*2x2=1x2 -> rank ?
    (0,1,2) -> (1,2,2) -> 1x2*2x2=1x2 -> rank ?
    (0,2,0) -> (1,2,1) -> 1x2*2x1=1x1 -> rank 2
    (0,2,1) -> (1,2,2) -> 1x2*2x2=1x2 -> rank ?
    (0,2,2) -> (1,2,2) -> 1x2*2x2=1x2 -> rank ?
    (1,0,0) -> (2,1,1) -> 2x1*1x1=2x1 -> rank 2
    (1,0,1) -> (2,1,2) -> 2x1*1x2=2x2 -> rank 4
    (1,0,2) -> (2,1,2) -> 2x1*1x2=2x2 -> rank 4
    (1,1,0) -> (2,2,1) -> 2x2*2x1=2x1 -> rank ?
    (1,1,1) -> (2,2,2) -> 2x2*2x2=2x2 -> rank 7 (Strassen)
    (1,1,2) -> (2,2,2) -> 2x2*2x2=2x2 -> rank 7
    (1,2,0) -> (2,2,1) -> 2x2*2x1=2x1 -> rank ?
    (1,2,1) -> (2,2,2) -> 2x2*2x2=2x2 -> rank 7
    (1,2,2) -> (2,2,2) -> 2x2*2x2=2x2 -> rank 7
    (2,0,0) -> (2,1,1) -> 2x1*1x1=2x1 -> rank 2
    (2,0,1) -> (2,1,2) -> 2x1*1x2=2x2 -> rank 4
    (2,0,2) -> (2,1,2) -> 2x1*1x2=2x2 -> rank 4
    (2,1,0) -> (2,2,1) -> 2x2*2x1=2x1 -> rank ?
    (2,1,1) -> (2,2,2) -> 2x2*2x2=2x2 -> rank 7
    (2,1,2) -> (2,2,2) -> 2x2*2x2=2x2 -> rank 7
    (2,2,0) -> (2,2,1) -> 2x2*2x1=2x1 -> rank ?
    (2,2,1) -> (2,2,2) -> 2x2*2x2=2x2 -> rank 7
    (2,2,2) -> (2,2,2) -> 2x2*2x2=2x2 -> rank 7
    
    The "rank ?" cases need analysis:
    - 1x2 * 2x2 = 1x2: C[j] = sum_k A[k]*B[k,j] for j=0,1
      This is a 1x2 times 2x2. Rank: we can do it in rank 3?
      C0 = A0*B00 + A1*B10
      C1 = A0*B01 + A1*B11
      Naive: 4 terms. Can we do better?
      Let M1 = A0, M2 = A1, M3 = A0+A1
      C0 = M1*B00 + M2*B10
      C1 = M1*B01 + M2*B11
      Still 4. Actually the rank of 1x2*2x2 is 4 (it's just 2 independent dot products of length 2).
      Wait, no. The tensor is T[a,b,c] where a in {0,1} (row of A), b in {0,1} (col of A = row of B), c in {0,1} (col of B = col of C).
      T[a,b,c] = 1 if the multiplication gives C[c] += A[a]*B[b,c].
      So T[0,0,0]=1, T[0,1,0]=0, T[0,0,1]=0, T[0,1,1]=1, T[1,0,0]=0, T[1,1,0]=1, T[1,0,1]=1, T[1,1,1]=0
      Wait no. A is 1x2 (one row, two cols), B is 2x2.
      C[0,j] = sum_k A[0,k] * B[k,j]
      So C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
      C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
      
      Tensor indices: i (A row*col) -> just 0,1 for the two A entries
      j (B row*col) -> 0,1,2,3 for B[0,0],B[0,1],B[1,0],B[1,1]
      k (C row*col) -> 0,1 for C[0,0],C[0,1]
      
      T[i,j,k] = 1 if A[i] * B[j] contributes to C[k]
      T[0,0,0] = 1 (A[0]*B[0,0] -> C[0,0])
      T[0,1,1] = 1 (A[0]*B[0,1] -> C[0,1])
      T[1,2,0] = 1 (A[1]*B[1,0] -> C[0,0])
      T[1,3,1] = 1 (A[1]*B[1,1] -> C[0,1])
      
      This is a 2x4x2 tensor. Its rank: we need sum_r u_r[i]*v_r[j]*w_r[k] = T[i,j,k]
      This is equivalent to: for each k, C[k] = sum_i A[i] * (sum_r v_r[j]*w_r[k] * u_r[i])
      Actually it's just 4 rank-1 terms (one per nonzero entry). The rank is 4.
      
    - 2x2 * 2x1 = 2x1: similar, rank 4.
    
    So the naive total rank is:
    (0,0,0): 1
    (0,0,1): 2
    (0,0,2): 2
    (0,1,0): 2
    (0,1,1): 4
    (0,1,2): 4
    (0,2,0): 2
    (0,2,1): 4
    (0,2,2): 4
    (1,0,0): 2
    (1,0,1): 4
    (1,0,2): 4
    (1,1,0): 4
    (1,1,1): 7
    (1,1,2): 7
    (1,2,0): 4
    (1,2,1): 7
    (1,2,2): 7
    (2,0,0): 2
    (2,0,1): 4
    (2,0,2): 4
    (2,1,0): 4
    (2,1,1): 7
    (2,1,2): 7
    (2,2,0): 4
    (2,2,1): 7
    (2,2,2): 7
    
    Sum = 1+2+2+2+4+4+2+4+4+2+4+4+4+7+7+4+7+7+2+4+4+4+7+7+4+7+7
    Let me add: 
    Row 0: 1+2+2+2+4+4+2+4+4 = 25
    Row 1: 2+4+4+4+7+7+4+7+7 = 46
    Row 2: 2+4+4+4+7+7+4+7+7 = 46
    Total = 25+46+46 = 117
    
    That's better than 123 (the 2+3 split result) but worse than the 1+2+2 split claim of 76.
    
    Wait, the inspiration says "1+2+2 split instead of 2+3 allows three 2×2×2 Strassen blocks (saving 3 terms) versus one (saving 1 term), reducing rank from 123 to 76". Let me re-examine.
    
    Actually, I think the key is that in the 1+2+2 split, the cross terms between the 1-block and the 2-blocks can be handled more efficiently by sharing the Strassen structure. Let me think about this differently.
    
    Actually, I think the 76 claim might come from a more clever treatment where the 1-dimension blocks interact with the 2-dimension blocks in a way that shares rank-1 terms across multiple (i,k,j) triples.
    
    Let me just implement the naive 117-rank version first, verify it works, and then try to reduce.
    """
    n, m, p = 5, 5, 5
    nm, mp, pn = n*m, m*p, p*n
    
    # Block boundaries: block 0 = {0}, block 1 = {1,2}, block 2 = {3,4}
    block_sizes = [1, 2, 2]
    block_starts = [0, 1, 3]
    
    # Build the full decomposition
    U = np.zeros((nm, 0), dtype=np.float64)
    V = np.zeros((mp, 0), dtype=np.float64)
    W = np.zeros((pn, 0), dtype=np.float64)
    
    def add_term(u, v, w):
        """Add a rank-1 term."""
        nonlocal U, V, W
        U = np.column_stack([U, u]) if U.shape[1] > 0 else u.reshape(-1, 1)
        V = np.column_stack([V, v]) if V.shape[1] > 0 else v.reshape(-1, 1)
        W = np.column_stack([W, w]) if W.shape[1] > 0 else w.reshape(-1, 1)
    
    def add_block_product(i, k, j):
        """
        Add terms for C_block[i,j] += A_block[i,k] * B_block[k,j]
        i: row block of A and C
        k: col block of A / row block of B
        j: col block of B and C
        """
        si = block_sizes[i]
        sk = block_sizes[k]
        sj = block_sizes[j]
        
        # Global index offsets
        a_row_start = block_starts[i]
        a_col_start = block_starts[k]
        b_row_start = block_starts[k]
        b_col_start = block_starts[j]
        c_row_start = block_starts[i]
        c_col_start = block_starts[j]
        
        if si == 2 and sk == 2 and sj == 2:
            # Use Strassen for 2x2 * 2x2 = 2x2
            U2, V2, W2 = _build_strassen_2x2()
            for r in range(7):
                u = np.zeros(nm)
                v = np.zeros(mp)
                w = np.zeros(pn)
                for local_a in range(2):
                    for local_b in range(2):
                        u_idx = (a_row_start + local_a) * m + (a_col_start + local_b)
                        u[u_idx] = U2[local_a * 2 + local_b][r]
                for local_b in range(2):
                    for local_c in range(2):
                        v_idx = (b_row_start + local_b) * p + (b_col_start + local_c)
                        v[v_idx] = V2[local_b * 2 + local_c][r]
                for local_c in range(2):
                    for local_a in range(2):
                        w_idx = (c_col_start + local_c) * n + (c_row_start + local_a)
                        w[w_idx] = W2[local_a * 2 + local_c][r]
                add_term(u, v, w)
        
        elif si == 1 and sk == 1 and sj == 1:
            # 1x1 * 1x1 = 1x1: rank 1
            u = np.zeros(nm)
            v = np.zeros(mp)
            w = np.zeros(pn)
            u[a_row_start * m + a_col_start] = 1
            v[b_row_start * p + b_col_start] = 1
            w[c_col_start * n + c_row_start] = 1
            add_term(u, v, w)
        
        elif si == 1 and sk == 1 and sj == 2:
            # 1x1 * 1x2 = 1x2: rank 2
            for local_c in range(2):
                u = np.zeros(nm)
                v = np.zeros(mp)
                w = np.zeros(pn)
                u[a_row_start * m + a_col_start] = 1
                v[b_row_start * p + b_col_start + local_c] = 1
                w[(c_col_start + local_c) * n + c_row_start] = 1
                add_term(u, v, w)
        
        elif si == 1 and sk == 2 and sj == 1:
            # 1x2 * 2x1 = 1x1: rank 2
            for local_b in range(2):
                u = np.zeros(nm)
                v = np.zeros(mp)
                w = np.zeros(pn)
                u[a_row_start * m + a_col_start + local_b] = 1
                v[(b_row_start + local_b) * p + b_col_start] = 1
                w[c_col_start * n + c_row_start] = 1
                add_term(u, v, w)
        
        elif si == 2 and sk == 1 and sj == 1:
            # 2x1 * 1x1 = 2x1: rank 2
            for local_a in range(2):
                u = np.zeros(nm)
                v = np.zeros(mp)
                w = np.zeros(pn)
                u[(a_row_start + local_a) * m + a_col_start] = 1
                v[b_row_start * p + b_col_start] = 1
                w[c_col_start * n + c_row_start + local_a] = 1
                add_term(u, v, w)
        
        elif si == 2 and sk == 1 and sj == 2:
            # 2x1 * 1x2 = 2x2: rank 4
            for local_a in range(2):
                for local_c in range(2):
                    u = np.zeros(nm)
                    v = np.zeros(mp)
                    w = np.zeros(pn)
                    u[(a_row_start + local_a) * m + a_col_start] = 1
                    v[b_row_start * p + b_col_start + local_c] = 1
                    w[(c_col_start + local_c) * n + c_row_start + local_a] = 1
                    add_term(u, v, w)
        
        elif si == 1 and sk == 2 and sj == 2:
            # 1x2 * 2x2 = 1x2: rank 4
            for local_b in range(2):
                for local_c in range(2):
                    u = np.zeros(nm)
                    v = np.zeros(mp)
                    w = np.zeros(pn)
                    u[a_row_start * m + a_col_start + local_b] = 1
                    v[(b_row_start + local_b) * p + b_col_start + local_c] = 1
                    w[(c_col_start + local_c) * n + c_row_start] = 1
                    add_term(u, v, w)
        
        elif si == 2 and sk == 2 and sj == 1:
            # 2x2 * 2x1 = 2x1: rank 4
            for local_a in range(2):
                for local_b in range(2):
                    u = np.zeros(nm)
                    v = np.zeros(mp)
                    w = np.zeros(pn)
                    u[(a_row_start + local_a) * m + a_col_start + local_b] = 1
                    v[(b_row_start + local_b) * p + b_col_start] = 1
                    w[c_col_start * n + c_row_start + local_a] = 1
                    add_term(u, v, w)
        
        elif si == 1 and sk == 1 and sj == 1:
            pass  # handled above
    
    # Iterate over all (i, k, j) triples
    for i in range(3):
        for k in range(3):
            for j in range(3):
                add_block_product(i, k, j)
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return U_list, V_list, W_list


def _try_systematic_reduction(U, V, W, n, m, p, deadline):
    """
    Try to reduce rank by finding linear dependencies among columns.
    For each column, check if its tensor contribution is in the span of others.
    """
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    current_rank = U_arr.shape[1]
    nm, mp, pn = n*m, m*p, p*n
    
    # Build target tensor
    target = np.zeros((nm, mp, pn), dtype=np.float64)
    for a in range(n):
        for b in range(m):
            for c in range(p):
                target[a*m + b, b*p + c, c*n + a] = 1.0
    
    # For each column, try to express it as a combination of others
    # Use a subset of tensor entries for the linear system
    rng = np.random.RandomState(42)
    num_samples = 2000
    sample_idx = rng.randint(nm, size=num_samples)
    sample_jdx = rng.randint(mp, size=num_samples)
    sample_kdx = rng.randint(pn, size=num_samples)
    
    removed = []
    for col in range(current_rank):
        if time.time() > deadline - 15:
            break
        
        # Check if column 'col' is redundant
        other_cols = [c for c in range(current_rank) if c != col and c not in removed]
        if not other_cols:
            continue
        
        # Build system: for each sample, U[i,col]*V[j,col]*W[k,col] = sum_s alpha_s * U[i,s]*V[j,s]*W[k,s]
        A_mat = np.zeros((num_samples, len(other_cols)), dtype=np.float64)
        b_vec = np.zeros(num_samples, dtype=np.float64)
        
        for si, s in enumerate(other_cols):
            A_mat[:, si] = U_arr[sample_idx, s] * V_arr[sample_jdx, s] * W_arr[sample_kdx, s]
        
        b_vec = U_arr[sample_idx, col] * V_arr[sample_jdx, col] * W_arr[sample_kdx, col]
        
        if np.max(np.abs(b_vec)) < 1e-12:
            # Column contributes nothing, remove it
            removed.append(col)
            continue
        
        try:
            alpha, res, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        except:
            continue
        
        predicted = A_mat @ alpha
        if np.linalg.norm(b_vec - predicted) < 1e-8 * max(1.0, np.linalg.norm(b_vec)):
            removed.append(col)
    
    if not removed:
        return None
    
    # Remove columns
    keep = [c for c in range(current_rank) if c not in removed]
    U_new = [row[keep] for row in U]
    V_new = [row[keep] for row in V]
    W_new = [row[keep] for row in W]
    
    return U_new, V_new, W_new


def construct():
    deadline = time.time() + 135.0
    n, m, p = DIMENSIONS
    
    # Build the block decomposition
    U, V, W = _build_5x5_block_decomposition()
    current_rank = len(U[0])
    print(f"Initial block decomposition rank: {current_rank}")
    
    # Verify
    if _verify(U, V, W, n, m, p):
        print("Block decomposition verified OK")
        best_result = {"U": U, "V": V, "W": W}
    else:
        print("Block decomposition FAILED, falling back to schoolbook")
        rank_sb = n * m * p
        U_sb = [[0] * rank_sb for _ in range(n * m)]
        V_sb = [[0] * rank_sb for _ in range(m * p)]
        W_sb = [[0] * rank_sb for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][column] = 1
                    V_sb[j * p + k][column] = 1
                    W_sb[k * n + i][column] = 1
                    column += 1
        best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
        current_rank = rank_sb
    
    # Try systematic reduction
    if time.time() < deadline - 20:
        result = _try_systematic_reduction(U, V, W, n, m, p, deadline)
        if result is not None:
            U_new, V_new, W_new = result
            if _verify(U_new, V_new, W_new, n, m, p):
                new_rank = len(U_new[0])
                print(f"Reduced rank: {current_rank} -> {new_rank}")
                best_result = {"U": U_new, "V": V_new, "W": W_new}
                current_rank = new_rank
                U, V, W = U_new, V_new, W_new
    
    # Try iterative reduction
    if time.time() < deadline - 10:
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            result = _try_systematic_reduction(U, V, W, n, m, p, deadline)
            if result is not None:
                U_new, V_new, W_new = result
                if _verify(U_new, V_new, W_new, n, m, p):
                    new_rank = len(U_new[0])
                    print(f"Further reduced rank: {current_rank} -> {new_rank}")
                    best_result = {"U": U_new, "V": V_new, "W": W_new}
                    current_rank = new_rank
                    U, V, W = U_new, V_new, W_new
                else:
                    break
            else:
                break
    
    print(f"Best rank found: {current_rank}")
    return best_result

# EVOLVE-BLOCK-END
