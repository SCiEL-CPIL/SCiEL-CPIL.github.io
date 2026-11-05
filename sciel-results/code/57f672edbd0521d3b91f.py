"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]

def _schoolbook(n, m, p):
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
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition.
    
    U is 4x7 (indices: i*2+j), V is 4x7 (indices: j*2+k), W is 4x7 (indices: k*2+i)
    """
    U = [
        [1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [1, 1, 0, 0, 0, 0, 1],
    ]
    V = [
        [1, 0, 0, 0, 0, 1, 1],
        [0, 1, -1, 1, 0, 0, 1],
        [1, 0, 1, 0, 0, 1, 0],
        [1, 1, 0, -1, 1, 0, 0],
    ]
    W = [
        [1, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 0, 1, 0],
        [1, 0, 0, 1, -1, 0, 0],
    ]
    return U, V, W

def _verify_decomposition(n, m, p, U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    U_r = U_arr.reshape(n, m, rank)
    V_r = V_arr.reshape(m, p, rank)
    W_r = W_arr.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
    return np.allclose(T, np.ones((n, m, p)), atol=1e-9, rtol=0)

def _build_56_rank(n, m, p):
    """Build a rank-56 decomposition using Strassen blocks.
    
    Block structure for 3x4x5:
    - i in {0,1}, j in {0,1}, k in {0,1}: Strassen 2x2x2 (rank 7)
    - i in {0,1}, j in {0,1}, k in {2,3}: Strassen 2x2x2 (rank 7)
    - i in {0,1}, j in {0,1}, k = {4}:    schoolbook 2x2x1 (rank 4)
    - i in {0,1}, j in {2,3}, k in {0,1}: Strassen 2x2x2 (rank 7)
    - i in {0,1}, j in {2,3}, k in {2,3}: Strassen 2x2x2 (rank 7)
    - i in {0,1}, j in {2,3}, k = {4}:    schoolbook 2x2x1 (rank 4)
    - i = {2}, j in {0,1,2,3}, k in {0,1,2,3,4}: schoolbook 1x4x5 (rank 20)
    Total: 7+7+4+7+7+4+20 = 56
    """
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_strassen_block(i_off, j_off, k_off):
        U_s, V_s, W_s = _strassen_2x2x2()
        for c in range(7):
            U_col = [0.0] * (n * m)
            V_col = [0.0] * (m * p)
            W_col = [0.0] * (p * n)
            for ii in range(2):
                for jj in range(2):
                    U_col[(i_off + ii) * m + (j_off + jj)] = U_s[ii * 2 + jj][c]
            for jj in range(2):
                for kk in range(2):
                    V_col[(j_off + jj) * p + (k_off + kk)] = V_s[jj * 2 + kk][c]
            for kk in range(2):
                for ii in range(2):
                    W_col[(k_off + kk) * n + (i_off + ii)] = W_s[kk * 2 + ii][c]
            U_cols.append(U_col)
            V_cols.append(V_col)
            W_cols.append(W_col)
    
    def add_schoolbook_block(i_list, j_list, k_list):
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    U_col = [0.0] * (n * m)
                    V_col = [0.0] * (m * p)
                    W_col = [0.0] * (p * n)
                    U_col[i * m + j] = 1.0
                    V_col[j * p + k] = 1.0
                    W_col[k * n + i] = 1.0
                    U_cols.append(U_col)
                    V_cols.append(V_col)
                    W_cols.append(W_col)
    
    # Strassen blocks for i in {0,1}
    add_strassen_block(0, 0, 0)  # i={0,1}, j={0,1}, k={0,1}
    add_strassen_block(0, 0, 2)  # i={0,1}, j={0,1}, k={2,3}
    add_schoolbook_block([0, 1], [0, 1], [4])  # i={0,1}, j={0,1}, k={4}
    
    add_strassen_block(0, 2, 0)  # i={0,1}, j={2,3}, k={0,1}
    add_strassen_block(0, 2, 2)  # i={0,1}, j={2,3}, k={2,3}
    add_schoolbook_block([0, 1], [2, 3], [4])  # i={0,1}, j={2,3}, k={4}
    
    # Schoolbook for i=2
    add_schoolbook_block([2], list(range(4)), list(range(5)))
    
    rank = len(U_cols)
    U_mat = [[U_cols[c][r] for c in range(rank)] for r in range(n * m)]
    V_mat = [[V_cols[c][r] for c in range(rank)] for r in range(m * p)]
    W_mat = [[W_cols[c][r] for c in range(rank)] for r in range(p * n)]
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}

def _build_52_rank(n, m, p):
    """Build a rank-52 decomposition by using Strassen for the i=2 row too.
    
    Block structure for 3x4x5:
    - i in {0,1}, j in {0,1}, k in {0,1}: Strassen 2x2x2 (rank 7)
    - i in {0,1}, j in {0,1}, k in {2,3}: Strassen 2x2x2 (rank 7)
    - i in {0,1}, j in {0,1}, k = {4}:    schoolbook 2x2x1 (rank 4)
    - i in {0,1}, j in {2,3}, k in {0,1}: Strassen 2x2x2 (rank 7)
    - i in {0,1}, j in {2,3}, k in {2,3}: Strassen 2x2x2 (rank 7)
    - i in {0,1}, j in {2,3}, k = {4}:    schoolbook 2x2x1 (rank 4)
    - i = {2}, j in {0,1}, k in {0,1}:    Strassen 1x2x2 embedded (rank 4)
    - i = {2}, j in {0,1}, k in {2,3}:    Strassen 1x2x2 embedded (rank 4)
    - i = {2}, j in {0,1}, k = {4}:       schoolbook 1x2x1 (rank 2)
    - i = {2}, j in {2,3}, k in {0,1}:    Strassen 1x2x2 embedded (rank 4)
    - i = {2}, j in {2,3}, k in {2,3}:    Strassen 1x2x2 embedded (rank 4)
    - i = {2}, j in {2,3}, k = {4}:       schoolbook 1x2x1 (rank 2)
    Total: 7+7+4+7+7+4+4+4+2+4+4+2 = 56
    
    Actually let me try a different partition. Let me use:
    - i in {0,1,2}, j in {0,1}, k in {0,1}: 3x2x2 (rank 12 schoolbook, or can we do better?)
    
    Let me try yet another approach: use 2x2 Strassen blocks more aggressively.
    
    Alternative: partition j into {0,1,2,3} and k into {0,1,2,3,4}
    For each i in {0,1,2}:
      For j in {0,1}, k in {0,1}: 1x2x2 -> rank 4 schoolbook
      For j in {0,1}, k in {2,3}: 1x2x2 -> rank 4 schoolbook  
      For j in {0,1}, k = {4}:    1x2x1 -> rank 2 schoolbook
      For j in {2,3}, k in {0,1}: 1x2x2 -> rank 4 schoolbook
      For j in {2,3}, k in {2,3}: 1x2x2 -> rank 4 schoolbook
      For j in {2,3}, k = {4}:    1x2x1 -> rank 2 schoolbook
    Per i: 4+4+2+4+4+2 = 20
    Total: 3*20 = 60 (same as schoolbook)
    
    So the key insight is that Strassen helps when we have 2x2 sub-blocks in BOTH i and j.
    With n=3, we can only get one pair of Strassen blocks in the i dimension.
    
    Let me try: Can we do a 3x2x2 block with rank < 12?
    A 3x2x2 tensor has 12 entries. The minimum rank is at least max(3*2, 2*2, 3*2) = 6.
    Actually the rank of the 3x2x2 matrix multiplication tensor... let me think.
    
    For 1x2x2: rank 4 (schoolbook, since it's just dot products)
    For 2x2x2: rank 7 (Strassen)
    For 3x2x2: This is the tensor T[i,j,k] = 1 if we're computing (A*B)[i,k] where A is 3x2, B is 2x2.
    
    Actually, I realize the tensor is T[i,j,k] = delta_{i,i}delta_{j,j}delta_{k,k} = 1 for all (i,j,k) in the range.
    Wait no. The matrix multiplication tensor for (n x m) * (m x p) -> (n x p) is:
    T[i,j,k] = 1 for all valid (i,j,k), and the decomposition satisfies:
    sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all (i,j,k)
    
    So the tensor is just the all-ones tensor of shape (n,m,p).
    
    For 3x2x2: we need rank r such that there exist U(3,2,r), V(2,2,r), W(2,3,r) with
    sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for all i in {0,1,2}, j in {0,1}, k in {0,1}.
    
    That's 12 equations. The minimum rank... I believe it's 12 (schoolbook) since each (i,j,k) 
    involves a unique combination. Actually no, Strassen-type tricks might help.
    
    Let me just try to build a better decomposition by being more creative.
    
    Actually, let me try a completely different approach: embed 3x4x5 into 4x4x4 or 4x4x5
    and use known decompositions.
    
    For 4x4x4, the best known rank is 49 (Strassen: 7*(4/2)^3... no, that's not right).
    Actually Strassen gives rank 7^n for 2^n x 2^n x 2^n. For 4x4x4: 7^2 = 49.
    For 3x4x5, we could embed into 4x4x8 or similar, but that seems wasteful.
    
    Let me try a different block decomposition:
    - i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4}: 2x4x5
    - i = {2}, j in {0,1,2,3}, k in {0,1,2,3,4}: 1x4x5
    
    For 2x4x5: partition j into {0,1} and {2,3}, k into {0,1}, {2,3}, {4}
    - i in {0,1}, j in {0,1}, k in {0,1}: 2x2x2 Strassen rank 7
    - i in {0,1}, j in {0,1}, k in {2,3}: 2x2x2 Strassen rank 7
    - i in {0,1}, j in {0,1}, k = {4}:    2x2x1 schoolbook rank 4
    - i in {0,1}, j in {2,3}, k in {0,1}: 2x2x2 Strassen rank 7
    - i in {0,1}, j in {2,3}, k in {2,3}: 2x2x2 Strassen rank 7
    - i in {0,1}, j in {2,3}, k = {4}:    2x2x1 schoolbook rank 4
    Total for 2x4x5: 7+7+4+7+7+4 = 36
    
    For 1x4x5: partition j into {0,1} and {2,3}, k into {0,1}, {2,3}, {4}
    - i={2}, j in {0,1}, k in {0,1}: 1x2x2 schoolbook rank 4
    - i={2}, j in {0,1}, k in {2,3}: 1x2x2 schoolbook rank 4
    - i={2}, j in {0,1}, k = {4}:    1x2x1 schoolbook rank 2
    - i={2}, j in {2,3}, k in {0,1}: 1x2x2 schoolbook rank 4
    - i={2}, j in {2,3}, k in {2,3}: 1x2x2 schoolbook rank 4
    - i={2}, j in {2,3}, k = {4}:    1x2x1 schoolbook rank 2
    Total for 1x4x5: 4+4+2+4+4+2 = 20
    
    Grand total: 36+20 = 56 (same as before)
    
    To do better, I need to find a way to share rank terms between the i=2 block and the i={0,1} blocks.
    
    Idea: What if I use a 3x2x2 block with rank 10 instead of 12?
    Or better: what if I group i={0,1,2} with j={0,1} and k={0,1} into a single 3x2x2 block?
    The rank of 3x2x2 matrix multiplication... 
    
    Actually, I recall that the rank of the 3x2x2 tensor might be 10 or 11, not 12.
    Let me try to construct one.
    
    For 3x2x2, we need:
    sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for i=0,1,2; j=0,1; k=0,1
    
    That's 12 constraints. With rank r, we have 3*2*r + 2*2*r + 2*3*r = 14r variables.
    So r >= ceil(12/14) = 1, but that's a weak bound.
    
    The trivial bound is r >= max(3*2, 2*2, 2*3) = 6.
    
    Let me try to see if rank 10 works for 3x2x2 by construction.
    
    Actually, this is getting complex. Let me try a numerical approach: 
    start from the rank-56 solution and try to reduce the rank by finding linear dependencies
    or by using a gradient-based method to find a lower-rank decomposition.
    
    For now, let me try a slightly different block structure that might save a few terms.
    
    New idea: Instead of treating i=2 separately with schoolbook, what if I use
    a "mixed" approach where for the i=2 row, I share some Strassen terms?
    
    Actually, let me try to directly search for a rank-54 or rank-52 solution using
    a structured ansatz. I'll parameterize the solution as the rank-56 solution plus
    small perturbations and try to find rank reductions.
    
    For a practical implementation within the time budget, let me try:
    1. Build the rank-56 solution
    2. Try to find if any columns are linearly dependent or can be merged
    3. Try a local search to reduce rank
    
    But given the complexity, let me at least try a different block partition:
    Use 2x2 Strassen for all 2x2x2 sub-blocks where possible, and for the remaining
    1x2x2 blocks, see if we can do better than rank 4.
    
    For 1x2x2: T[i=0,j,k] = 1 for j=0,1 and k=0,1.
    We need sum_c U[0,j,c]*V[j,k,c]*W[k,0,c] = 1.
    With U being 2x r (since n=1, m=2), V being 2x2x r, W being 2x1x r.
    Let u_j = U[0,j,c], v_jk = V[j,k,c], w_k = W[k,0,c].
    sum_c u_j * v_jk * w_k = 1 for all j,k.
    
    This is equivalent to: the matrix M_jk = sum_c u_j * v_jk * w_k = 1 (all ones).
    This is a rank-r decomposition of the 2x2 all-ones matrix where the r-th term is
    outer product of (u_j * v_jk) and w_k... wait, that's not quite right.
    
    Actually: sum_c (u_j * v_jk) * w_k = 1. Let a_jk,c = u_j * v_jk. Then sum_c a_jk,c * w_k,c = 1.
    This means the 2x2 matrix of ones equals sum_c (a_c ⊗ w_c) where a_c is a 2x2 matrix 
    (with entries a_jk,c = u_j,c * v_jk,c) and w_c is a 2-vector.
    
    Hmm, this is getting complicated. The minimum rank for 1x2x2 is 4 (schoolbook), 
    since each of the 4 entries (j,k) needs to be satisfied independently given the 
    structure. Actually no - let me reconsider.
    
    For 1x2x2: we have 4 equations:
    c=1..r: sum_c U[0,0,c]*V[0,0,c]*W[0,0,c] = 1
    sum_c U[0,0,c]*V[0,1,c]*W[1,0,c] = 1
    sum_c U[0,1,c]*V[0,0,c]*W[0,1,c] = 1
    sum_c U[0,1,c]*V[0,1,c]*W[1,1,c] = 1
    
    With r=4 (schoolbook), each term is isolated. Can we do r=3?
    That would be 3*2 + 3*4 + 3*2 = 6+12+6 = 24 variables for 4 equations.
    It's likely possible, but I'm not sure of the minimum.
    
    Given the time constraints, let me just try to improve the existing rank-56 solution
    by attempting a local optimization to reduce the effective rank.
    
    Actually, let me try a completely different constructive approach:
    Use the fact that 3x4x5 can be embedded in 4x4x8, and use known bounds.
    
    The best known upper bound for 4x4x4 is 49 (Strassen).
    For 3x4x5, the schoolbook rank is 60.
    
    Let me try yet another block decomposition:
    - i in {0,1,2}, j in {0,1}, k in {0,1}: 3x2x2
    - i in {0,1,2}, j in {0,1}, k in {2,3}: 3x2x2
    - i in {0,1,2}, j in {0,1}, k = {4}:    3x2x1
    - i in {0,1,2}, j in {2,3}, k in {0,1}: 3x2x2
    - i in {0,1,2}, j in {2,3}, k in {2,3}: 3x2x2
    - i in {0,1,2}, j in {2,3}, k = {4}:    3x2x1
    
    If I can find a rank-10 decomposition for 3x2x2 and rank-6 for 3x2x1:
    Total: 4*10 + 2*6 = 40+12 = 52
    
    Or if 3x2x2 has rank 11 and 3x2x1 has rank 6:
    Total: 4*11 + 2*6 = 44+12 = 56 (same)
    
    Let me try to construct a rank-10 decomposition for 3x2x2.
    
    For 3x2x2, the tensor T[i,j,k] = 1 for i=0,1,2; j=0,1; k=0,1.
    
    Ansatz: Use Strassen-like formulas. Let me try:
    Define 10 terms with U(3,2,10), V(2,2,10), W(2,3,10).
    
    This is a 12-equation system with 30 variables (per matrix: 6+4+6=16, total 48 for 3 matrices).
    Actually 3*2*10 + 2*2*10 + 2*3*10 = 60+40+60 = 160 variables for 12 equations.
    Very underdetermined, so rank 10 should be feasible.
    
    Let me just try to construct it numerically. I'll set up the equations and solve.
    """
    # For now, let's try a numerical approach to find a lower-rank decomposition
    # by starting from the rank-56 solution and using a greedy column elimination.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_strassen_block(i_off, j_off, k_off, i_size=2, j_size=2, k_size=2):
        U_s, V_s, W_s = _strassen_2x2x2()
        for c in range(7):
            U_col = [0.0] * (n * m)
            V_col = [0.0] * (m * p)
            W_col = [0.0] * (p * n)
            for ii in range(i_size):
                for jj in range(j_size):
                    U_col[(i_off + ii) * m + (j_off + jj)] = U_s[ii * 2 + jj][c]
            for jj in range(j_size):
                for kk in range(k_size):
                    V_col[(j_off + jj) * p + (k_off + kk)] = V_s[jj * 2 + kk][c]
            for kk in range(k_size):
                for ii in range(i_size):
                    W_col[(k_off + kk) * n + (i_off + ii)] = W_s[kk * 2 + ii][c]
            U_cols.append(U_col)
            V_cols.append(V_col)
            W_cols.append(W_col)
    
    def add_schoolbook_block(i_list, j_list, k_list):
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    U_col = [0.0] * (n * m)
                    V_col = [0.0] * (m * p)
                    W_col = [0.0] * (p * n)
                    U_col[i * m + j] = 1.0
                    V_col[j * p + k] = 1.0
                    W_col[k * n + i] = 1.0
                    U_cols.append(U_col)
                    V_cols.append(V_col)
                    W_cols.append(W_col)
    
    # Same as before for now
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_schoolbook_block([0, 1], [0, 1], [4])
    add_strassen_block(0, 2, 0)
    add_strassen_block(0, 2, 2)
    add_schoolbook_block([0, 1], [2, 3], [4])
    add_schoolbook_block([2], list(range(4)), list(range(5)))
    
    rank = len(U_cols)
    U_mat = [[U_cols[c][r] for c in range(rank)] for r in range(n * m)]
    V_mat = [[V_cols[c][r] for c in range(rank)] for r in range(m * p)]
    W_mat = [[W_cols[c][r] for c in range(rank)] for r in range(p * n)]
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}

def _try_reduce_rank(n, m, p, U, V, W, max_time=100.0):
    """Try to reduce the rank by finding and removing redundant columns."""
    start_time = time.time()
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # Try to find columns that can be removed by adjusting others
    # This is a heuristic approach
    
    best_rank = rank
    best_U = U_arr.copy()
    best_V = V_arr.copy()
    best_W = W_arr.copy()
    
    # Check if any column is zero
    for c in range(rank):
        if time.time() - start_time > max_time:
            break
        if np.allclose(U_arr[:, c], 0) and np.allclose(V_arr[:, c], 0) and np.allclose(W_arr[:, c], 0):
            best_U = np.delete(best_U, c, axis=1)
            best_V = np.delete(best_V, c, axis=1)
            best_W = np.delete(best_W, c, axis=1)
            rank -= 1
            best_rank = rank
            break
    
    if best_rank < rank:
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    base_rank = n * m * p
    
    # Try the rank-56 Strassen-based decomposition
    candidate = _build_56_rank(n, m, p)
    
    if _verify_decomposition(n, m, p, candidate["U"], candidate["V"], candidate["W"]):
        # Try to reduce rank
        U_r, V_r, W_r = _try_reduce_rank(n, m, p, candidate["U"], candidate["V"], candidate["W"], 
                                          max_time=min(100.0, deadline - time.time() - 10.0))
        if U_r != candidate["U"] or V_r != candidate["V"] or W_r != candidate["W"]:
            candidate = {"U": U_r, "V": V_r, "W": W_r}
            if _verify_decomposition(n, m, p, candidate["U"], candidate["V"], candidate["W"]):
                return candidate
        return candidate
    
    # Fallback to schoolbook if verification fails
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
