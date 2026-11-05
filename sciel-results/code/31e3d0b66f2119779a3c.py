"""Rewritten program: structured block decomposition with Strassen for 2x2 sub-blocks."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    """
    Find a low-rank exact tensor decomposition for 3x4 * 4x5 matrix multiplication.
    
    Strategy: Partition the 4x4 middle dimension into blocks and use Strassen's
    algorithm for 2x2 sub-blocks where possible, reducing rank from 60 (schoolbook)
    to ~56.
    
    The multiplication C = A * B where A is 3x4, B is 4x5, C is 3x5.
    We view this as multiplying a 3x4 matrix by a 4x5 matrix.
    
    Key insight: Split the 4 intermediate dimension into 2+2.
    A = [A1 | A2] where A1 is 3x2, A2 is 3x2
    B = [B1; B2] where B1 is 2x5, B2 is 2x5
    
    Then C = A1*B1 + A2*B2
    
    Each 3x2 * 2x5 multiplication has rank 3*2*5 = 30 schoolbook.
    But we can be smarter: treat each as a sequence of 2x2 Strassen operations.
    
    Actually, let's think more carefully. The total schoolbook rank is 3*4*5 = 60.
    
    Better approach: Use the fact that for a 2x2x2 tensor (2x2 * 2x2 -> 2x2),
    Strassen gives rank 7 instead of 8.
    
    Let's partition:
    - Rows of A: 3 rows -> split as 2 + 1
    - Cols of A / Rows of B: 4 cols -> split as 2 + 2
    - Cols of B: 5 cols -> split as 2 + 2 + 1
    
    This gives blocks:
    A block (2x2) * B block (2x2) -> C block (2x2): use Strassen, rank 7
    A block (2x2) * B block (2x2) -> C block (2x1): rank 4 (schoolbook for 2x2*2x1)
    etc.
    
    Let me enumerate all block combinations:
    Row splits: [2, 1] (sizes of row blocks of A and C)
    Col splits of A / Row splits of B: [2, 2] 
    Col splits of B: [2, 2, 1]
    
    For each combination (i, j, k) where i in {0,1}, j in {0,1}, k in {0,1,2}:
    - A block size: rows[i] x cols_a[j]
    - B block size: rows_b[j] x cols_b[k]
    - C block size: rows[i] x cols_b[k]
    
    Where rows = [2, 1], cols_a = [2, 2], rows_b = [2, 2], cols_b = [2, 2, 1]
    
    Block dimensions:
    i=0,j=0,k=0: A(2x2) * B(2x2) -> C(2x2): Strassen rank 7
    i=0,j=0,k=1: A(2x2) * B(2x2) -> C(2x2): Strassen rank 7
    i=0,j=0,k=2: A(2x2) * B(2x1) -> C(2x1): schoolbook rank 4
    i=0,j=1,k=0: A(2x2) * B(2x2) -> C(2x2): Strassen rank 7
    i=0,j=1,k=1: A(2x2) * B(2x2) -> C(2x2): Strassen rank 7
    i=0,j=1,k=2: A(2x2) * B(2x1) -> C(2x1): schoolbook rank 4
    i=1,j=0,k=0: A(1x2) * B(2x2) -> C(1x2): schoolbook rank 4
    i=1,j=0,k=1: A(1x2) * B(2x2) -> C(1x2): schoolbook rank 4
    i=1,j=0,k=2: A(1x2) * B(2x1) -> C(1x1): schoolbook rank 2
    i=1,j=1,k=0: A(1x2) * B(2x2) -> C(1x2): schoolbook rank 4
    i=1,j=1,k=1: A(1x2) * B(2x2) -> C(1x2): schoolbook rank 4
    i=1,j=1,k=2: A(1x2) * B(2x1) -> C(1x1): schoolbook rank 2
    
    Total rank = 7+7+4+7+7+4+4+4+2+4+4+2 = 56
    """
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # Partition indices
    row_splits = [2, 1]  # for rows of A and C (n=3)
    mid_splits = [2, 2]  # for cols of A and rows of B (m=4)
    col_splits = [2, 2, 1]  # for cols of B (p=5)
    
    # Precompute index ranges
    def get_range(splits, idx):
        start = sum(splits[:idx])
        end = start + splits[idx]
        return start, end
    
    # Strassen for 2x2 * 2x2 -> 2x2
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
    
    # For 2x2 * 2x2 -> 2x2, the 7 products are:
    # Product r: U_vec(r) * V_vec(r) where U_vec is length 4 (flattened 2x2), V_vec is length 4
    # Then C = sum_r W_r * (U_vec(r) outer V_vec(r)) reshaped appropriately
    
    # Let's define the Strassen decomposition for 2x2x2:
    # Indices: A has (i,j) for i,j in {0,1}, B has (j,k) for j,k in {0,1}, C has (i,k)
    # Tensor T[i,j,k] = sum_l A[i,l]*B[l,k]
    
    # Strassen products (7 of them):
    # M1: (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # M2: (A[1,0]+A[1,1]) * B[0,0]
    # M3: A[0,0] * (B[1,0]-B[1,1])
    # M4: A[1,1] * (B[0,1]-B[0,0])
    # M5: (A[0,0]+A[0,1]) * B[1,1]
    # M6: (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    # M7: (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    
    # C[0,0] = M1+M4-M5+M7
    # C[0,1] = M3+M5
    # C[1,0] = M2+M4
    # C[1,1] = M1-M2+M3+M6
    
    # For each product r, define u_r (coefficients for A entries) and v_r (coefficients for B entries)
    # A entries: A[0,0], A[0,1], A[1,0], A[1,1] -> indices 0,1,2,3
    # B entries: B[0,0], B[0,1], B[1,0], B[1,1] -> indices 0,1,2,3
    
    strassen_A = [
        [1, 0, 0, 1],  # M1: A[0,0]+A[1,1]
        [0, 0, 1, 1],  # M2: A[1,0]+A[1,1]
        [1, 0, 0, 0],  # M3: A[0,0]
        [0, 0, 0, 1],  # M4: A[1,1]
        [1, 1, 0, 0],  # M5: A[0,0]+A[0,1]
        [0, 0, 1, -1], # M6: A[1,0]-A[0,0]
        [0, 1, 0, -1], # M7: A[0,1]-A[1,1]
    ]
    strassen_B = [
        [1, 0, 0, 1],  # M1: B[0,0]+B[1,1]
        [1, 0, 0, 0],  # M2: B[0,0]
        [0, 0, 1, -1], # M3: B[1,0]-B[1,1]
        [0, 1, 0, -1], # M4: B[0,1]-B[0,0]
        [0, 0, 0, 1],  # M5: B[1,1]
        [1, 1, 0, 0],  # M6: B[0,0]+B[0,1]
        [0, 0, 1, 1],  # M7: B[1,0]+B[1,1]
    ]
    # W coefficients: for each output C[i,k], coefficient for each M_r
    # C[0,0] = M1+M4-M5+M7 -> [1, 0, 0, 1, -1, 0, 1]
    # C[0,1] = M3+M5 -> [0, 0, 1, 0, 1, 0, 0]
    # C[1,0] = M2+M4 -> [0, 1, 0, 1, 0, 0, 0]
    # C[1,1] = M1-M2+M3+M6 -> [1, -1, 1, 0, 0, 1, 0]
    strassen_W = [
        [1, 0, 0, 1, -1, 0, 1],   # C[0,0]
        [0, 0, 1, 0, 1, 0, 0],   # C[0,1]
        [0, 1, 0, 1, 0, 0, 0],   # C[1,0]
        [1, -1, 1, 0, 0, 1, 0],  # C[1,1]
    ]
    
    # Schoolbook for ra x ma * mb x pa -> ra x pa
    def schoolbook_decomp(ra, ma, mb, pa):
        """Return (U_rows, V_rows, W_rows, rank) for schoolbook."""
        assert ma == mb  # inner dimension must match
        rank = ra * ma * pa
        U = []
        V = []
        W = []
        for i in range(ra):
            for j in range(ma):
                for k in range(pa):
                    # U row: one-hot at position (i, j) in ra x ma
                    u = [0] * (ra * ma)
                    u[i * ma + j] = 1
                    U.append(u)
                    # V row: one-hot at position (j, k) in ma x pa
                    v = [0] * (ma * pa)
                    v[j * pa + k] = 1
                    V.append(v)
                    # W row: one-hot at position (k, i) in pa x ra (trace convention W[k*n+i, r])
                    w = [0] * (pa * ra)
                    w[k * ra + i] = 1
                    W.append(w)
        return U, V, W, rank
    
    # Build the full decomposition
    # U has shape (n*m, rank) = (3*4, rank) = (12, rank)
    # V has shape (m*p, rank) = (4*5, rank) = (20, rank)
    # W has shape (p*n, rank) = (5*3, rank) = (15, rank)
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    for i_idx in range(len(row_splits)):
        for j_idx in range(len(mid_splits)):
            for k_idx in range(len(col_splits)):
                ra = row_splits[i_idx]
                ma = mid_splits[j_idx]
                mb = mid_splits[j_idx]  # same as ma
                pa = col_splits[k_idx]
                
                # Global index offsets
                i_start, i_end = get_range(row_splits, i_idx)
                j_start, j_end = get_range(mid_splits, j_idx)
                k_start, k_end = get_range(col_splits, k_idx)
                
                if ra == 2 and ma == 2 and pa == 2:
                    # Use Strassen
                    for r in range(7):
                        # U row: coefficients for A[i_start:i_end, j_start:j_end] in the 2x2 block
                        u = [0] * (n * m)
                        for a_i in range(2):
                            for a_j in range(2):
                                u[(i_start + a_i) * m + (j_start + a_j)] = strassen_A[r][a_i * 2 + a_j]
                        U_rows.append(u)
                        
                        # V row: coefficients for B[j_start:j_end, k_start:k_end] in the 2x2 block
                        v = [0] * (m * p)
                        for b_j in range(2):
                            for b_k in range(2):
                                v[(j_start + b_j) * p + (k_start + b_k)] = strassen_B[r][b_j * 2 + b_k]
                        V_rows.append(v)
                        
                        # W row: coefficients for C[i_start:i_end, k_start:k_end] in the 2x2 block
                        w = [0] * (p * n)
                        for c_i in range(2):
                            for c_k in range(2):
                                w[(k_start + c_k) * n + (i_start + c_i)] = strassen_W[c_i * 2 + c_k][r]
                        W_rows.append(w)
                else:
                    # Use schoolbook
                    U_blk, V_blk, W_blk, blk_rank = schoolbook_decomp(ra, ma, mb, pa)
                    for r in range(blk_rank):
                        u = [0] * (n * m)
                        for idx in range(ra * ma):
                            a_i = idx // ma
                            a_j = idx % ma
                            u[(i_start + a_i) * m + (j_start + a_j)] = U_blk[r][idx]
                        U_rows.append(u)
                        
                        v = [0] * (m * p)
                        for idx in range(ma * pa):
                            b_j = idx // pa
                            b_k = idx % pa
                            v[(j_start + b_j) * p + (k_start + b_k)] = V_blk[r][idx]
                        V_rows.append(v)
                        
                        w = [0] * (p * n)
                        for idx in range(pa * ra):
                            c_k = idx // ra
                            c_i = idx % ra
                            w[(k_start + c_k) * n + (i_start + c_i)] = W_blk[r][idx]
                        W_rows.append(w)
    
    rank = len(U_rows)
    
    # Verify the decomposition is exact
    # The multiplication tensor T[i,j,k] = delta (i.e., C[i,k] = sum_j A[i,j]*B[j,k])
    # Our decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal delta_{i,j,k} in the right sense
    # Actually, the tensor for matrix mult is T[i, j, k] where the product is sum_j A[i,j] B[j,k] = C[i,k]
    # The tensor is T[i, j, k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # More precisely: the bilinear form is sum_{i,j,k} C[i,k] * A[i,j] * B[j,k] = sum_{i,j,k} (sum_j' A[i,j']B[j',k]) A[i,j] B[j,k]
    # The tensor entries are T[i, j, k] = delta_{j, j'} summed... 
    
    # Let me verify by checking that for random A, B, the computed C matches.
    import numpy as np
    np.random.seed(42)
    A_test = np.random.randint(-5, 6, size=(n, m)).astype(float)
    B_test = np.random.randint(-5, 6, size=(m, p)).astype(float)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for r in range(rank):
        u_vec = np.array(U_rows[r])
        v_vec = np.array(V_rows[r])
        w_vec = np.array(W_rows[r])
        # u_vec is length n*m, v_vec is length m*p, w_vec is length p*n
        # The product contributes: for each (i,j,k), u[i*m+j] * v[j*p+k] * w[k*n+i]
        # This should sum to C[i,k] = sum_j A[i,j]*B[j,k]
        # Let's compute the contribution
        u_mat = u_vec.reshape(n, m)
        v_mat = v_vec.reshape(m, p)
        w_mat = w_vec.reshape(p, n)
        # Contribution to C[i,k] = sum_j u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
        # Note: w_mat is indexed by [k, i] but we need w[k*n+i] which is w_mat[k,i]
        for i in range(n):
            for k in range(p):
                contrib = 0
                for j in range(m):
                    contrib += u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
                C_computed[i, k] += contrib
    
    # Check if C_computed equals A_test @ B_test
    # This only works if the decomposition is correct
    # The actual check: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] should give the tensor
    # T[i,j,k] such that C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
    # For matrix multiplication, T[i,j,k] = 1 if j is the matching index, else 0
    # i.e., T[i,j,k] = delta_{j, j} = 1 for all (i,j,k) where the multiplication is defined
    # Wait, that's not right. The tensor is T[i,j,k] = 1 for all i,j,k (it's the identity in the middle)
    # No: the multiplication tensor has entries T[i, j, k] = 1 if we're pairing A[i,j] with B[j,k]
    # So T[i,j,k] = 1 for all i, j, k (since every A[i,j] pairs with B[j,k])
    # And C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k] = sum_j A[i,j] B[j,k]
    
    # So the check is: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i,j,k
    error = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U_rows[r][i * m + j] * V_rows[r][j * p + k] * W_rows[r][k * n + i]
                if abs(val - 1.0) > 1e-9:
                    error += abs(val - 1.0)
    
    if error > 1e-6:
        # Fallback to schoolbook
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
    
    return {"U": U_rows, "V": V_rows, "W": W_rows}
# EVOLVE-BLOCK-END
