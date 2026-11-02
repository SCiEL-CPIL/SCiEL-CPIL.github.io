"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strategy: Use a blocking decomposition.
    # We split the middle dimension m=4 into two blocks of size 2.
    # For each block, we split the first dimension n=3 into 1 and 2.
    # We also split the last dimension p=7 into blocks of size 2 (and one of size 1).
    
    # Rank calculation:
    # For each m-block (2 blocks):
    #   Part 1: n=1, m=2, p=7.
    #     Rank is 2*7 = 14 (Schoolbook, as rank(1xm x p) is mp).
    #   Part 2: n=2, m=2, p=7.
    #     We split p=7 into three 2s and one 1.
    #     Three 2x2x2 blocks: Rank 7 each -> 21.
    #     One 2x2x1 block: Rank 4 (Schoolbook).
    #     Total for Part 2: 25.
    #   Total per m-block: 14 + 25 = 39.
    # Total Rank = 2 * 39 = 78.
    
    # This is better than the schoolbook rank of 3*4*7 = 84.
    
    rank = 78
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        if col >= rank:
            raise ValueError("Rank exceeded")
        U[u_idx][col] = val
        V[v_idx][col] = val
        W[w_idx][col] = val
        col += 1

    # Strassen decomposition for 2x2x2
    # Indices: i in {0,1}, j in {0,1}, k in {0,1}
    # U size 4, V size 4, W size 4 (but W index is k*n+i, so for n=2, W is 4x? No, W is global.
    # In the local 2x2x2 block, the W indices correspond to global W indices.
    # Let's define a helper to add a Strassen block at offsets (i_off, j_off, k_off).
    
    def add_strassen_2x2x2(i_off, j_off, k_off):
        # The 7 products of Strassen for 2x2 matrices:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B11-B00)
        # M4 = A11(B00-B11)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Output C:
        # C00 = M1 - M4 - M3 + M7
        # C01 = M5 + M4
        # C10 = M2 + M3
        # C11 = M1 - M2 + M6 - M7
        
        # We need to add terms to U, V, W such that the tensor equation holds.
        # The tensor equation is sum_c U_ic V_jc W_kc = delta_{i, i'} delta_{j, j'} delta_{k, k'}?
        # No, the tensor is T_{ijk} = 1.
        # So for the local block, we need sum_{c in block} u_ic v_jc w_kc = 1 for all local i,j,k.
        
        # Let's verify Strassen vectors for the all-ones tensor of size 2x2x2.
        # Actually, Strassen is for matrix multiplication, which corresponds to the tensor
        # T_{ijk} = 1 if the indices are valid?
        # Yes, the multiplication tensor for 2x2x2 is the all-ones tensor.
        
        # Vectors (local indices 0,1):
        # u: 2x7, v: 2x7, w: 2x7 (but w is for output, which is 2x2?
        # Wait, the tensor is 2x2x2. The third dimension is k (0,1).
        # So w is a vector of length 2.
        # But Strassen computes a 2x2 matrix.
        # This means the "k" in the tensor T_{ijk} is NOT just the column of B.
        # In the problem, W[k*n+i].
        # If n=2, W has indices 0,1,2,3.
        # k=0 -> W[0], W[1] (for i=0,1)
        # k=1 -> W[2], W[3] (for i=0,1)
        # So the third dimension of the tensor is size p*n = 2*2 = 4.
        # So the tensor is 2x2x4.
        # The entries are T_{i, j, k*n+i'} = 1 if i=i' and j,j,k are valid?
        # No.
        # The condition is that the bilinear map reconstructs C.
        # C[i, k] = sum_j A[i, j] B[j, k].
        # The tensor T has indices (i, j, (k, i_out)).
        # T[i, j, k, i_out] = 1 if i == i_out.
        # So T_{i, j, w_idx} = 1 if w_idx corresponds to (k, i).
        # i.e., w_idx = k*n + i.
        # So for a fixed i, j, k, the term A[i,j]B[j,k] contributes to C[i,k].
        # So T[i, j, k*n+i] = 1.
        
        # Therefore, for a 2x2x2 block (n=2, m=2, p=2), the tensor size is 2x2x4.
        # The non-zero entries are at:
        # i=0, j=0, k=0 -> w=0
        # i=0, j=0, k=1 -> w=2
        # i=0, j=1, k=0 -> w=0
        # i=0, j=1, k=1 -> w=2
        # i=1, j=0, k=0 -> w=1
        # i=1, j=0, k=1 -> w=3
        # i=1, j=1, k=0 -> w=1
        # i=1, j=1, k=1 -> w=3
        
        # We need to decompose this tensor.
        # The Strassen algorithm provides a decomposition of the multiplication tensor.
        # Let's find the U, V, W vectors for Strassen.
        # The 7 products are:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B11-B00)
        # M4 = A11(B00-B11)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # The outputs are:
        # C00 = M1 - M4 - M3 + M7
        # C01 = M5 + M4
        # C10 = M2 + M3
        # C11 = M1 - M2 + M6 - M7
        
        # This means the tensor can be written as sum of 7 rank-1 terms.
        # Each term c has U_c, V_c, W_c.
        # U_c is a vector in R^4 (for i,j).
        # V_c is a vector in R^4 (for j,k).
        # W_c is a vector in R^4 (for k,i_out).
        
        # Let's derive the vectors.
        # Term 1 (M1):
        # U1: A00+A11 -> [1, 0, 0, 1] (indices 00, 01, 10, 11)
        # V1: B00+B11 -> [1, 0, 0, 1] (indices 00, 01, 10, 11)
        # W1: C00 + C11 -> [1, 0, 0, 1] (indices 00, 01, 10, 11)
        
        # Term 2 (M2):
        # U2: A10+A11 -> [0, 0, 1, 1]
        # V2: B00 -> [1, 0, 0, 0]
        # W2: C10 -> [0, 0, 1, 0]
        
        # Term 3 (M3):
        # U3: A00 -> [1, 0, 0, 0]
        # V3: B11-B00 -> [-1, 0, 0, 1]
        # W3: -C00 + C10 -> [-1, 0, 1, 0]
        
        # Term 4 (M4):
        # U4: A11 -> [0, 0, 0, 1]
        # V4: B00-B11 -> [1, 0, 0, -1]
        # W4: -C00 + C01 -> [-1, 1, 0, 0]
        
        # Term 5 (M5):
        # U5: A00+A01 -> [1, 1, 0, 0]
        # V5: B11 -> [0, 0, 0, 1]
        # W5: C01 -> [0, 1, 0, 0]
        
        # Term 6 (M6):
        # U6: A10-A00 -> [-1, 0, 1, 0]
        # V6: B00+B01 -> [1, 1, 0, 0]
        # W6: C11 -> [0, 0, 0, 1]
        
        # Term 7 (M7):
        # U7: A01-A11 -> [0, 1, 0, -1]
        # V7: B10+B11 -> [0, 0, 1, 1]
        # W7: C00 - C11 -> [1, 0, 0, -1]
        
        # Let's check if sum_{c=1}^7 U_c \otimes V_c \otimes W_c = T.
        # This is a standard result.
        
        # Now we map these local vectors to global indices.
        # Local i in {0,1} -> Global i = i_off + i
        # Local j in {0,1} -> Global j = j_off + j
        # Local k in {0,1} -> Global k = k_off + k
        # Local i_out in {0,1} -> Global i_out = i_off + i_out
        
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i_out
        
        # U1: [1, 0, 0, 1]
        # Indices: (0,0), (0,1), (1,0), (1,1)
        # Global: (i_off, j_off), (i_off, j_off+1), (i_off+1, j_off), (i_off+1, j_off+1)
        # Values: 1, 0, 0, 1
        # So U[i_off*m + j_off] += 1
        # U[i_off*m + j_off+1] += 0
        # U[i_off+1*m + j_off] += 0
        # U[i_off+1*m + j_off+1] += 1
        
        # Similarly for others.
        
        # We will add terms for each of the 7 products.
        # Each product corresponds to a column in U, V, W.
        # But wait, the decomposition is a sum of rank-1 tensors.
        # So each product c adds a column to U, V, W.
        # The entries in that column are the coefficients of the linear forms.
        
        # So for each c, we add a column.
        # U[:, c] = u_c
        # V[:, c] = v_c
        # W[:, c] = w_c
        
        # We will iterate c from 0 to 6.
        
        u_vecs = [
            [1, 0, 0, 1],  # M1
            [0, 0, 1, 1],  # M2
            [1, 0, 0, 0],  # M3
            [0, 0, 0, 1],  # M4
            [1, 1, 0, 0],  # M5
            [-1, 0, 1, 0], # M6
            [0, 1, 0, -1]  # M7
        ]
        
        v_vecs = [
            [1, 0, 0, 1],  # M1
            [1, 0, 0, 0],  # M2
            [-1, 0, 0, 1], # M3
            [1, 0, 0, -1], # M4
            [0, 0, 0, 1],  # M5
            [1, 1, 0, 0],  # M6
            [0, 0, 1, 1]   # M7
        ]
        
        w_vecs = [
            [1, 0, 0, 1],  # M1
            [0, 0, 1, 0],  # M2
            [-1, 0, 1, 0], # M3
            [-1, 1, 0, 0], # M4
            [0, 1, 0, 0],  # M5
            [0, 0, 0, 1],  # M6
            [1, 0, 0, -1]  # M7
        ]
        
        # Map local indices to global.
        # Local indices for U: (i, j)
        # Global U index: (i_off + i) * m + (j_off + j)
        # Local indices for V: (j, k)
        # Global V index: (j_off + j) * p + (k_off + k)
        # Local indices for W: (k, i_out)
        # Global W index: (k_off + k) * n + (i_off + i_out)
        
        # The vectors u_vecs[c] have 4 elements, corresponding to local (i,j) in order:
        # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
        
        for c in range(7):
            # Add column
            for idx in range(4):
                val = u_vecs[c][idx]
                if val != 0:
                    # Determine local i, j
                    li = idx // 2
                    lj = idx % 2
                    gi = i_off + li
                    gj = j_off + lj
                    U[gi * m + gj][col] = val
                    
                    val = v_vecs[c][idx]
                    if val != 0:
                        lj = idx // 2
                        lk = idx % 2
                        gj = j_off + lj
                        gk = k_off + lk
                        V[gj * p + gk][col] = val
                        
                    val = w_vecs[c][idx]
                    if val != 0:
                        lk = idx // 2
                        li_out = idx % 2
                        gk = k_off + lk
                        gi_out = i_off + li_out
                        W[gk * n + gi_out][col] = val
            
            col += 1

    # Iterate over m-blocks
    for block in range(2):
        j_off = block * 2
        
        # Part 1: n=1 (i=0), m=2, p=7
        # Rank 14 (Schoolbook)
        # T[i=0, j, k] = 1
        # U[0*m + j, c] = 1
        # V[j*p + k, c] = 1
        # W[k*n + 0, c] = 1
        for j in range(2):
            for k in range(7):
                U[0 * m + (j_off + j)][col] = 1
                V[(j_off + j) * p + k][col] = 1
                W[k * n + 0][col] = 1
                col += 1
        
        # Part 2: n=2 (i=1,2), m=2, p=7
        # Split p=7 into 2,2,2,1
        # k=0,1: Strassen
        add_strassen_2x2x2(1, j_off, 0)
        # k=2,3: Strassen
        add_strassen_2x2x2(1, j_off, 2)
        # k=4,5: Strassen
        add_strassen_2x2x2(1, j_off, 4)
        # k=6: Schoolbook 2x2x1
        # Rank 4
        for i in range(2):
            for j in range(2):
                U[(1 + i) * m + (j_off + j)][col] = 1
                V[(j_off + j) * p + 6][col] = 1
                W[6 * n + (1 + i)][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
