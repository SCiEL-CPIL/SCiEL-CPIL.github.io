import time
import sys

DEADLINE = 150  # seconds, but we need to be well under this
TIME_LIMIT = 140  # safety margin

# Laderman's 23-term algorithm for 3x3x3 matrix multiplication
# The 23 terms are given as (U_row_idx, V_row_idx, W_col_idx) for the 3x3 blocks
# We use the standard Laderman algorithm
# U: 9x23, V: 9x23, W: 9x23

LADERMAN_U = [
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
]

# Actually, let me use the proper Laderman algorithm.
# The standard Laderman 3x3x3 algorithm has 23 terms.
# Let me encode them properly.

# Laderman's algorithm for 3x3x3:
# The 23 multiplication terms involve linear combinations of entries of A and B.
# For each term t (0..22), we have:
#   U[t]: a 9-vector (coefficients for A's 9 entries)
#   V[t]: a 9-vector (coefficients for B's 9 entries)  
#   W[t]: a 9-vector (coefficients for C's 9 entries)

# Let me use a well-known formulation of Laderman's algorithm.
# A = [[a11, a12, a13],
#      [a21, a22, a23],
#      [a31, a32, a33]]
# B = [[b11, b12, b13],
#      [b21, b22, b23],
#      [b31, b32, b33]]
# C[i,j] = sum_k A[i,k] * B[k,j]

# Laderman's 23 terms (one common formulation):
# I'll use the formulation from the literature.

def build_laderman_3x3x3():
    """Return U, V, W as lists of 9-element lists, each with 23 columns."""
    # 23 terms
    # U[i][t] = coefficient of A entry i in term t
    # V[i][t] = coefficient of B entry i in term t  
    # W[i][t] = coefficient of C entry i in term t
    
    # Using a standard Laderman algorithm encoding
    # A entries indexed as: 0:a11, 1:a12, 2:a13, 3:a21, 4:a22, 5:a23, 6:a31, 7:a32, 8:a33
    # B entries indexed as: 0:b11, 1:b12, 2:b13, 3:b21, 4:b22, 5:b23, 6:b31, 7:b32, 8:b33
    # C entries indexed as: 0:c11, 1:c12, 2:c13, 3:c21, 4:c22, 5:c23, 6:c31, 7:c32, 8:c33
    
    # I'll use a known correct set of 23 terms
    # This is based on the standard Laderman algorithm
    
    U = [[0]*23 for _ in range(9)]
    V = [[0]*23 for _ in range(9)]
    W = [[0]*23 for _ in range(9)]
    
    # Term 0: a11*b11 -> c11
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    # Term 1: a12*b21 -> c11
    U[1][1] = 1; V[3][1] = 1; W[0][1] = 1
    # Term 2: a13*b31 -> c11
    U[2][2] = 1; V[6][2] = 1; W[0][2] = 1
    # Term 3: a21*b12 -> c12
    U[3][3] = 1; V[1][3] = 1; W[1][3] = 1
    # Term 4: a22*b22 -> c12
    U[4][4] = 1; V[4][4] = 1; W[1][4] = 1
    # Term 5: a23*b32 -> c12
    U[5][5] = 1; V[7][5] = 1; W[1][5] = 1
    # Term 6: a31*b13 -> c13
    U[6][6] = 1; V[2][6] = 1; W[2][6] = 1
    # Term 7: a32*b23 -> c13
    U[7][7] = 1; V[5][7] = 1; W[2][7] = 1
    # Term 8: a33*b33 -> c13
    U[8][8] = 1; V[8][8] = 1; W[2][8] = 1
    
    # Now we need 14 more terms to handle the remaining c21, c22, c23, c31, c32, c33
    # with some terms combining multiple outputs
    
    # Actually, the schoolbook 3x3x3 already uses 27 terms. Laderman reduces to 23.
    # Let me use a different approach - just use the schoolbook for now and add Strassen for 2x2x2 blocks.
    
    # For a 3x5x7 problem, let's partition:
    # Rows of A: 3, Cols of A: 5, Rows of B: 5, Cols of B: 7
    # We can identify disjoint 2x2x2 blocks and apply Strassen to them.
    
    return U, V, W

def build_strassen_2x2x2():
    """Return U, V, W for Strassen 2x2x2 algorithm (7 terms).
    A entries: 0:a11, 1:a12, 2:a21, 3:a22
    B entries: 0:b11, 1:b12, 2:b21, 3:b22
    C entries: 0:c11, 1:c12, 2:c21, 3:c22
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (a11+a22)(b11+b22)
    U[0][0] = 1; U[3][0] = 1; V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[1][0] = -1; W[2][0] = -1; W[3][0] = 1
    
    # M2 = (a21+a22)b11
    U[2][1] = 1; U[3][1] = 1; V[0][1] = 1
    W[0][1] = 0; W[1][1] = 0; W[2][1] = 1; W[3][1] = 1
    
    # M3 = a11(b21-b11)
    U[0][2] = 1; V[2][2] = 1; V[0][2] = -1
    W[0][2] = 0; W[1][2] = 0; W[2][2] = 1; W[3][2] = 0
    
    # M4 = a22(b12-b22)
    U[3][3] = 1; V[1][3] = 1; V[3][3] = -1
    W[0][3] = 0; W[1][3] = 1; W[2][3] = 0; W[3][3] = -1
    
    # M5 = (a11+a12)b22
    U[0][4] = 1; U[1][4] = 1; V[3][4] = 1
    W[0][4] = 1; W[1][4] = 0; W[2][4] = 0; W[3][4] = 1
    
    # M6 = (a21-a11)(b11+b12)
    U[2][5] = 1; U[0][5] = -1; V[0][5] = 1; V[1][5] = 1
    W[0][5] = 1; W[1][5] = 0; W[2][5] = 0; W[3][5] = 1
    
    # M7 = (a12-a22)(b21+b22)
    U[1][6] = 1; U[3][6] = -1; V[2][6] = 1; V[3][6] = 1
    W[0][6] = 0; W[1][6] = 1; W[2][6] = 1; W[3][6] = 0
    
    return U, V, W

def build_schoolbook_1x1x1():
    """Return U, V, W for a single 1x1x1 multiplication."""
    U = [[1]]
    V = [[1]]
    W = [[1]]
    return U, V, W

def construct():
    start_time = time.time()
    
    n, m, p = 3, 5, 7
    
    # We'll build the decomposition by partitioning into blocks.
    # Strategy: partition m=5 into 2+2+1, p=7 into 2+2+2+1
    # For each (i_block, j_block, k_block) triple, decide which algorithm to use.
    
    # Actually, let's take a simpler approach:
    # Use schoolbook for all 105 terms, but replace disjoint 2x2x2 subproblems with Strassen.
    
    # In a 3x5x7 multiplication:
    # C[i,j] = sum_{k=0}^{4} A[i,k] * B[k,j]
    # where i in 0..2, k in 0..4, j in 0..6
    
    # A 2x2x2 block corresponds to:
    # i in {i0, i0+1}, k in {k0, k0+1}, j in {j0, j0+1}
    # This contributes to C[i0,j0], C[i0,j0+1], C[i0+1,j0], C[i0+1,j0+1]
    # from A[i0,k0], A[i0,k0+1], A[i0+1,k0], A[i0+1,k0+1]
    # and B[k0,j0], B[k0,j0+1], B[k0+1,j0], B[k0+1,j0+1]
    
    # We need to find a set of disjoint (in terms of the tensor factors) 2x2x2 blocks.
    # Two 2x2x2 blocks are disjoint if they don't share any A entry, B entry, or C entry.
    # Actually for the decomposition, they just need to be such that their contributions
    # to C are to different C entries OR the same C entries can be summed.
    
    # Wait, the decomposition is a sum of rank-1 tensors. Each rank-1 term contributes
    # to all C[i,j] entries via the product U[i,j term] * V[term] * W[C index].
    # So multiple terms can contribute to the same C[i,j].
    
    # Let me just do a greedy packing of Strassen blocks.
    
    # Find all possible 2x2x2 blocks:
    # i0 in {0,1}, k0 in {0,1,2,3}, j0 in {0,1,2,3,4,5}
    # A block uses A entries: (i0,k0), (i0,k0+1), (i0+1,k0), (i0+1,k0+1)
    #                  B entries: (k0,j0), (k0,j0+1), (k0+1,j0), (k0+1,j0+1)
    #                  C entries: (i0,j0), (i0,j0+1), (i0+1,j0), (i0+1,j0+1)
    
    # For disjointness in the tensor decomposition sense, we need the A-entry sets,
    # B-entry sets, and C-entry sets to be pairwise disjoint across blocks.
    
    blocks = []
    for i0 in range(n - 1):
        for k0 in range(m - 1):
            for j0 in range(p - 1):
                a_set = frozenset([(i0, k0), (i0, k0+1), (i0+1, k0), (i0+1, k0+1)])
                b_set = frozenset([(k0, j0), (k0, j0+1), (k0+1, j0), (k0+1, j0+1)])
                c_set = frozenset([(i0, j0), (i0, j0+1), (i0+1, j0), (i0+1, j0+1)])
                blocks.append((i0, k0, j0, a_set, b_set, c_set))
    
    # Greedy: pick blocks that maximize the number of blocks (each saves 1 rank: 8->7)
    # Sort by some heuristic and greedily select disjoint ones.
    
    selected_blocks = []
    used_a = set()
    used_b = set()
    used_c = set()
    
    for block in blocks:
        i0, k0, j0, a_set, b_set, c_set = block
        if a_set & used_a or b_set & used_b or c_set & used_c:
            continue
        selected_blocks.append(block)
        used_a |= a_set
        used_b |= b_set
        used_c |= c_set
    
    # Now build the decomposition:
    # For each selected block, add 7 Strassen terms.
    # For all remaining (i,k,j) triples not covered by any selected block's C entries,
    # add 1 schoolbook term each.
    
    # First, determine which (i,j) C entries are covered by selected blocks:
    covered_c = set()
    for block in selected_blocks:
        i0, k0, j0, a_set, b_set, c_set = block
        covered_c |= c_set
    
    # Build U, V, W matrices
    # U has shape (n*m, rank) = (15, rank)
    # V has shape (m*p, rank) = (35, rank)
    # W has shape (p*n, rank) = (21, rank)
    
    # Indexing:
    # U row: i*m + j (i in 0..2, j in 0..4)
    # V row: j*p + k (j in 0..4, k in 0..6)  -- wait, V is for B matrix
    # Actually, let me re-read the convention.
    # U[i*m+j, c] corresponds to A[i,j]
    # V[j*p+k, c] corresponds to B[j,k]
    # W[k*n+i, c] corresponds to C[k,i]  -- wait, "W[k*n+i,r]"
    # So W row index is k*n + i where k is the column index of B (which is row of C) and i is row of A (which is col of C)?
    # Actually for C[i,j] = sum_k A[i,k]*B[k,j], the output is C[i,j].
    # The convention says W[k*n+i, r]. So if we think of C as having indices (row, col) = (i, j),
    # then W row = j*n + i? Or is it that the output matrix is indexed differently?
    
    # Let me re-read: "output factor follows the official trace convention W[k*n+i,r]"
    # In matrix multiplication C = AB where A is n x m, B is m x p, C is n x p.
    # C[i,j] = sum_{k=0}^{m-1} A[i,k] * B[k,j]
    # The trace convention W[k*n+i, r] suggests that for each term r, 
    # W[k*n+i, r] gives the coefficient for C[i,k]? That doesn't make sense dimensionally.
    
    # Actually, I think the convention is:
    # The multiplication tensor T has indices (i, j, k) where i indexes rows of A (and C),
    # j indexes columns of A (and rows of B), k indexes columns of B (and C).
    # T[i,j,k] = delta (it's 1 for the standard multiplication).
    # A decomposition T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] means:
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]  ... no that's not right either.
    
    # Let me think again. The standard bilinear decomposition:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_r (sum_i' U[i',j,r] * A[i',j]) ... no.
    
    # The correct form: C = sum_r (U_r * A) * (V_r * B) * W_r? No.
    
    # The rank-1 tensor for multiplication is:
    # For a specific (i0, j0, k0): A[i0,j0] * B[j0,k0] contributes to C[i0,k0].
    # So the rank-1 term has:
    #   U: 1 at position (i0, j0) [flattened as i0*m + j0]
    #   V: 1 at position (j0, k0) [flattened as j0*p + k0]
    #   W: 1 at position (k0, i0) [flattened as k0*n + i0]
    
    # So W[k0*n + i0, r] = 1 for the term corresponding to (i0, j0, k0).
    # This means W is indexed by (output_col, output_row) = (k, i).
    
    # OK so:
    # U row index: i*m + j  (for A[i,j])
    # V row index: j*p + k  (for B[j,k])
    # W row index: k*n + i  (for C[i,k])
    
    # Now for Strassen on a 2x2x2 block with i0, k0 (for B row), j0 (for B col):
    # Wait, I need to be careful. Let's say the block is:
    # A rows: i0, i0+1
    # A cols / B rows: k0, k0+1
    # B cols / C cols: j0, j0+1
    # C rows: i0, i0+1
    # C cols: j0, j0+1
    
    # For the 2x2 submatrices:
    # A_sub[i, j] = A[i0+i, k0+j] for i,j in {0,1}
    # B_sub[i, j] = B[k0+i, j0+j] for i,j in {0,1}
    # C_sub[i, j] = C[i0+i, j0+j] for i,j in {0,1}
    
    # Strassen terms for this block:
    # M1 = (A_sub[0,0]+A_sub[1,1]) * (B_sub[0,0]+B_sub[1,1])
    #   U contributions: A[i0,k0] and A[i0+1,k0+1]
    #   V contributions: B[k0,j0] and B[k0+1,j0+1]
    #   C contributions: C[i0,j0] += M1, C[i0,j0+1] -= M1, C[i0+1,j0] -= M1, C[i0+1,j0+1] += M1
    
    # etc.
    
    strassen_U, strassen_V, strassen_W = build_strassen_2x2x2()
    
    # Now build the full decomposition
    # Start with empty lists for U, V, W columns
    U_cols = []
    V_cols = []
    W_cols = []
    
    # For each selected Strassen block:
    for block in selected_blocks:
        i0, k0, j0, a_set, b_set, c_set = block
        
        # A_sub indices in global:
        # A_sub[0,0] = A[i0, k0] -> U row: i0*m + k0
        # A_sub[0,1] = A[i0, k0+1] -> U row: i0*m + k0+1
        # A_sub[1,0] = A[i0+1, k0] -> U row: (i0+1)*m + k0
        # A_sub[1,1] = A[i0+1, k0+1] -> U row: (i0+1)*m + k0+1
        
        # B_sub indices in global:
        # B_sub[0,0] = B[k0, j0] -> V row: k0*p + j0
        # B_sub[0,1] = B[k0, j0+1] -> V row: k0*p + j0+1
        # B_sub[1,0] = B[k0+1, j0] -> V row: (k0+1)*p + j0
        # B_sub[1,1] = B[k0+1, j0+1] -> V row: (k0+1)*p + j0+1
        
        # C_sub indices in global:
        # C_sub[0,0] = C[i0, j0] -> W row: j0*n + i0
        # C_sub[0,1] = C[i0, j0+1] -> W row: (j0+1)*n + i0
        # C_sub[1,0] = C[i0+1, j0] -> W row: j0*n + (i0+1)
        # C_sub[1,1] = C[i0+1, j0+1] -> W row: (j0+1)*n + (i0+1)
        
        # For each of the 7 Strassen terms:
        for t in range(7):
            u_col = [0] * (n * m)
            v_col = [0] * (m * p)
            w_col = [0] * (p * n)
            
            # U contributions
            for s in range(4):
                coeff = strassen_U[s][t]
                if coeff != 0:
                    # s=0: A_sub[0,0], s=1: A_sub[0,1], s=2: A_sub[1,0], s=3: A_sub[1,1]
                    if s == 0:
                        u_idx = i0 * m + k0
                    elif s == 1:
                        u_idx = i0 * m + (k0 + 1)
                    elif s == 2:
                        u_idx = (i0 + 1) * m + k0
                    else:
                        u_idx = (i0 + 1) * m + (k0 + 1)
                    u_col[u_idx] += coeff
            
            # V contributions
            for s in range(4):
                coeff = strassen_V[s][t]
                if coeff != 0:
                    if s == 0:
                        v_idx = k0 * p + j0
                    elif s == 1:
                        v_idx = k0 * p + (j0 + 1)
                    elif s == 2:
                        v_idx = (k0 + 1) * p + j0
                    else:
                        v_idx = (k0 + 1) * p + (j0 + 1)
                    v_col[v_idx] += coeff
            
            # W contributions
            for s in range(4):
                coeff = strassen_W[s][t]
                if coeff != 0:
                    if s == 0:
                        w_idx = j0 * n + i0
                    elif s == 1:
                        w_idx = (j0 + 1) * n + i0
                    elif s == 2:
                        w_idx = j0 * n + (i0 + 1)
                    else:
                        w_idx = (j0 + 1) * n + (i0 + 1)
                    w_col[w_idx] += coeff
            
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)
    
    # Now add schoolbook terms for all (i, k, j) triples where C[i,j] is NOT covered
    # by any Strassen block. Actually, we need to be more careful.
    # The Strassen blocks handle specific 2x2 subproblems completely.
    # For the remaining parts, we need to handle them with schoolbook.
    
    # Actually, the issue is that Strassen blocks only handle their own 2x2 subproblem.
    # The remaining contributions to C come from other (i,k) or (k,j) pairs.
    
    # Let me reconsider. The full multiplication is:
    # C[i,j] = sum_{k=0}^{m-1} A[i,k] * B[k,j]
    
    # If I select a Strassen block for (i0,k0,j0), it handles:
    # C[i0,j0] += A[i0,k0]*B[k0,j0] + A[i0,k0+1]*B[k0+1,j0] ... no wait.
    # Actually Strassen computes the full 2x2 matrix product:
    # C_sub = A_sub * B_sub
    # where A_sub is 2x2 (rows i0,i0+1; cols k0,k0+1)
    # and B_sub is 2x2 (rows k0,k0+1; cols j0,j0+1)
    # So C_sub[i,j] = sum_{s=0}^{1} A_sub[i,s] * B_sub[s,j]
    # = sum_{s=0}^{1} A[i0+i, k0+s] * B[k0+s, j0+j]
    
    # This handles the contributions to C[i0+i, j0+j] from k = k0 and k = k0+1 only.
    # But C[i0+i, j0+j] also gets contributions from other k values (k not in {k0, k0+1}).
    
    # So I still need schoolbook terms for all (i, k, j) where the k is not part of
    # a Strassen block's k-range for that particular (i, j) pair.
    
    # This is getting complicated. Let me just use schoolbook for everything not
    # in the Strassen blocks' k-range.
    
    # For each (i, j) in C, the Strassen blocks that cover C[i,j] handle certain k values.
    # I need to add schoolbook terms for the remaining k values.
    
    # Let me track which (i, j, k) triples are handled by Strassen blocks:
    handled_triples = set()
    for block in selected_blocks:
        i0, k0, j0, a_set, b_set, c_set = block
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    handled_triples.add((i0 + di, k0 + dk, j0 + dj))
    
    # Now add schoolbook terms for all (i, k, j) NOT in handled_triples:
    for i in range(n):
        for k in range(m):
            for j in range(p):
                if (i, k, j) not in handled_triples:
                    u_col = [0] * (n * m)
                    v_col = [0] * (m * p)
                    w_col = [0] * (p * n)
                    u_col[i * m + k] = 1
                    v_col[k * p + j] = 1
                    w_col[j * n + i] = 1
                    U_cols.append(u_col)
                    V_cols.append(v_col)
                    W_cols.append(w_col)
    
    rank = len(U_cols)
    
    # Convert to the required format: lists of rows
    U = [[U_cols[c][r] for c in range(rank)] for r in range(n * m)]
    V = [[V_cols[c][r] for c in range(rank)] for r in range(m * p)]
    W = [[W_cols[c][r] for c in range(rank)] for r in range(p * n)]
    
    return {"U": U, "V": V, "W": W}

