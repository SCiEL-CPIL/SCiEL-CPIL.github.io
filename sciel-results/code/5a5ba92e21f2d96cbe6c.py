"""Matrix multiplication 6x6x6 via 2x2 block decomposition of 3x3 blocks
with schoolbook rank-9 decomposition per block product.
Total rank: 4 * 9 = 36 (block-level) * ... wait, let me re-read.

Actually: 6x6 = 2x2 grid of 3x3 blocks.
C_{ij} = sum_k A_{ik} B_{kj} where each is a 3x3 matrix product.
So C has 4 blocks (top-left, top-right, bottom-left, bottom-right),
and for each output block position, we need one 3x3 matrix multiplication.

Each 3x3 multiplication has rank 9 (schoolbook).
So total rank = 4 * 9 = 36.

Wait, that's much better than 216! Let me think about this more carefully.

For 6x6 matrix multiplication:
C = A * B where A, B are 6x6.

Partition into 2x2 blocks of size 3x3:
A = [[A00, A01], [A10, A11]]
B = [[B00, B01], [B10, B11]]
C = [[C00, C01], [C10, C11]]

Where:
C00 = A00*B00 + A01*B10
C01 = A00*B01 + A01*B11
C10 = A10*B00 + A11*B10
C11 = A10*B01 + A11*B11

Each C block is a 3x3 matrix, and each A_ij, B_ij is a 3x3 matrix.
The product A00*B00 is a 3x3 matrix multiplication, which has a rank-9 tensor decomposition (schoolbook).

So we need:
- 8 individual 3x3 matrix multiplications: A00*B00, A01*B10, A00*B01, A01*B11, A10*B00, A11*B10, A10*B01, A11*B11
- Each has rank 9
- Total rank = 8 * 9 = 72

This is better than 216! Let me implement this.

The U matrix has shape (n*m, rank) = (36, rank)
The V matrix has shape (m*p, rank) = (36, rank)
The W matrix has shape (p*n, rank) = (36, rank)

For the schoolbook 3x3 decomposition:
For each pair of 3x3 matrices (X, Y) producing Z:
Z[i][j] = sum_k X[i][k] * Y[k][j]

The rank-9 decomposition has terms indexed by (i, k, j) where i, j, k are in 0..2.

For a single 3x3 multiplication, the rank-9 terms are:
For each (i, k, j):
  U part: X[i][k] -> index i*3+k in the 9x9 input
  V part: Y[k][j] -> index k*3+j in the 9x9 input  
  W part: Z[i][j] -> index j*3+i in the 9x9 output (following W[k*n+i,r] convention)

Wait, the task says: "The output factor follows the official trace convention W[k*n+i,r]."
So W[r][c] = sum over k of V[k*3 + something] * U[something + r]... 

Let me be more careful. In the standard formulation:
For matrix multiplication of (n x m) * (m x p) -> (n x p):
T[i,j,k] = A[i,k] * B[k,j] where i in 0..n-1, k in 0..m-1, j in 0..p-1.

The tensor decomposition gives:
C[i,j] = sum_r U[i*m+j, r] * V[j*p+k, r] ... no wait.

Actually in the standard notation for matrix multiplication tensor:
We have input A of size n x m, input B of size m x p, output C of size n x p.
The multiplication tensor has trilinear form: sum_{i,k,j} A[i,k] * B[k,j] * C[i,j].

The rank-r decomposition means:
C[i,j] = sum_r (sum_k U[i*m+k, r] * A[i,k]) * (sum_k V[k*p+j, r] * B[k,j])
... no, that's not right either.

Let me use the standard formulation. For matrix multiplication (n,m,p):
- U has shape (n*m, rank): U[a, r] where a = i*m + k
- V has shape (m*p, rank): V[b, r] where b = k*p + j  
- W has shape (p*n, rank): W[c, r] where c = j*n + i

And the condition is: for all i, k, j:
sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = 1
For all other (i,k,j) combinations that don't correspond to valid matrix mult, it should be 0.

Actually, the standard check is:
C[i,j] = sum_{k} A[i,k] * B[k,j]

In the tensor form:
sum_{i,k,j} A[i,k] B[k,j] C^*[i,j] where C^* is the dual.

The rank-r decomposition: T = sum_{r=0}^{rank-1} u_r ⊗ v_r ⊗ w_r

Where u_r is a vector in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}.

And: T[i,k;j,l] = sum_r u_r[i,k] * v_r[j,l] * w_r[?]... 

I'm getting confused with the index conventions. Let me just look at the current program:

Current schoolbook for (n,m,p) = (6,6,6), rank = 216:
For each (i, j, k):
  U[i*m + j][column] = 1  => U[i*6+j, col] = 1
  V[j*p + k][column] = 1  => V[j*6+k, col] = 1
  W[k*n + i][column] = 1  => W[k*6+i, col] = 1

So the convention is:
- U index: i*m + j (row i, col j of A)
- V index: j*p + k (row j, col k of B)
- W index: k*n + i (row k, col i of... C transposed? or C indexed as C[k][i]?)

Wait, the task says "W[k*n+i,r]" which matches. So W's first index is k*n+i, meaning it's indexed by (k, i) where k is the column of B / row of C's dual and i is the row of A / column of C's dual.

Actually, I think the convention is:
- A is n x m: entries A[i,j] for i in [0,n), j in [0,m)
- B is m x p: entries B[j,k] for j in [0,m), k in [0,p)
- C is n x p: entries C[i,k] for i in [0,n), k in [0,p)
- C[i,k] = sum_j A[i,j] * B[j,k]

Tensor: sum_{i,j,k} A[i,j] * B[j,k] * C^*[i,k]

Decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
where u_r has indices (i,j), v_r has indices (j,k), w_r has indices (i,k)

But the program says:
- U has shape (n*m, rank): indexed by i*m+j
- V has shape (m*p, rank): indexed by j*p+k
- W has shape (p*n, rank): indexed by k*n+i

So w_r has indices (k,i) not (i,k). That means W is indexed as W[k*n+i] which is (k, i) where k is the column index of C and i is the row index of C, but reversed in the flattening.

So C^*[i,k] corresponds to W[k*n+i]. This is just a reindexing.

OK so for my 6x6 = 2x2 blocks of 3x3:

Block A00 is rows 0-2, cols 0-2 of A
Block A01 is rows 0-2, cols 3-5 of A
Block A10 is rows 3-5, cols 0-2 of A
Block A11 is rows 3-5, cols 3-5 of A

Similarly for B and C.

C00 = A00 * B00 + A01 * B10 (all 3x3 matrix products)
C01 = A00 * B01 + A01 * B11
C10 = A10 * B00 + A11 * B10
C11 = A10 * B01 + A11 * B11

8 products total, each 3x3 with rank 9, so total rank 72.

For each 3x3 product, say X * Y = Z where X is rows x0-x2, cols y0-y2 of the full 6x6:

Schoolbook rank-9 for 3x3:
For each (i_local, k_local, j_local) where each is in 0..2:
  The local row i_local corresponds to global row x0 + i_local
  The local col of X, k_local corresponds to global col y0 + k_local
  The local row of Y, k_local corresponds to global row y1 + k_local (where y1 is start row of Y block)
  The local col of Y, j_local corresponds to global col z0 + j_local
  The local row of Z, i_local corresponds to global row x0 + i_local
  The local col of Z, j_local corresponds to global col z0 + j_local

U index: global_row_of_A * 6 + global_col_of_A
V index: global_row_of_B * 6 + global_col_of_B
W index: global_col_of_C * 6 + global_row_of_C

Let me define the 8 products:
1. A00 * B00 -> C00
2. A01 * B10 -> C00
3. A00 * B01 -> C01
4. A01 * B11 -> C01
5. A10 * B00 -> C10
6. A11 * B10 -> C10
7. A10 * B01 -> C11
8. A11 * B11 -> C11

For product (Ablock, Bblock, Cblock) where:
  Ablock starts at row ar, col ac
  Bblock starts at row br, col bc
  Cblock starts at row cr, col cc

For each (i, k, j) in range(3)^3:
  global A index: (ar+i) * 6 + (ac+k)
  global B index: (br+k) * 6 + (bc+j)
  global C index: (cc+j) * 6 + (cr+i)  [since W[k*n+i] means col*k + row... wait]

W index is k*n + i where k is the column of C and i is the row of C.
So W index = global_col_C * 6 + global_row_C = (cc+j) * 6 + (cr+i)

Let me code this up.
"""

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def _bclr_rank21_3x3():
    """Return BCLR rank-21 decomposition for 3x3 matrix multiplication.
    
    Returns list of 21 tuples (u, v, w) where:
    - u: 9-vector, u[i*3+j] is coefficient for A[i,j]
    - v: 9-vector, v[j*3+k] is coefficient for B[j,k]
    - w: 9-vector, w[k*3+i] is coefficient for C[i,k] (W[k*n+i] convention)
    
    Verification: for all i,j,k in {0,1,2}:
        sum_r u_r[i*3+j] * v_r[j*3+k] * w_r[k*3+i] == 1
    """
    terms = []
    
    # BCLR rank-21 decomposition for 3x3
    # Using the explicit decomposition from Bini, Capovani, Lotti, Romani (1978)
    # 
    # The decomposition is structured as follows. We define 21 terms.
    # Each term (u, v, w) contributes u[i,j]*v[j,k]*w[k,i] to the tensor.
    
    # I'll use a verified construction. The BCLR decomposition can be 
    # described using the following pattern:
    
    # Let's define helper to create sparse terms
    def make_term(u_dict, v_dict, w_dict):
        u = [0.0] * 9
        v = [0.0] * 9
        w = [0.0] * 9
        for idx, val in u_dict.items():
            u[idx] = val
        for idx, val in v_dict.items():
            v[idx] = val
        for idx, val in w_dict.items():
            w[idx] = val
        return (u, v, w)
    
    # BCLR rank-21 for 3x3:
    # The 21 terms are organized in 7 groups of 3.
    
    # Group 1: Terms involving A[0,*]
    # These handle contributions to C[0,*]
    
    # Term 0: A[0,0]*B[0,0] -> C[0,0]
    terms.append(make_term({0:1}, {0:1}, {0:1}))
    # Term 1: A[0,0]*B[0,1] -> C[0,1]
    terms.append(make_term({0:1}, {1:1}, {3:1}))
    # Term 2: A[0,0]*B[0,2] -> C[0,2]
    terms.append(make_term({0:1}, {2:1}, {6:1}))
    
    # Term 3: A[0,1]*B[1,0] -> C[0,0]
    terms.append(make_term({1:1}, {3:1}, {0:1}))
    # Term 4: A[0,1]*B[1,1] -> C[0,1]
    terms.append(make_term({1:1}, {4:1}, {3:1}))
    # Term 5: A[0,1]*B[1,2] -> C[0,2]
    terms.append(make_term({1:1}, {5:1}, {6:1}))
    
    # Term 6: A[0,2]*B[2,0] -> C[0,0]
    terms.append(make_term({2:1}, {6:1}, {0:1}))
    # Term 7: A[0,2]*B[2,1] -> C[0,1]
    terms.append(make_term({2:1}, {7:1}, {3:1}))
    # Term 8: A[0,2]*B[2,2] -> C[0,2]
    terms.append(make_term({2:1}, {8:1}, {6:1}))
    
    # Group 2: Terms involving A[1,*]
    # These handle contributions to C[1,*]
    
    # Term 9: A[1,0]*B[0,0] -> C[1,0]
    terms.append(make_term({3:1}, {0:1}, {1:1}))
    # Term 10: A[1,0]*B[0,1] -> C[1,1]
    terms.append(make_term({3:1}, {1:1}, {4:1}))
    # Term 11: A[1,0]*B[0,2] -> C[1,2]
    terms.append(make_term({3:1}, {2:1}, {7:1}))
    
    # Term 12: A[1,1]*B[1,0] -> C[1,0]
    terms.append(make_term({4:1}, {3:1}, {1:1}))
    # Term 13: A[1,1]*B[1,1] -> C[1,1]
    terms.append(make_term({4:1}, {4:1}, {4:1}))
    # Term 14: A[1,1]*B[1,2] -> C[1,2]
    terms.append(make_term({4:1}, {5:1}, {7:1}))
    
    # Term 15: A[1,2]*B[2,0] -> C[1,0]
    terms.append(make_term({5:1}, {6:1}, {1:1}))
    # Term 16: A[1,2]*B[2,1] -> C[1,1]
    terms.append(make_term({5:1}, {7:1}, {4:1}))
    # Term 17: A[1,2]*B[2,2] -> C[1,2]
    terms.append(make_term({5:1}, {8:1}, {7:1}))
    
    # Group 3: Terms involving A[2,*]
    # These handle contributions to C[2,*]
    
    # Term 18: A[2,0]*B[0,0] -> C[2,0]
    terms.append(make_term({6:1}, {0:1}, {2:1}))
    # Term 19: A[2,0]*B[0,1] -> C[2,1]
    terms.append(make_term({6:1}, {1:1}, {5:1}))
    # Term 20: A[2,0]*B[0,2] -> C[2,2]
    terms.append(make_term({6:1}, {2:1}, {8:1}))
    
    # Wait, that's only 21 terms but I'm missing A[2,1] and A[2,2] contributions!
    # C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0]
    # C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1] + A[2,2]*B[2,1]
    # C[2,2] = A[2,0]*B[0,2] + A[2,1]*B[1,2] + A[2,2]*B[2,2]
    
    # I've only covered A[2,0] contributions. I need A[2,1] and A[2,2] too.
    # That would add 6 more terms, making it 27 (schoolbook).
    
    # So my "21 terms" above are NOT a valid decomposition - they're missing terms.
    # This is just the schoolbook for the first 2 rows of A.
    
    # I clearly cannot produce a valid BCLR rank-21 decomposition from memory.
    # Let me fall back to schoolbook rank-27 for 3x3.
    
    # Actually, let me just return the full schoolbook (27 terms) and use that.
    # Total rank = 8 * 27 = 216, same as current.
    
    # Let me build the full schoolbook 3x3:
    schoolbook = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                u = [0.0]*9
                v = [0.0]*9
                w = [0.0]*9
                u[i*3+j] = 1.0
                v[j*3+k] = 1.0
                w[k*3+i] = 1.0
                schoolbook.append((u, v, w))
    
    return schoolbook  # rank 27, not 21, but correct

def _verify_decomposition(terms, n=3):
    """Verify that the terms form a valid decomposition for n x n matrix mult."""
    for i in range(n):
        for j in range(n):
            for k in range(n):
                s = 0.0
                for (u, v, w) in terms:
                    s += u[i*n+j] * v[j*n+k] * w[k*n+i]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6
    block_size = 3
    
    # 2x2 grid of 3x3 blocks
    block_offsets = [(0, 0), (0, 3), (3, 0), (3, 3)]
    
    # Define the 8 block products: (A_block_idx, B_block_idx, C_block_idx)
    # Block indices: 0=TL(0,0), 1=TR(0,3), 2=BL(3,0), 3=BR(3,3)
    products = [
        (0, 0, 0),  # A00 * B00 -> C00
        (1, 2, 0),  # A01 * B10 -> C00
        (0, 1, 1),  # A00 * B01 -> C01
        (1, 3, 1),  # A01 * B11 -> C01
        (2, 0, 2),  # A10 * B00 -> C10
        (3, 2, 2),  # A11 * B10 -> C10
        (2, 1, 3),  # A10 * B01 -> C11
        (3, 3, 3),  # A11 * B11 -> C11
    ]
    
    # Get 3x3 decomposition
    local_terms = _bclr_rank21_3x3()
    
    # Verify
    if not _verify_decomposition(local_terms, 3):
        # Fallback: build schoolbook
        local_terms = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    u = [0.0]*9
                    v = [0.0]*9
                    w = [0.0]*9
                    u[i*3+j] = 1.0
                    v[j*3+k] = 1.0
                    w[k*3+i] = 1.0
                    local_terms.append((u, v, w))
    
    local_rank = len(local_terms)
    total_rank = len(products) * local_rank
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for (a_idx, b_idx, c_idx) in products:
        a_row, a_col = block_offsets[a_idx]
        b_row, b_col = block_offsets[b_idx]
        c_row, c_col = block_offsets[c_idx]
        
        for (u_local, v_local, w_local) in local_terms:
            for i in range(3):
                for j in range(3):
                    if u_local[i*3+j] != 0:
                        gi = a_row + i
                        gj = a_col + j
                        U[gi * m + gj][col] = u_local[i*3+j]
            for j in range(3):
                for k in range(3):
                    if v_local[j*3+k] != 0:
                        gj = b_row + j
                        gk = b_col + k
                        V[gj * p + gk][col] = v_local[j*3+k]
            for k in range(3):
                for i in range(3):
                    if w_local[k*3+i] != 0:
                        gi = c_row + i
                        gk = c_col + k
                        W[gk * n + gi][col] = w_local[k*3+i]
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
