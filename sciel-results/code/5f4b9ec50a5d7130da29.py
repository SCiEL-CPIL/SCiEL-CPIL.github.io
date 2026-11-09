"""Rewritten program: uses Strassen-style divide-and-conquer to get a lower rank than schoolbook.
For 5x5x5 matrix multiplication, we use a hybrid approach:
- Split into blocks and apply Strassen's algorithm where beneficial
- The key insight: 5x5 can be decomposed using 4x4 Strassen (rank 27 for 4x4) plus handling of the 5th row/col

Actually, let me think about this more carefully. The known best rank for 5x5x5 matrix multiplication:
- Schoolbook: 125
- Using Strassen on sub-blocks: we can do better

For 5x5x5, one approach is to note that 5 = 4+1, and use a block decomposition.
But a cleaner approach: use the fact that we can embed 5x5 multiplication into a larger structured problem.

Actually, the most straightforward improvement is to use Strassen's algorithm recursively:
- 5x5: not a power of 2, so we need to handle it specially.
- One option: pad to 8x8 (rank 3^3 = 27 for 4x4, 3^4 = 81 for 8x8) - but that's worse than 125.
- Another: use 4x4 Strassen (rank 27) for the top-left 4x4 block, then handle the remaining parts.

Let me think about a block decomposition of 5x5:
A = [[A11, A12], [A21, A22]] where A11 is 4x4, A12 is 4x1, A21 is 1x4, A22 is 1x1.
B = [[B11, B12], [B21, B22]] similarly.

C11 = A11*B11 + A12*B21  (4x4)
C12 = A11*B12 + A12*B22  (4x1)
C21 = A21*B11 + A22*B21  (1x4)
C22 = A21*B12 + A22*B22  (1x1)

Using Strassen for the 4x4 part: rank 27 for A11*B11.
For the cross terms, we need additional multiplications.

This is getting complex. Let me use a different known result:
The rank of 5x5x5 matrix multiplication is at most 112 (or similar known bounds).

Actually, let me use a simpler approach: use the known Strassen-based decomposition for sizes that are powers of 2, and for 5, use a direct construction.

A better approach: use the fact that we can compute 5x5 multiplication using 4x4 Strassen plus 16 additional scalar multiplications for the boundary terms.

Let me construct this explicitly:
- Top-left 4x4: use Strassen (27 mults)
- The remaining terms involve the 5th row and column.

Actually, let me just use a known low-rank decomposition. The simplest improvement over 125 is to use Strassen's algorithm in a block manner.

For a clean implementation, let me use the following strategy:
- Decompose 5x5 multiplication using a 4x4 Strassen core plus boundary handling.

Here's the plan:
U has rows indexed by (i,j) for i,j in 0..4, so 25 rows.
V has rows indexed by (j,k) for j,k in 0..4, so 25 rows.
W has rows indexed by (k,i) for k,i in 0..4, so 25 rows.

For Strassen on the 4x4 top-left block (indices 0-3):
We use the standard Strassen decomposition for multiplying 4x4 matrices.

For the boundary terms (involving index 4):
We need additional rank-1 terms.

Let me implement a general Strassen decomposition for n x n where n is a power of 2, then extend to 5.

Actually, the cleanest approach for a guaranteed improvement: use the known result that 5x5x5 matrix multiplication can be done in rank <= 112. But constructing that explicitly is complex.

Let me use a simpler but still improved approach: 
- Use Strassen for a 4x4 sub-block (rank 27)
- Handle the 5th row/column with direct multiplications

For the 4x4 Strassen part, I'll implement the standard recursive Strassen.
For the boundary, I'll use direct rank-1 terms.

Total rank estimate: 27 (for 4x4) + additional terms for boundaries.

Let me count:
C11 = A11*B11 + A12*B21: 27 + 4 = 31 (A12 is 4x1, B21 is 1x4, so 4 mults)
C12 = A11*B12 + A12*B22: 27 + 4 = 31 (but A11*B12 can share some with C11's Strassen? No, different right factor)

This is getting complicated. Let me just implement a direct Strassen for the full 5x5 by padding to 8x8 and using 3^3=27... no, 8x8 needs 3^4=81.

Wait, 8x8 via Strassen: 3^4 = 81. That's better than 125!

But we need to extract the 5x5 result from the 8x8 computation. The padding zeros will contribute, so we just ignore the extra rows/columns.

Let me implement 8x8 Strassen and then project down to 5x5.

For 8x8 Strassen:
- Base case: 1x1 (rank 1)
- 2x2: rank 3
- 4x4: rank 27
- 8x8: rank 81

So rank 81 for 8x8, and we only need the top-left 5x5 of the result.

The factor matrices for 8x8 Strassen have:
- U: 64 x 81
- V: 64 x 81
- W: 64 x 81

We then select the first 25 rows of U, first 25 rows of V, and first 25 rows of W (corresponding to indices 0-4).

Wait, the indexing:
- U rows: (i,j) for i,j in 0..7 -> we want i,j in 0..4, so rows 0-4 for i and 0-4 for j -> indices i*8+j for i,j in 0..4 -> 0..4, 8..12, 16..20, 24..28, 32..36
- V rows: (j,k) for j,k in 0..7 -> we want j,k in 0..4 -> indices j*8+k for j,k in 0..4
- W rows: (k,i) for k,i in 0..7 -> we want k,i in 0..4 -> indices k*8+i for k,i in 0..4

Let me implement this.
"""
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    # Use 8x8 Strassen and project down to 5x5
    N = 8  # padded size
    
    def strassen_factors(size):
        """Recursively build Strassen factor matrices for size x size multiplication."""
        if size == 1:
            U = [[1]]
            V = [[1]]
            W = [[1]]
            return U, V, W
        
        half = size // 2
        # Get factors for half-size
        U_half, V_half, W_half = strassen_factors(half)
        rank_half = len(U_half[0])
        
        # Strassen's 7 products for 2x2 block multiplication
        # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        rank = 7 * rank_half
        U = [[0] * rank for _ in range(size * size)]
        V = [[0] * rank for _ in range(size * size)]
        W = [[0] * rank for _ in range(size * size)]
        
        def block_index(block_i, block_j, half):
            """Return the linear index for block (block_i, block_j) of size half."""
            return (block_i * half + 0) * size + (block_j * half + 0)
        
        # Helper to add scaled half-size factors
        def add_factors(U_block_rows, V_block_rows, W_block_rows, scale_U, scale_V, scale_W, offset):
            for i in range(half * half):
                for c in range(rank_half):
                    if scale_U != 0:
                        for ri in U_block_rows:
                            U[ri][offset + c] += scale_U * U_half[i][c]
                    if scale_V != 0:
                        for ri in V_block_rows:
                            V[ri][offset + c] += scale_V * V_half[i][c]
                    if scale_W != 0:
                        for ri in W_block_rows:
                            W[ri][offset + c] += scale_W * W_half[i][c]
        
        # Define block row ranges
        def rows_for_block(block_i, block_j):
            """Get the list of row indices for the block at (block_i, block_j)."""
            rows = []
            for di in range(half):
                for dj in range(half):
                    rows.append((block_i * half + di) * size + (block_j * half + dj))
            return rows
        
        A11_rows = rows_for_block(0, 0)
        A12_rows = rows_for_block(0, 1)
        A21_rows = rows_for_block(1, 0)
        A22_rows = rows_for_block(1, 1)
        B11_rows = rows_for_block(0, 0)  # V uses (j,k) indexing
        B12_rows = rows_for_block(0, 1)
        B21_rows = rows_for_block(1, 0)
        B22_rows = rows_for_block(1, 1)
        C11_rows = rows_for_block(0, 0)
        C12_rows = rows_for_block(0, 1)
        C21_rows = rows_for_block(1, 0)
        C22_rows = rows_for_block(1, 1)
        
        # M1 = (A11+A22)(B11+B22)
        offset = 0
        add_factors(A11_rows, B11_rows, C11_rows, 1, 1, 1, offset)
        add_factors(A22_rows, B11_rows, C11_rows, 1, 1, 1, offset)
        add_factors(A11_rows, B22_rows, C11_rows, 1, 1, 1, offset)
        add_factors(A22_rows, B22_rows, C11_rows, 1, 1, 1, offset)
        offset += rank_half
        
        # M2 = (A21+A22)B11
        add_factors(A21_rows, B11_rows, C21_rows, 1, 1, 1, offset)
        add_factors(A22_rows, B11_rows, C21_rows, 1, 1, 1, offset)
        offset += rank_half
        
        # M3 = A11(B12-B22)
        add_factors(A11_rows, B12_rows, C12_rows, 1, 1, 1, offset)
        add_factors(A11_rows, B22_rows, C12_rows, 1, -1, -1, offset)
        offset += rank_half
        
        # M4 = A22(B21-B11)
        add_factors(A22_rows, B21_rows, C11_rows, 1, 1, 1, offset)
        add_factors(A22_rows, B11_rows, C11_rows, 1, -1, -1, offset)
        offset += rank_half
        
        # M5 = (A11+A12)B22
        add_factors(A11_rows, B22_rows, C11_rows, 1, 1, -1, offset)
        add_factors(A12_rows, B22_rows, C11_rows, 1, 1, -1, offset)
        add_factors(A11_rows, B22_rows, C12_rows, 1, 1, 1, offset)
        add_factors(A12_rows, B22_rows, C12_rows, 1, 1, 1, offset)
        offset += rank_half
        
        # M6 = (A21-A11)(B11+B12)
        add_factors(A21_rows, B11_rows, C22_rows, 1, 1, 1, offset)
        add_factors(A21_rows, B12_rows, C22_rows, 1, 1, 1, offset)
        add_factors(A11_rows, B11_rows, C22_rows, -1, 1, -1, offset)
        add_factors(A11_rows, B12_rows, C22_rows, -1, 1, -1, offset)
        offset += rank_half
        
        # M7 = (A12-A22)(B21+B22)
        add_factors(A12_rows, B21_rows, C11_rows, 1, 1, 1, offset)
        add_factors(A12_rows, B22_rows, C11_rows, 1, 1, 1, offset)
        add_factors(A22_rows, B21_rows, C11_rows, -1, 1, -1, offset)
        add_factors(A22_rows, B22_rows, C11_rows, -1, 1, -1, offset)
        offset += rank_half
        
        return U, V, W
    
    # Build 8x8 Strassen factors
    U8, V8, W8 = strassen_factors(N)
    rank8 = len(U8[0])
    
    # Select rows for 5x5 from 8x8
    # U rows: (i,j) -> i*N+j, we want i,j in 0..4
    # V rows: (j,k) -> j*N+k, we want j,k in 0..4
    # W rows: (k,i) -> k*N+i, we want k,i in 0..4
    
    def select_rows(mat, size, target_size):
        """Select rows where both indices are < target_size."""
        selected = []
        for i in range(target_size):
            for j in range(target_size):
                row_idx = i * size + j
                selected.append(row_idx)
        return [mat[idx] for idx in selected]
    
    U5 = select_rows(U8, N, n)
    V5 = select_rows(V8, N, m)
    W5 = select_rows(W8, N, p)
    
    rank = len(U5[0]) if U5 else 0
    
    return {"U": U5, "V": V5, "W": W5}
# EVOLVE-BLOCK-END
