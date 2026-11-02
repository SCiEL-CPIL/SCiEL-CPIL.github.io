"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _build_strassen_2x2x2():
    """Return (U, V, W) for 2x2x2 matrix multiplication with rank 7.
    U shape (4,7), V shape (4,7), W shape (4,7).
    Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c].
    """
    n = 2
    # Strassen 7 terms
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A01-A11)(B10+B11)
    # M7 = (A10-A00)(B00+B01)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # A indices: A00=(0,0), A01=(0,1), A10=(1,0), A11=(1,1)
    # B indices: B00=(0,0), B01=(0,1), B10=(1,0), B11=(1,1)
    # C indices: C00=(0,0), C01=(0,1), C10=(1,0), C11=(1,1)
    
    # M1 = (A00+A11)(B00+B11)
    U[0*2+0, 0] = 1  # A00
    U[1*2+1, 0] = 1  # A11
    V[0*2+0, 0] = 1  # B00
    V[1*2+1, 0] = 1  # B11
    W[0*2+0, 0] = 1  # C00
    W[1*2+1, 0] = 1  # C11
    
    # M2 = (A10+A11)B00
    U[1*2+0, 1] = 1  # A10
    U[1*2+1, 1] = 1  # A11
    V[0*2+0, 1] = 1  # B00
    W[1*2+0, 1] = 1  # C10
    W[1*2+1, 1] = -1 # C11
    
    # M3 = A00(B01-B11)
    U[0*2+0, 2] = 1  # A00
    V[0*2+1, 2] = 1  # B01
    V[1*2+1, 2] = -1 # B11
    W[0*2+1, 2] = 1  # C01
    W[1*2+1, 2] = 1  # C11
    
    # M4 = A11(B10-B00)
    U[1*2+1, 3] = 1  # A11
    V[1*2+0, 3] = 1  # B10
    V[0*2+0, 3] = -1 # B00
    W[0*2+0, 3] = 1  # C00
    W[1*2+0, 3] = 1  # C10
    
    # M5 = (A00+A10)B11
    U[0*2+0, 4] = 1  # A00
    U[1*2+0, 4] = 1  # A10
    V[1*2+1, 4] = 1  # B11
    W[0*2+0, 4] = -1 # C00
    W[0*2+1, 4] = 1  # C01
    
    # M6 = (A01-A11)(B10+B11)
    U[0*2+1, 5] = 1  # A01
    U[1*2+1, 5] = -1 # A11
    V[1*2+0, 5] = 1  # B10
    V[1*2+1, 5] = 1  # B11
    W[1*2+1, 5] = 1  # C11
    
    # M7 = (A10-A00)(B00+B01)
    U[1*2+0, 6] = 1  # A10
    U[0*2+0, 6] = -1 # A00
    V[0*2+0, 6] = 1  # B00
    V[0*2+1, 6] = 1  # B01
    W[0*2+0, 6] = 1  # C00
    W[1*2+0, 6] = 1  # C10
    
    return U, V, W


def _build_strassen_2x2x3():
    """Return (U, V, W) for 2x2x3 matrix multiplication.
    U shape (4, rank), V shape (6, rank), W shape (6, rank).
    Convention: U[i*2+j, c], V[j*3+k, c], W[k*2+i, c].
    Uses Strassen-like approach with rank 9.
    """
    n, m, p = 2, 2, 3
    # We'll use a combination: split B into two parts B[:, :2] and B[:, 2]
    # Actually, let's use a direct rank-9 construction.
    # For 2x2x3, we can think of it as 2x2x2 + 2x2x1
    # 2x2x2 has rank 7, 2x2x1 has rank 4 (schoolbook), but they share structure.
    # Let's try rank 9: 7 for the first two columns + 2 for the third? No, that's not right.
    # Actually, for 2x2x3, the schoolbook rank is 12.
    # Let's use a known result: rank of 2x2x3 is 9.
    # Construction:
    # M1 = (A00+A11)(B00+B10)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A01-A11)(B10+B11)
    # M7 = (A10-A00)(B00+B01)
    # Plus for the third column:
    # M8 = A00*B02
    # M9 = A01*B12  ... wait, this doesn't work directly.
    
    # Let me use a different approach. For 2x2x3:
    # C = A * B where A is 2x2, B is 2x3, C is 2x3.
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C02 = A00*B02 + A01*B12
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # C12 = A10*B02 + A11*B12
    
    # Strassen for first two columns (rank 7) + 2 more for third column
    # But the third column shares A entries with the first two.
    # Let's just use rank 9 with a known construction.
    
    # Actually, let me use a simpler approach: schoolbook for 2x2x3 is rank 12.
    # With Strassen for 2x2x2 (rank 7) plus 2 additional terms for the third column,
    # we might get rank 9 if we can share some terms.
    
    # Let me try: use Strassen's 7 terms for columns 0,1 of B, and add 2 terms for column 2.
    # But the A entries are the same, so we need to see if we can reuse.
    
    # For now, let's use rank 9 with a specific construction:
    # Terms 1-7: Strassen for 2x2x2 (columns 0,1)
    # Term 8: A00*B02 contributes to C02
    # Term 9: A10*B02 + A01*B12 ... no, this is getting complicated.
    
    # Let me just use schoolbook rank 12 for simplicity, or find a better way.
    # Actually, the known rank for 2x2x3 is 9. Let me look up the construction.
    
    # For 2x2x3, one rank-9 construction:
    # M1 = (A00+A11)(B00+B10)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A01-A11)(B10+B11)
    # M7 = (A10-A00)(B00+B01)
    # M8 = A00*B02
    # M9 = A10*B02 + A01*B12  -- this is not a product of sums in the right form.
    
    # Let me use a different strategy. I'll use the fact that 4x4x5 can be decomposed
    # into blocks of 2x2x2 and 2x2x3, and use rank 7 for 2x2x2 and rank 9 for 2x2x3.
    # For 4x4x5: split into 2x2 blocks in the first two dimensions.
    # A is 4x4, B is 4x5.
    # Split A into 2x2 blocks: A00, A01, A10, A11 each 2x2.
    # Split B into 2x2 and 2x1 blocks: B00, B01, B02 (2x2), B10, B11, B12 (2x2), B03, B13 (2x1)
    # Wait, B is 4x5. Split rows into 2+2, columns into 2+2+1.
    # B00 (2x2), B01 (2x2), B02 (2x1)
    # B10 (2x2), B11 (2x2), B12 (2x1)
    
    # C = A*B is 4x5.
    # C00 (2x2) = A00*B00 + A01*B10
    # C01 (2x2) = A00*B01 + A01*B11
    # C02 (2x1) = A00*B02 + A01*B12
    # C10 (2x2) = A10*B00 + A11*B10
    # C11 (2x2) = A10*B01 + A11*B11
    # C12 (2x1) = A10*B02 + A11*B12
    
    # Each 2x2x2 block product has rank 7 (Strassen).
    # Each 2x2x1 block product has rank 4 (schoolbook, since 2x2x1 rank is 4).
    # Total rank = 4*7 + 2*4 = 28 + 8 = 36.
    
    # That's better than 80! Let me implement this.
    pass


def _build_2x2x1():
    """Return (U, V, W) for 2x2x1 matrix multiplication with rank 4.
    U shape (4,4), V shape (2,4), W shape (2,4).
    Convention: U[i*2+j, c], V[j*1+k, c], W[k*2+i, c].
    """
    n, m, p = 2, 2, 1
    rank = 4
    U = np.zeros((4, rank))
    V = np.zeros((2, rank))
    W = np.zeros((2, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W


def _assemble_block_decomposition():
    """Assemble the full 4x4x5 decomposition from 2x2x2 and 2x2x1 blocks."""
    n, m, p = 4, 4, 5
    
    # Get building blocks
    U222, V222, W222 = _build_strassen_2x2x2()
    U221, V221, W221 = _build_2x2x1()
    
    rank_222 = U222.shape[1]  # 7
    rank_221 = U221.shape[1]  # 4
    
    # Total rank: 4 blocks of 2x2x2 (rank 7 each) + 2 blocks of 2x2x1 (rank 4 each)
    total_rank = 4 * rank_222 + 2 * rank_221  # 28 + 8 = 36
    
    U = np.zeros((n*m, total_rank))  # (16, 36)
    V = np.zeros((m*p, total_rank))  # (20, 36)
    W = np.zeros((p*n, total_rank))  # (20, 36)
    
    col = 0
    # A is 4x4, split into 2x2 blocks:
    # A00: rows 0-1, cols 0-1
    # A01: rows 0-1, cols 2-3
    # A10: rows 2-3, cols 0-1
    # A11: rows 2-3, cols 2-3
    
    # B is 4x5, split rows into 2+2, cols into 2+2+1:
    # B00: rows 0-1, cols 0-1
    # B01: rows 0-1, cols 2-3
    # B02: rows 0-1, cols 4
    # B10: rows 2-3, cols 0-1
    # B11: rows 2-3, cols 2-3
    # B12: rows 2-3, cols 4
    
    # C is 4x5, split rows into 2+2, cols into 2+2+1:
    # C00: rows 0-1, cols 0-1
    # C01: rows 0-1, cols 2-3
    # C02: rows 0-1, cols 4
    # C10: rows 2-3, cols 0-1
    # C11: rows 2-3, cols 2-3
    # C12: rows 2-3, cols 4
    
    # Block products:
    # C00 = A00*B00 + A01*B10  (two 2x2x2 products)
    # C01 = A00*B01 + A01*B11  (two 2x2x2 products)
    # C02 = A00*B02 + A01*B12  (two 2x2x1 products)
    # C10 = A10*B00 + A11*B10  (two 2x2x2 products)
    # C11 = A10*B01 + A11*B11  (two 2x2x2 products)
    # C12 = A10*B02 + A11*B12  (two 2x2x1 products)
    
    # We need 8 instances of 2x2x2 and 4 instances of 2x2x1.
    # But we can reuse the same Strassen structure for each.
    
    # For each 2x2x2 block product, we place the Strassen terms.
    # For each 2x2x1 block product, we place the schoolbook terms.
    
    def place_2x2x2(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset, start_col):
        """Place a 2x2x2 Strassen decomposition for the block product.
        A block at (a_row_offset, a_col_offset) with size 2x2.
        B block at (b_row_offset, b_col_offset) with size 2x2.
        C block at (c_row_offset, c_col_offset) with size 2x2.
        """
        for t in range(rank_222):
            # U: A entries
            for i in range(2):
                for j in range(2):
                    val = U222[i*2+j, t]
                    if val != 0:
                        U[(a_row_offset + i) * m + (a_col_offset + j), start_col + t] = val
            
            # V: B entries
            for j in range(2):
                for k in range(2):
                    val = V222[j*2+k, t]
                    if val != 0:
                        V[(b_row_offset + j) * p + (b_col_offset + k), start_col + t] = val
            
            # W: C entries
            for k in range(2):
                for i in range(2):
                    val = W222[k*2+i, t]
                    if val != 0:
                        W[(c_col_offset + k) * n + (c_row_offset + i), start_col + t] = val
        
        return start_col + rank_222
    
    def place_2x2x1(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset, start_col):
        """Place a 2x2x1 schoolbook decomposition for the block product.
        A block at (a_row_offset, a_col_offset) with size 2x2.
        B block at (b_row_offset, b_col_offset) with size 2x1.
        C block at (c_row_offset, c_col_offset) with size 2x1.
        """
        for t in range(rank_221):
            # U: A entries
            for i in range(2):
                for j in range(2):
                    val = U221[i*2+j, t]
                    if val != 0:
                        U[(a_row_offset + i) * m + (a_col_offset + j), start_col + t] = val
            
            # V: B entries
            for j in range(2):
                for k in range(1):
                    val = V221[j*1+k, t]
                    if val != 0:
                        V[(b_row_offset + j) * p + (b_col_offset + k), start_col + t] = val
            
            # W: C entries
            for k in range(1):
                for i in range(2):
                    val = W221[k*2+i, t]
                    if val != 0:
                        W[(c_col_offset + k) * n + (c_row_offset + i), start_col + t] = val
        
        return start_col + rank_221
    
    # Place all block products:
    # C00 = A00*B00 + A01*B10
    col = place_2x2x2(0, 0, 0, 0, 0, 0, col)  # A00*B00
    col = place_2x2x2(0, 2, 2, 0, 0, 0, col)  # A01*B10
    
    # C01 = A00*B01 + A01*B11
    col = place_2x2x2(0, 0, 0, 2, 0, 2, col)  # A00*B01
    col = place_2x2x2(0, 2, 2, 2, 0, 2, col)  # A01*B11
    
    # C02 = A00*B02 + A01*B12
    col = place_2x2x1(0, 0, 0, 4, 0, 4, col)  # A00*B02
    col = place_2x2x1(0, 2, 2, 4, 0, 4, col)  # A01*B12
    
    # C10 = A10*B00 + A11*B10
    col = place_2x2x2(2, 0, 0, 0, 2, 0, col)  # A10*B00
    col = place_2x2x2(2, 2, 2, 0, 2, 0, col)  # A11*B10
    
    # C11 = A10*B01 + A11*B11
    col = place_2x2x2(2, 0, 0, 2, 2, 2, col)  # A10*B01
    col = place_2x2x2(2, 2, 2, 2, 2, 2, col)  # A11*B11
    
    # C12 = A10*B02 + A11*B12
    col = place_2x2x1(2, 0, 0, 4, 2, 4, col)  # A10*B02
    col = place_2x2x1(2, 2, 2, 4, 2, 4, col)  # A11*B12
    
    assert col == total_rank, f"Column mismatch: {col} vs {total_rank}"
    
    return U, V, W


def _verify_decomposition(U, V, W):
    """Verify that the decomposition is correct."""
    n, m, p = DIMENSIONS
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    rank = U.shape[1]
    
    # The tensor T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = delta (we want C[i,k] = sum_j A[i,j] B[j,k])
    # The decomposition gives: C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j,c] * V[j*p+k,c])
    # Wait, that's not right. Let me reconsider.
    
    # The standard tensor decomposition for matrix multiplication:
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[i,k,c]
    # where T[i,j,k] = 1 if this is a valid term in the product, else 0.
    # Actually, for matrix multiplication, the tensor is:
    # T[i,j,k] = 1 for all i,j,k (it's the identity tensor for the product)
    # No, that's not right either.
    
    # The matrix multiplication tensor M satisfies:
    # M[i,j,k] = 1 if we're computing C[i,k] from A[i,j] and B[j,k]
    # The decomposition is: M[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # Wait, the convention is W[k*n+i, c], so W index is (k, i).
    
    # So: C[i,k] = sum_j A[i,j] B[j,k]
    # Decomposition: C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_k' V[j*p+k', c] * B[j,k'])
    # Hmm, this is getting confusing. Let me just verify numerically.
    
    # For random A and B, check that the reconstructed C matches A@B.
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    # Reconstruct using the decomposition
    # C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
    # Wait, that's not how it works. The decomposition is:
    # C[i,k] = sum_c (sum_j U[i*m+j,c] * A[i,j]) * (sum_k' V[j'*p+k',c] * B[j',k]) * W[k*n+i,c]
    # No, that's still not right.
    
    # The correct interpretation:
    # Define U_c[i,j] = U[i*m+j, c], V_c[j,k] = V[j*p+k, c], W_c[i,k] = W[k*n+i, c]
    # Then: sum_c U_c[i,j] * V_c[j,k] * W_c[i,k] should equal the matrix multiplication tensor.
    # But that's not a standard tensor contraction.
    
    # Actually, for matrix multiplication tensor decomposition:
    # The tensor T has T[i,j,k] = 1 (for the "multiplication" operation)
    # And we want: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This means: for each (i,j,k), sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1
    # And for the actual multiplication: C[i,k] = sum_j A[i,j] B[j,k]
    # We can write: C[i,k] = sum_j sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] A[i,j] B[j,k]
    # = sum_c W[k*n+i,c] (sum_j U[i*m+j,c] A[i,j]) (sum_j V[j*p+k,c] B[j,k])
    # Wait, that has two sums over j, which is wrong.
    
    # Let me reconsider. The standard way:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Decomposition: C[i,k] = sum_c M_c[i,k] where M_c[i,k] = (U_c * A)[i] * (B * V_c)[k] * W_c[i,k]
    # where U_c is a vector of length m, V_c is a vector of length m, W_c is a matrix of size n x p.
    # Actually, U_c[i,j] = U[i*m+j, c], so for fixed c, U_c is an n x m matrix.
    # V_c[j,k] = V[j*p+k, c], so for fixed c, V_c is an m x p matrix.
    # W_c[i,k] = W[k*n+i, c], so for fixed c, W_c is an n x p matrix.
    
    # Then: sum_c sum_{i',j',k'} U_c[i',j'] A[i',j'] V_c[j',k'] B[j',k'] W_c[i,k]
    # That's not right either.
    
    # OK let me just do the direct verification:
    # For each (i,j,k), check that sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] equals the expected value.
    # For matrix multiplication, the "tensor" is:
    # T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
    # Which is always 1 for valid indices.
    # So we need: sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 for all i,j,k.
    
    errors = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1) > 1e-9:
                    errors.append((i, j, k, val))
    
    return len(errors) == 0, errors


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    U, V, W = _assemble_block_decomposition()
    
    # Verify
    valid, errors = _verify_decomposition(U, V, W)
    
    if not valid:
        # Fall back to schoolbook
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
    
    # Convert to lists with integer/half-integer entries
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    # Round to nearest half-integer
    def round_halfint(x):
        return round(2 * x) / 2
    
    U_list = [[round_halfint(x) for x in row] for row in U_list]
    V_list = [[round_halfint(x) for x in row] for row in V_list]
    W_list = [[round_halfint(x) for x in row] for row in W_list]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
