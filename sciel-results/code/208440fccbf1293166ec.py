"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication.
    
    Strategy: Use Strassen's 7-multiplication trick on 2x2 block matrices.
    
    Split A (4x5) into 2x2 blocks of size 2x3 and 2x2:
    A = [[A11, A12], [A21, A22]]
    where A11, A12, A21 are 2x3 blocks and A22 is 2x2 block
    - A11: rows 0-1, cols 0-2
    - A12: rows 0-1, cols 3-4
    - A21: rows 2-3, cols 0-2
    - A22: rows 2-3, cols 3-4
    
    Split B (5x6) into 2x2 blocks of size 3x3 and 2x3:
    B = [[B11, B12], [B21, B22]]
    where B11, B12 are 3x3 blocks and B21, B22 are 2x3 blocks
    - B11: rows 0-2, cols 0-2
    - B12: rows 0-2, cols 3-5
    - B21: rows 3-4, cols 0-2
    - B22: rows 3-4, cols 3-5
    
    However, Strassen requires square blocks. Let me use a different approach.
    
    Alternative: Pad to 6x6 * 6x6 and use Strassen, then extract the 4x5*5x6 part.
    But padding increases the rank.
    
    Better approach: Use the fact that we can decompose the multiplication tensor directly.
    
    For 4x5 * 5x6, the tensor has dimensions (4*5, 5*6, 6*4) = (20, 30, 24).
    
    Let me implement a valid bilinear algorithm using block Strassen on compatible blocks.
    
    Actually, let me use a simpler approach: decompose into 2x2 Strassen on the 4x4 core and handle the extra column/row with schoolbook.
    
    Split A (4x5) = [A_core (4x4) | A_extra (4x1)]
    Split B (5x6) = [B_top (4x6); B_bottom (1x6)]
    
    C = A_core * B_top + A_extra * B_bottom
    
    For A_core * B_top (4x4 * 4x6):
    We can view B_top as 6 columns, each of size 4x1.
    So this is equivalent to computing 4x4 * 4x1 for each of the 6 columns.
    
    But 4x4 * 4x1 is just 4x4 matrix-vector multiplication, which has rank 4*4 = 16 in schoolbook.
    Using Strassen on 2x2 blocks: 4x4 * 4x1 can be done with... actually, Strassen applies to matrix-matrix, not matrix-vector.
    
    Let me use a different strategy. I'll use the known result that 4x4 matrix multiplication can be done with rank 15 (using Strassen recursively: 7^2 = 49 for 8x8, but for 4x4 it's 7*2 = 14? No, Strassen on 4x4 gives 7*2 = 14 multiplications of 2x2, which is 14*4 = 56 scalar multiplications, worse than schoolbook 64? No, 4x4 schoolbook is 4*4*4 = 64, and Strassen gives 7*2*2 = 28? Let me recalculate.
    
    Strassen for 2x2: 7 multiplications of 2x2 matrices.
    For 4x4: split into 2x2 blocks, use Strassen: 7 multiplications of 2x2 matrices, each requiring 4 scalar multiplications = 28 scalar multiplications.
    Schoolbook for 4x4: 4*4*4 = 64 scalar multiplications.
    So Strassen gives 28 < 64. Good.
    
    But for 4x4 * 4x6, we need to be more careful.
    
    Let me just implement a correct decomposition using the following approach:
    
    Use the "border rank" or "tensor rank" known results. For small matrices, the minimal rank is known.
    
    Actually, let me implement a valid bilinear algorithm that achieves rank less than 120.
    
    Here's a concrete approach using Strassen on 2x2 blocks:
    
    For the 4x4 * 4x4 sub-problem (embedded in our 4x5*5x6), Strassen uses 7 multiplications.
    
    Let me define the full decomposition:
    
    A (4x5): rows 0-3, cols 0-4
    B (5x6): rows 0-4, cols 0-5
    C (4x6): rows 0-3, cols 0-5
    
    Split into:
    A = [A1 | A2] where A1 is 4x4 (cols 0-3), A2 is 4x1 (col 4)
    B = [B1; B2] where B1 is 4x6 (rows 0-3), B2 is 1x6 (row 4)
    
    C = A1*B1 + A2*B2
    
    A2*B2: outer product of 4x1 and 1x6, rank 4.
    Terms: for i in 0..3, k in 0..5: C[i,k] += A2[i] * B2[k]
    This can be done with 4*6 = 24 rank-1 terms? No, outer product has rank 4 (as a matrix), but as a tensor decomposition, it requires 4*6 = 24 terms? 
    
    Wait, the outer product C_outer[i,k] = A2[i] * B2[k].
    This is a rank-1 matrix, so as a bilinear form, it can be expressed with 1 term? No.
    
    C_outer[i,k] = sum over nothing, it's just A2[i] * B2[k].
    As a tensor: T[i,k,?,?] ... 
    
    The tensor for outer product: T_{i,k,j} where j indexes the "multiplication".
    C[i,k] = A2[i] * B2[k]
    This is a single bilinear term: U has A2[i], V has B2[k], W has C[i,k].
    So it's 1 term? No, because A2 has 4 elements and B2 has 6 elements.
    
    Actually, C[i,k] = A2[i] * B2[k] is a single product for each (i,k) pair, but the "multiplication" is just scalar multiplication.
    
    In the tensor decomposition framework:
    C[i,k] = sum_t U[i*m+j_t, t] * V[j_t*p+k, t]
    
    For the outer product part:
    C[i,k] = A2[i] * B2[k]
    
    We need: U[i*m + j, t] * V[j*p + k, t] = A2[i] * B2[k]
    
    If we set j = 4 (the extra column/row), then:
    U[i*5 + 4, t] * V[4*6 + k, t] = A2[i] * B2[k]
    
    This requires that for each (i,k), there's a t such that U[i*5+4, t] = A2[i] and V[4*6+k, t] = B2[k].
    
    But A2[i] and B2[k] are variables, so we need:
    U[i*5+4, t] to be 1 when i is fixed, and V[4*6+k, t] to be 1 when k is fixed.
    
    This means we need 4*6 = 24 terms, one for each (i,k) pair, with:
    U[i*5+4, t_{i,k}] = 1
    V[4*6+k, t_{i,k}] = 1
    W[k*4+i, t_{i,k}] = 1
    
    So the outer product part has rank 24.
    
    For A1*B1 (4x4 * 4x6):
    This is the main term. Let me use Strassen on the 4x4 part.
    
    A1 (4x4) split into 2x2 blocks:
    A1 = [[A11, A12], [A21, A22]] where each is 2x2
    
    B1 (4x6) split into 2x2 blocks of size 2x3:
    B1 = [[B11, B12], [B21, B22]] where B11, B12 are 2x3, B21, B22 are 2x3
    
    C1 (4x6) split into 2x2 blocks of size 2x3:
    C1 = [[C11, C12], [C21, C22]] where each is 2x3
    
    C11 = A11*B11 + A12*B21
    C12 = A11*B12 + A12*B22
    C21 = A21*B11 + A22*B21
    C22 = A21*B12 + A22*B22
    
    Using Strassen:
    M1 = (A11+A22)*(B11+B22)  -> but B11+B22 is 4x3, not square
    This doesn't work directly because B1 is not square.
    
    Let me use a different approach. I'll implement the schoolbook for the 4x4*4x6 part, which has rank 4*4*6 = 96.
    Total rank = 96 + 24 = 120. Same as before.
    
    To do better, I need to use Strassen or other optimizations.
    
    Given the complexity, let me implement a valid decomposition with rank 120 (schoolbook) to ensure correctness, and note that optimization is needed for improvement.
    """
    n, m, p = DIMENSIONS  # 4, 5, 6
    rank = n * m * p  # 120
    
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
# EVOLVE-BLOCK-END
