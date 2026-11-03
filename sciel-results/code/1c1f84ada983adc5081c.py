import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = 2, 4, 6
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen terms for 2x2 block matrix multiplication
    # Each term: (a_coeffs, b_coeffs, c_coeffs)
    # a_coeffs: list of (row, col, coeff) for the 2x2 A block
    # b_coeffs: list of (row, col, coeff) for the 2x2 B block
    # c_coeffs: list of (row, col, coeff) for the 2x2 C block
    strassen_terms = [
        # M1 = (A11+A22)(B11+B22) -> C11 += M1, C12 += M1, C21 += M1, C22 += M1 (partially)
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (0,1,1), (1,0,1), (1,1,1)]),
        # M2 = A21(B11+B12) -> C21 += M2, C22 += M2
        ([(1,0,1), (1,1,1)], [(0,0,1), (0,1,1)], [(1,0,1), (1,1,1)]),
        # M3 = (A11+A12)B22 -> C11 += M3, C12 += M3
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,1), (0,1,1)]),
        # M4 = A22(B21-B11) -> C21 += M4, C22 -= M4
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(1,0,1), (1,1,-1)]),
        # M5 = (A11-A22)(B11+B21) -> C11 += M5, C21 -= M5
        ([(0,0,1), (1,1,-1)], [(0,0,1), (1,0,1)], [(0,0,1), (1,0,-1)]),
        # M6 = (A21-A11)(B12+B22) -> C12 -= M6, C22 += M6
        ([(1,0,1), (0,0,-1)], [(0,1,1), (1,1,1)], [(0,1,-1), (1,1,1)]),
        # M7 = (A12-A22)(B21+B22) -> C12 += M7, C22 -= M7
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,1,1), (1,1,-1)]),
    ]

    # Blocks: A is 2x4, split into 1x2 blocks (2 blocks: A1=row0 cols0-1, A2=row1 cols2-3)
    # B is 4x6, split into 2x3 blocks (3 blocks: B1=rows0-1 cols0-2, B2=rows0-1 cols2-4, B3=rows0-1 cols4-6)
    # Wait, B is 4x6. For 2x2 block multiplication, we need B split into 2x2 blocks.
    # But B is 4x6, not 4x4. So we can't directly use 2x2 block structure.
    
    # Let me reconsider. The standard approach:
    # A is 2x4, B is 4x6. 
    # Split A into two 1x4 rows? No, that doesn't help with Strassen.
    
    # Actually, the original approach treats it as 2x2 block multiplication where:
    # A is split into 1x2 blocks (top: row0 cols0-1, bottom: row1 cols2-3) -- this is wrong
    # 
    # Let me re-examine. For 2x4 by 4x6:
    # We can view A as [[A1, A2]] where A1, A2 are 1x2
    # B as [[B1, B2, B3]] where B1, B2, B3 are 2x2... no, B is 4x6
    # 
    # Actually the correct block structure for applying Strassen:
    # A (2x4) = [[A11, A12]] where A11, A12 are 1x2
    # B (4x6) = [[B11, B12, B13], [B21, B22, B23]] where each Bij is 2x2
    # But 2x2 blocks in B would make B 4x6 only if we have 3 column blocks of width 2.
    # So B11, B12, B13 are 2x2 (rows 0-1, cols 0-1, 2-3, 4-5)
    # B21, B22, B23 are 2x2 (rows 2-3, cols 0-1, 2-3, 4-5)
    # 
    # Then C (2x6) = [[C11, C12, C13]] where Cij = A11*Bij + A12*B2j... 
    # Wait, C[i,k] = sum_j A[i,j] B[j,k]
    # C[0,k] = A[0,0]B[0,k] + A[0,1]B[1,k] + A[0,2]B[2,k] + A[0,3]B[3,k]
    # 
    # With block structure:
    # C1j = A11 * Bj1 + A12 * Bj2 where Bj1 is top 2 rows of column block j, Bj2 is bottom 2 rows
    # 
    # So for each column block j (3 blocks), we have a 1x2 by 2x2 multiplication... 
    # but that's not 2x2 by 2x2.
    
    # I think the original approach was: treat the 2x4 by 4x6 as 6 separate 1x2 by 2x3 multiplications
    # (one for each pair of A-row-block and B-column-block), and apply Strassen to each.
    # But 1x2 by 2x3 doesn't have a Strassen structure.
    
    # Let me just use the schoolbook approach for now and verify, then try to optimize.
    # Schoolbook: rank = 2*4*6/4 = 12? No, schoolbook is 2*6 = 12 dot products of length 4? 
    # No: C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k]. There are 2*6 = 12 such sums, each of length 4.
    # Total scalar multiplications: 12*4 = 48. So schoolbook rank is 48.
    
    # The Strassen approach claims rank 42. Let me verify by constructing it properly.
    
    # Approach: Partition A into two 1x2 row-blocks and B into three 2x2 column-blocks... 
    # This doesn't quite work. Let me think again.
    
    # Alternative: Use the fact that we can write the multiplication as:
    # C = A * B where A is 2x4, B is 4x6.
    # 
    # We can use a rank-6 algorithm for 1x2 by 2x3 (which is just schoolbook for that size).
    # There are 2 row-blocks of A (each 1x2) and 3 column-blocks of B (each 2x2)... 
    # 
    # I'm overcomplicating this. Let me just verify the current program's logic is sound
    # by checking if the decomposition is valid, and if not, fix it.
    
    # For now, let me return a known-good schoolbook decomposition of rank 48,
    # then try to find a better one.
    
    # Actually, let me just carefully implement the Strassen-based approach.
    # The key insight: view A as a 2x2 block matrix where each "block" is 1x2,
    # and B as a 2x3 block matrix where each "block" is 2x2.
    # This gives a 2x2 by 2x3 multiplication, which we can handle with 7 Strassen-like terms
    # applied to each of the 3 column positions.
    
    # A (2x4) viewed as 2x2 blocks of size 1x2:
    #   A = [[A11, A12], [A21, A22]] where A11=A[0,0:2], A12=A[0,2:4], A21=A[1,0:2], A22=A[1,2:4]
    # B (4x6) viewed as 2x3 blocks of size 2x2:
    #   B = [[B11, B12, B13], [B21, B22, B23]] where Bij is 2x2
    # C (2x6) = A*B, viewed as 2x3 blocks of size 1x2:
    #   C = [[C11, C12, C13], [C21, C22, C23]] where Cij is 1x2
    
    # For each column j, C1j = A11*B1j + A12*B2j (1x2 * 2x2 + 1x2 * 2x2 = 1x2)
    #            C2j = A21*B1j + A22*B2j
    
    # This is a 2x2 by 2x3 block multiplication. We can apply Strassen-like formulas.
    # For a 2x2 by 2x3 multiplication, we need 7 terms for the 2x2 part, applied to each column.
    # But actually, the 2x2 by 2x3 multiplication can be done with 7*2 = 14 scalar multiplications?
    # No, each "block multiplication" here is 1x2 by 2x2, which is 4 scalar multiplications.
    # 7 terms * 4 = 28 per column, 3 columns = 84. That's worse than schoolbook.
    
    # I think I'm misunderstanding the original approach. Let me just go with a clean schoolbook
    # implementation and see if I can find improvements.
    
    # Schoolbook: For each i in 0..1, k in 0..5, C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k]
    # This requires 2*6*4 = 48 scalar multiplications.
    
    # Let me build the schoolbook decomposition:
    U = [[0] * 48 for _ in range(n * m)]
    V = [[0] * 48 for _ in range(m * p)]
    W = [[0] * 48 for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
