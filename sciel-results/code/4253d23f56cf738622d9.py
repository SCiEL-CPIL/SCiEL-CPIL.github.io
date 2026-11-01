"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = DIMENSIONS
    
    # Schoolbook baseline: rank = n*m*p = 36
    def schoolbook():
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
    
    # Verify a decomposition by checking the tensor identity
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1 for all i,j,k
    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1) > 1e-10:
                        return False
        return True
    
    # Strassen-like decomposition for 2x2x2 (rank 7)
    # The 7 terms for multiplying two 2x2 matrices:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b10)
    # M7 = (a01-a11)(b01+b11)
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M4
    # C10 = M2 + M6
    # C11 = M1 - M3 + M6 + M7
    
    # For 2x2x9, we can split the 9 columns into groups and use efficient decompositions.
    # Strategy: Split p=9 into three blocks of 3.
    # For each block of 3 columns, we're multiplying 2x2 by 2x3.
    # The rank for 2x2x3 is at most 12 (schoolbook), but we can do better.
    
    # Actually, let's try a different approach: use the Kronecker product structure.
    # If we can write the 2x2x9 tensor as a sum of rank-1 tensors, we want to minimize the count.
    
    # Let's try to use a block decomposition where we treat the 9 as 3x3.
    # Split A (2x2) into 2x2 blocks (just A itself), B (2x9) into 2x3 blocks.
    # This doesn't immediately help.
    
    # Alternative: Use the fact that for 2x2 matrices, Strassen reduces 8 mults to 7.
    # For 2x2x9, if we could apply Strassen-like tricks across the 9 columns...
    
    # Let's try a constructive approach: 
    # We know that 2x2x2 has rank 7. 
    # 2x2x9 = 2x2x(3*3). 
    # Can we decompose using 3 copies of 2x2x3?
    # Rank of 2x2x3: schoolbook is 12. Can we do better?
    # Actually, the minimum rank of 2x2x3 is known to be 10 (or maybe less).
    
    # Let's try a simple improvement: 
    # For each pair of columns in the 9, use a 2x2x2 Strassen decomposition.
    # 9 columns = 4 pairs + 1 single.
    # 4 * 7 + 4 (for the single, since 2x2x1 has rank 4) = 28 + 4 = 32.
    # This is better than 36!
    
    # Let's construct this:
    # Columns 0,1: use Strassen for 2x2x2 (rank 7)
    # Columns 2,3: use Strassen for 2x2x2 (rank 7)
    # Columns 4,5: use Strassen for 2x2x2 (rank 7)
    # Columns 6,7: use Strassen for 2x2x2 (rank 7)
    # Column 8: use schoolbook for 2x2x1 (rank 4)
    # Total rank: 7*4 + 4 = 32
    
    # Strassen U, V, W factors for 2x2x2:
    # U factors (for A):
    # U1 = [[1,0],[1,1]] -> rows: [1,0], [1,1] (for a00, a10)
    # Wait, let's be more careful.
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    
    # M1 = (a00+a11)(b00+b11)
    #   U1 for A: [[1, 0], [0, 1]] -> no, we need linear combinations.
    #   Actually, U1 is a 2x1 vector for each row of A? No.
    #   The decomposition is: (AB)_{ik} = sum_c U_{i,c} V_{j,c} W_{k,c}
    #   Wait, the standard form is: (AB)_{ik} = sum_j A_{ij} B_{jk}
    #   = sum_c (sum_j U_{ij,c} A_{ij}) ... no, that's not right either.
    
    # Let me use the correct tensor decomposition form:
    # (AB)_{ik} = sum_{j} A_{ij} B_{jk}
    # We want to write this as sum_c f_c(A) g_c(B) h_c
    # where f_c(A) = sum_{ij} U_{ij,c} A_{ij}
    #              g_c(B) = sum_{jk} V_{jk,c} B_{jk}
    #              h_c is a scalar, and the result is sum_c f_c(A) g_c(B) h_c
    # But that's not how matrix multiplication works directly.
    
    # The correct form for the tensor T_{ijk} = 1 (for all i,j,k in appropriate ranges) is:
    # T_{ijk} = sum_c U_{ij,c} V_{jk,c} W_{ki,c}
    # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # and the indices are: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    
    # For Strassen 2x2x2:
    # The 7 terms correspond to:
    # c=0: M1 = (a00+a11)(b00+b11)
    #   U[0,0]=1, U[1,0]=0 (a00), U[2,0]=0 (a10), U[3,0]=1 (a11) -> wait, indices are i*m+j
    #   For 2x2: i in {0,1}, j in {0,1}
    #   U[0*2+0, c] = U[0,c] for a00
    #   U[0*2+1, c] = U[1,c] for a01
    #   U[1*2+0, c] = U[2,c] for a10
    #   U[1*2+1, c] = U[3,c] for a11
    
    # M1 = (a00+a11)(b00+b11)
    #   U[:,0] = [1, 0, 0, 1]  (a00 + a11)
    #   V[:,0] = [1, 0, 0, 1]  (b00 + b11) where V[j*p+k, c] with p=2
    #     V[0*2+0,0]=V[0,0]=1 (b00)
    #     V[0*2+1,0]=V[1,0]=0 (b01)
    #     V[1*2+0,0]=V[2,0]=0 (b10)
    #     V[1*2+1,0]=V[3,0]=1 (b11)
    #   W[:,0] = ? The output C = AB, C_{ik}
    #   M1 contributes to C00, C01, C10, C11? No, M1 is a scalar.
    #   Actually, in the tensor decomposition, W_{k*n+i, c} is the coefficient for C_{ik}.
    #   For M1, it contributes to all four entries? No, that's not right.
    
    # I think I'm confusing the Strassen algorithm with the tensor decomposition.
    # Let me look up the correct Strassen tensor decomposition.
    
    # Actually, the simplest approach is to just use the schoolbook for now and try to find
    # a better decomposition through optimization. But given the time constraints, let's
    # try the block approach with Strassen.
    
    # Let me try a different strategy: use the known fact that the rank of 2x2x9 is at most 32
    # by using 4 Strassen blocks (2x2x2) and 1 schoolbook block (2x2x1).
    
    # For the Strassen blocks, I need the correct U, V, W factors.
    # Let me derive them:
    
    # For 2x2x2, the Strassen decomposition has rank 7.
    # The 7 terms are:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b10)
    # M7 = (a01-a11)(b01+b11)
    
    # And the output is:
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M4
    # C10 = M2 + M6
    # C11 = M1 - M3 + M6 + M7
    
    # So for each term M_c, we have:
    # M_c = f_c(A) * g_c(B)
    # where f_c(A) is a linear combination of A's entries
    # and g_c(B) is a linear combination of B's entries
    
    # Then C_{ik} = sum_c (coefficient of M_c in C_{ik}) * M_c
    
    # In the tensor decomposition:
    # (AB)_{ik} = sum_j A_{ij} B_{jk} = sum_c U_{ij,c} V_{jk,c} W_{ki,c}
    
    # Wait, I think the correct interpretation is:
    # U_{ij,c} is the coefficient of A_{ij} in f_c(A)
    # V_{jk,c} is the coefficient of B_{jk} in g_c(B)
    # W_{ki,c} is the coefficient of M_c in C_{ik}
    
    # So for M1 = (a00+a11)(b00+b11):
    # U[:,c1] = [1, 0, 0, 1]  (a00 + a11)
    # V[:,c1] = [1, 0, 0, 1]  (b00 + b11)
    # W[:,c1] = [1, 1, 1, 1]  (M1 contributes 1 to C00, C01, C10, C11)? 
    # No, that's not right. M1 is a scalar, and it contributes to multiple C_{ik}.
    
    # Actually, I think the decomposition is:
    # C_{ik} = sum_c W_{ki,c} * (sum_{ij} U_{ij,c} A_{ij}) * (sum_{jk} V_{jk,c} B_{jk})
    
    # But that would make W_{ki,c} depend on k and i, which is what we want.
    
    # Let me try a different approach: just use the schoolbook and see if we can verify it.
    # Then try to find a better decomposition through search.
    
    # For now, let's return the schoolbook solution.
    elapsed = time.time() - start_time
    if elapsed > deadline - 20:
        return schoolbook()
    
    # Try to find a rank-32 decomposition using Strassen blocks.
    # This requires careful construction of U, V, W.
    
    # Let's try a simpler improvement: use the fact that 9 = 3*3 and try to
    # find a decomposition that exploits this structure.
    
    # Actually, let's just return the schoolbook for now, as constructing the
    # Strassen-based decomposition is complex and error-prone.
    
    return schoolbook()
# EVOLVE-BLOCK-END
