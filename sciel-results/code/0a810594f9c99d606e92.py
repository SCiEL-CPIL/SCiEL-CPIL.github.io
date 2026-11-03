"""Kronecker-style decomposition of 3x5x6 matrix multiplication."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 5, 6
    
    # We decompose the 3x5x6 multiplication using a block structure.
    # m=5 can be split as 2+3, p=6 can be split as 2+4 or 3+3.
    # Let's use m=2+3 and p=3+3 for symmetry, or m=2+3 and p=2+4.
    # Actually, let's try m=2+3 and p=3+3 to use 3x3x3 blocks.
    
    # But 3x3x3 has rank 19 (Laderman).
    # 3x2x2 has rank 7 (Strassen-like, actually rank 7 is known for 2x2x2, but for 3x2x2 it's different).
    # Wait, 2x2x2 has rank 7 (Strassen). 3x3x3 has rank 19 (Laderman).
    
    # Let's split m=5 as 2+3 and p=6 as 3+3.
    # Blocks:
    # (i,j,k) where i in [0,3), j in [0,5), k in [0,6)
    # Block A: i in [0,3), j in [0,2), k in [0,3) -> 3x2x3
    # Block B: i in [0,3), j in [2,5), k in [0,3) -> 3x3x3
    # Block C: i in [0,3), j in [0,2), k in [3,6) -> 3x2x3
    # Block D: i in [0,3), j in [2,5), k in [3,6) -> 3x3x3
    
    # 3x2x3: This is a 3x2 matrix times 2x3 matrix. Rank is 3*2*3=18 schoolbook.
    # Can we do better? 3x2x3 is equivalent to 2x3x3 transposed. 
    # Actually, let's just use schoolbook for the 3x2x3 blocks since I'm not sure of optimal rank.
    # 3x3x3: Laderman rank 19.
    
    # Total rank: 18 + 19 + 18 + 19 = 74. This is better than 90.
    
    # Let's implement this.
    
    # First, let's define the Laderman decomposition for 3x3x3.
    # The Laderman algorithm has 19 multiplications.
    # A and B are 3x3 matrices, C = A*B.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01+A11)(B10+B11)
    # M8 = (A01-A10)(B00+B10)
    # M9 = (A11-A01)(B01+B11)
    # M10 = (A10+A01)(B11-B01)  # Wait, this is not standard. Let me look up Laderman.
    
    # Actually, let me use a known correct Laderman decomposition.
    # Here's a standard one:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01+A11)(B10+B11)
    # M8 = (A01-A10)(B00+B10)
    # M9 = (A11-A01)(B01+B11)
    # M10 = (A10+A01)(B11-B01)
    # M11 = (A11+A00)(B11-B00)  # This doesn't look right either.
    
    # Let me use a different approach. I'll use the schoolbook for 3x3x3 (rank 27) 
    # if I can't get Laderman right. But 19 is known.
    # Let me try to find a correct Laderman decomposition.
    
    # Actually, for the purpose of this task, let me first get a working solution 
    # with rank < 90, then optimize.
    
    # Let's use:
    # 3x2x3: schoolbook rank 18
    # 3x3x3: schoolbook rank 27
    # Total: 18 + 27 + 18 + 27 = 90. Same as before.
    
    # So I need to use better decompositions for the blocks.
    
    # Let me try m=2+3, p=2+4.
    # Blocks:
    # 3x2x2: rank 7 (Strassen for 2x2x2, but we have 3 rows. Actually 3x2x2 is 3x2 * 2x2.
    #   The rank of 3x2x2 tensor is at most 3*2*2=12, but can be less.
    #   Actually, 3x2x2 is equivalent to (3x2) x (2x2). The tensor is T_{ijk} = sum_l A_{il} B_{lk}.
    #   Wait, no. The multiplication tensor for (n x m) * (m x p) -> (n x p) has T_{i,j,k} where 
    #   i is row of A, j is column of A (which is row of B), k is column of B.
    #   So for 3x2x2: n=3, m=2, p=2. T_{i,j,k} with i in [0,3), j in [0,2), k in [0,2).
    #   This is the tensor for multiplying a 3x2 matrix by a 2x2 matrix.
    #   The rank of this tensor is at most 3*2*2 = 12.
    #   Can we do better? The 2x2x2 part has rank 7. The extra row adds complexity.
    #   Actually, 3x2x2 can be seen as two 1x2x2 tensors plus one 1x2x2 tensor, but they share the B part.
    #   Let me just use schoolbook for 3x2x2: rank 12.
    
    # This is getting complicated. Let me just use a known good decomposition.
    
    # Alternative approach: Use the fact that 3x5x6 can be decomposed using 
    # the Kronecker product structure.
    
    # Let me try a different split: m=5, p=6.
    # What if I use a single block with a known rank?
    # The rank of 3x5x6 is at most 90 (schoolbook).
    # Known lower bound: max(3*5, 3*6, 5*6) = max(15, 18, 30) = 30.
    # Actually, the rank is at least max(n*m, m*p, n*p) / something? No.
    # The rank is at least max(n*m, n*p, m*p) is not correct.
    # The rank is at least n*m + n*p + m*p - something? No.
    # Actually, the rank is at least max(n*m, m*p, n*p) is a lower bound for the 
    # "slice rank" but not the tensor rank.
    
    # Let me just implement a correct schoolbook decomposition first to make sure 
    # the format is right, then try to optimize.
    
    rank = n * m * p  # 90
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
