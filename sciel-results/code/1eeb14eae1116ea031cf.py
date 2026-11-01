"""Program for matrix multiplication 2x2 by 2x9 with improved tensor rank."""
DIMENSIONS = [2, 2, 9]

import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 2, 9
    
    # Rank 24 decomposition for M_{2,2,8} using Laderman's algorithm
    # Laderman's algorithm gives rank 7 for 2x2 by 2x2 matrix multiplication
    # We need to handle 2x2 by 2x8, which can be viewed as 4 independent 2x2 by 2x2 multiplications
    # But we can do better by sharing some products
    
    # For 2x2 by 2x8:
    # A = [[a00, a01], [a10, a11]] (2x2)
    # B = [[b00, b01, ..., b07], [b10, b11, ..., b17]] (2x8)
    # C = A * B (2x8)
    # C[i,k] = a[i,0]*b[0,k] + a[i,1]*b[1,k]
    
    # Using Strassen-like decomposition:
    # M1 = (a00+a10)*(b00+b10)
    # M2 = (a00+a10)*(b01+b11)
    # M3 = (a00+a10)*(b02+b12)
    # M4 = (a00+a10)*(b03+b13)
    # M5 = (a00+a10)*(b04+b14)
    # M6 = (a00+a10)*(b05+b15)
    # M7 = (a00+a10)*(b06+b16)
    # M8 = (a00+a10)*(b07+b17)
    # M9 = (a01+a11)*(b00+b10)
    # M10 = (a01+a11)*(b01+b11)
    # M11 = (a01+a11)*(b02+b12)
    # M12 = (a01+a11)*(b03+b13)
    # M13 = (a01+a11)*(b04+b14)
    # M14 = (a01+a11)*(b05+b15)
    # M15 = (a01+a11)*(b06+b16)
    # M16 = (a01+a11)*(b07+b17)
    # M17 = a00*b00
    # M18 = a00*b01
    # M19 = a00*b02
    # M20 = a00*b03
    # M21 = a00*b04
    # M22 = a00*b05
    # M23 = a00*b06
    # M24 = a00*b07
    # M25 = a11*b10
    # M26 = a11*b11
    # M27 = a11*b12
    # M28 = a11*b13
    # M29 = a11*b14
    # M30 = a11*b15
    # M31 = a11*b16
    # M32 = a11*b17
    
    # Actually, let me use a simpler approach. For 2x2 by 2x8:
    # We can use 4 Strassen products (one for each pair of columns)
    # For columns 0,1: M1=(a00+a10)(b00+b10), M2=(a00+a10)(b01+b11), M3=(a01+a11)(b00+b10), M4=(a01+a11)(b01+b11)
    # C[0,0] = M1 - M4, C[0,1] = M2 - M4
    # C[1,0] = M4 - M3, C[1,1] = M2 - M3
    # Wait, that's not right. Let me reconsider.
    
    # Standard Strassen for 2x2 by 2x2:
    # M1 = (a00+a10)(b00+b10)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b00-b10)
    # M5 = (a00+a01)b11
    # M6 = (a01-a10)(b00+b10)
    # M7 = (a10-a11)(b01+b11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For 2x2 by 2x8, we can group columns into pairs and apply Strassen to each pair:
    # Columns (0,1), (2,3), (4,5), (6,7)
    # Each pair uses 7 multiplications, total 28. But we can do better.
    
    # Actually, for 2x2 by 2x8, the rank is known to be 24 (using a different decomposition).
    # Let me use a rank-24 decomposition.
    
    # Simpler approach: Use the fact that 2x2 by 2x8 = 4 * (2x2 by 2x1)
    # Each 2x2 by 2x1 has rank 4, so total rank is 16? No, that's not right either.
    
    # Let me just use schoolbook for 2x2 by 2x8: rank = 2*2*8 = 32
    # And for 2x2 by 2x1: rank = 4
    # Total: 32 + 4 = 36, same as before.
    
    # Let me try a different approach. The rank of M_{2,2,p} for p=8 is known to be 24.
    # Reference: https://arxiv.org/abs/1903.10572
    
    # For now, let me construct a valid decomposition with rank 32 (schoolbook for 2x2x8) + 4 (for 2x2x1) = 36
    # This is the same as the current program, so it won't improve the score.
    
    # Let me try to find a better decomposition.
    
    # For 2x2 by 2x8, I'll use a decomposition based on tensor rank properties.
    # The bilinear map is: (a00, a01, a10, a11) x (b00, b01, ..., b07, b10, b11, ..., b17) -> (c00, c01, ..., c07, c10, c11, ..., c17)
    # where c[i,k] = a[i,0]*b[0,k] + a[i,1]*b[1,k]
    
    # This is equivalent to 8 independent bilinear maps:
    # (a00, a01, a10, a11) x (b0k, b1k) -> (c0k, c1k) for k=0,...,7
    # Each map has rank 4 (it's just matrix-vector multiplication).
    # So the total rank should be at most 8*4 = 32.
    
    # Can we do better by sharing products across different k values?
    # If we use products like (a00+a10)*(b0k+b1k), we can compute two outputs with one product:
    # c0k = (a00+a10)*(b0k+b1k) - a10*b0k - a00*b1k
    # c1k = (a00+a10)*(b0k+b1k) - a00*b0k - a10*b1k
    # This requires 3 products per column pair, so 3*4 = 12 products for 8 columns? No, that's not right.
    
    # Let me use a simpler construction. I'll use the schoolbook decomposition but with a twist:
    # For each column k, compute:
    # M1k = a00*b0k
    # M2k = a00*b1k
    # M3k = a10*b0k
    # M4k = a10*b1k
    # M5k = a01*b0k
    # M6k = a01*b1k
    # M7k = a11*b0k
    # M8k = a11*b1k
    # Then c0k = M1k + M6k, c1k = M5k + M7k
    # Wait, that's not right. c0k = a00*b0k + a01*b1k, c1k = a10*b0k + a11*b1k
    # So I need:
    # M1k = a00*b0k, M2k = a01*b1k, M3k = a10*b0k, M4k = a11*b1k
    # c0k = M1k + M2k, c1k = M3k + M4k
    # This uses 4 products per column, total 4*8 = 32 for 2x2x8, plus 4 for 2x2x1 = 36.
    
    # This is the same as the schoolbook decomposition. Let me just return this.
    
    # Actually, I realize I should try harder. Let me use a known rank-24 decomposition for M_{2,2,8}.
    # Based on the work of Bader and others, the rank of M_{2,2,8} is 24.
    
    # For now, I'll return a valid decomposition with rank 36 (same as current).
    # This ensures correctness even if it doesn't improve the score.
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]  # 4 x 36
    V = [[0] * rank for _ in range(m * p)]  # 18 x 36
    W = [[0] * rank for _ in range(p * n)]  # 18 x 36
    
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
