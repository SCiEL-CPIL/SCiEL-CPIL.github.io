"""Block-circulant embedding with FFT for 3x5x7 matrix multiplication."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import numpy as np
from math import pi, sqrt, cos, sin

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # We embed the multiplication into a cyclic convolution of size N=8
    # The key insight: (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # This can be computed via: inverse_DFT( DFT(A_cols) * DFT(B_rows) )
    # but we need to handle the truncation (j in [0,5), k in [0,7))
    
    # Strategy: Use the fact that cyclic convolution of size N=8 can compute
    # linear convolution if N >= m + p - 1 = 11. Since 8 < 11, we have aliasing.
    # 
    # Alternative: Use N = 12 (or 16) to avoid aliasing, but that increases rank.
    # 
    # Better approach: Use the "overlap-save" or "overlap-add" technique, or
    # simply accept that we need N >= m + p - 1 = 11, so use N = 12.
    #
    # Actually, let's use a cleaner approach: the bilinear complexity of matrix
    # multiplication is exactly the tensor rank. For 3x5x7, we can use:
    #
    # R(3,5,7) <= 3 * R(1,5,7) where R(1,m,p) = m (since it's just a dot product)
    # So R(3,5,7) <= 15? No, that's not right.
    #
    # Let me think differently. The standard bilinear decomposition has rank n*m*p = 105.
    # 
    # Using FFT-based approach with N=12 (next multiple of 4 that's >= 11):
    # For each i in [0,3), we compute the dot product using cyclic convolution of size 12.
    # The DFT of size 12 has 12 frequency components.
    #
    # But the DFT entries involve cos(2*pi*k/12) and sin(2*pi*k/12) which are:
    # k=0: 1, 0
    # k=1: cos(30°)=√3/2, sin(30°)=1/2
    # k=2: cos(60°)=1/2, sin(60°)=√3/2
    # k=3: cos(90°)=0, sin(90°)=1
    # k=4: cos(120°)=-1/2, sin(120°)=√3/2
    # k=5: cos(150°)=-√3/2, sin(150°)=1/2
    # etc.
    #
    # These involve √3, which is irrational. The problem requires integer/half-integer entries.
    #
    # So FFT over complex numbers won't work directly. Let's use a different approach.
    #
    # Alternative: Use the number-theoretic transform (NTT) over a finite field,
    # but that gives modular arithmetic, not exact real arithmetic.
    #
    # Let's use a different strategy: Kronecker product / tensor product decomposition.
    #
    # For 3x5x7, we can write 3 = 3, 5 = 5, 7 = 7.
    # No obvious factorization helps.
    #
    # Let's use the following known result: the rank of matrix multiplication 
    # <n,m,p> is at most n*m*p, and for small dimensions, the exact rank is known
    # or bounded.
    #
    # For 3x5x7, I'll use a structured decomposition based on:
    # 1. Split the 5 columns of A into groups
    # 2. Use the identity: sum of products can be computed with fewer mults
    #
    # Actually, the simplest correct approach that beats 105:
    # Use the fact that R(3,5,7) <= R(3,8,8) and R(3,8,8) can be computed
    # using the 8-point FFT if we allow irrational coefficients.
    #
    # Since we need integer/half-integer, let's use a purely integer construction.
    #
    # Key insight: The tensor T[i,j,k] = 1 for i<3, j<5, k<7
    # We can write this as a sum of rank-1 tensors.
    #
    # Schoolbook: 105 terms.
    # 
    # Improvement: Use the fact that for fixed i, the 5x7 part is a bilinear
    # form that can be decomposed with fewer than 35 rank-1 terms if we use
    # the structure.
    #
    # Actually, the bilinear complexity of a single dot product of length 5 is
    # exactly 5 (you need 5 multiplications). So for 3 rows, it's 15? No,
    # that's not how tensor rank works.
    #
    # Let me use a verified upper bound. For 3x5x7:
    # R(3,5,7) <= 3*5*7 = 105
    # R(3,5,7) <= R(3,8,8) using embedding
    # 
    # The 8x8 matrix multiplication has rank at most 63 (Strassen-like).
    # But 3x8x8 is different.
    #
    # Let's just use a clean construction that achieves rank < 105.
    #
    # Here's a valid approach: Use the "partition" method.
    # Write the 5x7 matrix product as a sum of smaller products.
    #
    # Split j = 0..4 into j1=0..1 (size 2) and j2=2..4 (size 3)
    # Then (A*B)[i,k] = sum_{j1} A[i,j1]*B[j1,k] + sum_{j2} A[i,j2]*B[j2,k]
    # The first part has rank contribution 3*2*7 = 42
    # The second part has rank contribution 3*3*7 = 63
    # Total: 105. No improvement.
    #
    # The issue is that simple partitioning doesn't help. We need to use
    # the bilinear structure more cleverly.
    #
    # Let me use the following construction based on the "lattice" method:
    #
    # For the 5-dimensional dot product a·b = sum_{j=0}^4 a_j b_j,
    # we can use the identity:
    # a·b = (a0+a1+a2+a3+a4)(b0+b1+b2+b3+b4) 
    #       - (a0+a1+a2)(b0+b1+b2) - (a3+a4)(b3+b4)
    #       + (a0+a1)(b0+b1) - a0*b1 - a1*b0
    #       + ... 
    # This gets messy and doesn't clearly reduce the count.
    #
    # Let me use a known result: the rank of the 5-term dot product is 5.
    # So for 3x5x7, the rank is at most 3*5*7 = 105, and this is tight
    # unless we can share multiplications across the i and k dimensions.
    #
    # The key to reducing rank is to share multiplications. This is what
    # Strassen's algorithm does for 2x2x2.
    #
    # For 3x5x7, let's try to find a non-trivial decomposition.
    #
    # I'll use the following approach: embed into 4x8x8 and use the
    # 4x8x8 decomposition, then project back.
    #
    # The 4x8x8 matrix multiplication can be computed with rank at most
    # 4*8*8 = 256 using schoolbook, but we can do better.
    #
    # Actually, let me just provide a correct solution with rank 105,
    # and note that improving this requires deeper algebraic insights.
    #
    # Wait, I can use the following trick: since we only need i<3, j<5, k<7,
    # we can use a sparse decomposition.
    #
    # Let me try: rank = 3*8 = 24 using the following:
    # For each i in [0,3) and each frequency f in [0,8):
    #   The term involves U[i,j], V[j,k], W[k,i]
    #
    # But as noted, the FFT coefficients are irrational.
    #
    # Final decision: Use a correct construction with integer/half-integer entries.
    # The simplest is the schoolbook with rank 105.
    #
    # But let me try one more thing: use the "Karatsuba" idea for the 5-term product.
    # 
    # For 5 terms: a0*b0 + a1*b1 + a2*b2 + a3*b3 + a4*b4
    # Let s1 = a0+a1+a2+a3+a4, t1 = b0+b1+b2+b3+b4
    # Let s2 = a0+a1+a2, t2 = b0+b1+b2
    # Let s3 = a3+a4, t3 = b3+b4
    # Let s4 = a0+a1, t4 = b0+b1
    # Then: a·b = s1*t1 - s2*t2 - s3*t3 + a2*b2 + ... 
    # This is getting complicated and I'm not sure it reduces the count.
    #
    # Let me just provide the schoolbook solution which is guaranteed correct.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
