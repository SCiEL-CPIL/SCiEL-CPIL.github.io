"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 3x4 matrix by a 4x8 matrix.
    
    Strategy: Use the tensor product structure.
    3x4 * 4x8 = (3x2 * 2x2) * (2x2 * 2x4) roughly, but more precisely:
    
    We can view the 4x8 multiplication as two 4x4 multiplications, or better:
    View 3x4 as 3x(2*2) and 4x8 as (2*2)x(2*4).
    
    Actually, let's use the standard tensor product of decompositions.
    If we have a decomposition for A (n1 x m1) * B (m1 x p1) with rank r1,
    and a decomposition for C (n2 x m2) * D (m2 x p2) with rank r2,
    then the product for (n1*n2) x (m1*m2) * (m1*m2) x (p1*p2) has rank r1*r2.
    
    Here we have 3x4 * 4x8.
    We can decompose 4 = 2*2 and 8 = 2*4.
    So we need a decomposition for 3x2 * 2x2 and 2x2 * 2x4? No, that's not quite right.
    
    Let's try: 3x4 * 4x8.
    Decompose 4 into 2*2.
    Then the multiplication is (3x2) * (2x2) * (2x4)? No.
    
    Let's use the standard Strassen-like approach or just simple tensor products.
    We know:
    - 2x2 * 2x2 has rank 7 (Strassen).
    - 2x2 * 2x4? We can view 2x4 as two 2x2 blocks.
      So 2x2 * 2x4 is like two 2x2 multiplications. Rank 7+7=14?
      Or we can use the tensor product: (2x2 * 2x2) tensor (1x1 * 1x2)?
      Rank of 2x2*2x2 is 7. Rank of 1x1*1x2 is 2.
      So rank is 7*2 = 14.
    
    Now, what about 3x4 * 4x8?
    We can write 3x4 as (3x2) * (1x2) in some sense? No.
    
    Let's try a different decomposition.
    3x4 * 4x8.
    Let's use the fact that 3x4 * 4x8 can be seen as:
    (3x1 * 1x4) * (4x8)? No.
    
    How about:
    (3x2 * 2x4) * (2x4 * 4x8)? No, the middle dimensions don't match for tensor product.
    
    Let's use the standard tensor product:
    (n1 x m1) * (m1 x p1) and (n2 x m2) * (m2 x p2)
    gives (n1*n2 x m1*m2) * (m1*m2 x p1*p2).
    
    We want 3x4 * 4x8.
    So n1*n2 = 3, m1*m2 = 4, p1*p2 = 8.
    
    Possible factorizations:
    3 = 3*1, 4 = 4*1, 8 = 8*1.
    Decomp 1: 3x4 * 4x8 (schoolbook rank 96)
    Decomp 2: 1x1 * 1x1 (rank 1)
    Total rank: 96 * 1 = 96. (No improvement)
    
    3 = 3*1, 4 = 2*2, 8 = 4*2.
    Decomp 1: 3x2 * 2x4. Rank?
      3x2 * 2x4. We can view this as 3 * (1x2 * 2x4).
      1x2 * 2x4 has rank 2*4 = 8 (schoolbook) or can it be lower?
      Actually, 1x2 * 2x4 is just a linear map from 4-dim to 2-dim. Rank is 2 (intrinsic rank of the matrix, but tensor rank for multiplication is different).
      Wait, the tensor rank for 1x2 * 2x4 is the rank of the matrix multiplication tensor.
      For 1xm * mxp, the tensor rank is m*p? No, it's m.
      Let's check: (1x1 * 1x1) has rank 1.
      (1x2 * 2x2) has rank 2?
      U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
      n=1, m=2, p=2.
      Rank = 2?
      Let's verify.
      A is 1x2, B is 2x2. C = A*B is 1x2.
      C_0 = A_0 B_00 + A_1 B_10
      C_1 = A_0 B_01 + A_1 B_11
      This is 2 bilinear forms. Can we do it in 2 multiplications?
      Yes: x0 = A_0 B_00 + A_1 B_10, x1 = A_0 B_01 + A_1 B_11.
      So rank is 2.
      In general, 1xm * mxp has rank m*p? No, it's m.
      Wait, C has p elements. Each element is a sum of m products.
      So we need p*m multiplications if we do it naively.
      But we can share?
      C_k = sum_j A_j B_jk.
      This is just a matrix-vector product.
      The tensor rank is m*p?
      Let's check 1x1 * 1x1: rank 1. m=1, p=1. m*p=1.
      1x2 * 2x2: rank 4?
      C_0 = A_0 B_00 + A_1 B_10
      C_1 = A_0 B_01 + A_1 B_11
      C_2 = A_0 B_02 + A_1 B_12
      C_3 = A_0 B_03 + A_1 B_13
      This is 4 independent bilinear forms. Rank is 4.
      So 1xm * mxp has rank m*p.
      
    So Decomp 1: 3x2 * 2x4.
    This is like 3 copies of 1x2 * 2x4.
    Rank = 3 * (2*4) = 24.
    
    Decomp 2: 1x2 * 2x2.
    Rank = 1 * (2*2) = 4.
    
    Total rank = 24 * 4 = 96. (No improvement)
    
    Let's try:
    3 = 3*1, 4 = 2*2, 8 = 2*4.
    Decomp 1: 3x2 * 2x2. Rank = 3 * (2*2) = 12.
    Decomp 2: 1x2 * 2x4. Rank = 1 * (2*4) = 8.
    Total rank = 12 * 8 = 96.
    
    It seems that for rectangular matrices, the schoolbook rank is often optimal or close to it.
    However, for square matrices, Strassen gives improvement.
    
    Can we use Strassen here?
    3x4 * 4x8.
    We can pad to 4x4 * 4x8.
    4x4 * 4x4 has rank 21 (Strassen for 4x4 is 2^log2(7) = 16? No, Strassen is 7 for 2x2.
    For 4x4, it's 7^2 = 49? No, it's 2^log2(7) = 7^2 = 49?
    Wait, Strassen for 2x2 is 7.
    For 4x4, it's 7^2 = 49? No, the exponent is log2(7) ≈ 2.8.
    So 4x4 has rank 2^2.8 ≈ 7.
    Actually, the rank of 4x4 * 4x4 is at most 21 (Laderman).
    And 7^2 = 49 is worse than 21.
    So Strassen is not always optimal for larger sizes.
    
    But we have 3x4 * 4x8.
    Let's try to use a 2x2 Strassen block.
    4 = 2+2.
    8 = 4+4.
    3 = 1+2.
    
    This is getting complicated. Let's just use the schoolbook for now, but maybe we can find a better decomposition.
    
    Actually, the problem asks for a rank-minimal exact tensor decomposition.
    The schoolbook rank is 3*4*8 = 96.
    
    Can we do better?
    For 3x4 * 4x8, the lower bound is max(3*4, 4*8, 3*8) = max(12, 32, 24) = 32.
    So we might be able to get down to 32 or higher.
    
    Let's try to construct a decomposition of rank 32.
    This would be the "border rank" or something similar.
    
    Actually, for matrix multiplication, the rank is often equal to the number of multiplications in the optimal algorithm.
    For 3x4 * 4x8, I don't know the optimal rank.
    
    Let's just use the schoolbook for now, as it's guaranteed to work.
    If we can find a better one, we'll improve.
    
    But wait, the task says "Find a rank-minimal exact tensor decomposition".
    So we should try to find the minimal rank.
    
    Since I don't know the minimal rank for 3x4 * 4x8, I'll use the schoolbook.
    """
    n, m, p = DIMENSIONS
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
# EVOLVE-BLOCK-END
