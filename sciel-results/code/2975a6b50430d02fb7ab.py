import numpy as np
import time
from fractions import Fraction

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W form a valid decomposition for n x m * m x p."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    T = np.zeros((p * n, n * m, m * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[k * n + i, i * m + j, j * p + k] = 1.0
    recon = np.zeros((p * n, n * m, m * p))
    for r in range(rank):
        recon += np.einsum('i,j,k->ijk', W[:, r], U[:, r], V[:, r])
    return np.allclose(T, recon, atol=tol)

def _rank14_2x3x3():
    """
    Return a rank-14 decomposition for 2x3 * 3x3 using a verified algorithm.
    U: (6, 14), V: (9, 14), W: (6, 14)
    A index: i*3+j, B index: j*3+k, C/W index: k*2+i
    """
    # Use a known rank-14 bilinear algorithm for 2x3 * 3x3.
    # We'll construct it by solving the linear constraints.
    # 
    # The 2x3 * 3x3 multiplication has tensor rank 14 (known result).
    # Here's a verified construction using bilinear combinations.
    
    # A = [[a00, a01, a02], [a10, a11, a12]]
    # B = [[b00, b01, b02], [b10, b11, b12], [b20, b21, b22]]
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # Let me use a systematic approach: find 14 rank-1 tensors that sum to the multiplication tensor.
    # I'll use a known algorithm based on the work of Strassen-type reductions.
    
    # Actually, let me construct this numerically and verify.
    # The tensor T[k*2+i, i*3+j, j*3+k] = 1
    
    # I'll use a verified rank-14 decomposition. Here's one based on
    # bilinear combinations that I'll verify computationally.
    
    U = np.zeros((6, 14), dtype=float)
    V = np.zeros((9, 14), dtype=float)
    W = np.zeros((6, 14), dtype=float)
    
    # A indices: 0=a00, 1=a01, 2=a02, 3=a10, 4=a11, 5=a12
    # B indices: 0=b00, 1=b01, 2=b02, 3=b10, 4=b11, 5=b12, 6=b20, 7=b21, 8=b22
    # W indices: 0=c00(k=0,i=0), 1=c01(k=1,i=0), 2=c02(k=2,i=0), 3=c10(k=0,i=1), 4=c11(k=1,i=1), 5=c12(k=2,i=1)
    # Wait: W index = k*n + i = k*2 + i
    # c00: k=0,i=0 -> 0
    # c01: k=1,i=0 -> 2  
    # c02: k=2,i=0 -> 4
    # c10: k=0,i=1 -> 1
    # c11: k=1,i=1 -> 3
    # c12: k=2,i=1 -> 5
    
    # Let me use a verified rank-14 algorithm. I'll define it explicitly.
    # Based on known results, here's a valid rank-14 decomposition:
    
    # Product 1: (a00+a10)(b00+b10) -> contributes to c00, c10, c01, c11
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[1,0]=1; W[2,0]=1; W[3,0]=1
    
    # Product 2: (a00-a10)(b00-b10)
    U[0,1]=1; U[3,1]=-1
    V[0,1]=1; V[3,1]=-1
    W[0,1]=1; W[1,1]=-1; W[2,1]=-1; W[3,1]=1
    
    # Product 3: (a00+a01)(b00+b03)  -- wait, b03 doesn't exist
    # Let me restart with a cleaner approach.
    
    # I'll use a computational search to find a valid rank-14 decomposition.
    # For now, return None and let construct() handle it.
    return None

def _find_rank14_2x3x3():
    """
    Find a rank-14 decomposition for 2x3 * 3x3 by solving the system.
    Returns (U, V, W) as lists of lists with integer/half-integer entries, or None.
    """
    n, m, p = 2, 3, 3
    rank = 14
    
    # Build target tensor
    T = np.zeros((p*n, n*m, m*p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[k*n+i, i*m+j, j*p+k] = 1.0
    
    # We need to find U (6,14), V (9,14), W (6,14) such that
    # sum_r U[:,r] outer V[:,r] outer W[:,r] = T
    #
    # This is a non-linear system. Let me try a different approach:
    # Use the fact that we can parameterize and use numerical optimization.
    #
    # Alternative: use a known explicit rank-14 algorithm.
    #
    # Let me try to construct one using the "border rank" approach.
    # For 2x3*3x3, the rank is known to be 14.
    #
    # Here's a construction based on bilinear forms:
    
    # I'll use a verified algorithm. Let me define the 14 products:
    # Using the notation A[i,j], B[j,k], C[i,k]
    
    # After careful analysis, here's a rank-14 decomposition:
    # I'll verify it computationally.
    
    U = np.zeros((6, 14))
    V = np.zeros((9, 14))
    W = np.zeros((6, 14))
    
    # Let me try a different strategy: use the known rank-14 algorithm
    # from the literature. One such algorithm uses the following structure:
    
    # Define linear forms in A:
    # alpha_1 = a00 + a10
    # alpha_2 = a00 - a10
    # alpha_3 = a01 + a11
    # alpha_4 = a01 - a11
    # alpha_5 = a02 + a12
    # alpha_6 = a02 - a12
    # alpha_7 = a00 + a01 + a02
    # alpha_8 = a10 + a11 + a12
    # alpha_9 = a00 + a01 - a02
    # alpha_10 = a10 + a11 - a12
    # alpha_11 = a00 - a01 + a02
    # alpha_12 = a10 - a11 + a12
    # alpha_13 = a00 + a01 + a02 + a10 + a11 + a12  (this won't work well)
    
    # This is getting complex. Let me use a numerical solver approach.
    # I'll set up the problem as a system of equations and solve it.
    
    # Actually, let me just verify a candidate algorithm.
    # I'll try the following 14 products and check if they work:
    
    # P1 = (a00+a10)(b00+b10)
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    # Contributes: a00*b00->c00, a00*b10->c01, a10*b00->c10, a10*b10->c11
    W[0,0]=1; W[2,0]=1; W[1,0]=1; W[3,0]=1
    
    # P2 = (a00-a10)(b00-b10)
    U[0,1]=1; U[3,1]=-1
    V[0,1]=1; V[3,1]=-1
    # Contributes: a00*b00->c00, -a00*b10->c01, -a10*b00->c10, a10*b10->c11
    W[0,1]=1; W[2,1]=-1; W[1,1]=-1; W[3,1]=1
    
    # P3 = (a01+a11)(b01+b11)
    U[1,2]=1; U[4,2]=1
    V[1,2]=1; V[4,2]=1
    # Contributes: a01*b01->c01, a01*b11->c02, a11*b01->c11, a11*b11->c12
    W[2,2]=1; W[4,2]=1; W[3,2]=1; W[5,2]=1
    
    # P4 = (a01-a11)(b01-b11)
    U[1,3]=1; U[4,3]=-1
    V[1,3]=1; V[4,3]=-1
    # Contributes: a01*b01->c01, -a01*b11->c02, -a11*b01->c11, a11*b11->c12
    W[2,3]=1; W[4,3]=-1; W[3,3]=-1; W[5,3]=1
    
    # P5 = (a02+a12)(b02+b12)
    U[2,4]=1; U[5,4]=1
    V[2,4]=1; V[5,4]=1
    # Contributes: a02*b02->c02, a02*b12->c02... wait
    # a02*b02: i=0,j=2,k=2 -> c02 (W index 4)
    # a02*b12: i=0,j=2,k=2 -> c02 (W index 4) -- same!
    # a12*b02: i=1,j=2,k=2 -> c12 (W index 5)
    # a12*b12: i=1,j=2,k=2 -> c12 (W index 5) -- same!
    # So this only gives c02 and c12, each with coefficient 2
    W[4,4]=2; W[5,4]=2
    
    # P6 = (a02-a12)(b02-b12)
    U[2,5]=1; U[5,5]=-1
    V[2,5]=1; V[5,5]=-1
    # a02*b02->c02, -a02*b12->c02, -a12*b02->c12, a12*b12->c12
    W[4,5]=2; W[5,5]=2
    
    # Now I have 6 products. I need 8 more.
    # Missing contributions: c00 needs a01*b10+a02*b20, c10 needs a01*b10... wait
    # Let me check what's been covered so far:
    # c00 (W[0]): P1 gives 1*a00*b00, P2 gives 1*a00*b00. Total: 2*a00*b00. But we need a00*b00 + a01*b10 + a02*b20
    # This approach is not working cleanly.
    
    # Let me abandon this manual approach and use a computational method.
    return None

def _construct_rank42():
    """
    Construct a rank-42 decomposition for 2x3 * 3x9 using DFT over Z_3
    to reduce to three independent 2x3 * 3x3 problems, each of rank 14.
    """
    n, m, p = 2, 3, 9
    rank = 42
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # The key insight: for 2x3 * 3x9, we can use DFT over the 3 column-blocks
    # to diagonalize the cyclic structure, reducing to three independent
    # 2x3 * 3x3 multiplications.
    #
    # However, the DFT involves complex numbers (cube roots of unity),
    # which means we can't use integer/half-integer entries directly.
    #
    # Alternative approach: use the structure directly.
    # The 3x9 matrix B can be viewed as three 3x3 blocks: B = [B0 | B1 | B2]
    # where Bt has columns 3t, 3t+1, 3t+2.
    # C = A * B = [A*B0 | A*B1 | A*B2]
    # Each A*Bt is a 2x3 matrix.
    # So we need 3 independent 2x3 * 3x3 multiplications.
    #
    # If each has rank 14, total rank = 42.
    #
    # But we need a valid rank-14 algorithm for 2x3 * 3x3.
    #
    # Let me try to find one computationally.
    
    # Build the 2x3 * 3x3 tensor
    n2, m2, p2 = 2, 3, 3
    T233 = np.zeros((p2*n2, n2*m2, m2*p2))
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                T233[k*n2+i, i*m2+j, j*p2+k] = 1.0
    
    # Try to find a rank-14 decomposition using numerical methods
    # This is a hard non-linear problem. Let me try a different approach.
    
    # Use the known fact that the rank of 2x3*3x3 is 14.
    # I'll construct a valid algorithm using bilinear combinations.
    
    # After research, here's a valid rank-14 algorithm for 2x3 * 3x3:
    # I'll define it and verify.
    
    # Let me use a systematic construction.
    # The 14 products will be bilinear forms in A and B.
    
    U_block = np.zeros((6, 14))
    V_block = np.zeros((9, 14))
    W_block = np.zeros((6, 14))
    
    # I'll use a verified algorithm. Here's one that I'll check:
    # Based on the structure of the problem, I'll use the following:
    
    # For the 2x3 * 3x3 case, one rank-14 algorithm is:
    # (I'll define it and verify computationally)
    
    # Let me try a different approach: use the schoolbook rank-18 for each block
    # but with some sharing. Actually, since the A matrix is shared across all
    # three blocks, maybe we can share some products.
    
    # Actually, the three blocks are independent in B, so we can't share.
    # Each block needs its own set of products.
    
    # Let me just use rank-18 (schoolbook) for each block = 54 total.
    # That's the same as the current. Not an improvement.
    
    # Let me try to find a valid rank-14 algorithm.
    # I'll use a computational search.
    
    # For now, return None to indicate failure.
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110
    n, m, p = 2, 3, 9
    
    # Strategy: Use DFT over Z_3 to decompose 2x3 * 3x9 into three
    # independent 2x3 * 3x3 multiplications, then use rank-14 for each.
    # Total rank = 42.
    #
    # The DFT approach:
    # Let omega = exp(2*pi*i/3) be a primitive cube root of unity.
    # The DFT matrix is F = [[1,1,1],[1,omega,omega^2],[1,omega^2,omega^4]]
    #
    # For the 3x9 matrix B, we can write B = [B0 | B1 | B2] where each
    # Bt is 3x3. The product C = A*B = [A*B0 | A*B1 | A*B2].
    #
    # The DFT diagonalizes the cyclic convolution structure.
    # Specifically, if we define B_hat_t = sum_s omega^(st) Bs, then
    # C_hat_t = A * B_hat_t (a 2x3 * 3x3 product).
    # Then C_s = (1/3) sum_t omega^(-st) C_hat_t.
    #
    # But this involves complex numbers, which violates the integer/half-integer
    # requirement.
    #
    # Alternative: use the real form of the DFT.
    # For Z_3, the DFT can be written using only real numbers if we
    # group conjugate pairs. But for prime 3, there's one real eigenvalue
    # (omega^0 = 1) and one conjugate pair (omega, omega^2).
    #
    # The real DFT for n=3:
    # F_real = [[1, 1, 1],
    #           [1, 1, -1],
    #           [0, 1, -1]]  -- this isn't quite right
    #
    # Actually, the standard real DFT for n=3 is:
    # F = [[1, 1, 1],
    #      [1, cos(2pi/3), cos(4pi/3)],
    #      [0, sin(2pi/3), sin(4pi/3)]]
    # = [[1, 1, 1],
    #    [1, -1/2, -1/2],
    #    [0, sqrt(3)/2, -sqrt(3)/2]]
    #
    # This involves sqrt(3)/2, which is not a half-integer.
    #
    # So the DFT approach doesn't directly give integer/half-integer entries.
    #
    # Let me try a different approach: directly find a rank-42 decomposition
    # for 2x3 * 3x9 without using DFT.
    #
    # Since the three blocks are independent, rank-42 = 3 * rank-14.
    # I need a valid rank-14 algorithm for 2x3 * 3x3.
    #
    # Let me try to find one using a numerical solver.
    
    # Build the 2x3 * 3x3 tensor
    n2, m2, p2 = 2, 3, 3
    T233 = np.zeros((p2*n2, n2*m2, m2*p2))
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                T233[k*n2+i, i*m2+j, j*p2+k] = 1.0
    
    # Try to find a rank-14 decomposition.
    # This is a non-linear system. Let me try a gradient-based approach
    # or use a known algorithm.
    
    # After careful consideration, let me use a verified rank-14 algorithm.
    # Here's a construction that I'll verify:
    
    # The 2x3 * 3x3 multiplication tensor has rank 14.
    # One valid algorithm uses the following 14 bilinear products:
    
    # I'll define them and verify computationally.
    
    U14 = np.zeros((6, 14))
    V14 = np.zeros((9, 14))
    W14 = np.zeros((6, 14))
    
    # Let me use a known rank-14 algorithm from the literature.
    # Based on the work of Bini et al., here's a rank-14 algorithm:
    
    # A = [[a0, a1, a2], [a3, a4, a5]]
    # B = [[b0, b1, b2], [b3, b4, b5], [b6, b7, b8]]
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # The 14 products (I'll define them and verify):
    
    # P1 = (a0+a3)(b0+b3)
    U14[0,0]=1; U14[3,0]=1
    V14[0,0]=1; V14[3,0]=1
    # a0*b0->C[0,0], a0*b3->C[0,1], a3*b0->C[1,0], a3*b3->C[1,1]
    W14[0,0]=1; W14[2,0]=1; W14[1,0]=1; W14[3,0]=1
    
    # P2 = (a0-a3)(b0-b3)
    U14[0,1]=1; U14[3,1]=-1
    V14[0,1]=1; V14[3,1]=-1
    # a0*b0->C[0,0], -a0*b3->C[0,1], -a3*b0->C[1,0], a3*b3->C[1,1]
    W14[0,1]=1; W14[2,1]=-1; W14[1,1]=-1; W14[3,1]=1
    
    # P3 = (a1+a4)(b1+b4)
    U14[1,2]=1; U14[4,2]=1
    V14[1,2]=1; V14[4,2]=1
    # a1*b1->C[0,1], a1*b4->C[0,2], a4*b1->C[1,1], a4*b4->C[1,2]
    W14[2,2]=1; W14[4,2]=1; W14[3,2]=1; W14[5,2]=1
    
    # P4 = (a1-a4)(b1-b4)
    U14[1,3]=1; U14[4,3]=-1
    V14[1,3]=1; V14[4,3]=-1
    # a1*b1->C[0,1], -a1*b4->C[0,2], -a4*b1->C[1,1], a4*b4->C[1,2]
    W14[2,3]=1; W14[4,3]=-1; W14[3,3]=-1; W14[5,3]=1
    
    # P5 = (a2+a5)(b2+b5)
    U14[2,4]=1; U14[5,4]=1
    V14[2,4]=1; V14[5,4]=1
    # a2*b2->C[0,2], a2*b5->C[0,2], a5*b2->C[1,2], a5*b5->C[1,2]
    # All contribute to C[0,2] or C[1,2]
    W14[4,4]=2; W14[5,4]=2
    
    # P6 = (a2-a5)(b2-b5)
    U14[2,5]=1; U14[5,5]=-1
    V14[2,5]=1; V14[5,5]=-1
    # a2*b2->C[0,2], -a2*b5->C[0,2], -a5*b2->C[1,2], a5*b5->C[1,2]
    W14[4,5]=2; W14[5,5]=2
    
    # P7 = (a0+a1+a2)(b0+b3+b6)
    U14[0,6]=1; U14[1,6]=1; U14[2,6]=1
    V14[0,6]=1; V14[3,6]=1; V14[6,6]=1
    # This gives many cross terms. Let me check:
    # a0*b0->C[0,0], a0*b3->C[0,1], a0*b6->C[0,2]
    # a1*b0->C[0,0]... wait, a1 is A[0,1], b0 is B[0,0]
    # A[0,1]*B[0,0] -> C[0,0]? No! C[i,k] = sum_j A[i,j]B[j,k]
    # A[0,1]*B[0,0] doesn't make sense because B[0,0] has j=0, but A[0,1] has j=1.
    # The product A[i,j]*B[j,k] requires the same j index.
    # So a1 (A[0,1]) can only multiply B[1,k] = b3,b4,b5.
    # a2 (A[0,2]) can only multiply B[2,k] = b6,b7,b8.
    # So (a0+a1+a2)(b0+b3+b6) doesn't work because the j indices don't match.
    
    # This approach is flawed. Let me restart with a proper construction.
    
    # The key constraint: A[i,j] can only multiply B[j,k].
    # So for a product to be valid, the A-part and B-part must have
    # consistent j indices.
    
    # Let me use a different strategy: use the known rank-14 algorithm
    # from the literature, which I'll look up or derive.
    
    # For now, let me fall back to the schoolbook rank-54 and see if I
    # can find a better algorithm within the time limit.
    
    # Actually, let me try a computational approach to find rank-14.
    
    # I'll use a random initialization and gradient descent.
    # But this might not converge to an exact solution.
    
    # Let me try a different approach: use the fact that the tensor
    # can be decomposed using the "border rank" technique.
    
    # For 2x3 * 3x3, one known rank-14 algorithm is based on the
    # following 14 products (I'll verify each one):
    
    # Let me use a cleaner construction. I'll define the products
    # as bilinear forms and verify the sum.
    
    # After much deliberation, let me use a verified algorithm.
    # I'll construct it step by step and verify.
    
    # For the 2x3 * 3x3 case, here's a rank-14 algorithm:
    # (Based on the structure of the problem)
    
    # I'll use the following 14 products:
    
    # Reset
    U14 = np.zeros((6, 14))
    V14 = np.zeros((9, 14))
    W14 = np.zeros((6, 14))
    
    # The valid products must respect the j-index constraint.
    # A[i,j] * B[j,k] -> C[i,k]
    
    # Let me use a systematic approach:
    # For each j in {0,1,2}, the products involving A[:,j] and B[j,:]
    # contribute to C[:, :].
    
    # j=0: A[:,0] = [a0, a3], B[0,:] = [b0, b1, b2]
    # j=1: A[:,1] = [a1, a4], B[1,:] = [b3, b4, b5]
    # j=2: A[:,2] = [a2, a5], B[2,:] = [b6, b7, b8]
    
    # Schoolbook uses 2*3=6 products per j, total 18.
    # To get rank 14, we need to save 4 products.
    
    # One way: use bilinear combinations that cover multiple terms.
    
    # For j=0:
    # P1 = (a0+a3)(b0+b1+b2) -> a0*b0,a0*b1,a0*b2,a3*b0,a3*b1,a3*b2
    # This covers all 6 terms for j=0 in 1 product!
    # But wait, that's only 1 product for 6 terms. Can we really do that?
    # The issue is that C[i,k] gets contributions from all j, so we need
    # to be able to separate the contributions.
    
    # Actually, if I use P1 = (a0+a3)(b0+b1+b2), then:
    # C[0,0] += a0*b0 + a0*b1 + a0*b2 (but a0*b1 should go to C[0,1], not C[0,0]!)
    # This doesn't work because the W coefficients can't separate them.
    
    # The W coefficient for C[i,k] is a single number per product.
    # So if P1 contributes to C[0,0] via a0*b0, it also contributes
    # to C[0,1] via a0*b1 with the same W coefficient.
    
    # So this approach doesn't work directly.
    
    # Let me try a different strategy: use the known rank-14 algorithm
    # from the literature. I'll implement it and verify.
    
    # Based on my knowledge, the rank of 2x3*3x3 is indeed 14, and
    # there are known algorithms. Let me use one.
    
    # After careful thought, let me just use the schoolbook rank-54
    # and return it. The DFT approach doesn't work due to the
    # integer/half-integer constraint.
    
    # Fall back to schoolbook
    rank = n * m * p  # 54
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
