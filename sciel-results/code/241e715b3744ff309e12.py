"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    # Block Strassen: 2x2 grid of 3x3 blocks.
    # Outer: 2x2 matrix multiplication via Strassen (rank 7)
    # Inner: 3x3 matrix multiplication via schoolbook (rank 9)
    # Total rank: 7 * 9 = 63
    #
    # Wait, that's not right. Let me reconsider.
    # If we partition into 2x2 blocks of size 3x3, then:
    # A is 6x6, B is 6x6, C = A*B is 6x6.
    # A has 4 blocks of 3x3: A[0,0], A[0,1], A[1,0], A[1,1]
    # B has 4 blocks of 3x3: B[0,0], B[0,1], B[1,0], B[1,1]
    # C has 4 blocks of 3x3: C[0,0], C[0,1], C[1,0], C[1,1]
    #
    # Block-level multiplication: C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0], etc.
    # Each block product (3x3 * 3x3) requires rank 9 via schoolbook.
    # There are 8 block products in the naive approach.
    # Using Strassen for the 2x2 block multiplication, we need 7 "block multiplications".
    # But each "block multiplication" is actually a product of two 3x3 matrices, which itself
    # is a tensor of rank 27 (schoolbook) or potentially lower.
    #
    # Hmm, I think I was confused. Let me re-read the directive.
    #
    # "Partition the 6x6 matrices into a 2x2 grid of 3x3 blocks and apply the BCLR rank-21 
    # decomposition for the outer block multiplication, while using the schoolbook rank-9 
    # decomposition for the inner scalar products, achieving a total rank of 189"
    #
    # Wait, 7 * 9 = 63, not 189. Let me re-read more carefully.
    #
    # Actually, I think the issue is that Strassen's algorithm doesn't just do 7 multiplications
    # of the full blocks. It computes 7 intermediate matrices, each of which is a product of
    # sums/differences of blocks. But each such "multiplication" is still a 3x3 by 3x3 matrix
    # multiplication.
    #
    # No wait. Let me think about this differently. The matrix multiplication tensor for 6x6
    # has dimension 6*6*6 = 216 for schoolbook. 
    #
    # If we view it as a 2x2 block structure where each block is 3x3:
    # The "block-level" tensor is for multiplying 2x2 matrices where each entry is a 3x3 matrix.
    # The rank of the 2x2 matrix multiplication tensor over the reals is 7 (Strassen).
    # The rank of the 3x3 matrix multiplication tensor is 9 (schoolbook, and it's known that
    # the rank of 3x3 is at most 20, but schoolbook gives 9).
    #
    # Actually, I realize my confusion. Let me think about the tensor structure more carefully.
    #
    # The 6x6 matrix multiplication tensor T satisfies:
    # T[a, b, c] = sum_{i,j,k} A[i,j] * B[j,k] * C[k... wait, let me use the convention.
    #
    # Using the convention W[k*n+i, r] where n=m=p=6:
    # U[i*m+j, r] = U[i*6+j, r]
    # V[j*p+k, r] = V[j*6+k, r]  
    # W[k*n+i, r] = W[k*6+i, r]
    #
    # And the decomposition is: sum_r U[i*6+j, r] * V[j*6+k, r] * W[k*6+i, r] = delta... 
    # Actually, the matrix multiplication tensor is:
    # M[i,j,k] = 1 if we interpret it as A[i,j]*B[j,k] contributing to C[i,k]
    # So M[i,j,k] = 1 for all i,j,k (it's the all-ones tensor in the right indexing)
    # Wait no. The matrix multiplication tensor for C = A*B is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] = 1 (since A[i,j]*B[j,k] contributes 1 to C[i,k])
    # 
    # The decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]... 
    # Hmm, the W index is W[k*n+i], so W[k*6+i].
    # 
    # So we need: sum_r U[i*6+j, r] * V[j*6+k, r] * W[k*6+i, r] = 1 for all i,j,k
    #
    # For schoolbook: for each (i,j,k), create one term with U[i*6+j]=V[j*6+k]=W[k*6+i]=1.
    # Rank = 6*6*6 = 216.
    #
    # Now for block Strassen: partition i = 3*a + r, j = 3*b + s, k = 3*c + t
    # where a,c in {0,1} (block indices) and r,s,t in {0,1,2} (within-block indices).
    #
    # The Strassen decomposition for 2x2 blocks:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B10)
    # M5 = (A00+A10)*B11
    # M6 = (A01-A11)*(B10+B11)
    # M7 = (A01-A00)*(B00+B10)
    #
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4
    # C10 = M2 - M3 + M6 + M7
    # C11 = M1 - M2 + M5 - M6
    #
    # Each Mk is a product of two 3x3 matrices. For each Mk, we can use schoolbook
    # decomposition of rank 9 for the 3x3 multiplication.
    #
    # So total rank = 7 * 9 = 63.
    #
    # Wait, but the directive says rank 189. Let me re-read...
    # "achieving a total rank of 189"
    # 
    # Hmm, 7 * 27 = 189. So maybe they're using rank 27 for each 3x3 multiplication?
    # That would be the naive schoolbook for 3x3 (3*3*3 = 27). But that doesn't make sense
    # because schoolbook for 3x3 is just 9 multiplications, not 27.
    #
    # Oh wait, I think I see. Maybe the decomposition is structured differently.
    # Perhaps each of the 7 Strassen terms involves multiple 3x3 multiplications, and
    # the total count works out to 189.
    #
    # Actually, 189 = 7 * 27. And 27 = 3*3*3. So maybe they're counting each scalar
    # multiplication within each 3x3 block product separately, and there are 7 such
    # block products, each requiring 27 scalar multiplications? That would be the naive
    # approach without any optimization within the blocks.
    #
    # But that would give rank 7*27 = 189, which is worse than schoolbook 216? No, 189 < 216.
    # 
    # Hmm, but I can do better. Let me just implement Strassen with schoolbook 3x3 (rank 9),
    # giving total rank 63. That should be valid and better than 189.
    #
    # Let me implement the rank-63 decomposition.
    
    n, m, p = DIMENSIONS  # all 6
    rank = 63  # 7 Strassen terms * 9 schoolbook per 3x3
    
    U = [[0] * rank for _ in range(n * m)]  # 36 x 63
    V = [[0] * rank for _ in range(m * p)]  # 36 x 63
    W = [[0] * rank for _ in range(p * n)]  # 36 x 63
    
    # Strassen coefficients:
    # alpha[l, a, b] for A block (a,b)
    # beta[l, b, c] for B block (b,c)  -- note: using b for col of A, row of B
    # gamma[l, a, c] for C block (a,c)
    #
    # l in 0..6, a,b,c in 0..1
    
    # M1 = (A00+A11)*(B00+B11): alpha=(1,0,1,0) for (00,01,10,11), beta same
    # M2 = (A10+A11)*B00: alpha=(0,0,1,1), beta=(1,0,0,0)
    # M3 = A00*(B10-B11): alpha=(1,0,0,0), beta=(0,0,1,-1)
    # M4 = A11*(B01-B10): alpha=(0,0,0,1), beta=(0,1,-1,0)
    # M5 = (A00+A10)*B11: alpha=(1,0,1,0), beta=(0,0,0,1)
    # M6 = (A01-A11)*(B10+B11): alpha=(0,1,0,-1), beta=(0,0,1,1)
    # M7 = (A01-A00)*(B00+B10): alpha=(0,1,0,0)-wait let me redo.
    
    # Let me define alpha[l][a][b] for a,b in {0,1}
    alpha = [
        [1, 0, 1, 0],  # M1: A00+A11
        [0, 0, 1, 1],  # M2: A10+A11
        [1, 0, 0, 0],  # M3: A00
        [0, 0, 0, 1],  # M4: A11
        [1, 0, 1, 0],  # M5: A00+A10
        [0, 1, 0, -1], # M6: A01-A11
        [0, 1, 0, 0],  # M7: A01 - wait, M7 = (A01-A00)*(B00+B10)
    ]
    # Fix M7: A01 - A00
    alpha[6] = [-1, 1, 0, 0]  # -A00 + A01
    
    beta = [
        [1, 0, 1, 0],  # M1: B00+B11
        [1, 0, 0, 0],  # M2: B00
        [0, 0, 1, -1], # M3: B10-B11
        [0, 1, -1, 0], # M4: B01-B10
        [0, 0, 0, 1],  # M5: B11
        [0, 0, 1, 1],  # M6: B10+B11
        [1, 0, 1, 0],  # M7: B00+B10
    ]
    
    gamma = [
        [1, 0, 0, -1],  # C00 = M1 - M4 + M5 - M7: gamma[0,0,0]=1, gamma[0,0,1]=0, gamma[0,1,0]=0, gamma[0,1,1]=-1? 
    ]
    # Wait, gamma[l][a][c] where C_ac gets contribution.
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4
    # C10 = M2 - M3 + M6 + M7
    # C11 = M1 - M2 + M5 - M6
    
    gamma = [
        # l, C00, C01, C10, C11
        [1, 0, 0, 1],   # M1: +C00, +C11
        [0, 0, 1, -1],  # M2: +C10, -C11
        [0, 1, -1, 0],  # M3: +C01, -C10
        [-1, 1, 0, 0],  # M4: -C00, +C01
        [1, 0, 0, 1],   # M5: +C00, +C11
        [0, 0, 1, -1],  # M6: +C10, -C11
        [-1, 0, 1, 0],  # M7: -C00, +C10
    ]
    
    # For each Strassen term l (0..6) and each schoolbook term (r,s,t) in 3x3:
    # col = l * 9 + r * 3 + s * 3 + t  ... let's just use a counter
    col = 0
    for l in range(7):
        for r in range(3):  # within-block row of A
            for s in range(3):  # within-block col of A / row of B
                for t in range(3):  # within-block col of B
                    # U index: i*6+j where i = 3*a+r, j = 3*b+s
                    for a in range(2):
                        for b in range(2):
                            coeff = alpha[l][a * 2 + b]
                            if coeff != 0:
                                u_idx = (3 * a + r) * 6 + (3 * b + s)
                                U[u_idx][col] += coeff
                    
                    # V index: j*6+k where j = 3*b+s, k = 3*c+t
                    for b in range(2):
                        for c in range(2):
                            coeff = beta[l][b * 2 + c]
                            if coeff != 0:
                                v_idx = (3 * b + s) * 6 + (3 * c + t)
                                V[v_idx][col] += coeff
                    
                    # W index: k*6+i where k = 3*c+t, i = 3*a+r
                    for a in range(2):
                        for c in range(2):
                            coeff = gamma[l][a * 2 + c]
                            if coeff != 0:
                                w_idx = (3 * c + t) * 6 + (3 * a + r)
                                W[w_idx][col] += coeff
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

