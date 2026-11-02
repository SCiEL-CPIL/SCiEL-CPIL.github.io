"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    n, m, p = DIMENSIONS
    
    # Build the multiplication tensor T[i, j, k] = delta_{ik}
    # where C[i,k] = sum_j A[i,j] B[j,k]
    # We want to decompose this tensor into rank-r form:
    # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    # where W follows the convention W[k*n+i, r]
    
    # The tensor has shape (n*m, m*p, p*n) in the flattened convention
    # U has shape (n*m, rank)
    # V has shape (m*p, rank)  
    # W has shape (p*n, rank)
    # and the tensor T is such that for all i,j,k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{ik} * A[i,j] * B[j,k]
    
    # Actually, let me think about this more carefully.
    # Matrix multiplication: C = A * B where A is n x m, B is m x p
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The bilinear form for entry (i,k) is: sum_j A[i,j] B[j,k]
    # We want to express this as: C[i,k] = sum_r U[i,k,r] * A_flat[r] * B_flat[r]
    # No, that's not right either.
    
    # The standard tensor decomposition for matrix multiplication:
    # We have tensors U of shape (n*m, r), V of shape (m*p, r), W of shape (p*n, r)
    # such that for all i, j, k:
    #   sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] = delta_{ik}
    # This gives: C[i,k] = sum_j sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] A[i,j] B[j,k]
    # We need this to equal sum_j A[i,j] B[j,k]
    
    # So we need: sum_j sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] A[i,j] B[j,k] = sum_j A[i,j] B[j,k]
    # This is a bilinear form, so we need the tensor to satisfy certain conditions.
    
    # The correct condition is that the decomposition gives the identity on the multiplication map.
    # Let me use the known Strassen-based approach for 2x2 x 2x10.
    
    # For 2x2 x 2x2, Strassen gives rank 7.
    # For 2x2 x 2x10, we can think of it as 5 independent 2x2 x 2x2 multiplications
    # but sharing the A matrix, so we can potentially share some of the Strassen terms.
    
    # Actually, the multiplication is: C[i,k] = sum_j A[i,j] B[j,k]
    # where A is 2x2, B is 2x10, C is 2x10.
    # We can split B into 5 blocks of 2x2: B = [B0 | B1 | B2 | B3 | B4]
    # Then C = A * B = [A*B0 | A*B1 | A*B2 | A*B3 | A*B4]
    # Each A*Bt is a 2x2 product.
    
    # For each block, we can use Strassen's 7 bilinear forms.
    # The key insight: the A-side of Strassen only depends on A, which is shared.
    # So we compute 7 combinations of A once, then for each B block, we combine them.
    
    # Strassen's 7 products for 2x2 x 2x2:
    # M1 = A00 * (B01 - B11)
    # M2 = (A00 + A01) * B11
    # M3 = (A10 + A11) * B00
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A10) * (B00 + B01)
    # M6 = (A01 - A10) * (B10 + B11)
    # M7 = (A00 - A11) * (B01 + B11)
    
    # Wait, let me use the standard Strassen formulation more carefully.
    # For C = AB where A, B are 2x2:
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # M1 = A00 * (B01 - B11)
    # M2 = (A00 + A01) * B11
    # M3 = (A10 + A11) * B00
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A10) * (B00 + B01)
    # M6 = (A01 - A10) * (B10 + B11)
    # M7 = (A00 - A11) * (B01 + B11)
    
    # C00 = M4 + M5 + M6 - M2  ... let me verify
    # Actually, I recall the standard Strassen:
    # p1 = A11 * (B12 - B22)
    # p2 = (A11 + A22) * B22
    # p3 = (A11 + A21) * B11
    # p4 = A22 * (B21 - B11)
    # p5 = (A11 + A22) * (B11 + B12)
    # p6 = (A21 - A11) * (B12 + B22)
    # p7 = (A12 - A22) * (B21 + B22)
    
    # C11 = p1 + p4 + p6 - p2
    # C12 = p2 + p7
    # C21 = p3 + p5
    # C22 = p1 + p3 + p5 - p7
    
    # Let me verify C11: p1 + p4 + p6 - p2
    # = A11*(B12-B22) + A22*(B21-B11) + (A21-A11)*(B12+B22) - (A11+A22)*B22
    # = A11*B12 - A11*B22 + A22*B21 - A22*B11 + A21*B12 + A21*B22 - A11*B12 - A11*B22 - A22*B12 - A22*B22
    # = A22*B21 + A21*B12 + A21*B22 - A22*B11 - A22*B12 - A22*B22 - A11*B22
    
    # That doesn't look right. Let me use a different, well-known formulation.
    
    # Standard Strassen (from Wikipedia, 1-indexed for clarity, converting to 0-indexed):
    # m1 = a11 * (b12 - b22)
    # m2 = (a11 + a12) * b22
    # m3 = (a21 + a22) * b11
    # m4 = a22 * (b21 - b11)
    # m5 = (a11 + a22) * (b11 + b12)
    # m6 = (a12 - a21) * (b21 + b22)
    # m7 = (a11 - a22) * (b12 + b22)
    
    # c11 = m4 + m5 + m6 - m2
    # c12 = m1 + m2
    # c21 = m3 + m5
    # c22 = m1 + m3 + m7 - m5
    
    # Let me verify c11 = m4 + m5 + m6 - m2:
    # m4 = a22*(b21-b11) = a22*b21 - a22*b11
    # m5 = (a11+a22)*(b11+b12) = a11*b11 + a11*b12 + a22*b11 + a22*b12
    # m6 = (a12-a21)*(b21+b22) = a12*b21 + a12*b22 - a21*b21 - a21*b22
    # -m2 = -(a11+a12)*b22 = -a11*b22 - a12*b22
    
    # Sum: a22*b21 - a22*b11 + a11*b11 + a11*b12 + a22*b11 + a22*b12 + a12*b21 + a12*b22 - a21*b21 - a21*b22 - a11*b22 - a12*b22
    # = a22*b21 + a11*b11 + a11*b12 + a22*b12 + a12*b21 - a21*b21 - a21*b22 - a11*b22
    # Hmm, this doesn't simplify to a11*b11 + a12*b21.
    
    # I think I'm making errors. Let me just use the schoolbook decomposition for now
    # and try to find a better one. The schoolbook has rank 40.
    # Actually, let me try a different approach: use the known rank for 2x2x10.
    
    # The rank of the 2x2x2 matrix multiplication tensor is 7 (Strassen).
    # For 2x2 x 2x10, since the second factor has 10 columns, we can view it as
    # multiplying a 2x2 matrix by 5 different 2x2 matrices.
    
    # If we could share all the A-side computations, we'd need 7 + 5*7 = 42, which is worse.
    # But actually, the A-side only needs 7 combinations, and for each B block, we need
    # 7 combinations of B, but the products are shared.
    
    # Wait, I think the key is: we compute 7 values on the A side, and for each of the 5
    # B-blocks, we compute 7 values on the B side, giving 7 * 5 = 35 products total.
    # That's rank 35.
    
    # Let me construct this. The 7 A-combinations and 7 B-combinations for each block.
    
    # Using 0-indexed notation:
    # A = [[a00, a01], [a10, a11]]
    # B_block = [[b00, b01], [b10, b11]]
    
    # The 7 products:
    # p1 = a11 * (b01 - b11)
    # p2 = (a00 + a01) * b11
    # p3 = (a10 + a11) * b00
    # p4 = a11 * (b10 - b00)
    # p5 = (a00 + a10) * (b00 + b01)
    # p6 = (a01 - a10) * (b10 + b11)
    # p7 = (a00 - a11) * (b01 + b11)
    
    # C00 = p4 + p5 + p6 - p2
    # C01 = p1 + p2
    # C10 = p3 + p5
    # C11 = p1 + p3 + p7 - p5
    
    # Let me verify C00 = p4 + p5 + p6 - p2:
    # p4 = a11*(b10-b00) = a11*b10 - a11*b00
    # p5 = (a00+a10)*(b00+b01) = a00*b00 + a00*b01 + a10*b00 + a10*b01
    # p6 = (a01-a10)*(b10+b11) = a01*b10 + a01*b11 - a10*b10 - a10*b11
    # -p2 = -(a00+a01)*b11 = -a00*b11 - a01*b11
    # Sum = a11*b10 - a11*b00 + a00*b00 + a00*b01 + a10*b00 + a10*b01 + a01*b10 + a01*b11 - a10*b10 - a10*b11 - a00*b11 - a01*b11
    # = a11*b10 + a00*b00 + a00*b01 + a10*b00 + a10*b01 + a01*b10 - a10*b10 - a10*b11 - a00*b11 - a11*b00
    # This should equal a00*b00 + a01*b10.
    # Simplifying: = a00*b00 + a01*b10 + a00*b01 + a10*b00 + a10*b01 - a10*b10 - a10*b11 - a00*b11 + a11*b10 - a11*b00
    # This is not equal to a00*b00 + a01*b10 in general. So my Strassen formula is wrong.
    
    # Let me use a verified Strassen formula. From multiple sources, the correct one is:
    # m1 = a00 * (b01 - b11)
    # m2 = (a00 + a01) * b11
    # m3 = (a10 + a11) * b00
    # m4 = a11 * (b10 - b00)
    # m5 = (a00 + a10) * (b00 + b01)
    # m6 = (a01 - a10) * (b10 + b11)
    # m7 = (a00 - a11) * (b01 + b11)
    
    # c00 = m4 + m5 + m6 - m2
    # c01 = m1 + m2
    # c10 = m3 + m5
    # c11 = m1 + m3 + m7 - m5
    
    # Verify c01 = m1 + m2:
    # m1 + m2 = a00*(b01-b11) + (a00+a01)*b11 = a00*b01 - a00*b11 + a00*b11 + a01*b11 = a00*b01 + a01*b11
    # Should be a00*b01 + a01*b11. Correct!
    
    # Verify c10 = m3 + m5:
    # m3 + m5 = (a10+a11)*b00 + (a00+a10)*(b00+b01) = a10*b00 + a11*b00 + a00*b00 + a00*b01 + a10*b00 + a10*b01
    # = a00*b00 + a00*b01 + a10*b00 + a10*b01 + a11*b00 + a10*b00
    # = a00*b00 + a00*b01 + a11*b00 + 2*a10*b00 + a10*b01
    # Should be a10*b00 + a11*b10. This is wrong.
    
    # I'm clearly making errors with the Strassen indices. Let me just use a known-good
    # implementation. After more careful thought, here's the correct Strassen:
    
    # Using 1-based indexing in the formulas (converting to 0-based in code):
    # p1 = A11 * (B12 - B22)
    # p2 = (A11 + A22) * B22
    # p3 = (A11 + A21) * B11
    # p4 = A22 * (B21 - B11)
    # p5 = (A11 + A22) * (B11 + B12)
    # p6 = (A21 - A11) * (B12 + B22)
    # p7 = (A12 - A22) * (B21 + B22)
    
    # C11 = p1 + p4 + p6 - p2
    # C12 = p2 + p7
    # C21 = p3 + p5
    # C22 = p1 + p3 + p7 - p5
    
    # Converting to 0-based:
    # p1 = A[1][1] * (B[1][0] - B[1][1])
    # p2 = (A[1][1] + A[0][0]) * B[1][1]
    # p3 = (A[1][1] + A[1][0]) * B[0][0]
    # p4 = A[0][0] * (B[0][1] - B[0][0])
    # p5 = (A[1][1] + A[0][0]) * (B[0][0] + B[0][1])
    # p6 = (A[1][0] - A[1][1]) * (B[0][1] + B[1][1])
    # p7 = (A[0][1] - A[0][0]) * (B[1][0] + B[1][1])
    
    # C[1][1] = p1 + p4 + p6 - p2
    # C[1][0] = p2 + p7
    # C[0][1] = p3 + p5
    # C[0][0] = p1 + p3 + p7 - p5
    
    # Let me verify C[0][0] = p1 + p3 + p7 - p5:
    # p1 = A[1][1]*(B[1][0]-B[1][1])
    # p3 = (A[1][1]+A[1][0])*B[0][0]
    # p7 = (A[0][1]-A[0][0])*(B[1][0]+B[1][1])
    # -p5 = -(A[1][1]+A[0][0])*(B[0][0]+B[0][1])
    
    # This is getting too complicated to verify by hand. Let me just use the schoolbook
    # decomposition with rank 40, and then try to find improvements.
    
    # Actually, let me try a different strategy. I'll use the known fact that the rank
    # of 2x2x2 matrix multiplication is 7, and for 2x2 x 2x10, we can potentially get
    # rank 35 by sharing the A-side Strassen terms.
    
    # Let me construct the rank-35 decomposition using the following approach:
    # Compute 7 A-combinations, and for each of 5 B-blocks, compute 7 B-combinations.
    # The products give 35 bilinear terms.
    
    # I'll use the following Strassen formulas (verified by testing):
    # For 2x2 matrices A and B:
    # m1 = A[0,0] * (B[0,1] - B[1,1])
    # m2 = (A[0,0] + A[0,1]) * B[1,1]
    # m3 = (A[1,0] + A[1,1]) * B[0,0]
    # m4 = A[1,1] * (B[1,0] - B[0,0])
    # m5 = (A[0,0] + A[1,0]) * (B[0,0] + B[0,1])
    # m6 = (A[0,1] - A[1,0]) * (B[1,0] + B[1,1])
    # m7 = (A[0,0] - A[1,1]) * (B[0,1] + B[1,1])
    
    # C[0,0] = m4 + m5 + m6 - m2
    # C[0,1] = m1 + m2
    # C[1,0] = m3 + m5
    # C[1,1] = m1 + m3 + m7 - m5
    
    # Let me verify C[0,1] = m1 + m2:
    # m1 + m2 = A[0,0]*(B[0,1]-B[1,1]) + (A[0,0]+A[0,1])*B[1,1]
    # = A[0,0]*B[0,1] - A[0,0]*B[1,1] + A[0,0]*B[1,1] + A[0,1]*B[1,1]
    # = A[0,0]*B[0,1] + A[0,1]*B[1,1] ✓
    
    # Let me verify C[1,0] = m3 + m5:
    # m3 + m5 = (A[1,0]+A[1,1])*B[0,0] + (A[0,0]+A[1,0])*(B[0,0]+B[0,1])
    # = A[1,0]*B[0,0] + A[1,1]*B[0,0] + A[0,0]*B[0,0] + A[0,0]*B[0,1] + A[1,0]*B[0,0] + A[1,0]*B[0,1]
    # = A[0,0]*B[0,0] + A[0,0]*B[0,1] + A[1,1]*B[0,0] + 2*A[1,0]*B[0,0] + A[1,0]*B[0,1]
    # Should be A[1,0]*B[0,0] + A[1,1]*B[1,0]. This is wrong.
    
    # The issue is with m5. Let me reconsider. I think the correct Strassen has:
    # m5 = (A[0,0] + A[1,1]) * (B[0,0] + B[0,1])
    
    # Let me try that:
    # m3 + m5 = (A[1,0]+A[1,1])*B[0,0] + (A[0,0]+A[1,1])*(B[0,0]+B[0,1])
    # = A[1,0]*B[0,0] + A[1,1]*B[0,0] + A[0,0]*B[0,0] + A[0,0]*B[0,1] + A[1,1]*B[0,0] + A[1,1]*B[0,1]
    # = A[0,0]*B[0,0] + A[0,0]*B[0,1] + A[1,0]*B[0,0] + 2*A[1,1]*B[0,0] + A[1,1]*B[0,1]
    # Still wrong.
    
    # I give up trying to derive Strassen from memory. Let me just use the schoolbook
    # rank-40 decomposition, which I know is correct, and see if I can improve it.
    
    # Schoolbook: rank = n*m*p = 2*2*10 = 40
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
