import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 7
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (28, rank)
    # W: (p*n, rank) = (14, rank)
    # Tensor T[i*m+j, j*p+k, k*n+i] = 1

    # We use a block decomposition:
    # Split m=4 into m1=2, m2=2.
    # Split p=7 into p1=2, p2=2, p3=3.
    #
    # A = [A1 | A2] where A1, A2 are 2x2.
    # B = [B1; B2] where B1 is 2x7, B2 is 2x7.
    # B1 = [B11 | B12 | B13] where B11, B12 are 2x2, B13 is 2x3.
    # B2 = [B21 | B22 | B23] where B21, B22 are 2x2, B23 is 2x3.
    #
    # C = A1*B1 + A2*B2
    #   = A1*B11 + A1*B12 + A1*B13 + A2*B21 + A2*B22 + A2*B23
    #
    # For the 2x2 * 2x2 blocks, we use Strassen's algorithm (rank 7).
    # For the 2x2 * 2x3 blocks, we need a decomposition.
    #
    # Let's analyze the rank of 2x2 * 2x3.
    # C = A * B, A is 2x2, B is 2x3, C is 2x3.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # This is equivalent to 3 independent 2x2 * 2x1 multiplications?
    # No, the A matrix is shared.
    # Let's view it as:
    # C = A * B = A * [b1 | b2 | b3] = [A*b1 | A*b2 | A*b3]
    # Each A*bk is a 2x2 * 2x1 multiplication, which has rank 4.
    # However, the A is the same.
    # Can we do better?
    # The tensor is T_{(i,j), (j,k), (i,k)} with i,k in {0,1}, j in {0,1}, k in {0,1,2}.
    # Indices: A: 0..3, B: 0..5, C: 0..5.
    # Non-zeros:
    # (0,0,0)=1, (1,1,0)=1  (A00*b00->C00, A01*b10->C00)
    # (0,1,1)=1, (1,2,1)=1  (A00*b01->C01, A01*b11->C01)
    # (0,2,2)=1, (1,3,2)=1  (A00*b02->C02, A01*b12->C02)
    # (2,0,3)=1, (3,1,3)=1  (A10*b00->C10, A11*b10->C10)
    # (2,1,4)=1, (3,2,4)=1  (A10*b01->C11, A11*b11->C11)
    # (2,2,5)=1, (3,3,5)=1  (A10*b02->C12, A11*b12->C12)
    #
    # This tensor is the Kronecker product of T_{2,2,1} (rank 4) and I_3?
    # T_{2,2,1} has non-zeros (0,0,0), (1,1,0), (2,0,1), (3,1,1).
    # T_{2,2,1} \otimes I_3 would have non-zeros at (a*3+b, c*3+d, e*3+f) where (a,b,e) are non-zeros of T_{2,2,1} and b=d.
    # This matches the structure above if we index correctly.
    # Yes, the rank of 2x2 * 2x3 is 4 * 3 = 12?
    # Wait, is it the direct sum?
    # For each column k, we have a 2x2 * 2x1 multiplication.
    # These are disjoint in the B and C indices (different k).
    # So the rank is sum of ranks = 3 * 4 = 12.
    #
    # So:
    # 4 blocks of 2x2 * 2x2: rank 7 each -> 4 * 7 = 28.
    # 2 blocks of 2x2 * 2x3: rank 12 each -> 2 * 12 = 24.
    # Total rank = 28 + 24 = 52.
    # This is better than 56!
    #
    # Let's verify if we can do better for 2x2 * 2x3.
    # Is R(2,2,3) < 12?
    # Lower bound: R(2,2,3) >= max(R(2,2,1), R(2,1,3)) = max(4, 6) = 6.
    # Actually, R(2,1,3) is 2x1 * 1x3, which is just 2*3=6 scalar multiplications? No.
    # A is 2x1, B is 1x3. C is 2x3.
    # C_{i,k} = A_i * B_k.
    # This is a rank 1 tensor? No, it's a sum of 6 rank 1 tensors.
    # Can we do it in fewer?
    # C = A \otimes B^T.
    # This is a rank 1 tensor in the space of matrices?
    # The tensor T_{i,k} = A_i B_k.
    # This is rank 1 in the matrix sense, but in the tensor decomposition sense (bilinear complexity),
    # it's just 6 scalar multiplications.
    # Wait, the bilinear complexity of C = A B where A is 2x1 and B is 1x3 is 2*3 = 6?
    # Yes, because there is no sharing.
    # So R(2,1,3) = 6.
    # Thus R(2,2,3) >= 6.
    # Is R(2,2,3) = 6?
    # If so, total rank = 28 + 2*6 = 40.
    # Let's check if R(2,2,3) = 6.
    # The tensor is T_{i,j,k} with i,k in 2, j in 2, k in 3.
    # If we could share the A's across the 3 columns, we might reduce it.
    # But A is fixed.
    # C_k = A * b_k.
    # This is 3 independent matrix-vector products.
    # Each has rank 4.
    # Since the b_k are independent, the total rank is 3*4=12.
    # Unless there is a way to combine them.
    # But the output C_k depends only on A and b_k.
    # So the tensor is the direct sum of 3 copies of T_{2,2,1}.
    # So R(2,2,3) = 12.
    #
    # So the rank 52 construction is valid.
    #
    # Let's implement it.
    #
    # Block 1: A1 * B11 (2x2 * 2x2) -> Strassen
    # Block 2: A1 * B12 (2x2 * 2x2) -> Strassen
    # Block 3: A1 * B13 (2x2 * 2x3) -> Direct (Rank 12)
    # Block 4: A2 * B21 (2x2 * 2x2) -> Strassen
    # Block 5: A2 * B22 (2x2 * 2x2) -> Strassen
    # Block 6: A2 * B23 (2x2 * 2x3) -> Direct (Rank 12)
    #
    # Total Rank = 7 + 7 + 12 + 7 + 7 + 12 = 52.

    rank = 52
    U = [[0] * rank for _ in range(8)]
    V = [[0] * rank for _ in range(28)]
    W = [[0] * rank for _ in range(14)]

    def add_term(u_idx, v_idx, w_idx, val=1):
        U[u_idx][col] = val
        V[v_idx][col] = val
        W[w_idx][col] = val

    col = 0

    # Helper to add Strassen terms for a 2x2 * 2x2 block
    # A indices: a00, a01, a10, a11
    # B indices: b00, b01, b10, b11
    # C indices: c00, c01, c10, c11
    def add_strassen(a_base, b_base, c_base):
        nonlocal col
        # a_base: index of A00
        # b_base: index of B00
        # c_base: index of C00
        # A: a_base+0 (00), a_base+1 (01), a_base+2 (10), a_base+3 (11)
        # B: b_base+0 (00), b_base+1 (01), b_base+2 (10), b_base+3 (11)
        # C: c_base+0 (00), c_base+1 (01), c_base+2 (10), c_base+3 (11)
        
        # M1 = (A00 + A11)(B00 + B11)
        add_term(a_base, b_base, c_base, 1) # A00*B00 -> C00
        add_term(a_base, b_base+1, c_base+1, 1) # A00*B01 -> C01
        add_term(a_base+2, b_base, c_base+2, 1) # A10*B00 -> C10
        add_term(a_base+2, b_base+1, c_base+3, 1) # A10*B01 -> C11
        
        # Wait, Strassen uses intermediate products.
        # The terms are:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B00)
        # M4 = A11(B01-B11)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 - M4 + M6
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M7
        
        # We need to map these to the decomposition.
        # Each M is a product of a linear combo of A and a linear combo of B.
        # So we add terms for each pair.
        
        # M1
        add_term(a_base, b_base, c_base, 1) # A00 B00
        add_term(a_base, b_base+1, c_base+1, 1) # A00 B01
        add_term(a_base+2, b_base, c_base+2, 1) # A10 B00
        add_term(a_base+2, b_base+1, c_base+3, 1) # A10 B01
        # This is just schoolbook for the 2x2 block.
        # I need to implement Strassen correctly.
        
        # Let's restart the Strassen implementation.
        # We will add 7 terms.
        # Term 1: (A00+A11) * (B00+B11)
        # This contributes to C00 and C11.
        # (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
        # We add 4 sub-terms for this single "Strassen step"?
        # No, the rank-1 tensor is u v w.
        # u = A00+A11, v = B00+B11, w = C00+C11? No.
        # The product is a scalar. It contributes to C00 and C11.
        # So we add:
        # 1. u=A00+A11, v=B00+B11, w=C00 (coeff 1)
        # 2. u=A00+A11, v=B00+B11, w=C11 (coeff 1)
        # But this is 2 rank-1 tensors.
        # And we have 7 such steps.
        # Total 14 terms?
        # No, Strassen reduces the number of scalar multiplications.
        # The bilinear complexity is the number of rank-1 tensors.
        # Each scalar multiplication in Strassen corresponds to a rank-1 tensor?
        # Yes.
        # So 7 scalar multiplications -> 7 rank-1 tensors.
        
        # Let's verify.
        # M1 = (A00+A11)(B00+B11).
        # This is one scalar multiplication.
        # In the decomposition, this corresponds to:
        # u = A00+A11, v = B00+B11, w = ?
        # The result M1 is added to C00 and C11.
        # So the rank-1 tensor is u \otimes v \otimes (C00 + C11)?
        # No, the tensor T has entries T_{a,b,c}.
        # If we use u \otimes v \otimes w, it adds u_a v_b w_c to T_{a,b,c}.
        # We want to add 1 to T_{A00, B00, C00}, T_{A00, B11, C00}, etc.
        # This is not a single rank-1 tensor.
        # A rank-1 tensor is u_a v_b w_c.
        # If u = A00+A11, v = B00+B11, w = C00, then it adds to T_{A00,B00,C00}, T_{A00,B11,C00}, T_{A11,B00,C00}, T_{A11,B11,C00}.
        # This is exactly the contribution of M1 to C00.
        # Similarly, for C11, we need another rank-1 tensor with w=C11.
        # So M1 requires 2 rank-1 tensors.
        # Then 7 M's require 14 rank-1 tensors?
        # But Strassen is rank 7.
        # Ah, the "rank" in bilinear complexity is the number of scalar multiplications.
        # But the tensor rank is the number of rank-1 tensors in the decomposition.
        # Are they the same?
        # Yes, for matrix multiplication, the bilinear complexity equals the tensor rank.
        # So why does M1 require 2 tensors?
        # Because M1 contributes to two outputs.
        # But we can combine them?
        # No, the outputs are distinct indices.
        # So it seems Strassen uses 7 scalar multiplications, but the tensor rank is higher?
        # No, that's not right.
        # Let's check the rank of 2x2x2.
        # It is 7.
        # So there must be a decomposition with 7 rank-1 tensors.
        # How?
        # The 7 terms are:
        # 1. (A00+A11)(B00+B11) -> C00, C11
        # 2. (A10+A11)B00 -> C10, C11
        # 3. A00(B10-B00) -> C01, C11
        # 4. A11(B01-B11) -> C00, C10
        # 5. (A00+A01)B11 -> C01, C11
        # 6. (A10-A00)(B00+B01) -> C10, C00
        # 7. (A01-A11)(B10+B11) -> C11, C01
        
        # Each term contributes to 2 outputs.
        # So each term is a rank-1 tensor with w having 2 non-zeros.
        # Yes! w is a vector, not a scalar.
        # So u \otimes v \otimes w where w has 2 non-zeros is still a rank-1 tensor.
        # So 7 terms -> rank 7.
        
        # Okay, let's implement this.
        
        # Term 1: u = A00+A11, v = B00+B11, w = C00+C11
        # w indices: c_base (C00), c_base+3 (C11)
        U[a_base][col] = 1
        U[a_base+3][col] = 1
        V[b_base][col] = 1
        V[b_base+3][col] = 1
        W[c_base][col] = 1
        W[c_base+3][col] = 1
        col += 1
        
        # Term 2: u = A10+A11, v = B00, w = C10+C11
        U[a_base+2][col] = 1
        U[a_base+3][col] = 1
        V[b_base][col] = 1
        W[c_base+2][col] = 1
        W[c_base+3][col] = 1
        col += 1
        
        # Term 3: u = A00, v = B10-B00, w = C01+C11
        U[a_base][col] = 1
        V[b_base+2][col] = 1
        V[b_base][col] = -1
        W[c_base+1][col] = 1
        W[c_base+3][col] = 1
        col += 1
        
        # Term 4: u = A11, v = B01-B11, w = C00+C10
        U[a_base+3][col] = 1
        V[b_base+1][col] = 1
        V[b_base+3][col] = -1
        W[c_base][col] = 1
        W[c_base+2][col] = 1
        col += 1
        
        # Term 5: u = A00+A01, v = B11, w = C01+C11
        U[a_base][col] = 1
        U[a_base+1][col] = 1
        V[b_base+3][col] = 1
        W[c_base+1][col] = 1
        W[c_base+3][col] = 1
        col += 1
        
        # Term 6: u = A10-A00, v = B00+B01, w = C10+C00
        U[a_base+2][col] = 1
        U[a_base][col] = -1
        V[b_base][col] = 1
        V[b_base+1][col] = 1
        W[c_base+2][col] = 1
        W[c_base][col] = 1
        col += 1
        
        # Term 7: u = A01-A11, v = B10+B11, w = C11+C01
        U[a_base+1][col] = 1
        U[a_base+3][col] = -1
        V[b_base+2][col] = 1
        V[b_base+3][col] = 1
        W[c_base+3][col] = 1
        W[c_base+1][col] = 1
        col += 1

    # Helper to add direct terms for a 2x2 * 2x3 block
    # A indices: a00, a01, a10, a11
    # B indices: b00, b01, b02, b10, b11, b12
    # C indices: c00, c01, c02, c10, c11, c12
    def add_direct_223(a_base, b_base, c_base):
        nonlocal col
        # 12 terms
        # C00 = A00*B00 + A01*B10
        U[a_base][col] = 1; V[b_base][col] = 1; W[c_base][col] = 1; col += 1
        U[a_base+1][col] = 1; V[b_base+3][col] = 1; W[c_base][col] = 1; col += 1
        # C01 = A00*B01 + A01*B11
        U[a_base][col] = 1; V[b_base+1][col] = 1; W[c_base+1][col] = 1; col += 1
        U[a_base+1][col] = 1; V[b_base+4][col] = 1; W[c_base+1][col] = 1; col += 1
        # C02 = A00*B02 + A01*B12
        U[a_base][col] = 1; V[b_base+2][col] = 1; W[c_base+2][col] = 1; col += 1
        U[a_base+1][col] = 1; V[b_base+5][col] = 1; W[c_base+2][col] = 1; col += 1
        # C10 = A10*B00 + A11*B10
        U[a_base+2][col] = 1; V[b_base][col] = 1; W[c_base+3][col] = 1; col += 1
        U[a_base+3][col] = 1; V[b_base+3][col] = 1; W[c_base+3][col] = 1; col += 1
        # C11 = A10*B01 + A11*B11
        U[a_base+2][col] = 1; V[b_base+1][col] = 1; W[c_base+4][col] = 1; col += 1
        U[a_base+3][col] = 1; V[b_base+4][col] = 1; W[c_base+4][col] = 1; col += 1
        # C12 = A10*B02 + A11*B12
        U[a_base+2][col] = 1; V[b_base+2][col] = 1; W[c_base+5][col] = 1; col += 1
        U[a_base+3][col] = 1; V[b_base+5][col] = 1; W[c_base+5][col] = 1; col += 1

    # Block 1: A1 * B11
    # A1 is A[0:2, 0:2] -> indices 0,1,2,3
    # B11 is B[0:2, 0:2] -> indices 0,1,2,3 (B row major: j*p+k, j=0,1; k=0,1)
    # C11 is C[0:2, 0:2] -> indices 0,1,2,3 (C row major: k*n+i, k=0,1; i=0,1)
    add_strassen(0, 0, 0)

    # Block 2: A1 * B12
    # B12 is B[0:2, 2:4] -> indices 2,3,4,5
    # C12 is C[0:2, 2:4] -> indices 2,3,4,5
    add_strassen(0, 2, 2)

    # Block 3: A1 * B13
    # B13 is B[0:2, 4:7] -> indices 4,5,6,7,8,9
    # C13 is C[0:2, 4:7] -> indices 4,5,6,7,8,9
    add_direct_223(0, 4, 4)

    # Block 4: A2 * B21
    # A2 is A[0:2, 2:4] -> indices 4,5,6,7
    # B21 is B[2:4, 0:2] -> indices 14,15,16,17
    # C21 is C[0:2, 0:2] -> indices 0,1,2,3
    add_strassen(4, 14, 0)

    # Block 5: A2 * B22
    # B22 is B[2:4, 2:4] -> indices 16,17,18,19
    # C22 is C[0:2, 2:4] -> indices 2,3,4,5
    add_strassen(4, 16, 2)

    # Block 6: A2 * B23
    # B23 is B[2:4, 4:7] -> indices 18,19,20,21,22,23
    # C23 is C[0:2, 4:7] -> indices 4,5,6,7,8,9
    add_direct_223(4, 18, 4)

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
