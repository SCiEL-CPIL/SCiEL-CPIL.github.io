import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # We want to decompose the tensor T[a,b,c] = 1 iff a=i*m+j, b=j'*p+k, c=k'*n+i
    # where i in [0,n), j in [0,m), j' in [0,m), k in [0,p), k' in [0,p)
    # Actually, the tensor for matrix multiplication C = A * B where A is n x m, B is m x p, C is n x p
    # T[a, b, c] = 1 if a = i*m + j, b = j*p + k, c = i + k*n ... wait
    # The trace convention: W[k*n+i, r]. So C index c = k*n + i.
    # A index a = i*m + j.
    # B index b = j*p + k.
    # The product is C[i, k] = sum_j A[i, j] * B[j, k].
    # So T[i*m+j, j*p+k, k*n+i] = 1.
    # Note: j appears in both A and B indices, linking them.
    # i appears in A and C indices.
    # k appears in B and C indices.
    
    # We can use Strassen's algorithm for 2x2x2 blocks.
    # Let's split the matrices into blocks.
    # A is 2x4. Split into two 2x2 blocks: A = [A0, A1] where A0 = A[:, 0:2], A1 = A[:, 2:4].
    # B is 4x8. Split into two 2x8 blocks: B = [B0; B1] where B0 = B[0:2, :], B1 = B[2:4, :].
    # C = A * B = A0 * B0 + A1 * B1.
    
    # So we need to compute two 2x2 * 2x8 products and add them.
    # Let's use Strassen for the 2x2 * 2x2 part, but B is 2x8.
    # We can split B into four 2x2 blocks: B0 = [B00, B01], B1 = [B10, B11] where each is 2x2.
    # Then C0 = A0 * B0 = A0 * [B00, B01] = [A0*B00, A0*B01].
    # C1 = A1 * B1 = A1 * [B10, B11] = [A1*B10, A1*B11].
    # C = [C0, C1] = [A0*B00, A0*B01, A1*B10, A1*B11].
    # Wait, C is 2x8. So C = [C_left, C_right] where C_left = A0*B0 + A1*B1 is 2x2? No.
    # A is 2x4, B is 4x8. C is 2x8.
    # C[i, k] = sum_{j=0}^{3} A[i, j] B[j, k].
    # Let's split j into j0 (0,1) and j1 (2,3).
    # C = A[:, 0:2] * B[0:2, :] + A[:, 2:4] * B[2:4, :].
    # Let A0 = A[:, 0:2] (2x2), A1 = A[:, 2:4] (2x2).
    # Let B0 = B[0:2, :] (2x8), B1 = B[2:4, :] (2x8).
    # C = A0 * B0 + A1 * B1.
    
    # Now, A0 * B0 is a 2x2 * 2x8 product. We can use Strassen for the 2x2 part.
    # Strassen computes a 2x2 * 2x2 product with 7 multiplications.
    # For 2x2 * 2x8, we can split B0 into four 2x2 blocks: B0 = [B00, B01, B02, B03] where each is 2x2.
    # Then A0 * B0 = [A0*B00, A0*B01, A0*B02, A0*B03].
    # Each A0*B0j is a 2x2 * 2x2 product, which takes 7 multiplications.
    # So A0 * B0 takes 4 * 7 = 28 multiplications.
    # Similarly, A1 * B1 takes 4 * 7 = 28 multiplications.
    # Total rank = 28 + 28 = 56.
    
    # Let's implement this.
    # We need to define the Strassen factors for 2x2 * 2x2.
    # Strassen's algorithm:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # The tensor for 2x2 * 2x2 has rank 7.
    # We can represent this with U, V, W matrices of size (4, 7), (4, 7), (4, 7).
    # Indices:
    # A: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # B: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # C: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    # Let's define the Strassen factors.
    # For each multiplication Mk, we have U[:, k], V[:, k], W[:, k].
    # M1: U1 = [1,0,1,0], V1 = [1,0,1,0], W1 = [1,0,0,1] (C11 and C22)
    # Wait, W1 should be the output coefficients.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # So for M1, W1 = [1, 0, 0, 1] (contributes to C11 and C22)
    # For M2, W2 = [0, 0, 1, -1] (contributes to C21 and C22)
    # For M3, W3 = [0, 1, 0, 1] (contributes to C12 and C22)
    # For M4, W4 = [1, 0, 1, 0] (contributes to C11 and C21)
    # For M5, W5 = [-1, 1, 0, 0] (contributes to C11 and C12)
    # For M6, W6 = [0, 0, 0, 1] (contributes to C22)
    # For M7, W7 = [1, 0, 0, 0] (contributes to C11)
    
    # Let's verify:
    # C11: M1(1) + M4(1) - M5(1) + M7(1) = 1+1-1+1 = 2? No, M5 is -1 in W5, so -M5 is -(-1)=1.
    # Wait, W5 = [-1, 1, 0, 0]. So M5 contributes -1 to C11 and 1 to C12.
    # C11 = M1*1 + M4*1 + M5*(-1) + M7*1 = 1+1-1+1 = 2? That's wrong.
    # Let's re-check Strassen's formulas.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # So W1 = [1, 0, 0, 1]
    # W2 = [0, 0, 1, -1]
    # W3 = [0, 1, 0, 1]
    # W4 = [1, 0, 1, 0]
    # W5 = [-1, 1, 0, 0]
    # W6 = [0, 0, 0, 1]
    # W7 = [1, 0, 0, 0]
    
    # Let's check C11: 1 (from M1) + 1 (from M4) - 1 (from M5) + 1 (from M7) = 2. Still wrong.
    # Ah, the coefficients in the Strassen formulas are 1, -1, etc.
    # C11 = M1 + M4 - M5 + M7. So the coefficient of M5 is -1.
    # So W5[0] = -1.
    # Then C11 = 1*1 + 1*1 + (-1)*(-1) + 1*1 = 1+1+1+1 = 4. Still wrong.
    # I think I'm confusing the W matrix definition.
    # T[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t].
    # For a single multiplication Mk, U[:,k], V[:,k] define the inputs, and W[:,k] defines the outputs.
    # For M1 = (A11+A22)(B11+B22), U1 = [1,0,1,0], V1 = [1,0,1,0].
    # This product contributes to C11 and C22 with coefficient 1.
    # So W1 = [1,0,0,1].
    # For M2 = (A21+A22)B11, U2 = [0,0,1,1], V2 = [1,0,0,0].
    # This contributes to C21 with 1 and C22 with -1.
    # So W2 = [0,0,1,-1].
    # For M3 = A11(B12-B22), U3 = [1,0,0,0], V3 = [0,1,0,-1].
    # This contributes to C12 with 1 and C22 with 1.
    # So W3 = [0,1,0,1].
    # For M4 = A22(B21-B11), U4 = [0,0,0,1], V4 = [0,0,1,-1].
    # This contributes to C11 with 1 and C21 with 1.
    # So W4 = [1,0,1,0].
    # For M5 = (A11+A12)B22, U5 = [1,1,0,0], V5 = [0,0,0,1].
    # This contributes to C11 with -1 and C12 with 1.
    # So W5 = [-1,1,0,0].
    # For M6 = (A21-A11)(B11+B12), U6 = [0,-1,1,0], V6 = [1,1,0,0].
    # This contributes to C22 with 1.
    # So W6 = [0,0,0,1].
    # For M7 = (A12-A22)(B21+B22), U7 = [0,1,0,-1], V7 = [0,0,1,1].
    # This contributes to C11 with 1.
    # So W7 = [1,0,0,0].
    
    # Let's verify C11: M1(1) + M4(1) + M5(-1) + M7(1) = 1+1-1+1 = 2. Still wrong.
    # I see the issue. The Strassen formulas are:
    # C11 = M1 + M4 - M5 + M7
    # So the coefficient of M5 is -1. In the sum, we have W5[0] = -1.
    # So the contribution to C11 from M5 is U5*V5*W5[0] = (A11+A12)B22 * (-1).
    # So C11 = M1 + M4 - M5 + M7 is correct.
    # But my calculation 1+1-1+1 = 2 is wrong because the M's are not 1, they are products.
    # The W matrix just stores the coefficients. The actual values are computed by the tensor product.
    # So the W matrix is correct as defined.
    
    # Let's construct the full U, V, W for the 2x4x8 case.
    # We have 56 terms.
    # For each of the 4 blocks in B0 and B1, we have 7 Strassen terms.
    # Total 8 * 7 = 56 terms.
    
    rank = 56
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Strassen factors for 2x2x2
    U_strassen = [
        [1,0,1,0],  # M1
        [0,0,1,1],  # M2
        [1,0,0,0],  # M3
        [0,0,0,1],  # M4
        [1,1,0,0],  # M5
        [0,-1,1,0], # M6
        [0,1,0,-1]  # M7
    ]
    V_strassen = [
        [1,0,1,0],  # M1
        [1,0,0,0],  # M2
        [0,1,0,-1], # M3
        [0,0,1,-1], # M4
        [0,0,0,1],  # M5
        [1,1,0,0],  # M6
        [0,0,1,1]   # M7
    ]
    W_strassen = [
        [1,0,0,1],  # M1
        [0,0,1,-1], # M2
        [0,1,0,1],  # M3
        [1,0,1,0],  # M4
        [-1,1,0,0], # M5
        [0,0,0,1],  # M6
        [1,0,0,0]   # M7
    ]
    
    # Map global indices to local indices for each block
    # A is 2x4. A0 = A[:, 0:2], A1 = A[:, 2:4].
    # B is 4x8. B0 = B[0:2, :], B1 = B[2:4, :].
    # B0 = [B00, B01, B02, B03] where B0j is 2x2.
    # B1 = [B10, B11, B12, B13] where B1j is 2x2.
    
    # For each block (b_block in 0,1) and each sub-block (s in 0,1,2,3):
    # b_block=0: A0, B0s
    # b_block=1: A1, B1s
    
    col = 0
    for b_block in range(2):  # 0 for A0/B0, 1 for A1/B1
        for s in range(4):  # 0,1,2,3 for the four 2x2 blocks in B
            for t in range(7):  # 7 Strassen terms
                # U: A indices
                # A0: i in 0,1, j in 0,1 -> global a = i*4 + j
                # A1: i in 0,1, j in 2,3 -> global a = i*4 + (j+2)
                for i in range(2):
                    for j_local in range(2):
                        j_global = j_local + 2 * b_block
                        a = i * 4 + j_global
                        U[a, col] = U_strassen[t][i * 2 + j_local]
                
                # V: B indices
                # B0s: j' in 0,1, k in 2*s, 2*s+1 -> global b = j'*8 + (2*s + k_local)
                # B1s: j' in 2,3, k in 2*s, 2*s+1 -> global b = (j'+2)*8 + (2*s + k_local)
                for j_prime_local in range(2):
                    j_prime_global = j_prime_local + 2 * b_block
                    for k_local in range(2):
                        k_global = 2 * s + k_local
                        b = j_prime_global * 8 + k_global
                        V[b, col] = V_strassen[t][j_prime_local * 2 + k_local]
                
                # W: C indices
                # C is 2x8. c = k_global * 2 + i
                for k_local in range(2):
                    k_global = 2 * s + k_local
                    for i in range(2):
                        c = k_global * 2 + i
                        W[c, col] = W_strassen[t][i * 2 + k_local]
                
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
