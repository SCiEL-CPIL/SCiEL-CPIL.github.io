import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 7
    t0 = time.time()
    deadline = t0 + 130  # safety margin (evaluator kills at 150s)

    # Laderman algorithm for 3x3 matrix multiplication (rank 23)
    # A is 3x3, B is 3x3, C = A*B
    # The 23 products are:
    #   p0 = A00 * B00
    #   p1 = (A00 + A01) * (B10 + B11)
    #   p2 = (A00 + A01 - A02) * (B20 + B21 - B22)
    #   p3 = A11 * B11
    #   p4 = (A00 - A11) * (B11 + B22)
    #   p5 = (A00 + A11 - A22) * (B22 + B00)
    #   p6 = (A11 - A22) * (B00 + B01)
    #   p7 = (A01 + A11) * (B10 + B20)
    #   p8 = (A01 - A11) * (B10 + B11)
    #   p9 = (A01 + A11 - A22) * (B20 + B22)
    #   p10 = (A02 + A11) * (B11 + B21)
    #   p11 = (A02 - A11) * (B11 + B12)
    #   p12 = (A02 + A11 - A22) * (B21 + B22)
    #   p13 = (A10 + A11) * (B11 + B12)
    #   p14 = (A10 - A11) * (B01 + B11)
    #   p15 = (A10 + A11 - A22) * (B01 + B22)
    #   p16 = (A10 + A11 + A22) * (B12 + B22)
    #   p17 = (A20 + A11) * (B11 + B21)
    #   p18 = (A20 - A11) * (B11 + B12)
    #   p19 = (A20 + A11 - A22) * (B01 + B22)
    #   p20 = (A20 + A11 + A22) * (B01 + B12)
    #   p21 = (A20 + A21) * (B21 + B22)
    #   p22 = (A20 - A21) * (B11 + B22)

    # C = A*B where C[i][k] = sum_j A[i][j] * B[j][k]
    # C[0][0] = p0
    # C[0][1] = p1 - p0
    # C[0][2] = p2 - p1 + p0
    # C[1][0] = p7 - p0
    # C[1][1] = p1 - p7 + p0
    # C[1][2] = p8 - p1 + p7 - p0
    # C[2][0] = p17 - p7
    # C[2][1] = p21 - p17 + p7
    # C[2][2] = p2 - p21 + p17 - p7

    # Let me verify this carefully.
    # Actually, let me use the standard Laderman formulation:
    #
    # p0 = A[0][0] * B[0][0]
    # p1 = (A[0][0] + A[0][1]) * (B[1][0] + B[1][1])
    # p2 = (A[0][0] + A[0][1] - A[0][2]) * (B[2][0] + B[2][1] - B[2][2])
    # p3 = A[1][1] * B[1][1]
    # p4 = (A[0][0] - A[1][1]) * (B[1][1] + B[2][2])
    # p5 = (A[0][0] + A[1][1] - A[2][2]) * (B[2][2] + B[0][0])
    # p6 = (A[1][1] - A[2][2]) * (B[0][0] + B[0][1])
    # p7 = (A[0][1] + A[1][1]) * (B[1][0] + B[2][0])
    # p8 = (A[0][1] - A[1][1]) * (B[1][0] + B[1][1])
    # p9 = (A[0][1] + A[1][1] - A[2][2]) * (B[2][0] + B[2][2])
    # p10 = (A[0][2] + A[1][1]) * (B[1][1] + B[2][1])
    # p11 = (A[0][2] - A[1][1]) * (B[1][1] + B[1][2])
    # p12 = (A[0][2] + A[1][1] - A[2][2]) * (B[2][1] + B[2][2])
    # p13 = (A[1][0] + A[1][1]) * (B[1][1] + B[1][2])
    # p14 = (A[1][0] - A[1][1]) * (B[0][1] + B[1][1])
    # p15 = (A[1][0] + A[1][1] - A[2][2]) * (B[0][1] + B[2][2])
    # p16 = (A[1][0] + A[1][1] + A[2][2]) * (B[1][2] + B[2][2])
    # p17 = (A[2][0] + A[1][1]) * (B[1][1] + B[2][1])
    # p18 = (A[2][0] - A[1][1]) * (B[1][1] + B[1][2])
    # p19 = (A[2][0] + A[1][1] - A[2][2]) * (B[0][1] + B[2][2])
    # p20 = (A[2][0] + A[1][1] + A[2][2]) * (B[0][1] + B[1][2])
    # p21 = (A[2][0] + A[2][1]) * (B[2][1] + B[2][2])
    # p22 = (A[2][0] - A[2][1]) * (B[1][1] + B[2][2])
    #
    # C[0][0] = p0
    # C[0][1] = p1 - p0
    # C[0][2] = p2 - p1 + p0
    # C[1][0] = p7 - p0
    # C[1][1] = p1 - p7 + p0
    # C[1][2] = p8 - p1 + p7 - p0
    # C[2][0] = p17 - p7
    # C[2][1] = p21 - p17 + p7
    # C[2][2] = p2 - p21 + p17 - p7

    # Let me verify C[0][0] = A[0][0]*B[0][0] + A[0][1]*B[1][0] + A[0][2]*B[2][0]
    # p0 = A[0][0]*B[0][0]
    # So C[0][0] should equal p0 only if the other terms cancel, which they don't.
    # 
    # I think I'm confusing myself. Let me look up the actual Laderman formula.
    #
    # The correct Laderman decomposition:
    # p0 = A00*B00
    # p1 = (A00+A01)*(B10+B11)
    # p2 = (A00+A01-A02)*(B20+B21-B22)
    # p3 = A11*B11
    # p4 = (A00-A11)*(B11+B22)
    # p5 = (A00+A11-A22)*(B22+B00)
    # p6 = (A11-A22)*(B00+B01)
    # p7 = (A01+A11)*(B10+B20)
    # p8 = (A01-A11)*(B10+B11)
    # p9 = (A01+A11-A22)*(B20+B22)
    # p10 = (A02+A11)*(B11+B21)
    # p11 = (A02-A11)*(B11+B12)
    # p12 = (A02+A11-A22)*(B21+B22)
    # p13 = (A10+A11)*(B11+B12)
    # p14 = (A10-A11)*(B01+B11)
    # p15 = (A10+A11-A22)*(B01+B22)
    # p16 = (A10+A11+A22)*(B12+B22)
    # p17 = (A20+A11)*(B11+B21)
    # p18 = (A20-A11)*(B11+B12)
    # p19 = (A20+A11-A22)*(B01+B22)
    # p20 = (A20+A11+A22)*(B01+B12)
    # p21 = (A20+A21)*(B21+B22)
    # p22 = (A20-A21)*(B11+B22)
    #
    # C00 = p0
    # C01 = p1 - p0
    # C02 = p2 - p1 + p0
    # C10 = p7 - p0
    # C11 = p1 - p7 + p0
    # C12 = p8 - p1 + p7 - p0
    # C20 = p17 - p7
    # C21 = p21 - p17 + p7
    # C22 = p2 - p21 + p17 - p7

    # Let me verify C00:
    # C00 = A00*B00 + A01*B10 + A02*B20
    # p0 = A00*B00
    # So C00 = p0 + A01*B10 + A02*B20
    # This doesn't equal p0 alone. So the formula must be different.
    #
    # I think I need to be more careful. Let me just use a known correct formulation.
    #
    # Actually, I recall now. The Laderman algorithm works as follows:
    # Define 23 bilinear forms. The output C[i][k] is a linear combination of these.
    #
    # Let me just implement it by constructing the U, V, W matrices directly
    # by enumerating the 23 terms and their contributions.

    # For the Laderman 3x3x3:
    # Each term t has a coefficient for each (i,j) in U (row i*3+j), 
    # a coefficient for each (j,k) in V (row j*3+k),
    # and a coefficient for each (k,i) in W (row k*3+i).
    
    # U is 9x23, V is 9x23, W is 9x23
    
    # Let me define the 23 terms as (u_coeff, v_coeff, w_coeff) where
    # u_coeff is a dict from (i,j) to coefficient
    # v_coeff is a dict from (j,k) to coefficient  
    # w_coeff is a dict from (k,i) to coefficient

    # Actually, it's easier to think of it this way:
    # For each of the 23 terms, we have:
    # - A linear combination of entries of A (3x3) -> gives a row of U
    # - A linear combination of entries of B (3x3) -> gives a row of V
    # - A linear combination of entries of C (3x3) -> gives a row of W
    
    # The tensor decomposition says: T[i,j,k] = sum_t U[i,j,t] * V[j,k,t] * W[k,i,t]
    # where T[i,j,k] = delta (the multiplication tensor)
    
    # For matrix multiplication, T[i,j,k] = 1 if (i,j,k) corresponds to A[i,j]*B[j,k] contributing to C[i,k]
    # In the trace convention W[k*n+i, r], the output is C[i,k] = sum_j A[i,j]*B[j,k]
    
    # So the tensor T has T[i,j,k] = 1 for all i,j,k (it's the "multiplication" tensor)
    # Wait, no. The tensor for matrix multiplication is:
    # T[i, j, k] where i indexes rows of A, j indexes cols of A / rows of B, k indexes cols of B
    # T[i,j,k] = 1 (all ones) because A[i,j] * B[j,k] contributes to C[i,k]
    
    # The decomposition is: sum_t U[i,j,t] * V[j,k,t] * W[k,i,t] = T[i,j,k] = 1 for all i,j,k
    # Wait, that's not right either. The tensor is:
    # T[i, j, k] = 1 if this is a valid multiplication term, which is always true.
    # But actually the tensor has T[i,j,k] = 1 for all i,j,k in the range.
    
    # Hmm, let me think about this differently.
    # The matrix multiplication tensor T ∈ R^{n×m} ⊗ R^{m×p} ⊗ R^{p×n}
    # T[i, j, k] = 1 for all i ∈ {0..n-1}, j ∈ {0..m-1}, k ∈ {0..p-1}
    # No wait, that's not right.
    
    # The multiplication tensor is defined as:
    # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k * (coefficient)
    # where the coefficient is 1 if A[i,j]*B[j,k] contributes to C[i,k]
    # Since every (i,j,k) triple corresponds to a valid product, T[i,j,k] = 1 for all i,j,k.
    
    # Actually no. The tensor T is such that:
    # <T, A ⊗ B> = C
    # where A is n×m, B is m×p, C is p×n (in trace convention C[k*n+i] = C[i,k])
    # <T, A ⊗ B> = sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] = C[i,k]
    # So T[i,j,k] must be such that sum_j T[i,j,k]*A[i,j]*B[j,k] = C[i,k] = sum_j A[i,j]*B[j,k]
    # This means T[i,j,k] = 1 for all i,j,k.
    
    # So the tensor is just the all-ones tensor of shape (n*m, m*p, p*n) = (15, 35, 21)
    # And we want to decompose it as sum_t U[:,t] ⊗ V[:,t] ⊗ W[:,t]
    # where U is 15×rank, V is 35×rank, W is 21×rank
    
    # For the schoolbook: rank = n*m*p = 105
    # U[i*m+j, t] = 1 if t = i*m*p + j*p + k
    # V[j*p+k, t] = 1 if t = i*m*p + j*p + k  
    # W[k*n+i, t] = 1 if t = i*m*p + j*p + k
    
    # For the Laderman 3x3x3 part (i<3, j<3, k<3):
    # We replace the 27 schoolbook terms with 23 Laderman terms.
    
    # The Laderman terms for the 3x3x3 sub-block:
    # Each term t (0..22) has:
    # - u_vec: 9-dim vector (for i,j in 0..2)
    # - v_vec: 9-dim vector (for j,k in 0..2)
    # - w_vec: 9-dim vector (for k,i in 0..2)
    
    # And we need: sum_{t=0}^{22} u_vec[t][i*m+j] * v_vec[t][j*p+k] * w_vec[t][k*n+i] = 1
    # for all i,j,k in 0..2
    
    # Let me define the Laderman terms explicitly.
    # I'll use the standard Laderman formulation.
    
    # Term 0: p0 = A[0,0] * B[0,0]
    # u: (0,0) -> 1
    # v: (0,0) -> 1
    # w: (0,0) -> 1
    
    # Term 1: p1 = (A[0,0]+A[0,1]) * (B[1,0]+B[1,1])
    # u: (0,0) -> 1, (0,1) -> 1
    # v: (1,0) -> 1, (1,1) -> 1
    # w: contributes to C[0,1] and C[0,0]? No...
    
    # I think the issue is that I need to be very careful about how the Laderman
    # decomposition maps to the tensor format.
    
    # Let me just directly construct the U, V, W matrices for the full problem
    # by combining the Laderman 3x3x3 with schoolbook for the rest.
    
    # Strategy:
    # 1. For the 3x3x3 sub-block (i<3, j<3, k<3), use Laderman (23 terms)
    # 2. For all other (i,j,k) triples, use schoolbook (1 term each)
    # 3. Total rank = 23 + (105 - 27) = 23 + 78 = 101
    
    # Let me define the Laderman terms in the tensor format.
    # For the 3x3x3 sub-block, the tensor entries are T[i,j,k] = 1 for i,j,k in 0..2
    # We need: sum_{t=0}^{22} U[i*3+j, t] * V[j*3+k, t] * W[k*3+i, t] = 1 for all i,j,k in 0..2
    
    # Here's the Laderman decomposition (I'll verify by computation):
    
    # I'll define each term as (u_dict, v_dict, w_dict) where:
    # u_dict maps (i,j) -> coefficient
    # v_dict maps (j,k) -> coefficient
    # w_dict maps (k,i) -> coefficient
    
    laderman_terms = []
    
    # p0 = A00 * B00
    laderman_terms.append((
        {(0,0): 1},
        {(0,0): 1},
        {(0,0): 1}
    ))
    
    # p1 = (A00+A01) * (B10+B11)
    laderman_terms.append((
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (1,1): 1},
        {(1,0): 1, (0,0): -1}  # C01 = p1 - p0, C00 gets -p0 from this term
    ))
    
    # Hmm, this is getting complicated. Let me just hardcode the full U, V, W
    # matrices for the Laderman 3x3x3 by looking up the correct values.
    
    # Actually, let me take a different approach. I'll construct the Laderman
    # decomposition programmatically by solving for the coefficients.
    
    # But that might be slow. Let me just use a known correct Laderman formulation.
    
    # Here's the correct Laderman algorithm (verified):
    # 
    # p0 = A[0][0] * B[0][0]
    # p1 = (A[0][0] + A[0][1]) * (B[1][0] + B[1][1])
    # p2 = (A[0][0] + A[0][1] - A[0][2]) * (B[2][0] + B[2][1] - B[2][2])
    # p3 = A[1][1] * B[1][1]
    # p4 = (A[0][0] - A[1][1]) * (B[1][1] + B[2][2])
    # p5 = (A[0][0] + A[1][1] - A[2][2]) * (B[2][2] + B[0][0])
    # p6 = (A[1][1] - A[2][2]) * (B[0][0] + B[0][1])
    # p7 = (A[0][1] + A[1][1]) * (B[1][0] + B[2][0])
    # p8 = (A[0][1] - A[1][1]) * (B[1][0] + B[1][1])
    # p9 = (A[0][1] + A[1][1] - A[2][2]) * (B[2][0] + B[2][2])
    # p10 = (A[0][2] + A[1][1]) * (B[1][1] + B[2][1])
    # p11 = (A[0][2] - A[1][1]) * (B[1][1] + B[1][2])
    # p12 = (A[0][2] + A[1][1] - A[2][2]) * (B[2][1] + B[2][2])
    # p13 = (A[1][0] + A[1][1]) * (B[1][1] + B[1][2])
    # p14 = (A[1][0] - A[1][1]) * (B[0][1] + B[1][1])
    # p15 = (A[1][0] + A[1][1] - A[2][2]) * (B[0][1] + B[2][2])
    # p16 = (A[1][0] + A[1][1] + A[2][2]) * (B[1][2] + B[2][2])
    # p17 = (A[2][0] + A[1][1]) * (B[1][1] + B[2][1])
    # p18 = (A[2][0] - A[1][1]) * (B[1][1] + B[1][2])
    # p19 = (A[2][0] + A[1][1] - A[2][2]) * (B[0][1] + B[2][2])
    # p20 = (A[2][0] + A[1][1] + A[2][2]) * (B[0][1] + B[1][2])
    # p21 = (A[2][0] + A[2][1]) * (B[2][1] + B[2][2])
    # p22 = (A[2][0] - A[2][1]) * (B[1][1] + B[2][2])
    #
    # C[0][0] = p0
    # C[0][1] = p1 - p0
    # C[0][2] = p2 - p1 + p0
    # C[1][0] = p7 - p0
    # C[1][1] = p1 - p7 + p0
    # C[1][2] = p8 - p1 + p7 - p0
    # C[2][0] = p17 - p7
    # C[2][1] = p21 - p17 + p7
    # C[2][2] = p2 - p21 + p17 - p7

    # Let me verify C[0][0]:
    # C[0][0] = A[0][0]*B[0][0] + A[0][1]*B[1][0] + A[0][2]*B[2][0]
    # p0 = A[0][0]*B[0][0]
    # So C[0][0] = p0 + A[0][1]*B[1][0] + A[0][2]*B[2][0]
    # But the formula says C[0][0] = p0. That's wrong unless A[0][1]*B[1][0] + A[0][2]*B[2][0] = 0.
    
    # I think I have the wrong Laderman formula. Let me look it up more carefully.
    
    # Actually, I think the issue is that the Laderman algorithm I'm remembering
    # might not be correct, or I'm misremembering the output formula.
    
    # Let me just use a simpler approach: I'll construct the Laderman decomposition
    # by directly specifying the U, V, W matrices and verifying they work.
    
    # For now, let me just return the schoolbook solution to be safe.
    # The Laderman approach is correct in principle but I need to get the
    # exact coefficients right.
    
    # Actually, let me try a different strategy. I'll use the known fact that
    # Laderman's algorithm works, and I'll construct it carefully.
    
    # The key insight: for the 3x3x3 sub-tensor, we need 23 terms such that
    # sum_t U[i,j,t] * V[j,k,t] * W[k,i,t] = 1 for all i,j,k in {0,1,2}
    
    # Let me just hardcode a known correct Laderman decomposition.
    # I'll use the one from the Wikipedia article on Laderman's algorithm.
    
    # From Wikipedia (Laderman's algorithm):
    # p1 = a11 * b11
    # p2 = (a11 + a12) * (b21 + b22)
    # p3 = (a11 + a12 - a13) * (b31 + b32 - b33)
    # p4 = a22 * b22
    # p5 = (a11 - a22) * (b22 + b33)
    # p6 = (a11 + a22 - a33) * (b33 + b11)
    # p7 = (a22 - a33) * (b11 + b12)
    # p8 = (a12 + a22) * (b21 + b31)
    # p9 = (a12 - a22) * (b21 + b22)
    # p10 = (a12 + a22 - a33) * (b31 + b33)
    # p11 = (a13 + a22) * (b22 + b32)
    # p12 = (a13 - a22) * (b22 + b23)
    # p13 = (a13 + a22 - a33) * (b32 + b33)
    # p14 = (a21 + a22) * (b22 + b23)
    # p15 = (a21 - a22) * (b12 + b22)
    # p16 = (a21 + a22 - a33) * (b12 + b33)
    # p17 = (a21 + a22 + a33) * (b23 + b33)
    # p18 = (a31 + a22) * (b22 + b32)
    # p19 = (a31 - a22) * (b22 + b23)
    # p20 = (a31 + a22 - a33) * (b12 + b33)
    # p21 = (a31 + a32) * (b32 + b33)
    # p22 = (a31 - a32) * (b22 + b33)
    #
    # c11 = p1
    # c12 = p2 - p1
    # c13 = p3 - p2 + p1
    # c21 = p8 - p1
    # c22 = p2 - p8 + p1
    # c23 = p9 - p2 + p8 - p1
    # c31 = p18 - p8
    # c32 = p21 - p18 + p8
    # c33 = p3 - p21 + p18 - p8

    # Let me verify c11:
    # c11 = a11*b11 + a12*b21 + a13*b31
    # p1 = a11*b11
    # So c11 = p1 + a12*b21 + a13*b31
    # But the formula says c11 = p1. This is still wrong.
    
    # I think the problem is that I'm not accounting for all the contributions.
    # The Laderman algorithm must have a different structure than what I'm assuming.
    
    # Let me just give up on Laderman for now and return the schoolbook solution.
    # I can try to verify the Laderman decomposition numerically in a separate step.
    
    rank_school = n * m * p  # 105
    U_sb = np.zeros((n * m, rank_school), dtype=int)
    V_sb = np.zeros((m * p, rank_school), dtype=int)
    W_sb = np.zeros((p * n, rank_school), dtype=int)
    
    for t in range(rank_school):
        i = t // (m * p)
        rem = t % (m * p)
        j = rem // p
        k = rem % p
        U_sb[i * m + j, t] = 1
        V_sb[j * p + k, t] = 1
        W_sb[k * n + i, t] = 1
    
    if time.time() > deadline:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
