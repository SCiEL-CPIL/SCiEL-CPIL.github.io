"""Matrix multiplication 4x5x5 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 5, 5
    nm, mp, pn = n * m, m * p, p * n  # 20, 25, 20
    
    def idx_A(i, j):
        """Index in U for A[i,j], where A is n x m, i in 0..n-1, j in 0..m-1"""
        return i * m + j
    
    def idx_B(j, k):
        """Index in V for B[j,k], where B is m x p, j in 0..m-1, k in 0..p-1"""
        return j * p + k
    
    def idx_C(i, k):
        """Index in W for C[i,k], where C is n x p, i in 0..n-1, k in 0..p-1"""
        return k * n + i  # W[k*n+i] per the convention
    
    terms = []  # List of (u_idx, v_idx, w_idx, coeff)
    
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        terms.append((u_idx, v_idx, w_idx, coeff))
    
    # Strategy: Use Strassen for the 4x4x4 core (A[0:4,0:4] * B[0:4,0:4] -> C[0:4,0:4])
    # and schoolbook for all other terms.
    
    # The full schoolbook decomposition has n*m*p = 4*5*5 = 100 terms.
    # The 4x4x4 core has 4*4*4 = 64 terms.
    # Strassen reduces the 4x4x4 core to 7 multiplications of 2x2x2, 
    # each with rank 7 (using Strassen) = 49 terms.
    # Or 7 multiplications of 2x2x2, each with rank 8 (schoolbook) = 56 terms.
    # 
    # The remaining terms (non-core) are 100 - 64 = 36 terms.
    # Total with one-level Strassen: 56 + 36 = 92 terms.
    # Total with full Strassen: 49 + 36 = 85 terms.
    
    # Let's implement Strassen for the 4x4x4 core.
    # A_core = A[0:4, 0:4], B_core = B[0:4, 0:4], C_core = C[0:4, 0:4]
    
    # Split into 2x2 blocks:
    # A00 = A[0:2, 0:2], A01 = A[0:2, 2:4]
    # A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    # B00 = B[0:2, 0:2], B01 = B[0:2, 2:4]
    # B10 = B[2:4, 0:2], B11 = B[2:4, 2:4]
    # C00 = C[0:2, 0:2], C01 = C[0:2, 2:4]
    # C10 = C[2:4, 0:2], C11 = C[2:4, 2:4]
    
    # Strassen 7 multiplications:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 - M4 + M6 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M2 + M7
    
    # Each Mk is a 2x2 matrix product. For schoolbook 2x2x2, rank = 8.
    # For each Mk = X * Y where X, Y are 2x2:
    # C[i,j] = sum_k X[i,k] * Y[k,j]
    # Each term: U[A_idx(i,k)] = coeff_x, V[B_idx(k,j)] = coeff_y, W[C_idx(i,j)] = 1
    
    # But wait, in the Strassen formula, the coefficients are applied to the whole 2x2 matrix.
    # So for M1 = (A00+A11)*(B00+B11), we expand:
    # M1 = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # Each of these is a 2x2*2x2 schoolbook product (8 terms each).
    # So M1 contributes 32 terms.
    
    # Similarly for other M's.
    
    # Let's write a helper to add schoolbook terms for a 2x2*2x2 product.
    # X_u[i][k] = U row index for X[i,k]
    # Y_v[k][j] = V row index for Y[k,j]
    # C_w[i][j] = W row index for C[i,j]
    
    def add_2x2_schoolbook(X_u, Y_v, C_w, coeff=1):
        """Add 8 schoolbook terms for a 2x2*2x2 product."""
        for i in range(2):
            for k in range(2):
                for j in range(2):
                    add_term(X_u[i][k], Y_v[k][j], C_w[i][j], coeff)
    
    # Define the 2x2 blocks as 2x2 lists of indices
    A00_u = [[idx_A(0,0), idx_A(0,1)], [idx_A(1,0), idx_A(1,1)]]
    A01_u = [[idx_A(0,2), idx_A(0,3)], [idx_A(1,2), idx_A(1,3)]]
    A10_u = [[idx_A(2,0), idx_A(2,1)], [idx_A(3,0), idx_A(3,1)]]
    A11_u = [[idx_A(2,2), idx_A(2,3)], [idx_A(3,2), idx_A(3,3)]]
    
    B00_v = [[idx_B(0,0), idx_B(0,1)], [idx_B(1,0), idx_B(1,1)]]
    B01_v = [[idx_B(0,2), idx_B(0,3)], [idx_B(1,2), idx_B(1,3)]]
    B10_v = [[idx_B(2,0), idx_B(2,1)], [idx_B(3,0), idx_B(3,1)]]
    B11_v = [[idx_B(2,2), idx_B(2,3)], [idx_B(3,2), idx_B(3,3)]]
    
    C00_w = [[idx_C(0,0), idx_C(0,1)], [idx_C(1,0), idx_C(1,1)]]
    C01_w = [[idx_C(0,2), idx_C(0,3)], [idx_C(1,2), idx_C(1,3)]]
    C10_w = [[idx_C(2,0), idx_C(2,1)], [idx_C(3,0), idx_C(3,1)]]
    C11_w = [[idx_C(2,2), idx_C(2,3)], [idx_C(3,2), idx_C(3,3)]]
    
    # M1 = (A00+A11)*(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
    add_2x2_schoolbook(A00_u, B00_v, C00_w, 1)
    add_2x2_schoolbook(A00_u, B11_v, C00_w, 1)
    add_2x2_schoolbook(A11_u, B00_v, C00_w, 1)
    add_2x2_schoolbook(A11_u, B11_v, C00_w, 1)
    # M1 contributes 32 terms.
    
    # M2 = (A10+A11)*B00 = A10*B00 + A11*B00
    add_2x2_schoolbook(A10_u, B00_v, C10_w, 1)
    add_2x2_schoolbook(A11_u, B00_v, C10_w, 1)
    # M2 contributes 16 terms.
    
    # M3 = A00*(B01-B11) = A00*B01 - A00*B11
    add_2x2_schoolbook(A00_u, B01_v, C01_w, 1)
    add_2x2_schoolbook(A00_u, B11_v, C01_w, -1)
    # M3 contributes 16 terms.
    
    # M4 = A11*(B10-B00) = A11*B10 - A11*B00
    add_2x2_schoolbook(A11_u, B10_v, C11_w, 1)
    add_2x2_schoolbook(A11_u, B00_v, C11_w, -1)
    # M4 contributes 16 terms.
    
    # M5 = (A00+A01)*B11 = A00*B11 + A01*B11
    add_2x2_schoolbook(A00_u, B11_v, C01_w, 1)
    add_2x2_schoolbook(A01_u, B11_v, C01_w, 1)
    # M5 contributes 16 terms.
    
    # M6 = (A10-A00)*(B00+B01) = A10*B00 + A10*B01 - A00*B00 - A00*B01
    add_2x2_schoolbook(A10_u, B00_v, C10_w, 1)
    add_2x2_schoolbook(A10_u, B01_v, C10_w, 1)
    add_2x2_schoolbook(A00_u, B00_v, C10_w, -1)
    add_2x2_schoolbook(A00_u, B01_v, C10_w, -1)
    # M6 contributes 32 terms.
    
    # M7 = (A01-A11)*(B10+B11) = A01*B10 + A01*B11 - A11*B10 - A11*B11
    add_2x2_schoolbook(A01_u, B10_v, C11_w, 1)
    add_2x2_schoolbook(A01_u, B11_v, C11_w, 1)
    add_2x2_schoolbook(A11_u, B10_v, C11_w, -1)
    add_2x2_schoolbook(A11_u, B11_v, C11_w, -1)
    # M7 contributes 32 terms.
    
    # Total for Strassen part: 32+16+16+16+16+32+32 = 160 terms.
    # That's way more than the schoolbook 64 terms! Something is wrong.
    
    # Oh I see the issue. The Strassen formula computes the 2x2 block products,
    # but I'm expanding them incorrectly. Let me reconsider.
    
    # Actually, the Strassen algorithm for 2x2*2x2 uses 7 scalar multiplications.
    # For 4x4*4x4, if I use Strassen at the top level, I get 7 multiplications of 2x2*2x2.
    # Each 2x2*2x2 multiplication can be done with schoolbook (rank 8) or Strassen (rank 7).
    # So the total rank for the 4x4*4x4 core is 7*8 = 56 (one-level) or 7*7 = 49 (two-level).
    
    # But in my expansion above, I'm adding 160 terms, which is more than 64 (schoolbook).
    # This means my expansion is wrong.
    
    # Let me think more carefully. For M1 = (A00+A11)*(B00+B11):
    # This is a single 2x2*2x2 product. The result is a 2x2 matrix M1.
    # To compute M1, I need 8 scalar multiplications (schoolbook) or 7 (Strassen).
    # But I'm expanding M1 as A00*B00 + A00*B11 + A11*B00 + A11*B11,
    # which is 4 separate 2x2*2x2 products, each with 8 terms = 32 terms.
    # That's the mistake!
    
    # The correct way: M1 is a 2x2*2x2 product where the input matrices are sums.
    # X = A00 + A11, Y = B00 + B11.
    # M1[i,j] = sum_k X[i,k] * Y[k,j] = sum_k (A00[i,k]+A11[i,k])*(B00[k,j]+B11[k,j])
    # = sum_k (A00[i,k]*B00[k,j] + A00[i,k]*B11[k,j] + A11[i,k]*B00[k,j] + A11[i,k]*B11[k,j])
    # = 4 terms, each a product of scalars.
    # So M1 has 4*4 = 16 scalar products? No, for each (i,j), there are 4 terms.
    # Total: 4 (i,j pairs) * 4 (k and expansion) = 16 terms? No.
    
    # For a 2x2*2x2 schoolbook: C[i,j] = sum_{k=0}^{1} X[i,k]*Y[k,j].
    # So for each (i,j), there are 2 terms. Total 4*2 = 8 terms.
    # If X and Y are sums of 2x2 matrices, then each X[i,k] and Y[k,j] is a sum of 2 scalars.
    # So each term X[i,k]*Y[k,j] expands to 4 scalar products.
    # Total: 8 (schoolbook terms) * 4 (expansion) = 32 terms.
    # That matches what I had.
    
    # But that's 32 terms for M1 alone, and there are 7 M's, so 7*32 = 224 terms?
    # No, the other M's have fewer terms because they involve products with only one sum.
    
    # M2 = (A10+A11)*B00: X = A10+A11, Y = B00.
    # Each X[i,k] is a sum of 2 scalars, Y[k,j] is a single scalar.
    # So each term X[i,k]*Y[k,j] expands to 2 scalar products.
    # Total: 8 * 2 = 16 terms.
    
    # M3 = A00*(B01-B11): X = A00, Y = B01-B11.
    # Each X[i,k] is a single scalar, Y[k,j] is a sum of 2 scalars (with sign).
    # So each term expands to 2 scalar products.
    # Total: 8 * 2 = 16 terms.
    
    # Similarly M4, M5: 16 terms each.
    
    # M6 = (A10-A00)*(B00+B01): X = A10-A00, Y = B00+B01.
    # Each expands to 2 scalars.
    # Total: 8 * 4 = 32 terms.
    
    # M7 = (A01-A11)*(B10+B11): 32 terms.
    
    # Total: 32+16+16+16+16+32+32 = 160 terms.
    # That's still more than 64 (schoolbook for 4x4*4x4).
    
    # I think the issue is that Strassen is not helping here because the matrices are not
    # being treated as atomic 2x2 blocks. In the standard Strassen algorithm, you treat
    # the 2x2 blocks as atomic and do 7 "multiplications" of 2x2 matrices.
    # Each "multiplication" of 2x2 matrices has rank 8 (schoolbook) or 7 (Strassen).
    # So the total rank is 7*8 = 56 or 7*7 = 49.
    
    # But in my expansion, I'm getting 160 terms. This means my expansion is wrong.
    
    # Let me try a different approach. I'll just use schoolbook for everything.
    # Rank = 100.
    
    # Actually, let me just return the schoolbook decomposition to be safe.
    
    # Clear terms and use schoolbook
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                add_term(idx_A(i, j), idx_B(j, k), idx_C(i, k), 1)
    
    rank = len(terms)
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    for r, (u_idx, v_idx, w_idx, coeff) in enumerate(terms):
        if coeff == 1:
            U[u_idx, r] = 1
            V[v_idx, r] = 1
            W[w_idx, r] = 1
        elif coeff == -1:
            U[u_idx, r] = 1
            V[v_idx, r] = -1
            W[w_idx, r] = -1
        else:
            U[u_idx, r] = coeff
            V[v_idx, r] = 1
            W[w_idx, r] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
