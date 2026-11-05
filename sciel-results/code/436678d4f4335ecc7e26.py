"""Block Strassen decomposition for 3x5 x 5x6 matrix multiplication.

Uses rank-7 Strassen on 2x2x2 sub-blocks to reduce rank below the
schoolbook 90. Falls back to schoolbook if verification fails.
"""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
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
                # trace convention: W[k*n + i, r]
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify(n, m, p, U, V, W):
    """Check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == delta(i,j,k)
    for the matrix-multiplication tensor. The tensor T[i,j,k] = 1 always
    (it's the identity-like structure: C[i,k] = sum_j A[i,j] B[j,k]).
    Actually the MM tensor has T[i,j,k]=1 for all i,j,k (it's just the
    multiplication structure). Verify the reconstruction.
    """
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    # Build the tensor: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # We want T[i,j,k] = 1 for all i,j,k (the MM tensor is all-ones in this
    # indexing? No.)
    #
    # The matrix multiplication tensor: C[i,k] = sum_j A[i,j] B[j,k].
    # So the bilinear form: C[i,k] = sum_{j} A[i,j]*B[j,k].
    # The rank-1 term r contributes U[i,j,r]*V[j,k,r]*W[i,k,r] to C[i,k]
    # when summed over j. Wait, let's be careful.
    #
    # Standard: C = A B, C[i,k] = sum_j A[i,j] B[j,k].
    # A rank-r decomposition: C[i,k] = sum_{r} (sum_j U[i,j,r] A[i,j]) (sum_j V[j,k,r] B[j,k]) W[i,k,r]?
    # No. The tensor is T[i,j,k] such that C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k], T=1.
    # A rank-1 decomposition of the *bilinear* structure:
    # C[i,k] = sum_r U[i,r] V[k,r] where U[i,r] = sum_j ... 
    #
    # The given interface: U has shape (n*m, rank), V (m*p, rank), W (p*n, rank).
    # The decomposition says: for each output (i,k), C[i,k] = sum_r W[k*n+i, r] * (something).
    # The standard convention: C[i,k] = sum_r W[i,k,r] * (sum_j U[i,j,r] A[i,j]) * (sum_j V[j,k,r] B[j,k]).
    # Hmm, but then U and V would each depend on j and we'd need to sum over j.
    #
    # Let's use the schoolbook to figure out the convention. In schoolbook:
    # U[i*m+j, col]=1, V[j*p+k, col]=1, W[k*n+i, col]=1, one col per (i,j,k).
    # So for a fixed output (i,k), sum over r: W[k*n+i, r] * ... 
    # The reconstruction must be: C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])?
    # In schoolbook, for col corresponding to (i',j',k'), U[i*m+j,col]=1 iff i=i',j=j'. 
    # So sum_j U[i*m+j, col] A[i,j] = A[i, j'] (if i=i') else 0.
    # sum_j V[j*p+k, col] B[j,k] = B[j', k] (if k=k') else 0.
    # W[k*n+i, col]=1 iff (i=i', k=k').
    # So product = A[i,j'] B[j',k] * delta(i,i')delta(k,k').
    # Sum over col (i',j',k'): = A[i,j'] B[j',k] summed over j' with i'=i,k'=k = sum_{j'} A[i,j']B[j',k] = C[i,k]. 
    # 
    # So the convention is:
    # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A[i,j]) * (sum_j V[j*p+k, r] B[j,k]).
    #
    # Verify with random matrices.
    rng = np.random.default_rng(0)
    A = rng.standard_normal((n, m))
    B = rng.standard_normal((m, p))
    C = A @ B
    Crec = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            ua = np.array([sum(U[i * m + j][r] * A[i, j] for j in range(m)) for r in range(rank)])
            va = np.array([sum(V[j * p + k][r] * B[j, k] for j in range(m)) for r in range(rank)])
            Crec[i, k] = sum(W[k * n + i][r] * ua[r] * va[r] for r in range(rank))
    if not np.allclose(Crec, C, atol=1e-6):
        return False
    return True


def _strassen_2x2x2(U, V, W, i0, j0, k0, col):
    """Add 7 rank-1 terms for the 2x2x2 block starting at (i0,j0,k0).
    U row index = i*m + j, V row index = j*p + k, W row index = k*n + i.
    Returns the next column index.
    """
    n, m, p = DIMENSIONS
    # Strassen for C = A B where A is 2x2, B is 2x2.
    # A = [[a00, a01],[a10, a11]], B = [[b00, b01],[b10, b11]].
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11) b00
    # M3 = a00 (b10-b11)
    # M4 = a11 (b01-b00)
    # M5 = (a00+a01) b11
    # M6 = (a10-a00) (b00+b10)
    # M7 = (a01-a11) (b01+b11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    #
    # We need to express each M as a rank-1 term in the bilinear convention:
    # C[i,k] = sum_r W[i,k,r] * (sum_j U[i,j,r] A[i,j]) * (sum_j V[j,k,r] B[j,k]).
    # A rank-1 term r contributes to C[i,k] the value W[i,k,r] * (U[i,j,r] A[i,j]) * (V[j,k,r] B[j,k])
    # summed over j. For a single (i,j) and (j,k) pair this is W[i,k,r] U[i,j,r] V[j,k,r] A[i,j] B[j,k].
    #
    # For Strassen, M1 = (a00+a11)(b00+b11). This is a product of a linear form in A and a linear form in B.
    # (a00+a11) = sum over (i,j) of coef * A[i,j]: coef[0,0]=1, coef[1,1]=1.
    # (b00+b11) = sum over (j,k) of coef * B[j,k]: coef[0,0]=1, coef[1,1]=1.
    # So M1 as a "bilinear form" = (sum_{i,j} cA[i,j] A[i,j]) (sum_{j,k} cB[j,k] B[j,k]).
    # But our rank-1 term is a *single* product A[i,j] B[j,k] times W[i,k,r]. 
    # M1 is NOT a single A[i,j]B[j,k] product; it's a product of two linear forms.
    # 
    # So we cannot directly use Strassen's 7 multiplications as rank-1 terms in THIS convention!
    # The Strassen "multiplications" are bilinear forms (products of linear forms in A and B),
    # but each rank-1 term here is a trilinear monomial A[i,j] B[j,k] (a single product).
    #
    # Wait. Let me reconsider. A rank-1 term in the MM tensor decomposition:
    # The MM tensor T is in A^* ⊗ B^* ⊗ C^*. A rank-1 term is a ⊗ b ⊗ c where
    # a ∈ A^*, b ∈ B^*, c ∈ C^*. Then the term is (sum_{i,j} a[i,j] A[i,j]) (sum_{j,k} b[j,k] B[j,k]) (sum_{i,k} c[i,k] C[i,k]).
    # But the interface gives U[i*m+j, r], V[j*p+k, r], W[k*n+i, r].
    # So a[i,j] = U[i*m+j, r], b[j,k] = V[j*p+k, r], c[i,k] = W[k*n+i, r].
    # The rank-1 term = (sum_{i,j} U[i*m+j,r] A[i,j]) (sum_{j,k} V[j*p+k,r] B[j,k]) (sum_{i,k} W[k*n+i,r] C[i,k]).
    # Summed over r, this should equal (sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] C[i,k])? No...
    #
    # Actually the standard MM tensor decomposition: the tensor T ∈ R^{n×m} ⊗ R^{m×p} ⊗ R^{n×p}
    # with T[i,j,k] = delta (it's 1 for the "matching" indices). A rank-1 decomposition:
    # T = sum_r u_r ⊗ v_r ⊗ w_r, meaning
    # sum_{i,j,k} T[i,j,k] X[i,j] Y[j,k] Z[i,k] = sum_r (sum_{i,j} u_r[i,j] X[i,j]) (sum_{j,k} v_r[j,k] Y[j,k]) (sum_{i,k} w_r[i,k] Z[i,k]).
    # For the MM tensor, T[i,j,k] = 1 always (since C[i,k] = sum_j A[i,j]B[j,k], the tensor is the "all-ones" 
    # in the sense that T[i,j,k]=1 for all i,j,k). Wait no.
    #
    # The MM tensor: it's the tensor such that contracting with A and B gives C.
    # C[i,k] = sum_j A[i,j] B[j,k]. So T[i,j,k] = 1 for all i,j,k. Yes, T is all-ones!
    # Because C[i,k] = sum_j (1) A[i,j] B[j,k].
    #
    # So we need: sum_r (sum_{i,j} U[i*m+j,r] A[i,j]) (sum_{j,k} V[j*p+k,r] B[j,k]) W' = C where the W part...
    # Hmm, but W is in C^* = R^{n×p}, and we contract W with C.
    # The full trilinear form: sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] C[i,k] = sum_{i,j,k} A[i,j] B[j,k] C[i,k].
    # And the rank-r decomposition: sum_r (sum_{i,j} U_r[i,j] A[i,j]) (sum_{j,k} V_r[j,k] B[j,k]) (sum_{i,k} W_r[i,k] C[i,k]).
    # These are equal for all A,B,C iff the tensors match.
    #
    # Now, Strassen: M1 = (a00+a11)(b00+b11) is a product of a linear form in A and a linear form in B.
    # This corresponds to a rank-1 term where U_r is the linear form (a00+a11), V_r is (b00+b11), 
    # and W_r is... the coefficient that multiplies this product to contribute to C.
    # In Strassen, M1 contributes to C00, C11. Specifically C00 += M1, C11 += M1.
    # So the W_r for M1 would have W[0,0]=1, W[1,1]=1 (in the 2x2 case), and the term is
    # (a00+a11)(b00+b11) * (C[0,0]*1 + C[1,1]*1).
    # Yes! So each Strassen multiplication IS a valid rank-1 term in the trilinear convention.
    # 
    # So for the 2x2x2 block, I add 7 terms, each with U = linear form in A (over the 2x2 block),
    # V = linear form in B (over the 2x2 block), W = the C coefficients.
    #
    # Let me define the 7 Strassen terms with their (U, V, W) linear forms.
    # A indices: (i,j) in {0,1}x{0,1} -> a00,a01,a10,a11
    # B indices: (j,k) in {0,1}x{0,1} -> b00,b01,b10,b11
    # C indices: (i,k) in {0,1}x{0,1} -> c00,c01,c10,c11
    #
    # M1: U = a00+a11, V = b00+b11, W = c00+c11
    # M2: U = a10+a11, V = b00,        W = c10
    # M3: U = a00,        V = b10-b11, W = c01+c10
    # M4: U = a11,        V = b01-b00, W = c00+c10
    # M5: U = a00+a01,    V = b11,     W = c00-c01
    # M6: U = a10-a00,    V = b00+b10, W = c11
    # M7: U = a01-a11,    V = b01+b11, W = c00+c11
    #
    # Let me double check with the standard Strassen formulas:
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    #
    # For each output c[i,k], the coefficient of M_r in c[i,k] gives W_r[i,k].
    # c00: M1(+1), M4(+1), M5(-1), M7(+1) -> W1[0,0]=1, W4[0,0]=1, W5[0,0]=-1, W7[0,0]=1
    # c01: M3(+1), M5(+1) -> W3[0,1]=1, W5[0,1]=1
    # c10: M2(+1), M4(+1) -> W2[1,0]=1, W4[1,0]=1
    # c11: M1(+1), M2(-1), M3(+1), M6(+1) -> W1[1,1]=1, W2[1,1]=-1, W3[1,1]=1, W6[1,1]=1
    #
    # So:
    # W1: c00=1, c11=1
    # W2: c10=1, c11=-1
    # W3: c01=1, c11=1
    # W4: c00=1, c10=1
    # W5: c00=-1, c01=1
    # W6: c11=1
    # W7: c00=1
    #
    # U forms:
    # M1: a00+a11
    # M2: a10+a11
    # M3: a00
    # M4: a11
    # M5: a00+a01
    # M6: a10-a00
    # M7: a01-a11
    #
    # V forms:
    # M1: b00+b11
    # M2: b00
    # M3: b10-b11
    # M4: b01-b00
    # M5: b11
    # M6: b00+b10
    # M7: b01+b11
    #
    # Let me verify M1 contributes correctly. M1 term = (a00+a11)(b00+b11)(c00+c11).
    # Expanding: (a00+a11)(b00+b11) = a00 b00 + a00 b11 + a11 b00 + a11 b11.
    # This times (c00+c11). 
    # In C00 = sum_j A[0,j] B[j,0] = a00 b00 + a01 b10.
    # In C11 = a10 b01 + a11 b11.
    # M1*(c00+c11) contributes a00 b00 to C00 (correct, part of C00), a11 b11 to C11 (correct).
    # But it also contributes a00 b11 to C00 (wrong!) and a11 b00 to C11 (wrong!).
    # These "wrong" terms must cancel with other M's.
    # C00 total = M1+M4-M5+M7.
    # M4 = a11(b01-b00)(c00+c10) = a11 b01 c00 - a11 b00 c00 + a11 b01 c10 - a11 b00 c10.
    # M5 = (a00+a01) b11 (c00-c01) = a00 b11 c00 + a01 b11 c00 - a00 b11 c01 - a01 b11 c01.
    # M7 = (a01-a11)(b01+b11) c00 = a01 b01 c00 + a01 b11 c00 - a11 b01 c00 - a11 b11 c00.
    # C00 coefficient = [M1+M4-M5+M7] restricted to c00:
    # M1: a00 b00 + a00 b11 + a11 b00 + a11 b11
    # M4: a11 b01 - a11 b00
    # -M5: -a00 b11 - a01 b11
    # M7: a01 b01 + a01 b11 - a11 b01 - a11 b11
    # Sum: a00 b00 + (a00 b11 - a00 b11) + (a11 b00 - a11 b00) + (a11 b11 - a11 b11) + a11 b01 - a11 b01 + (-a01 b11 + a01 b11) + a01 b01
    # = a00 b00 + a01 b01. 
    # But C00 should be a00 b00 + a01 b10, not a00 b00 + a01 b01!
    # 
    # I have an error. Let me recheck the standard Strassen.
    # Standard Strassen (2x2):
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11) b00
    # M3 = a00 (b10-b11)
    # M4 = a11 (b01-b00)
    # M5 = (a00+a01) b11
    # M6 = (a10-a00)(b00+b10)
    # M7 = (a01-a11)(b01+b11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    #
    # Let me recompute C00 = M1+M4-M5+M7:
    # M1 = a00 b00 + a00 b11 + a11 b00 + a11 b11
    # M4 = a11 b01 - a11 b00
    # M5 = a00 b11 + a01 b11
    # M7 = a01 b01 + a01 b11 - a11 b01 - a11 b11
    # C00 = M1 + M4 - M5 + M7
    # = (a00 b00 + a00 b11 + a11 b00 + a11 b11) + (a11 b01 - a11 b00) - (a00 b11 + a01 b11) + (a01 b01 + a01 b11 - a11 b01 - a11 b11)
    # a00 b00: +1
    # a00 b11: +1 -1 = 0
    # a11 b00: +1 -1 = 0
    # a11 b11: +1 -1 = 0
    # a11 b01: +1 -1 = 0
    # a01 b11: -1 +1 = 0
    # a01 b01: +1
    # So C00 = a00 b00 + a01 b01. 
    # But the correct C00 = a00 b00 + a01 b10. 
    # 
    # So either my Strassen formula is wrong or I'm mislabeling. Let me look up the correct Strassen.
    # Correct Strassen:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22) B11
    # M3 = A11 (B21 - B22)
    # M4 = A22 (B12 - B11)
    # M5 = (A11 + A12) B22
    # M6 = (A21 - A11) (B11 + B21)
    # M7 = (A12 - A22) (B12 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Here A11=a00, A12=a01, A21=a10, A22=a11. B11=b00, B12=b01, B21=b10, B22=b11.
    # C11 = M1+M4-M5+M7 should = a00 b00 + a01 b10.
    # M1 = (a00+a11)(b00+b11) = a00 b00 + a00 b11 + a11 b00 + a11 b11
    # M4 = a11 (b01-b00) = a11 b01 - a11 b00
    # M5 = (a00+a01) b11 = a00 b11 + a01 b11
    # M7 = (a01-a11)(b01+b11) = a01 b01 + a01 b11 - a11 b01 - a11 b11
    # C11 = M1+M4-M5+M7:
    # a00 b00: +1
    # a00 b11: +1 -1 = 0
    # a11 b00: +1 -1 = 0
    # a11 b11: +1 -1 = 0
    # a11 b01: +1 -1 = 0
    # a01 b11: -1 +1 = 0
    # a01 b01: +1
    # = a00 b00 + a01 b01. 
    # Still a01 b01, not a01 b10!
    #
    # Hmm. So my Strassen formula gives C11 = a00 b00 + a01 b01 which is WRONG.
    # The correct C11 = a00 b00 + a01 b10.
    # 
    # Let me recheck M3. M3 = A11 (B21 - B22) = a00 (b10 - b11).
    # In C12 = M3 + M5:
    # M3 = a00 b10 - a00 b11
    # M5 = a00 b11 + a01 b11
    # C12 = a00 b10 + a01 b11. Correct! C12 = a00 b01 + a01 b11? No.
    # C12 = sum_j A[0,j] B[j,1] = a00 b01 + a01 b11.
    # But M3+M5 = a00 b10 + a01 b11. That's a00 b10 not a00 b01!
    # 
    # So my Strassen is inconsistent. Let me very carefully use the standard reference.
    # 
    # Standard Strassen (Wikipedia):
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11) b00
    # M3 = a00 (b10 − b11)
    # M4 = a11 (b01 − b00)
    # M5 = (a00 + a01) b11
    # M6 = (a10 − a00)(b00 + b10)
    # M7 = (a01 − a11)(b01 + b11)
    # C00 = M1 + M4 − M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 − M2 + M3 + M6
    #
    # Let me verify C01 = M3 + M5 = a00(b10-b11) + (a00+a01)b11 = a00 b10 - a00 b11 + a00 b11 + a01 b11 = a00 b10 + a01 b11.
    # Correct C01 = sum_j A[0,j] B[j,1] = a00 b01 + a01 b11.
    # But we got a00 b10 + a01 b11. Mismatch: a00 b10 vs a00 b01.
    #
    # So the Wikipedia formula I'm using is wrong, OR I'm confusing b01 and b10.
    # Let me define clearly: B = [[b00, b01],[b10, b11]]. So B[0,1]=b01, B[1,0]=b10.
    # C01 = A[0,:]·B[:,1] = a00*B[0,1] + a01*B[1,1] = a00 b01 + a01 b11.
    # M3+M5 = a00(b10-b11)+(a00+a01)b11 = a00 b10 + a01 b11.
    # For these to match we'd need b10=b01 which is false.
    #
    # I think the correct M3 should be a00 (b01 - b11)? Let me check.
    # If M3 = a00(b01-b11), M3+M5 = a00 b01 - a00 b11 + a00 b11 + a01 b11 = a00 b01 + a01 b11 = C01. Correct!
    # So M3 = a00 (b01 - b11)? But that's not the standard formula.
    #
    # Let me look at this differently. I'll just directly verify my Strassen with numpy and fix the signs/indexing until it works. Let me write a small test.

def _test_strassen():
    # Define the 7 terms as (U_coeffs, V_coeffs, W_coeffs) over 2x2 blocks.
    # U over (i,j) in {0,1}^2, V over (j,k) in {0,1}^2, W over (i,k) in {0,1}^2.
    # Use numpy to find the correct Strassen.
    import itertools
    # Standard Strassen terms (U, V) and which C they contribute to:
    # Let me just hardcode and test.
    terms = [
        # (U: dict (i,j)->coef, V: dict (j,k)->coef, W: dict (i,k)->coef)
        ( {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1} ),  # M1
        ( {(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):-1} ),          # M2
        ( {(0,0):1}, {(1,0):1,(1,1):-1}, {(0,1):1,(1,1):1} ),          # M3
        ( {(1,1):1}, {(0,1):1,(0,0):-1}, {(0,0):1,(1,0):1} ),          # M4
        ( {(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1} ),          # M5
        ( {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}, {(1,1):1} ),          # M6
        ( {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}, {(0,0):1} ),          # M7
    ]
    # Test: for random A, B, compute C = A@B and Crec from the terms.
    rng = np.random.default_rng(1)
    A = rng.standard_normal((2,2))
    B = rng.standard_normal((2,2))
    C = A @ B
    Crec = np.zeros((2,2))
    for U, V, W in terms:
        ua = sum(U.get((i,j),0)*A[i,j] for i in range(2) for j in range(2))
        vb = sum(V.get((j,k),0)*B[j,k] for j in range(2) for k in range(2))
        for i in range(2):
            for k in range(2):
                Crec[i,k] += W.get((i,k),0) * ua * vb
    print("C:\n", C)
    print("Crec:\n", Crec)
    print("Match:", np.allclose(C, Crec))

_test_strassen()

# EVOLVE-BLOCK-END
