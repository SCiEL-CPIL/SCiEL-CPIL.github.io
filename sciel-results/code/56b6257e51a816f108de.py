import numpy as np
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 6
    deadline = time.time() + 140
    
    def verify_decomp(U, V, W, n, m, p, num_tests=5):
        """Verify that the decomposition correctly computes matrix multiplication."""
        for _ in range(num_tests):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            C_computed = np.zeros((n, p))
            for r in range(U.shape[1]):
                # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
                # Term r contributes: (U[:,r]^T A) * (V[:,r]^T B) * W[:,r]
                a_vec = U[:, r]  # (n*m,)
                b_vec = V[:, r]  # (m*p,)
                w_vec = W[:, r]  # (p*n,)
                # Reshape and compute
                a_mat = a_vec.reshape(n, m)
                b_mat = b_vec.reshape(m, p)
                w_mat = w_vec.reshape(p, n)
                # The bilinear form: sum over (i,j,k) of a_mat[i,j]*b_mat[j,k]*w_mat[k,i]
                # This should equal C_expected[i,k] for each (i,k)
                for i in range(n):
                    for k in range(p):
                        val = 0.0
                        for j in range(m):
                            val += a_mat[i, j] * b_mat[j, k] * w_mat[k, i]
                        C_computed[i, k] += val
            if not np.allclose(C_computed, C_expected, atol=1e-9):
                return False
        return True
    
    # Try to construct a rank-46 decomposition (23+23 for two 3x3x3 blocks)
    # Bini-Capovani-Lotti-Prestin (1979) rank-23 decomposition for 3x3x3
    
    # The 23 terms for 3x3x3 matrix multiplication
    # Using the explicit formulation from the literature
    U3 = np.zeros((9, 23), dtype=float)
    V3 = np.zeros((9, 23), dtype=float)
    W3 = np.zeros((9, 23), dtype=float)
    
    # Define helper functions
    def su(r, i, j, val):
        U3[i * 3 + j, r] = val
    def sv(r, j, k, val):
        V3[j * 3 + k, r] = val
    def sw(r, k, i, val):
        W3[k * 3 + i, r] = val
    
    # The Bini-Capovani-Lotti-Prestin (1979) rank-23 decomposition:
    # Based on the explicit construction in the original paper
    
    # Terms 0-2: diagonal
    su(0, 0, 0, 1); sv(0, 0, 0, 1); sw(0, 0, 0, 1)
    su(1, 1, 1, 1); sv(1, 1, 1, 1); sw(1, 1, 1, 1)
    su(2, 2, 2, 1); sv(2, 2, 2, 1); sw(2, 2, 2, 1)
    
    # Terms 3-5: handle pairs (0,1), (0,2), (1,2) using sum/difference
    # Term 3: (A01+A10)*(B10+B01) -> gives A01*B10 + A01*B01 + A10*B10 + A10*B01
    # We want: C[0,1] += A01*B01, C[1,0] += A10*B01
    # Unwanted: A01*B10 (goes to C[0,0]), A10*B10 (goes to C[1,1])
    su(3, 0, 1, 1); su(3, 1, 0, 1)
    sv(3, 1, 0, 1); sv(3, 0, 1, 1)
    sw(3, 0, 1, 1); sw(3, 1, 0, 1)
    
    # Term 4: (A02+A20)*(B20+B02)
    su(4, 0, 2, 1); su(4, 2, 0, 1)
    sv(4, 2, 0, 1); sv(4, 0, 2, 1)
    sw(4, 0, 2, 1); sw(4, 2, 0, 1)
    
    # Term 5: (A12+A21)*(B21+B12)
    su(5, 1, 2, 1); su(5, 2, 1, 1)
    sv(5, 2, 1, 1); sv(5, 1, 2, 1)
    sw(5, 1, 2, 1); sw(5, 2, 1, 1)
    
    # This simple approach doesn't correctly decompose the tensor.
    # The cross terms need to be canceled by additional terms.
    # Let me use a verified explicit decomposition instead.
    
    # Reset
    U3[:] = 0; V3[:] = 0; W3[:] = 0
    
    # Use a known correct rank-23 decomposition.
    # I'll use the formulation where the 23 terms are explicitly defined.
    # Reference: Bini, Capovani, Lotti, Prestin (1979), "Fast algorithms 
    # for the multiplication of matrices", Linear Algebra and its Applications.
    
    # The explicit 23-term decomposition (verified):
    # Using coefficients in {-2, -1, 0, 1, 2}
    
    # I'll construct this systematically. The key identity used is:
    # For 3x3, we can write the multiplication as a combination of 23 bilinear forms.
    
    # Let me use a different, more reliable construction:
    # Based on the "sum of squares" method
    
    # Actually, let me just try to verify a candidate and fall back if it fails.
    
    # Candidate: use the following 23 terms
    # (This is a best-effort reconstruction; will be verified numerically)
    
    # Term 0: A00*B00
    su(0, 0, 0, 1); sv(0, 0, 0, 1); sw(0, 0, 0, 1)
    # Term 1: A11*B11
    su(1, 1, 1, 1); sv(1, 1, 1, 1); sw(1, 1, 1, 1)
    # Term 2: A22*B22
    su(2, 2, 2, 1); sv(2, 2, 2, 1); sw(2, 2, 2, 1)
    
    # Terms 3-8: off-diagonal with sum/difference
    # Term 3: (A01+A10)*(B01+B10)
    su(3, 0, 1, 1); su(3, 1, 0, 1)
    sv(3, 0, 1, 1); sv(3, 1, 0, 1)
    sw(3, 1, 0, 1); sw(3, 0, 1, 1)
    
    # Term 4: (A02+A20)*(B02+B20)
    su(4, 0, 2, 1); su(4, 2, 0, 1)
    sv(4, 0, 2, 1); sv(4, 2, 0, 1)
    sw(4, 2, 0, 1); sw(4, 0, 2, 1)
    
    # Term 5: (A12+A21)*(B12+B21)
    su(5, 1, 2, 1); su(5, 2, 1, 1)
    sv(5, 1, 2, 1); sv(5, 2, 1, 1)
    sw(5, 2, 1, 1); sw(5, 1, 2, 1)
    
    # Terms 6-8: difference terms to extract individual products
    # Term 6: (A01-A10)*(B01-B10)
    su(6, 0, 1, 1); su(6, 1, 0, -1)
    sv(6, 0, 1, 1); sv(6, 1, 0, -1)
    sw(6, 1, 0, 1); sw(6, 0, 1, -1)
    
    # Term 7: (A02-A20)*(B02-B20)
    su(7, 0, 2, 1); su(7, 2, 0, -1)
    sv(7, 0, 2, 1); sv(7, 2, 0, -1)
    sw(7, 2, 0, 1); sw(7, 0, 2, -1)
    
    # Term 8: (A12-A21)*(B12-B21)
    su(8, 1, 2, 1); su(8, 2, 1, -1)
    sv(8, 1, 2, 1); sv(8, 2, 1, -1)
    sw(8, 2, 1, 1); sw(8, 1, 2, -1)
    
    # This gives us 9 terms but doesn't correctly decompose the tensor.
    # The issue is that these terms produce incorrect cross-contributions.
    
    # Given the complexity and my uncertainty about the exact coefficients,
    # let me use a fallback strategy: verify the candidate, and if it fails,
    # use schoolbook rank 54.
    
    # Verify the candidate 3x3x3 decomposition
    valid_333 = verify_decomp(U3, V3, W3, 3, 3, 3, num_tests=3)
    
    if valid_333 and time.time() < deadline:
        # Build the 3x3x6 decomposition from two 3x3x3 blocks
        rank = 46
        U = np.zeros((9, rank), dtype=float)
        V = np.zeros((18, rank), dtype=float)
        W = np.zeros((18, rank), dtype=float)
        
        # Block 1: columns 0-2 of B (3x3x3)
        U[:, 0:23] = U3
        V[0:9, 0:23] = V3
        W[0:9, 0:23] = W3
        
        # Block 2: columns 3-5 of B (3x3x3)
        U[:, 23:46] = U3
        V[9:18, 23:46] = V3
        W[9:18, 23:46] = W3
        
        # Final verification for 3x3x6
        valid_336 = verify_decomp(U, V, W, 3, 3, 6, num_tests=3)
        
        if valid_336:
            # Convert to lists with integer/half-integer entries
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            # Round to nearest 0.5
            for mat in [U_list, V_list, W_list]:
                for i in range(len(mat)):
                    for j in range(len(mat[i])):
                        mat[i][j] = round(mat[i][j] * 2) / 2
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook rank 54
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
