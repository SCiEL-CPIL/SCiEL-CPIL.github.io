import numpy as np
import time

DIMENSIONS = [4, 4, 5]
TIME_LIMIT = 120

def _verify_decomposition(U, V, W, n=4, m=4, p=5, num_tests=10, tol=1e-10):
    """
    Verify that the tensor decomposition correctly computes C = A * B.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    """
    rank = U.shape[1]
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    for _ in range(num_tests):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    a_val = 0.0
                    for j in range(m):
                        a_val += U[i*m + j, r] * A[i, j]
                    b_val = 0.0
                    for j in range(m):
                        b_val += V[j*p + k, r] * B[j, k]
                    val += W[k*n + i, r] * a_val * b_val
                C_computed[i, k] = val
        
        if not np.allclose(C_computed, C_expected, atol=tol):
            return False
    
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Rank 44 decomposition:
    # 1. Block Strassen for A(4x4) * B(4x4) -> C(4x4). Rank 28.
    # 2. Schoolbook for A(4x4) * B(4x1) -> C(4x1). Rank 16.
    
    rank_bs = 28
    rank_sb = 16
    rank_total = rank_bs + rank_sb
    
    U = np.zeros((n*m, rank_total))
    V = np.zeros((m*p, rank_total))
    W = np.zeros((p*n, rank_total))
    
    # --- Part 1: Block Strassen for cols 0-3 of B ---
    # A blocks: A00, A01, A10, A11 (2x2 each)
    # B blocks: B00, B01, B10, B11 (2x2 each)
    # C blocks: C00, C01, C10, C11 (2x2 each)
    
    # Strassen Products M1..M7
    # M1 = A00 * (B00 - B11)
    # M2 = (A00 + A10) * B11
    # M3 = (A11 - A00) * B00
    # M4 = A11 * (B11 - B00)
    # M5 = (A00 + A11) * B00
    # M6 = (A01 - A11) * (B00 + B10)
    # M7 = (A10 - A00) * (B01 + B11)
    
    # Each M is 2x2, computed by 4 scalar mults (schoolbook).
    # We map each scalar mult to a rank factor.
    
    # Helper to add a schoolbook factor for 2x2 * 2x2
    def add_strassen_factor(r_start, A_coeffs, B_coeffs, C_coeffs):
        # A_coeffs: dict (i,j) -> coeff for A block
        # B_coeffs: dict (i,j) -> coeff for B block
        # C_coeffs: dict (i,j) -> coeff for C block
        # This adds 4 factors for the 4 entries of the 2x2 result?
        # No, a 2x2 * 2x2 multiplication is rank 4.
        # We need to add 4 rank-1 factors corresponding to the 4 scalar products.
        
        # The 4 scalar products are:
        # (sum A_ij a_ij) * (sum B_kl b_kl)
        # This is not quite right. The "product" M is a matrix.
        # M = A_lin * B_lin.
        # M_ij = sum_k (A_lin)_{ik} (B_lin)_{kj}.
        # We decompose M_ij into rank-1 terms?
        # No, we are constructing the global U, V, W.
        # Each global rank factor r corresponds to one scalar multiplication z_r.
        # z_r = (sum u_{ij} A_{ij}) * (sum v_{kl} B_{kl}).
        # Then C_{mn} += w_{mn} z_r.
        
        # For a 2x2 block multiplication M = A_lin B_lin, we can use 4 scalar mults.
        # z_1 = A_lin[0,0] * B_lin[0,0]
        # z_2 = A_lin[0,1] * B_lin[1,0]
        # z_3 = A_lin[1,0] * B_lin[0,1]
        # z_4 = A_lin[1,1] * B_lin[1,1]
        # M[0,0] = z_1 + z_2
        # M[0,1] = z_3 + z_4
        # M[1,0] = z_5 + z_6 ... wait.
        # Schoolbook for 2x2:
        # M[0,0] = A00 B00 + A01 B10
        # M[0,1] = A00 B01 + A01 B11
        # M[1,0] = A10 B00 + A11 B10
        # M[1,1] = A10 B01 + A11 B11
        
        # So we need 4 scalar products for each M?
        # No, M is defined by linear forms in A and B.
        # M = A_lin B_lin.
        # We can compute A_lin B_lin using 4 scalar multiplications if we treat A_lin and B_lin as constant matrices?
        # No, A_lin is a linear combination of A blocks.
        # Let A_lin = sum c_a A_a.
        # Let B_lin = sum c_b B_b.
        # M = (sum c_a A_a) (sum c_b B_b) = sum_{a,b} c_a c_b A_a B_b.
        # This expands to many terms.
        # The rank of the bilinear form M = A_lin B_lin is 4 (since it's a 2x2 matmul).
        # So we need 4 rank factors for each M.
        
        # For each of the 4 scalar mults in the schoolbook decomposition of M:
        # 1. z = A_lin[0,0] * B_lin[0,0] -> contributes to M[0,0]
        #    U: A_lin[0,0] coeffs. V: B_lin[0,0] coeffs. W: C[0,0] += 1.
        # 2. z = A_lin[0,1] * B_lin[1,0] -> contributes to M[0,0]
        #    U: A_lin[0,1] coeffs. V: B_lin[1,0] coeffs. W: C[0,0] += 1.
        # 3. z = A_lin[0,0] * B_lin[0,1] -> contributes to M[0,1]
        #    U: A_lin[0,0] coeffs. V: B_lin[0,1] coeffs. W: C[0,1] += 1.
        # 4. z = A_lin[0,1] * B_lin[1,1] -> contributes to M[0,1]
        #    U: A_lin[0,1] coeffs. V: B_lin[1,1] coeffs. W: C[0,1] += 1.
        # ... and so on for rows 1.
        
        # This seems correct.
        
        pass # Placeholder, will implement in code

    # I will implement the logic directly in the loop below.
    
    # Define the 7 Strassen linear forms
    # A forms:
    # A1 = A00
    # A2 = A00 + A10
    # A3 = A11 - A00
    # A4 = A11
    # A5 = A00 + A11
    # A6 = A01 - A11
    # A7 = A10 - A00
    
    # B forms:
    # B1 = B00 - B11
    # B2 = B11
    # B3 = B00
    # B4 = B11 - B00
    # B5 = B00
    # B6 = B00 + B10
    # B7 = B01 + B11
    
    # C forms (linear combinations of M1..M7):
    # C00 = M1 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We iterate through the 7 M's.
    # For each M, we have A_form and B_form.
    # We generate 4 scalar mults for M.
    # For each scalar mult, we update U, V, W.
    # Then we accumulate the contribution to C blocks.
    
    # This is complex to map directly.
    # Easier: Just build the U, V, W matrices for the 28 factors.
    
    # Factor indices 0..27
    # For each M_k (k=0..6):
    #   4 factors: 4*k, 4*k+1, 4*k+2, 4*k+3
    
    # M_k = A_k * B_k
    # Schoolbook for 2x2:
    # M[0,0] = A_k[0,0] B_k[0,0] + A_k[0,1] B_k[1,0]
    # M[0,1] = A_k[0,0] B_k[0,1] + A_k[0,1] B_k[1,1]
    # M[1,0] = A_k[1,0] B_k[0,0] + A_k[1,1] B_k[1,0]
    # M[1,1] = A_k[1,0] B_k[0,1] + A_k[1,1] B_k[1,1]
    
    # So for factor r = 4*k + 0:
    #   U: A_k[0,0] coeffs
    #   V: B_k[0,0] coeffs
    #   W: C[0,0] += 1
    # For factor r = 4*k + 1:
    #   U: A_k[0,1] coeffs
    #   V: B_k[1,0] coeffs
    #   W: C[0,0] += 1
    # For factor r = 4*k + 2:
    #   U: A_k[0,0] coeffs
    #   V: B_k[0,1] coeffs
    #   W: C[0,1] += 1
    # For factor r = 4*k + 3:
    #   U: A_k[0,1] coeffs
    #   V: B_k[1,1] coeffs
    #   W: C[0,1] += 1
    # ...
    
    # We need the coefficients of A_k and B_k in terms of global A and B entries.
    
    # A_k is a 2x2 matrix of linear forms in A.
    # B_k is a 2x2 matrix of linear forms in B.
    
    # Example: M1 = A00 * (B00 - B11)
    # A1 = A00. Coeffs: A[0,0]=1, A[0,1]=1, A[1,0]=1, A[1,1]=1? No, A00 is the top-left 2x2 block.
    # A00[i,j] = A[i,j] for i,j in 0,1.
    # So A1[0,0] = A[0,0]. A1[0,1] = A[0,1]. etc.
    
    # B1 = B00 - B11.
    # B1[0,0] = B[0,0] - B[2,2].
    # B1[0,1] = B[0,1] - B[2,3].
    # B1[1,0] = B[1,0] - B[3,2].
    # B1[1,1] = B[1,1] - B[3,3].
    
    # I will create a helper to get the global coefficients for a 2x2 linear form.
    
    def get_A_coeffs(form_id, i, j):
        # form_id 0..6
        # i,j in 0,1
        # Returns dict (global_i, global_j) -> coeff
        pass
        
    def get_B_coeffs(form_id, i, j):
        # form_id 0..6
        # i,j in 0,1
        # Returns dict (global_i, global_j) -> coeff
        pass
        
    # This is getting too long for the thought block. I will write the code directly.
    
    # Fallback to schoolbook if time runs out or verification fails.
    
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
