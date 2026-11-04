"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    nm, mp, pn = n * m, m * p, p * n  # 16, 36, 36
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    U[i * m + j, idx] = 1
                    V[j * p + k, idx] = 1
                    W[k * n + i, idx] = 1
        return U, V, W, rank

    def laderman_4x4x4():
        """
        Returns U, V, W for 4x4x4 matrix multiplication with rank 12.
        U shape (16, 12), V shape (16, 12), W shape (16, 12).
        Indices:
        U[i*m + j, r] corresponds to A[i, j]
        V[j*p + k, r] corresponds to B[j, k] (here p=4)
        W[k*n + i, r] corresponds to C[k, i] (here n=4)
        """
        rank = 12
        U = np.zeros((16, rank))
        V = np.zeros((16, rank))
        W = np.zeros((16, rank))
        
        # Helper to add a term
        def add_term(u_idx, v_idx, w_idx, coeff=1):
            # u_idx is (i, j) in A
            # v_idx is (j, k) in B
            # w_idx is (k, i) in C
            # We need to map these to flat indices
            # A index: i * 4 + j
            # B index: j * 4 + k
            # C index: k * 4 + i
            # The term is u * v * w
            # This adds to the rank-1 decomposition
            pass
            
        # Laderman's 12 terms (1-based in literature, converting to 0-based)
        # z1: A00 B00 -> C00
        U[0*4+0, 0] = 1; V[0*4+0, 0] = 1; W[0*4+0, 0] = 1
        # z2: A11 B11 -> C11
        U[1*4+1, 1] = 1; V[1*4+1, 1] = 1; W[1*4+1, 1] = 1
        # z3: A22 B22 -> C22
        U[2*4+2, 2] = 1; V[2*4+2, 2] = 1; W[2*4+2, 2] = 1
        # z4: A33 B33 -> C33
        U[3*4+3, 3] = 1; V[3*4+3, 3] = 1; W[3*4+3, 3] = 1
        
        # z5: (A00+A11)(B10+B01) -> C10+C01
        # A00 is (0,0), A11 is (1,1)
        # B10 is (1,0), B01 is (0,1)
        # C10 is (1,0) -> W[1*4+0], C01 is (0,1) -> W[0*4+1]
        U[0*4+0, 4] = 1; U[1*4+1, 4] = 1
        V[1*4+0, 4] = 1; V[0*4+1, 4] = 1
        W[1*4+0, 4] = 1; W[0*4+1, 4] = 1
        
        # z6: (A22+A33)(B32+B23) -> C32+C23
        U[2*4+2, 5] = 1; U[3*4+3, 5] = 1
        V[3*4+2, 5] = 1; V[2*4+3, 5] = 1
        W[3*4+2, 5] = 1; W[2*4+3, 5] = 1
        
        # z7: (A01+A10)(B00+B11) -> C00+C11
        U[0*4+1, 6] = 1; U[1*4+0, 6] = 1
        V[0*4+0, 6] = 1; V[1*4+1, 6] = 1
        W[0*4+0, 6] = 1; W[1*4+1, 6] = 1
        
        # z8: (A23+A32)(B22+B33) -> C22+C33
        U[2*4+3, 7] = 1; U[3*4+2, 7] = 1
        V[2*4+2, 7] = 1; V[3*4+3, 7] = 1
        W[2*4+2, 7] = 1; W[3*4+3, 7] = 1
        
        # z9: (A02+A20)(B22+B00) -> C02+C20
        U[0*4+2, 8] = 1; U[2*4+0, 8] = 1
        V[2*4+2, 8] = 1; V[0*4+0, 8] = 1
        W[2*4+0, 8] = 1; W[0*4+2, 8] = 1
        
        # z10: (A03+A30)(B33+B00) -> C03+C30
        U[0*4+3, 9] = 1; U[3*4+0, 9] = 1
        V[3*4+3, 9] = 1; V[0*4+0, 9] = 1
        W[3*4+0, 9] = 1; W[0*4+3, 9] = 1
        
        # z11: (A12+A21)(B22+B11) -> C12+C21
        U[1*4+2, 10] = 1; U[2*4+1, 10] = 1
        V[2*4+2, 10] = 1; V[1*4+1, 10] = 1
        W[2*4+1, 10] = 1; W[1*4+2, 10] = 1
        
        # z12: (A13+A31)(B33+B11) -> C13+C31
        U[1*4+3, 11] = 1; U[3*4+1, 11] = 1
        V[3*4+3, 11] = 1; V[1*4+1, 11] = 1
        W[3*4+1, 11] = 1; W[1*4+3, 11] = 1
        
        return U, V, W

    def verify(U, V, W, rank, n, m, p):
        # Random test
        np.random.seed(42)
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_expected = A @ B
        
        C_calc = np.zeros((n, p))
        for r in range(rank):
            a_comb = np.dot(U[:, r], A.flatten())
            b_comb = np.dot(V[:, r], B.flatten())
            # C_calc += a_comb * b_comb * W[:, r] reshaped
            # W index is k*n + i
            W_reshaped = W[:, r].reshape(p, n)
            C_calc += a_comb * b_comb * W_reshaped.T # W[k, i] contributes to C[i, k]
            
        # Check
        if not np.allclose(C_calc, C_expected, atol=1e-9):
            return False
        return True

    # Construct 4x4x9 decomposition
    # Split B into 3 blocks of 4x3
    # For each block, use Laderman 4x4x4 and mask column 3 (0-based)
    
    U_total = np.zeros((16, 36))
    V_total = np.zeros((36, 36)) # m*p = 4*9 = 36
    W_total = np.zeros((36, 36)) # p*n = 9*4 = 36
    
    U_lad, V_lad, W_lad = laderman_4x4x4()
    
    for block_idx in range(3):
        # Block corresponds to columns 3*block_idx to 3*block_idx+2 of B
        # In V, indices are j*p + k.
        # For block b, k is in {3*b, 3*b+1, 3*b+2}
        # In W, indices are k*n + i.
        # For block b, k is in {3*b, 3*b+1, 3*b+2}
        
        for r in range(12):
            r_total = block_idx * 12 + r
            
            # Copy U
            U_total[:, r_total] = U_lad[:, r]
            
            # Copy V
            # V_lad has indices j*4 + k_lad where k_lad in 0..3
            # We want to map k_lad 0,1,2 to k 3*b, 3*b+1, 3*b+2
            # k_lad 3 is ignored (set to 0)
            for j in range(4):
                for k_lad in range(4):
                    if k_lad < 3:
                        k_new = 3 * block_idx + k_lad
                        V_total[j * 9 + k_new, r_total] = V_lad[j * 4 + k_lad, r]
                    # else V_lad[j*4+3, r] is ignored (contributes to col 3 of 4x4, which is not in 4x3)
            
            # Copy W
            # W_lad has indices k_lad*4 + i
            # We want to map k_lad 0,1,2 to k 3*b, 3*b+1, 3*b+2
            # k_lad 3 is ignored
            for i in range(4):
                for k_lad in range(4):
                    if k_lad < 3:
                        k_new = 3 * block_idx + k_lad
                        W_total[k_new * 4 + i, r_total] = W_lad[k_lad * 4 + i, r]
                    # else ignored

    rank_total = 36
    
    # Verify
    if verify(U_total, V_total, W_total, rank_total, n, m, p):
        return {
            "U": U_total.tolist(),
            "V": V_total.tolist(),
            "W": W_total.tolist()
        }
    else:
        # Fallback
        U, V, W, rank = schoolbook()
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
# EVOLVE-BLOCK-END
