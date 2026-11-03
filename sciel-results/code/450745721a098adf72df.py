"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 3
    
    def _verify(U_mat, V_mat, W_mat, n_, m_, p_, rank_):
        """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
        for i in range(n_):
            for k in range(p_):
                for j1 in range(m_):
                    for j2 in range(m_):
                        s = 0.0
                        for r in range(rank_):
                            s += W_mat[k*n_+i, r] * U_mat[i*m_+j1, r] * V_mat[j2*p_+k, r]
                        target = 1.0 if j1 == j2 else 0.0
                        if abs(s - target) > 1e-9:
                            return False
        return True
    
    def _schoolbook(n_, m_, p_):
        rank_sb = n_ * m_ * p_
        U_fb = np.zeros((n_ * m_, rank_sb))
        V_fb = np.zeros((m_ * p_, rank_sb))
        W_fb = np.zeros((p_ * n_, rank_sb))
        col = 0
        for i in range(n_):
            for j in range(m_):
                for k in range(p_):
                    U_fb[i*m_+j, col] = 1
                    V_fb[j*p_+k, col] = 1
                    W_fb[k*n_+i, col] = 1
                    col += 1
        return U_fb, V_fb, W_fb, rank_sb
    
    # Rank-17 decomposition using Strassen for 2x2x2 + schoolbook for remaining
    rank17 = 17
    U17 = np.zeros((n * m, rank17))
    V17 = np.zeros((m * p, rank17))
    W17 = np.zeros((p * n, rank17))
    
    # Part 1: Strassen for 2x2x2 (factors 0-6)
    U17[0, 0] = 1; U17[4, 0] = 1
    V17[0, 0] = 1; V17[4, 0] = 1
    W17[0, 0] = 1; W17[3, 0] = 1
    
    U17[3, 1] = 1; U17[4, 1] = 1
    V17[0, 1] = 1
    W17[1, 1] = 1; W17[3, 1] = -1
    
    U17[0, 2] = 1
    V17[1, 2] = 1; V17[4, 2] = -1
    W17[2, 2] = 1; W17[3, 2] = 1
    
    U17[4, 3] = 1
    V17[3, 3] = 1; V17[0, 3] = -1
    W17[0, 3] = 1; W17[1, 3] = 1
    
    U17[0, 4] = 1; U17[1, 4] = 1
    V17[4, 4] = 1
    W17[0, 4] = -1; W17[2, 4] = 1
    
    U17[3, 5] = 1; U17[0, 5] = -1
    V17[0, 5] = 1; V17[1, 5] = 1
    W17[3, 5] = 1
    
    U17[1, 6] = 1; U17[4, 6] = -1
    V17[3, 6] = 1; V17[4, 6] = 1
    W17[0, 6] = 1
    
    # Part 2: Schoolbook for middle column (factors 7-10)
    U17[2, 7] = 1; V17[6, 7] = 1; W17[0, 7] = 1
    U17[2, 8] = 1; V17[7, 8] = 1; W17[2, 8] = 1
    U17[5, 9] = 1; V17[6, 9] = 1; W17[1, 9] = 1
    U17[5, 10] = 1; V17[7, 10] = 1; W17[3, 10] = 1
    
    # Part 3: Schoolbook for last column (factors 11-16)
    U17[0, 11] = 1; V17[2, 11] = 1; W17[4, 11] = 1
    U17[1, 12] = 1; V17[5, 12] = 1; W17[4, 12] = 1
    U17[2, 13] = 1; V17[8, 13] = 1; W17[4, 13] = 1
    U17[3, 14] = 1; V17[2, 14] = 1; W17[5, 14] = 1
    U17[4, 15] = 1; V17[5, 15] = 1; W17[5, 15] = 1
    U17[5, 16] = 1; V17[8, 16] = 1; W17[5, 16] = 1
    
    # Verify rank-17 is correct
    if not _verify(U17, V17, W17, n, m, p, rank17):
        U_fb, V_fb, W_fb, rank_sb = _schoolbook(n, m, p)
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    best_rank = rank17
    best_U, best_V, best_W = U17.copy(), V17.copy(), W17.copy()
    
    # Strategy: Use the known rank-23 3x3x3 decomposition (Strassen-based) and restrict to 2-row subspace
    # The 3x3x3 Strassen decomposition has rank 23. When we restrict to 2 rows (i=0,1), we get
    # a 2x3x3 decomposition. We can try to eliminate redundant terms to get rank 16.
    
    # First, let's try a direct approach: fix U and V to a structured basis and solve for W
    # Try multiple structured bases
    
    # Base 1: Strassen for 2x2 part + schoolbook for remaining (current rank-17 structure)
    # Try to merge some schoolbook terms using linearity
    
    # Key insight: For the 2x3x3 tensor, we have:
    # - 7 factors for the 2x2x2 part (Strassen)
    # - 9 factors for the remaining terms (schoolbook)
    # Total: 16 factors if we can merge properly
    
    # The remaining terms involve:
    # - A02, A12 (third column of A) multiplied by B20, B21, B22 (third column of B)
    # - A00, A10 (first column of A) multiplied by B02 (third column of B) 
    # - A01, A11 (second column of A) multiplied by B12 (third column of B)
    
    # Let's try to combine some of these using Strassen-like tricks
    
    # Attempt 1: Combine (A02 + A12) * B2k for k=0,1,2
    # This gives 3 factors instead of 6, saving 3 factors
    # But we need to separate the contributions to different output positions
    
    # Let's try a different approach: use the 3x3x3 Strassen decomposition and project
    
    # The 3x3x3 Strassen decomposition (rank 23) can be written as:
    # M0 = (A00+A11+A22)(B00+B11+B22)
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A00+A11)(B00+B11)
    # M3 = A00*B00
    # M4 = A11*B11
    # M5 = A22*B22
    # M6 = (A00-A11)(B11-B00)
    # M7 = (A11-A22)(B22-B11)
    # M8 = (A22-A00)(B00-B22)
    # ... and 14 more terms for the off-diagonal blocks
    
    # This is getting complex. Let's try a more systematic approach.
    
    # Try rank-16 by fixing U and V to a specific structure and solving for W
    
    rank16 = 16
    U16 = np.zeros((n * m, rank16))
    V16 = np.zeros((m * p, rank16))
    
    # Factors 0-6: Strassen for 2x2x2
    U16[0, 0] = 1; U16[4, 0] = 1
    V16[0, 0] = 1; V16[4, 0] = 1
    
    U16[3, 1] = 1; U16[4, 1] = 1
    V16[0, 1] = 1
    
    U16[0, 2] = 1
    V16[1, 2] = 1; V16[4, 2] = -1
    
    U16[4, 3] = 1
    V16[3, 3] = 1; V16[0, 3] = -1
    
    U16[0, 4] = 1; U16[1, 4] = 1
    V16[4, 4] = 1
    
    U16[3, 5] = 1; U16[0, 5] = -1
    V16[0, 5] = 1; V16[1, 5] = 1
    
    U16[1, 6] = 1; U16[4, 6] = -1
    V16[3, 6] = 1; V16[4, 6] = 1
    
    # Factors 7-15: 9 factors for remaining terms
    # Try to combine some terms using linearity
    
    # Factor 7: (A02 + A12) * B20
    U16[2, 7] = 1; U16[5, 7] = 1
    V16[6, 7] = 1
    
    # Factor 8: (A02 + A12) * B21
    U16[2, 8] = 1; U16[5, 8] = 1
    V16[7, 8] = 1
    
    # Factor 9: (A02 + A12) * B22
    U16[2, 9] = 1; U16[5, 9] = 1
    V16[8, 9] = 1
    
    # Factor 10: (A00 + A10) * B02
    U16[0, 10] = 1; U16[3, 10] = 1
    V16[2, 10] = 1
    
    # Factor 11: A01 * B12
    U16[1, 11] = 1
    V16[5, 11] = 1
    
    # Factor 12: A11 * B12
    U16[4, 12] = 1
    V16[5, 12] = 1
    
    # Factor 13: (A00 - A10) * B02
    U16[0, 13] = 1; U16[3, 13] = -1
    V16[2, 13] = 1
    
    # Factor 14: (A01 - A11) * B12
    U16[1, 14] = 1; U16[4, 14] = -1
    V16[5, 14] = 1
    
    # Factor 15: A22 * B22 (already covered by factor 9, so use for something else)
    # Actually, factor 9 covers A02*B22 and A12*B22, so we don't need this
    # Use factor 15 for a correction term if needed
    U16[2, 15] = 0; U16[5, 15] = 0
    V16[8, 15] = 0
    
    # Solve for W16 from the linear constraints
    num_vars = p * n * rank16
    num_eqs = n * m * m * p * p
    
    A = np.zeros((num_eqs, num_vars))
    b = np.zeros(num_eqs)
    
    eq_idx = 0
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        target = 1.0 if (j == jp and k == kp) else 0.0
                        b[eq_idx] = target
                        
                        for r in range(rank16):
                            coeff = U16[i*m+j, r] * V16[jp*p+kp, r]
                            var_idx = kp * n + i
                            var_idx_full = var_idx * rank16 + r
                            A[eq_idx, var_idx_full] = coeff
                        
                        eq_idx += 1
    
    # Solve the linear system
    try:
        W_flat, residuals, rank_sv, sv = np.linalg.lstsq(A, b, rcond=None)
        W16 = W_flat.reshape((p * n, rank16))
        
        # Check if solution is valid
        if _verify(U16, V16, W16, n, m, p, rank16):
            best_rank = rank16
            best_U, best_V, best_W = U16.copy(), V16.copy(), W16.copy()
    except:
        pass
    
    # If rank-16 didn't work, try another structured basis
    if best_rank == rank17:
        U16b = np.zeros((n * m, rank16))
        V16b = np.zeros((m * p, rank16))
        
        # Factors 0-6: Same Strassen
        U16b[0, 0] = 1; U16b[4, 0] = 1
        V16b[0, 0] = 1; V16b[4, 0] = 1
        
        U16b[3, 1] = 1; U16b[4, 1] = 1
        V16b[0, 1] = 1
        
        U16b[0, 2] = 1
        V16b[1, 2] = 1; V16b[4, 2] = -1
        
        U16b[4, 3] = 1
        V16b[3, 3] = 1; V16b[0, 3] = -1
        
        U16b[0, 4] = 1; U16b[1, 4] = 1
        V16b[4, 4] = 1
        
        U16b[3, 5] = 1; U16b[0, 5] = -1
        V16b[0, 5] = 1; V16b[1, 5] = 1
        
        U16b[1, 6] = 1; U16b[4, 6] = -1
        V16b[3, 6] = 1; V16b[4, 6] = 1
        
        # Factors 7-15: Different structure
        # Factor 7: A02 * (B20 + B21 + B22)
        U16b[2, 7] = 1
        V16b[6, 7] = 1; V16b[7, 7] = 1; V16b[8, 7] = 1
        
        # Factor 8: A12 * (B20 + B21 + B22)
        U16b[5, 8] = 1
        V16b[6, 8] = 1; V16b[7, 8] = 1; V16b[8, 8] = 1
        
        # Factor 9: (A02 - A12) * (B20 + B21 + B22)
        U16b[2, 9] = 1; U16b[5, 9] = -1
        V16b[6, 9] = 1; V16b[7, 9] = 1; V16b[8, 9] = 1
        
        # Factor 10: (A00 + A10) * B02
        U16b[0, 10] = 1; U16b[3, 10] = 1
        V16b[2, 10] = 1
        
        # Factor 11: (A01 + A11) * B12
        U16b[1, 11] = 1; U16b[4, 11] = 1
        V16b[5, 11] = 1
        
        # Factor 12: (A00 - A10) * B02
        U16b[0, 12] = 1; U16b[3, 12] = -1
        V16b[2, 12] = 1
        
        # Factor 13: (A01 - A11) * B12
        U16b[1, 13] = 1; U16b[4, 13] = -1
        V16b[5, 13] = 1
        
        # Factor 14: A02 * B22
        U16b[2, 14] = 1
        V16b[8, 14] = 1
        
        # Factor 15: A12 * B22
        U16b[5, 15] = 1
        V16b[8, 15] = 1
        
        # Solve for W16b
        A = np.zeros((num_eqs, num_vars))
        b = np.zeros(num_eqs)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            target = 1.0 if (j == jp and k == kp) else 0.0
                            b[eq_idx] = target
                            
                            for r in range(rank16):
                                coeff = U16b[i*m+j, r] * V16b[jp*p+kp, r]
                                var_idx = kp * n + i
                                var_idx_full = var_idx * rank16 + r
                                A[eq_idx, var_idx_full] = coeff
                            
                            eq_idx += 1
        
        try:
            W_flat, residuals, rank_sv, sv = np.linalg.lstsq(A, b, rcond=None)
            W16b = W_flat.reshape((p * n, rank16))
            
            if _verify(U16b, V16b, W16b, n, m, p, rank16):
                best_rank = rank16
                best_U, best_V, best_W = U16b.copy(), V16b.copy(), W16b.copy()
        except:
            pass
    
    # Final verification
    if not _verify(best_U, best_V, best_W, n, m, p, best_rank):
        U_fb, V_fb, W_fb, rank_sb = _schoolbook(n, m, p)
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
