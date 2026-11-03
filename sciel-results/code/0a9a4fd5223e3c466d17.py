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
    
    # Try to find rank-16 decomposition by solving linear system
    # Fix U and V to a structured basis, solve for W
    rank16 = 16
    U16 = np.zeros((n * m, rank16))
    V16 = np.zeros((m * p, rank16))
    
    # Factors 0-6: Same Strassen as before
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
    # Handle j=2 (third column of A) and k=2 (third column of B)
    U16[2, 7] = 1; V16[6, 7] = 1   # A02*B20
    U16[2, 8] = 1; V16[7, 8] = 1   # A02*B21
    U16[2, 9] = 1; V16[8, 9] = 1   # A02*B22
    U16[5, 10] = 1; V16[6, 10] = 1  # A12*B20
    U16[5, 11] = 1; V16[7, 11] = 1  # A12*B21
    U16[5, 12] = 1; V16[8, 12] = 1  # A12*B22
    U16[0, 13] = 1; V16[2, 13] = 1  # A00*B02
    U16[1, 14] = 1; V16[5, 14] = 1  # A01*B12
    U16[3, 15] = 1; V16[2, 15] = 1  # A10*B02
    
    # Solve for W16 from the linear constraints
    # For each (i, j, j', k, k'), we need:
    # sum_r U16[i*m+j, r] * V16[j'*p+k', r] * W16[k'*n+i, r] = delta_{j,j'}*delta_{k,k'}
    
    W16 = np.zeros((p * n, rank16))
    
    # Build linear system for W entries
    # Unknowns: W16[k'*n+i, r] for all k', i, r
    # For each constraint (i, j, j', k, k'), we get an equation
    
    # Let's solve this by direct substitution since the structure is sparse
    # For each factor r, W16[k'*n+i, r] appears in constraints where U16[i*m+j, r] and V16[j'*p+k', r] are nonzero
    
    # Try to solve by iterating over factors and determining W values
    solved = True
    for r in range(rank16):
        # For this factor, find all (i, j) where U16[i*m+j, r] != 0
        # and all (j', k') where V16[j'*p+k', r] != 0
        u_indices = [(i, j) for i in range(n) for j in range(m) if U16[i*m+j, r] != 0]
        v_indices = [(jp, kp) for jp in range(m) for kp in range(p) if V16[jp*p+kp, r] != 0]
        
        for (i, j) in u_indices:
            for (jp, kp) in v_indices:
                # This factor contributes to constraint (i, j, jp, k_target, kp) 
                # where k_target must equal kp for the target to be 1 (if j==jp)
                # The contribution is U16[i*m+j, r] * V16[jp*p+kp, r] * W16[kp*n+i, r]
                # For this to match the target, we need to check all k values
                
                # For each k in range(p):
                for k in range(p):
                    target = 1.0 if (j == jp and k == kp) else 0.0
                    # The total contribution from all factors to this constraint must equal target
                    # This factor's contribution: U16[i*m+j, r] * V16[jp*p+kp, r] * W16[kp*n+i, r]
                    # We'll accumulate and solve at the end
    
    # Alternative: Set up the full linear system
    # Variables: W16[k'*n+i, r] for k' in 0..p-1, i in 0..n-1, r in 0..rank16-1
    # Total variables: p * n * rank16 = 6 * 16 = 96
    
    # Equations: for each (i, j, jp, k, kp) in 0..n-1, 0..m-1, 0..m-1, 0..p-1, 0..p-1
    # Total equations: n * m * m * p * p = 2 * 3 * 3 * 3 * 3 = 162
    
    # Build matrix A and vector b
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
                        
                        # For each factor r, the variable W16[kp*n+i, r] appears with coefficient
                        # U16[i*m+j, r] * V16[jp*p+kp, r]
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
        else:
            # Try a different structured basis
            # Use half-integers to allow more flexibility
            U16b = U16.copy()
            V16b = V16.copy()
            
            # Modify factor 15 to combine A10*B02 and A11*B12
            U16b[3, 15] = 1; U16b[4, 15] = 1
            V16b[2, 15] = 1; V16b[5, 15] = 1
            # Remove the separate A01*B12 factor by adjusting
            # This is getting complex, fall back to rank 17
            best_rank = rank17
            best_U, best_V, best_W = U17.copy(), V17.copy(), W17.copy()
    except:
        best_rank = rank17
        best_U, best_V, best_W = U17.copy(), V17.copy(), W17.copy()
    
    # Final verification
    if not _verify(best_U, best_V, best_W, n, m, p, best_rank):
        U_fb, V_fb, W_fb, rank_sb = _schoolbook(n, m, p)
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
