import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Build the matrix multiplication tensor for verification
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Rank 17 decomposition for 2x3x3 matrix multiplication (fallback)
    U17 = np.zeros((n * m, 17))
    V17 = np.zeros((m * p, 17))
    W17 = np.zeros((p * n, 17))
    
    # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
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
    
    # Part 2: Schoolbook for A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
    U17[2, 7] = 1
    V17[6, 7] = 1
    W17[0, 7] = 1
    
    U17[2, 8] = 1
    V17[7, 8] = 1
    W17[2, 8] = 1
    
    U17[5, 9] = 1
    V17[6, 9] = 1
    W17[1, 9] = 1
    
    U17[5, 10] = 1
    V17[7, 10] = 1
    W17[3, 10] = 1
    
    # Part 3: Schoolbook for A[0:2, 0:3] * B[0:3, 2] -> C[0:2, 2]
    U17[0, 11] = 1
    V17[2, 11] = 1
    W17[4, 11] = 1
    
    U17[1, 12] = 1
    V17[5, 12] = 1
    W17[4, 12] = 1
    
    U17[2, 13] = 1
    V17[8, 13] = 1
    W17[4, 13] = 1
    
    U17[3, 14] = 1
    V17[2, 14] = 1
    W17[5, 14] = 1
    
    U17[4, 15] = 1
    V17[5, 15] = 1
    W17[5, 15] = 1
    
    U17[5, 16] = 1
    V17[8, 16] = 1
    W17[5, 16] = 1
    
    # Verify the rank-17 decomposition
    T_recon17 = np.einsum('ar,br,cr->abc', U17, V17, W17)
    if np.max(np.abs(T_recon17 - T)) > 1e-10:
        return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}
    
    # Try to find a rank-16 decomposition using alternating linearization
    # Strategy: fix two matrices, solve for the third, iterate
    # The tensor has 18 non-zero entries; we need 16 rank-1 terms to span them
    
    rank_target = 16
    R = rank_target
    
    # Initialize with random sparse patterns inspired by the structure
    # Each term should have support on a subset of {a}x{b}x{c}
    
    # The 18 non-zero tensor entries are:
    # (a,b,c) where a=i*m+j, b=j*p+k, c=k*n+i
    # i in [0,2), j in [0,3), k in [0,3)
    nz_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nz_entries.append((a, b, c))
    
    # For rank-16, we need to find U (6x16), V (9x16), W (6x16)
    # such that sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    
    # Use alternating minimization with exact arithmetic
    # Initialize with a structured guess: take the rank-17 and try to merge terms
    
    # Approach: try all pairs of terms to merge, check if the remaining 15 + merged = 16 works
    # This is equivalent to checking if any two terms in the rank-17 are "redundant"
    
    # Better approach: direct search using the fact that the tensor is sparse
    # Set up: for each of the 18 entries, we have an equation
    # sum_{r=1}^{16} U[a_r, r] * V[b_r, r] * W[c_r, r] = 1
    
    # Fix W to be identity-like (6x16 with first 6 columns as identity)
    # Then V[b,r] * W[c,r] = V[b,r] if c=r, else 0
    # This doesn't work well. Let's try a different initialization.
    
    # Alternative: use the structure that each term in a good decomposition
    # typically has small support. Try to find 16 terms with support size 1-3.
    
    # Direct algebraic approach:
    # The tensor T has 18 non-zeros. A rank-r decomposition means T is in
    # the r-th secant variety. For r=16, we need to solve 18 equations
    # in 3*16*6 = 288 variables (but with scaling redundancy).
    
    # Use iterative linearization:
    # 1. Fix V and W, solve for U (linear in U)
    # 2. Fix U and W, solve for V (linear in V)  
    # 3. Fix U and V, solve for W (linear in W)
    # Repeat until convergence
    
    # Initialize with the rank-17 truncated to 16 terms + one merged term
    # Try removing each of the 17 terms and see if the remaining 16 can be
    # adjusted to still represent T
    
    best_result = None
    
    for elim in range(17):
        if time.time() > deadline - 10:
            break
            
        # Try to find a rank-16 decomposition by removing term 'elim'
        # and adjusting the remaining 16 terms
        
        # Initialize: copy the 16 remaining terms
        U_curr = np.zeros((n*m, 16))
        V_curr = np.zeros((m*p, 16))
        W_curr = np.zeros((p*n, 16))
        
        count = 0
        for r in range(17):
            if r != elim:
                U_curr[:, count] = U17[:, r]
                V_curr[:, count] = V17[:, r]
                W_curr[:, count] = W17[:, r]
                count += 1
        
        # Check if these 16 terms already represent T
        T_recon = np.einsum('ar,br,cr->abc', U_curr, V_curr, W_curr)
        if np.max(np.abs(T_recon - T)) < 1e-10:
            # Found it! Return this rank-16 decomposition
            return {"U": U_curr.tolist(), "V": V_curr.tolist(), "W": W_curr.tolist()}
        
        # Try alternating linearization to see if we can converge to a solution
        # from this initialization
        U_alt = U_curr.copy()
        V_alt = V_curr.copy()
        W_alt = W_curr.copy()
        
        converged = False
        for iteration in range(50):
            if time.time() > deadline - 5:
                break
                
            # Solve for U given V_alt and W_alt
            # For each (a,b,c) with T[a,b,c] != 0:
            # sum_r U[a,r] * V_alt[b,r] * W_alt[c,r] = T[a,b,c]
            # This is a linear system in U[a,:] for each a
            
            U_new = np.zeros((n*m, R))
            for a in range(n*m):
                # Build linear system for U[a, :]
                # For each (b,c) where there exists i,j,k with a=i*m+j, b=j*p+k, c=k*n+i
                # and T[a,b,c] = 1:
                # sum_r U[a,r] * V_alt[b,r] * W_alt[c,r] = 1
                
                # Collect all (b,c) pairs that are relevant for this a
                A_sys = np.zeros((0, R))
                b_sys = np.zeros(0)
                
                for i in range(n):
                    for j in range(m):
                        if i * m + j != a:
                            continue
                        for k in range(p):
                            b = j * p + k
                            c = k * n + i
                            if T[a, b, c] != 0:
                                row = V_alt[b, :] * W_alt[c, :]
                                A_sys = np.vstack([A_sys, row])
                                b_sys = np.append(b_sys, T[a, b, c])
                
                if len(b_sys) > 0:
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                        U_new[a, :] = sol
                    except:
                        U_new[a, :] = U_alt[a, :]
                else:
                    U_new[a, :] = U_alt[a, :]
            
            # Solve for V given U_new and W_alt
            V_new = np.zeros((m*p, R))
            for b in range(m*p):
                A_sys = np.zeros((0, R))
                b_sys = np.zeros(0)
                
                for j in range(m):
                    for k in range(p):
                        if j * p + k != b:
                            continue
                        for i in range(n):
                            a = i * m + j
                            c = k * n + i
                            if T[a, b, c] != 0:
                                row = U_new[a, :] * W_alt[c, :]
                                A_sys = np.vstack([A_sys, row])
                                b_sys = np.append(b_sys, T[a, b, c])
                
                if len(b_sys) > 0:
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                        V_new[b, :] = sol
                    except:
                        V_new[b, :] = V_alt[b, :]
                else:
                    V_new[b, :] = V_alt[b, :]
            
            # Solve for W given U_new and V_new
            W_new = np.zeros((p*n, R))
            for c in range(p*n):
                A_sys = np.zeros((0, R))
                b_sys = np.zeros(0)
                
                for k in range(p):
                    for i in range(n):
                        if k * n + i != c:
                            continue
                        for j in range(m):
                            a = i * m + j
                            b = j * p + k
                            if T[a, b, c] != 0:
                                row = U_new[a, :] * V_new[b, :]
                                A_sys = np.vstack([A_sys, row])
                                b_sys = np.append(b_sys, T[a, b, c])
                
                if len(b_sys) > 0:
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                        W_new[c, :] = sol
                    except:
                        W_new[c, :] = W_alt[c, :]
                else:
                    W_new[c, :] = W_alt[c, :]
            
            # Check convergence
            U_alt, V_alt, W_alt = U_new, V_new, W_new
            T_recon = np.einsum('ar,br,cr->abc', U_alt, V_alt, W_alt)
            err = np.max(np.abs(T_recon - T))
            
            if err < 1e-10:
                converged = True
                break
        
        if converged:
            # Found a rank-16 solution!
            # Try to rationalize
            U_rational = np.zeros((n*m, R))
            V_rational = np.zeros((m*p, R))
            W_rational = np.zeros((p*n, R))
            
            for a in range(n*m):
                for r in range(R):
                    U_rational[a, r] = float(Fraction(U_alt[a, r]).limit_denominator(1000))
            for b in range(m*p):
                for r in range(R):
                    V_rational[b, r] = float(Fraction(V_alt[b, r]).limit_denominator(1000))
            for c in range(p*n):
                for r in range(R):
                    W_rational[c, r] = float(Fraction(W_alt[c, r]).limit_denominator(1000))
            
            # Verify rationalized version
            T_recon_rat = np.einsum('ar,br,cr->abc', U_rational, V_rational, W_rational)
            if np.max(np.abs(T_recon_rat - T)) < 1e-8:
                return {"U": U_rational.tolist(), "V": V_rational.tolist(), "W": W_rational.tolist()}
            else:
                # Return the float version
                return {"U": U_alt.tolist(), "V": V_alt.tolist(), "W": W_alt.tolist()}
    
    # Return the rank-17 solution as fallback
    return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}
# EVOLVE-BLOCK-END
