"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill

    # Strassen coefficients for 2x2x2
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]), # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]), # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]), # M7
    ]

    # Build initial rank-56 decomposition
    U = np.zeros((n * m, 56), dtype=float)
    V = np.zeros((m * p, 56), dtype=float)
    W = np.zeros((p * n, 56), dtype=float)

    current_rank = 0
    for group in range(2):  # A1*Btop, A2*Bbottom
        for q in range(4):
            for s in range(7):
                u_loc, v_loc, w_loc = strassen_terms[s]
                r = current_rank
                for i in range(2):
                    for j_loc in range(2):
                        U[i * m + (j_loc + group * 2)][r] = u_loc[i * 2 + j_loc]
                for j_loc in range(2):
                    for k_loc in range(2):
                        V[(j_loc + group * 2) * p + (2 * q + k_loc)][r] = v_loc[j_loc * 2 + k_loc]
                for i in range(2):
                    for k_loc in range(2):
                        W[(2 * q + k_loc) * n + i][r] = w_loc[k_loc * 2 + i]
                current_rank += 1

    def verify_decomposition(U, V, W, n, m, p, rank):
        """Vectorized verification of decomposition."""
        expected = np.zeros((n, m, p))
        for j in range(m):
            expected[:, j, j] = 1.0
        
        U2 = U[:n*m, :rank].reshape(n, m, rank)
        V2 = V[:m*p, :rank].reshape(m, p, rank)
        W2 = W[:p*n, :rank].reshape(p, n, rank)
        
        actual = np.einsum('ijr,jkr,kir->ijk', U2, V2, W2)
        
        return np.allclose(actual, expected, atol=1e-10)

    def reduce_rank_iteratively(U, V, W, n, m, p, rank, deadline):
        """Iteratively try to reduce rank by finding removable terms."""
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for r_remove in range(rank - 1, -1, -1):
                if time.time() > deadline:
                    return U, V, W, rank
                
                # Try zeroing out term r_remove
                trial_U = U.copy()
                trial_V = V.copy()
                trial_W = W.copy()
                trial_U[:, r_remove] = 0
                trial_V[:, r_remove] = 0
                trial_W[:, r_remove] = 0
                
                if verify_decomposition(trial_U, trial_V, trial_W, n, m, p, rank):
                    # Remove the term by shifting columns
                    U = np.column_stack([U[:, :r_remove], U[:, r_remove+1:]])
                    V = np.column_stack([V[:, :r_remove], V[:, r_remove+1:]])
                    W = np.column_stack([W[:, :r_remove], W[:, r_remove+1:]])
                    rank -= 1
                    changed = True
                    break
        
        return U, V, W, rank

    # Try to reduce rank
    U, V, W, rank = reduce_rank_iteratively(U, V, W, n, m, p, 56, deadline)

    # If we still have rank > 49, try a different approach:
    # Use the structure of the problem to find a better decomposition
    if rank > 49 and time.time() < deadline:
        # Try using a 2x4x8 specific optimization
        # The key insight: we can share Strassen terms across the 4 column blocks
        # by exploiting the structure of the 4x8 matrix
        
        # Alternative: use a rank-49 construction based on known results
        # For 2x4x8, the tensor rank is at most 49 (by known bounds)
        
        # Let's try a direct construction
        # Split the 4x8 matrix into four 4x2 blocks
        # Use 7 Strassen terms per 2x2x2 subproblem
        # But share some terms across blocks
        
        # Actually, let's try to find a valid rank-49 decomposition
        # by modifying the rank-56 one
        
        # The 56 terms are: 2 groups x 4 blocks x 7 Strassen terms
        # We need to eliminate 7 terms
        
        # Key observation: terms with the same (group, strassen_idx) across different q
        # may have structure that allows merging
        
        # For a given group and Strassen term s, we have 4 terms (one per q)
        # with V columns at positions (j_loc+group*2)*p + (2*q + k_loc)
        # These are linearly independent in general
        
        # Let's try a different strategy: use numerical optimization to find
        # a rank-49 decomposition by minimizing the tensor norm
        
        if time.time() < deadline:
            # Use scipy to find a low-rank approximation
            from scipy.optimize import minimize
            
            # Flatten the tensor
            # T[i,j,k] = delta(j,k) for i in 0..1, j in 0..3, k in 0..7
            # We want: T = sum_{r=1}^{R} u_r ⊗ v_r ⊗ w_r
            # where u_r in R^2, v_r in R^4, w_r in R^8
            
            # This is a non-convex optimization problem
            # Let's try to find a rank-49 solution
            
            R = 49
            # Parameterize: U (8, R), V (32, R), W (16, R)
            # Total parameters: 8*49 + 32*49 + 16*49 = 49*56 = 2744
            
            # Use random initialization from the rank-56 solution
            # Take first 49 columns
            U_init = U[:, :49].copy()
            V_init = V[:, :49].copy()
            W_init = W[:, :49].copy()
            
            # This won't work directly since we removed 7 arbitrary columns
            # Let's instead try to find a valid rank-49 decomposition
            # by solving the system of equations
            
            # The tensor has 2*4*8 = 64 entries
            # Each rank-1 term has 2+4+8 = 14 parameters (minus scale invariance)
            # So 49 terms have 49*13 = 637 free parameters
            # We have 64 equations, so the system is overdetermined
            # But the tensor is very sparse (only 8 non-zero entries)
            
            # Actually, let's just verify if rank 49 is achievable
            # and return the best we can do
            
            pass

    # Convert to list format
    U_list = [[int(U[i, r]) for r in range(rank)] for i in range(n * m)]
    V_list = [[int(V[i, r]) for r in range(rank)] for i in range(m * p)]
    W_list = [[int(W[i, r]) for r in range(rank)] for i in range(p * n)]

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
