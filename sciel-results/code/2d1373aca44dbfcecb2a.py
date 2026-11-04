"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 5, 6
    schoolbook_rank = n * m * p  # 120
    
    def verify_decomp(U, V, W):
        """Check if sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k"""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Start with schoolbook decomposition
    U_school = np.zeros((n * m, schoolbook_rank))
    V_school = np.zeros((m * p, schoolbook_rank))
    W_school = np.zeros((p * n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    best_rank = schoolbook_rank
    
    # Strategy: Try to combine dimensions using known lower-rank algorithms
    # For 4x5x6, we can try:
    # 1. Treat as (4x4 + 4x1) * (4x4 + 4x1) style decompositions
    # 2. Use the fact that 4x4 multiplication can be done in rank 21 (Strassen-like)
    # 3. Try to exploit the structure
    
    # Approach: Build using block structure
    # 4x5 = [4x4, 4x1], 5x6 = [4x4, 4x2] on the right
    
    # Let's try a direct search for rank reduction
    # We'll use a greedy approach: try to eliminate terms
    
    def try_eliminate_terms(U, V, W, max_attempts=500):
        """Try to eliminate terms from the decomposition"""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        rank = U.shape[1]
        eliminated = 0
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Try each term
            found = False
            for r in range(rank):
                if time.time() > deadline - 10:
                    break
                
                # Save current state
                U_save, V_save, W_save = U.copy(), V.copy(), W.copy()
                
                # Zero out column r
                U[:, r] = 0
                V[:, r] = 0
                W[:, r] = 0
                
                # Check if still valid
                if verify_decomp(U, V, W):
                    # Remove this column
                    mask = np.ones(rank, dtype=bool)
                    mask[r] = False
                    U = U[:, mask]
                    V = V[:, mask]
                    W = W[:, mask]
                    rank -= 1
                    eliminated += 1
                    found = True
                    break
                else:
                    # Restore
                    U, V, W = U_save, V_save, W_save
            
            if not found:
                break
        
        return U, V, W, eliminated
    
    # First, try simple elimination on schoolbook
    U_elim, V_elim, W_elim, elim_count = try_eliminate_terms(U_school, V_school, W_school)
    
    if elim_count > 0:
        best_U, best_V, best_W = U_elim, V_elim, W_elim
        best_rank = U_elim.shape[1]
    
    # Now try more sophisticated approaches
    # Use the fact that we can sometimes combine terms
    
    # For 4x5x6, try to use a structured decomposition
    # Idea: 4x5 * 5x6 can be seen as blocks
    
    # Let's try a random perturbation + elimination approach
    np.random.seed(42)
    
    for trial in range(20):
        if time.time() > deadline - 15:
            break
        
        # Create a perturbed version of schoolbook
        U_trial = U_school.copy().astype(float)
        V_trial = V_school.copy().astype(float)
        W_trial = W_school.copy().astype(float)
        
        # Add small random perturbations to some entries
        num_perturb = 10
        for _ in range(num_perturb):
            idx = np.random.randint(0, schoolbook_rank)
            row_u = np.random.randint(0, n * m)
            row_v = np.random.randint(0, m * p)
            row_w = np.random.randint(0, p * n)
            
            U_trial[row_u, idx] += np.random.choice([-1, 1]) * 0.5
            V_trial[row_v, idx] += np.random.choice([-1, 1]) * 0.5
            W_trial[row_w, idx] += np.random.choice([-1, 1]) * 0.5
        
        # Try elimination
        U_t, V_t, W_t, e = try_eliminate_terms(U_trial, V_trial, W_trial, max_attempts=200)
        
        if e > elim_count:
            elim_count = e
            best_U, best_V, best_W = U_t, V_t, W_t
            best_rank = U_t.shape[1]
    
    # Final verification
    if not verify_decomp(best_U, best_V, best_W):
        # Fallback to schoolbook
        best_U, best_V, best_W = U_school, V_school, W_school
        best_rank = schoolbook_rank
    
    # Convert to lists
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
