import time
import random
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin for 120s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p # 36
    
    # 1. Generate Schoolbook Solution (Rank 36)
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U_sb[u_idx][col] = 1
                V_sb[v_idx][col] = 1
                W_sb[w_idx][col] = 1
                col += 1
    best_solution = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank

    def verify(U, V, W, R, num_tests=5):
        """
        Verifies if the decomposition U, V, W correctly computes A * B.
        U: (n*m, R), V: (m*p, R), W: (p*n, R)
        """
        for _ in range(num_tests):
            # Random integer matrices
            A = np.random.randint(-2, 3, size=(n, m))
            B = np.random.randint(-2, 3, size=(m, p))
            
            # Direct multiplication
            C_direct = A @ B
            
            # Decomposed multiplication
            # LA[i, r] = sum_j U[i*m+j, r] * A[i,j]
            # LB[k, r] = sum_j V[j*p+k, r] * B[j,k]
            
            U_arr = np.array(U)
            V_arr = np.array(V)
            W_arr = np.array(W)
            
            # Reshape for easier einsum
            # U: (n, m, R)
            U_reshaped = U_arr.reshape(n, m, R)
            # V: (m, p, R)
            V_reshaped = V_arr.reshape(m, p, R)
            # W: (p, n, R)
            W_reshaped = W_arr.reshape(p, n, R)
            
            # LA[i, r] = sum_j U[i, j, r] * A[i, j]
            LA = np.einsum('ij,ijr->ir', A, U_reshaped)
            # LB[k, r] = sum_j V[j, k, r] * B[j, k]
            LB = np.einsum('jk,jkr->kr', B, V_reshaped)
            
            # C_decomp[i, k] = sum_r W[k, i, r] * LA[i, r] * LB[k, r]
            C_decomp = np.einsum('kir,ir,kr->ik', W_reshaped, LA, LB)
            
            if not np.array_equal(C_direct, C_decomp):
                return False
                
        return True

    # 2. Search for lower ranks
    # We iterate downwards from schoolbook_rank - 1
    # We limit the number of trials per rank to avoid getting stuck
    # We prioritize higher ranks first as they are more likely to exist
    
    # To make the search more effective, we can use a "local search" approach
    # but for now, let's try random sparse matrices with small integers.
    
    # We can also try to "perturb" the schoolbook solution.
    # For rank R, we can take the first R columns of the schoolbook solution
    # and try to adjust them? No, that's not how it works.
    
    # Let's use a randomized search with a deadline check.
    # We will try ranks 35 down to 20.
    
    # To increase the chance of finding a solution, we can use a 
    # "structured" random search.
    # For each rank R, we generate random matrices with entries in {-1, 0, 1}.
    # We can also try {-2, -1, 0, 1, 2}.
    
    # Optimization: Use a faster verification for the initial screening?
    # The current verification is quite fast for small n, m, p.
    
    # Let's add a "local search" refinement.
    # If a random candidate is "close" (small error), we try to fix it.
    # But defining "close" and "fixing" is complex.
    
    # Alternative: Use a known algorithm for small tensor decompositions?
    # For 2x3x6, the minimal rank is not widely known, but likely close to 36.
    
    # Let's stick to random search but with a higher density of trials for higher ranks.
    
    for rank in range(schoolbook_rank - 1, 15, -1):
        if time.time() > deadline - 10.0:
            break
            
        # Adjust number of trials based on remaining time
        remaining_time = deadline - time.time()
        # Estimate time per trial (approx 1ms for verification?)
        # Let's just cap it.
        max_trials = 500
        if remaining_time < 30.0:
            max_trials = 100
        if remaining_time < 10.0:
            max_trials = 20
            
        for trial in range(max_trials):
            if time.time() > deadline - 5.0:
                break
                
            # Generate random sparse matrices
            # Density ~ 0.3
            U = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(n * m)]
            V = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(m * p)]
            W = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(p * n)]
            
            if verify(U, V, W, rank, num_tests=3):
                # If it passes 3 tests, do a final verification with more tests
                if verify(U, V, W, rank, num_tests=10):
                    best_rank = rank
                    best_solution = {"U": U, "V": V, "W": W}
                    return best_solution
                    
    return best_solution
# EVOLVE-BLOCK-END
