import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = 120.0 - 5.0  # safety margin
    n, m, p = DIMENSIONS  # n=2, m=3, p=4
    
    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, k*p+l, r*n+s] = 1 iff j==k and i==r
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for l in range(p):
                T[i * m + j, j * p + l, l * n + i] = 1
    
    def verify(U, V, W, tol=1e-9):
        """Check that sum_r U[:,r] outer V[:,r] outer W[:,r] == T exactly."""
        U = np.asarray(U, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        W = np.asarray(W, dtype=np.float64)
        r = U.shape[1]
        recon = np.zeros(T.shape, dtype=np.float64)
        for c in range(r):
            recon += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        if not np.allclose(recon, T, atol=tol, rtol=0):
            return False
        # Check integer/half-integer constraint
        for M in (U, V, W):
            for v in M.flat:
                if abs(v) > 1e-12:
                    if abs(v - round(v)) > 1e-9 and abs(2*v - round(2*v)) > 1e-9:
                        return False
        return True
    
    def make_base_solution():
        """Return the known rank-22 solution."""
        rank = 22
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        
        def add_term(u_indices, v_indices, w_indices, col):
            for idx, val in u_indices:
                U[idx, col] += val
            for idx, val in v_indices:
                V[idx, col] += val
            for idx, val in w_indices:
                W[idx, col] += val
        
        # Block 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
        add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
        add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
        add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
        add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
        add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
        add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
        add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)
        
        # Block 2: Strassen for A[0:2, 0:2] * B[0:2, 2:4]
        add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
        add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
        add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
        add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
        add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
        add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
        add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)
        
        # Outer: A[0:2, 2:3] * B[2:3, 0:4]
        add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
        add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
        add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
        add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
        add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
        add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
        add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
        add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)
        
        return U, V, W
    
    # Start with the base rank-22 solution
    U, V, W = make_base_solution()
    best_rank = 22
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to reduce rank by removing columns and checking if we can compensate
    # Strategy: try rank 21, 20, etc. by zeroing out one column and seeing if
    # the remaining decomposition still works, or if we can find a new decomposition
    
    # For each rank from 21 down to 18, try to find a valid decomposition
    for target_rank in range(21, 17, -1):
        if time.time() - start_time > deadline:
            break
        
        # Try multiple random initializations
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() - start_time > deadline:
                break
            
            # Random initialization with small integers/half-integers
            U_rand = np.zeros((n * m, target_rank), dtype=np.float64)
            V_rand = np.zeros((m * p, target_rank), dtype=np.float64)
            W_rand = np.zeros((p * n, target_rank), dtype=np.float64)
            
            # Initialize with random ±1 values sparsely
            for c in range(target_rank):
                # Pick random non-zero entries
                for _ in range(random.randint(1, 4)):
                    U_rand[random.randint(0, n*m-1), c] = random.choice([-1, 1])
                for _ in range(random.randint(1, 4)):
                    V_rand[random.randint(0, m*p-1), c] = random.choice([-1, 1])
                for _ in range(random.randint(1, 4)):
                    W_rand[random.randint(0, p*n-1), c] = random.choice([-1, 1])
            
            # Check if this random init works
            if verify(U_rand, V_rand, W_rand):
                best_rank = target_rank
                best_U, best_V, best_W = U_rand.copy(), V_rand.copy(), W_rand.copy()
                break
        
        if best_rank == target_rank:
            break
    
    # If we found a better rank, use it; otherwise use the base
    if best_rank < 22:
        U, V, W = best_U, best_V, best_W
        rank = best_rank
    else:
        U, V, W = make_base_solution()
        rank = 22
    
    # Convert to lists
    U_list = [[int(U[i, j]) if abs(U[i, j] - int(U[i, j])) < 1e-9 else float(U[i, j]) for j in range(rank)] for i in range(n * m)]
    V_list = [[int(V[i, j]) if abs(V[i, j] - int(V[i, j])) < 1e-9 else float(V[i, j]) for j in range(rank)] for i in range(m * p)]
    W_list = [[int(W[i, j]) if abs(W[i, j] - int(W[i, j])) < 1e-9 else float(W[i, j]) for j in range(rank)] for i in range(p * n)]
    
    # Final verification
    if not verify(U_list, V_list, W_list):
        # Fall back to base solution
        U, V, W = make_base_solution()
        rank = 22
        U_list = [[int(U[i, j]) for j in range(rank)] for i in range(n * m)]
        V_list = [[int(V[i, j]) for j in range(rank)] for i in range(m * p)]
        W_list = [[int(W[i, j]) for j in range(rank)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
