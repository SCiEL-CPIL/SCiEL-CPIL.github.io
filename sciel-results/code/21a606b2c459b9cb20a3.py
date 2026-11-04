import numpy as np
import time
import random
from itertools import combinations

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit (evaluator kills at 150s, but task says 120s, let's be safe)
    
    # Build the target tensor T
    # T is a 3D array of shape (n*m, m*p, p*n)
    # Index 1: i*m + j
    # Index 2: j*p + k
    # Index 3: k*n + i
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Helper to check if a decomposition is valid
    def check_decomposition(U, V, W, tol=1e-9):
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # Reconstruct T
        # T_recon[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        rank = U.shape[1]
        T_recon = np.zeros_like(T)
        # Using einsum for speed
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        # Check if T_recon matches T
        # The tensor is sparse. We only care about the 42 non-zero entries.
        # But np.linalg.norm(T - T_recon) < tol is sufficient and fast enough for rank ~40
        return np.linalg.norm(T - T_recon) < tol

    # Schoolbook decomposition
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return U, V, W

    U_sb, V_sb, W_sb = get_schoolbook()
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p
    
    # Strategy: 
    # 1. Try to find rank 41 decomposition by removing one schoolbook term and optimizing.
    # 2. If that fails, try rank 40, etc.
    # The optimization is a non-convex problem. We will use a simple iterative refinement.
    # Given the small size, we can afford some compute.
    
    # We will try to reduce rank from 42 down to, say, 30.
    # For each rank R, we try to find a decomposition.
    # Initialization: Take the first R columns of the schoolbook decomposition? 
    # That won't work because we lose the terms corresponding to the removed columns.
    # Better: Random initialization with small noise, or structured initialization.
    
    # Let's try a specific heuristic: 
    # The tensor is T[i,j,k] = delta_{i,j,k} (in flattened indices).
    # A rank 1 term is u v w^T.
    # We want sum_r u_r v_r w_r = T.
    
    # Since we need integer/half-integer entries, we can't just use continuous optimization and round.
    # We need to find exact integer/half-integer solutions.
    # This makes the problem discrete and harder.
    # However, for 2x3x7, it is highly likely that rank 42 is minimal or very close to it.
    # Strassen for 2x2x2 is 7 vs 8.
    # For 2x3x7, let's see if we can find a reduction.
    
    # Let's attempt a search for rank 41.
    # We can fix most of the structure and try to satisfy the remaining constraints.
    
    # Actually, a more robust approach for small dimensions is to use a SAT/SMT solver logic, 
    # but we don't have one. 
    # We can use a randomized local search.
    
    # Let's define a cost function: sum of squared errors.
    # We will use gradient-free optimization (e.g., Nelder-Mead or simple random perturbations) 
    # but constrained to half-integers.
    
    # Given the time limit, let's try a few random restarts with local search for rank 41.
    
    def local_search(U, V, W, max_iter=1000):
        # U, V, W are numpy arrays of shape (rows, rank)
        # We will perturb entries and accept if cost decreases.
        # Cost: ||T - T_recon||^2
        # We restrict entries to multiples of 0.5
        
        current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
        current_cost = np.linalg.norm(T - np.einsum('ir,jr,kr->ijk', current_U, current_V, current_W)) ** 2
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            # Pick a random matrix and a random entry
            mat_idx = random.randint(0, 2)
            if mat_idx == 0:
                mat = current_U
            elif mat_idx == 1:
                mat = current_V
            else:
                mat = current_W
            
            row = random.randint(0, mat.shape[0] - 1)
            col = random.randint(0, mat.shape[1] - 1)
            
            old_val = mat[row, col]
            
            # Try a small perturbation: +/- 0.5
            # Or pick a random new value from a small set
            # Let's try adding a random delta from {-1, -0.5, 0.5, 1}
            delta = random.choice([-1.0, -0.5, 0.5, 1.0])
            new_val = old_val + delta
            
            # Apply perturbation
            if mat_idx == 0:
                current_U[row, col] = new_val
            elif mat_idx == 1:
                current_V[row, col] = new_val
            else:
                current_W[row, col] = new_val
            
            new_cost = np.linalg.norm(T - np.einsum('ir,jr,kr->ijk', current_U, current_V, current_W)) ** 2
            
            if new_cost < current_cost:
                current_cost = new_cost
            else:
                # Revert
                if mat_idx == 0:
                    current_U[row, col] = old_val
                elif mat_idx == 1:
                    current_V[row, col] = old_val
                else:
                    current_W[row, col] = old_val
        
        return current_U, current_V, current_W, current_cost

    # Search for rank 41
    target_rank = best_rank - 1
    
    # We can try to initialize by taking schoolbook and zeroing out one column, then optimizing.
    # But that column's contribution is missing.
    # Better: Random initialization.
    
    found_better = False
    attempts = 0
    max_attempts = 50
    
    while attempts < max_attempts and time.time() < deadline - 5:
        attempts += 1
        
        # Random initialization
        # Entries should be small to avoid overflow, but we want to find exact solutions.
        # Let's initialize with small random half-integers.
        U = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(n*m, target_rank))
        V = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(m*p, target_rank))
        W = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(p*n, target_rank))
        
        U, V, W, cost = local_search(U, V, W, max_iter=200)
        
        if cost < 1e-9:
            best_U, best_V, best_W = U, V, W
            best_rank = target_rank
            found_better = True
            break
            
    # If we found rank 41, try to find 40?
    # Given the difficulty, let's stop if we found 41.
    # If not, return 42.
    
    # Convert to list of lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
