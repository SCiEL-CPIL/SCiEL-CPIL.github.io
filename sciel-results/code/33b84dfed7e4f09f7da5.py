"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]
import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Schoolbook decomposition: Rank = n * m * p = 100
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Precompute the target tensor T as a sparse representation
    # T is defined on indices (a, b, c) where:
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    # T[a,b,c] = 1 for these, 0 otherwise.
    
    ones = []
    ones_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                ones.append((a, b, c))
                ones_set.add((a, b, c))
    
    # Precompute all zero indices for faster checking
    all_indices = []
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                all_indices.append((a, b, c))
    
    zeros = [idx for idx in all_indices if idx not in ones_set]
    
    def check_exact(U, V, W):
        # U, V, W are numpy arrays of shape (nm, R), (mp, R), (pn, R)
        # Check all ones
        for a, b, c in ones:
            temp = U[a] * V[b]
            val = np.dot(temp, W[c])
            if abs(val - 1.0) > 1e-6:
                return False
        
        # Check all zeros
        for a, b, c in zeros:
            temp = U[a] * V[b]
            val = np.dot(temp, W[c])
            if abs(val) > 1e-6:
                return False
        return True
    
    # Strategy: Utilize the known rank-19 decomposition for 3x3 matrix multiplication
    # by partitioning the 4x5 and 5x5 matrices into blocks, then combining with
    # rank-8 (2x2 Strassen) and rank-4 (1x2/2x1) decompositions.
    
    # However, for simplicity and reliability, let's first try to find a better
    # decomposition using a more systematic approach.
    
    # Let's try to use the fact that matrix multiplication can be decomposed
    # using Strassen-like methods. For 4x5 * 5x5, we can try to use a
    # combination of smaller matrix multiplications.
    
    # Actually, let's try a different approach: use the known rank-19 for 3x3
    # and see if we can combine it with smaller blocks.
    
    # For now, let's try to improve upon the schoolbook by using a more
    # efficient search strategy.
    
    best_rank = rank_school
    best_solution = {"U": U_school, "V": V_school, "W": W_school}
    
    # Try to find a decomposition with lower rank using local search
    # Start with rank 99 and go down
    
    for R in range(rank_school - 1, max(rank_school - 20, 89), -1):
        if time.time() > deadline - 5:
            break
            
        # Try a few random initializations
        found = False
        for seed in range(5):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(seed)
            # Initialize with small integers
            U = np.random.randint(-2, 3, size=(n * m, R))
            V = np.random.randint(-2, 3, size=(m * p, R))
            W = np.random.randint(-2, 3, size=(p * n, R))
            
            # Compute error on a subset for speed
            def compute_error(U, V, W, subset=None):
                err = 0.0
                if subset is None:
                    # Use all ones and a random sample of zeros
                    sample_zeros = random.sample(zeros, min(100, len(zeros)))
                    indices = ones + sample_zeros
                else:
                    indices = subset
                    
                for a, b, c in indices:
                    temp = U[a] * V[b]
                    val = np.dot(temp, W[c])
                    target = 1.0 if (a, b, c) in ones_set else 0.0
                    err += (val - target) ** 2
                return err
            
            current_err = compute_error(U, V, W)
            
            # Local search steps
            for step in range(50):
                if time.time() > deadline - 5:
                    break
                # Pick a random matrix and element
                mat_choice = np.random.randint(3)
                if mat_choice == 0:
                    mat = U
                    rows = n * m
                elif mat_choice == 1:
                    mat = V
                    rows = m * p
                else:
                    mat = W
                    rows = p * n
                    
                r_idx = np.random.randint(rows)
                c_idx = np.random.randint(R)
                
                # Try increment/decrement
                old_val = mat[r_idx, c_idx]
                improved = False
                for delta in [-1, 1]:
                    mat[r_idx, c_idx] = old_val + delta
                    new_err = compute_error(U, V, W)
                    if new_err < current_err:
                        current_err = new_err
                        improved = True
                        break
                    else:
                        mat[r_idx, c_idx] = old_val
                
                if not improved:
                    continue
            
            # Check if we found an exact solution
            if current_err < 1e-6:
                if check_exact(U, V, W):
                    # Convert to lists of ints/half-ints
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    
    # If no better solution found, return the schoolbook decomposition
    return best_solution
# EVOLVE-BLOCK-END
