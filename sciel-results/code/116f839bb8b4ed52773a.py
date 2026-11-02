"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]
import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Schoolbook decomposition: Rank = n * m * p
    # This is always valid and produces integer entries.
    # U[i*m + j, c] = 1 if c corresponds to (i,j,k)
    # V[j*p + k, c] = 1 if c corresponds to (i,j,k)
    # W[k*n + i, c] = 1 if c corresponds to (i,j,k)
    
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
    # We only need to check the 1s and a sample of 0s for speed, 
    # but final verification must check all.
    
    ones = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                ones.append((a, b, c))
                
    def check_exact(U, V, W):
        # U, V, W are numpy arrays of shape (nm, R), (mp, R), (pn, R)
        R = U.shape[1]
        # Check all ones
        for a, b, c in ones:
            val = np.dot(U[a], np.dot(V[b], W[c])) # Wait, W is (pn, R)
            # W[c, r] is the entry.
            # Sum_r U[a,r] * V[b,r] * W[c,r]
            # Let's compute this efficiently.
            # term_r = U[a,r] * V[b,r] * W[c,r]
            # val = sum(term_r)
            # We can do: temp = U[a] * V[b] (vector of R)
            # val = dot(temp, W[c])
            temp = U[a] * V[b]
            val = np.dot(temp, W[c])
            if abs(val - 1.0) > 1e-6:
                return False
        
        # Check a sample of zeros
        # Total entries = nm * mp * pn
        # nm=20, mp=25, pn=25. Total=12500.
        # We can check all zeros if it's fast enough.
        # 12500 checks * (R multiplications).
        # If R=100, 1.25M operations. Fast in numpy.
        
        # Iterate over all a, b, c
        # But we can skip the ones we already checked.
        # To be safe and fast, we can just check all.
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    if (a, b, c) in ones: # This is slow. Use a set.
                        continue
                    temp = U[a] * V[b]
                    val = np.dot(temp, W[c])
                    if abs(val) > 1e-6:
                        return False
        return True

    # Optimize check_exact using a set for ones
    ones_set = set(ones)
    def check_exact_fast(U, V, W):
        R = U.shape[1]
        # Check ones
        for a, b, c in ones:
            temp = U[a] * V[b]
            val = np.dot(temp, W[c])
            if abs(val - 1.0) > 1e-6:
                return False
        
        # Check zeros
        # We can vectorize this?
        # Construct the full tensor? No, too big.
        # Just loop. 12500 iterations.
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    if (a, b, c) in ones_set:
                        continue
                    temp = U[a] * V[b]
                    val = np.dot(temp, W[c])
                    if abs(val) > 1e-6:
                        return False
        return True

    # Search for lower rank
    best_rank = rank_school
    best_solution = {"U": U_school, "V": V_school, "W": W_school}
    
    # Try ranks from 99 down to 90
    for R in range(rank_school - 1, 89, -1):
        if time.time() > deadline - 5:
            break
            
        # Try a few random initializations
        found = False
        for seed in range(10):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(seed)
            # Initialize with small integers
            U = np.random.randint(-2, 3, size=(n * m, R))
            V = np.random.randint(-2, 3, size=(m * p, R))
            W = np.random.randint(-2, 3, size=(p * n, R))
            
            # Simple local search: perturb one element at a time
            # If error decreases, keep it.
            # Error = sum (T - T_approx)^2
            # We can compute error on a subset for speed.
            
            def compute_error(U, V, W, subset=None):
                err = 0.0
                if subset is None:
                    # Use all ones and a random sample of zeros
                    indices = ones + [(a, b, c) for a in range(n*m) for b in range(m*p) for c in range(p*n) if (a,b,c) not in ones_set][:100]
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
            for step in range(100):
                if time.time() > deadline - 5:
                    break
                # Pick a random matrix and element
                mat_choice = np.random.randint(3)
                if mat_choice == 0:
                    mat = U
                    rows = n * m
                    name = 'U'
                elif mat_choice == 1:
                    mat = V
                    rows = m * p
                    name = 'V'
                else:
                    mat = W
                    rows = p * n
                    name = 'W'
                    
                r_idx = np.random.randint(rows)
                c_idx = np.random.randint(R)
                
                # Try increment/decrement
                old_val = mat[r_idx, c_idx]
                for delta in [-1, 1]:
                    mat[r_idx, c_idx] = old_val + delta
                    new_err = compute_error(U, V, W)
                    if new_err < current_err:
                        current_err = new_err
                        break
                    else:
                        mat[r_idx, c_idx] = old_val
                
            # Check if we found an exact solution
            if current_err < 1e-6:
                if check_exact_fast(U, V, W):
                    # Convert to lists of ints/half-ints
                    # Ensure they are integers or half-integers
                    # The search uses integers.
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    
    return best_solution
# EVOLVE-BLOCK-END
