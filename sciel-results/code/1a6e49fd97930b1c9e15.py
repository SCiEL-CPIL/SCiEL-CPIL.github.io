import time
import numpy as np
import random

DIMENSIONS = [3, 4, 7]

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix
    by a 4 by 7 matrix.
    Uses a stochastic local search strategy to find a low-rank exact decomposition.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # Safety margin
    
    # Dimensions of the factor matrices
    dim_u = n * m  # 12
    dim_v = m * p  # 28
    dim_w = p * n  # 21
    
    # Theoretical lower bound is max(n*m, m*p, p*n) = 28
    # Known upper bound is n*m*p = 84
    # We start search from a low rank and increase if necessary, or start high and decrease.
    # Given the complexity, starting from a moderate rank and trying to reduce is hard.
    # Starting from a low rank and trying to find a solution is also hard.
    # A common heuristic is to start near the lower bound and increase until a solution is found,
    # or use a specific known bound.
    
    # For small matrices like 3x4x7, the exact rank is often close to the max dimension product or slightly higher.
    # Let's try to find the minimal rank by increasing the rank from the theoretical minimum.
    
    # Theoretical min rank is max(12, 28, 21) = 28.
    # Let's try ranks from 28 upwards.
    # However, random search in high dimensions is inefficient.
    
    # Alternative strategy:
    # 1. Start with a known valid decomposition (schoolbook, rank 84).
    # 2. Try to reduce the rank by zeroing out columns and adjusting others? No, that's not how it works.
    # 3. Better: Randomly initialize U, V, W with small integers for a given rank R.
    #    Check if sum_k U[:,k] V[:,k] = Tensor.
    #    This is a system of linear equations.
    
    # Let's use a more direct approach for this specific small case:
    # Iterate R from 28 to 84.
    # For each R, run a short random search to see if a solution exists.
    
    best_rank = 84
    best_U, best_V, best_W = None, None, None
    
    # Pre-compute the target tensor T[i, j, k] = delta(i,j,k) in flattened form?
    # The condition is: for all i, j, k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if (i,j,k) is a valid triplet?
    # No, the matrix multiplication tensor T_{i,j,k} is 1 if the indices correspond to a valid multiplication term.
    # Actually, the tensor for matrix multiplication C = AB is defined such that:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor T has indices (i, j, k) where i is row of A, j is col of A/row of B, k is col of B.
    # T[i, j, k] = 1 if the term A_{i,j} B_{j,k} contributes to C_{i,k}.
    # Wait, the standard tensor decomposition of matrix multiplication is:
    # M_{i, j, k} = sum_r U_{i, j, r} V_{j, k, r} W_{k, i, r}
    # Here, the indices of the matrices are mapped to the tensor indices.
    # U shape: (n*m, R) -> U[i*m + j, r]
    # V shape: (m*p, R) -> V[j*p + k, r]
    # W shape: (p*n, R) -> W[k*n + i, r]
    
    # The product is sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This sum must equal 1 for all valid (i, j, k) triplets in the multiplication?
    # No, the matrix multiplication tensor T is defined such that:
    # T[i, j, k] = 1 if the indices (i, j, k) form a valid "corner" of the multiplication?
    # Actually, the tensor T is the structure tensor of matrix multiplication.
    # T[i, j, k] = 1 if the term A_{i,j} B_{j,k} is part of the product.
    # In the standard definition, T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # Wait, if T[i,j,k]=1 for all i,j,k, then the sum is 1 for all i,j,k.
    # But the output C_{i,k} is the sum over j.
    # The tensor decomposition represents the bilinear map.
    # The condition is:
    # For all i, j, k:
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
    # Is this correct?
    # Let's check the schoolbook case.
    # U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1.
    # Sum over r: only one r (the column) is non-zero for a specific (i,j,k).
    # So the sum is 1.
    # Yes, the target tensor T has 1s at all positions (i, j, k) in the n x m x p box.
    
    # So we need to find U, V, W such that for all i, j, k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a system of n*m*p = 84 equations.
    # Unknowns are 3*R entries.
    
    # Strategy:
    # 1. Try R from 28 upwards.
    # 2. For each R, use random initialization and local search (or just random restarts) to find a solution.
    #    Since the entries are integers/half-integers, we can search in a discrete space.
    #    However, the search space is huge.
    
    # A better approach for small matrices:
    # Use the fact that the rank is likely small.
    # We can use a randomized algorithm to find a solution for a fixed R.
    # If we find one, we return it.
    
    # Given the time limit, we can only try a few R values.
    # Let's try R = 28, 29, ..., up to some limit or time limit.
    
    # To make it efficient, we can use a greedy or heuristic approach.
    # But for now, let's implement a simple random search with a time budget per R.
    
    # Optimization:
    # We can fix one matrix, say W, to be sparse or simple, and solve for U and V?
    # No, that's not straightforward.
    
    # Let's try a different angle:
    # The rank of 3x4x7 matrix multiplication is known to be 84? No, it's likely lower.
    # For 2x2x2, rank is 7.
    # For 3x3x3, rank is 23 (Strassen-like).
    # For 3x4x7, it's unknown but likely less than 84.
    
    # Let's try to find a solution with R=28.
    # If not found, R=29, etc.
    
    # We will use a simple random walk in the space of integer matrices.
    # To speed up, we can use NumPy for the verification.
    
    # Verification function:
    def verify(U, V, W, R):
        # U: (12, R), V: (28, R), W: (21, R)
        # We need to check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k.
        
        # We can vectorize this.
        # Create arrays of indices.
        # This might be slow if done in a loop.
        # Let's reshape U, V, W to 3D and use einsum?
        # U is (n, m, R)
        # V is (m, p, R)
        # W is (p, n, R)
        
        U3 = U.reshape(n, m, R)
        V3 = V.reshape(m, p, R)
        W3 = W.reshape(p, n, R)
        
        # Product is sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
        # This is a tensor contraction.
        # We can use np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        
        prod = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        
        # Check if all entries are 1
        return np.all(prod == 1)
        
    # Random initialization
    def random_init(R):
        # Entries are small integers, e.g., -2, -1, 0, 1, 2
        # To increase chance of success, we can use a mix of 0s and 1s.
        # But random integers might be better for exploration.
        
        # Let's use -1, 0, 1 initially.
        U = np.random.choice([-1, 0, 1], size=(dim_u, R))
        V = np.random.choice([-1, 0, 1], size=(dim_v, R))
        W = np.random.choice([-1, 0, 1], size=(dim_w, R))
        
        return U, V, W
        
    # Local search step
    def local_search(U, V, W, R, steps=1000):
        # Pick a random entry in one of the matrices and change it.
        # Accept if it reduces the error or stays the same?
        # Since we need exact match, we can't use gradient descent easily.
        # We can use a "hit or miss" approach:
        # 1. Calculate current error (number of mismatches).
        # 2. Propose a change.
        # 3. If error decreases or stays same, accept.
        # 4. If error increases, accept with small probability (simulated annealing).
        
        # To be efficient, we only recompute the error for the affected (i,j,k) triplets.
        # But with R up to 84, and 84 triplets, full recomputation is fast enough with NumPy.
        
        current_error = np.sum(np.einsum('ijr,jkr,kir->ijk', U.reshape(n,m,R), V.reshape(m,p,R), W.reshape(p,n,R)) != 1)
        
        for _ in range(steps):
            if time.time() > deadline:
                break
                
            # Choose a random matrix and entry
            mat_idx = np.random.choice(3)
            if mat_idx == 0:
                i, j = np.random.randint(0, n), np.random.randint(0, m)
                r = np.random.randint(0, R)
                old_val = U[i*m+j, r]
                new_val = old_val + np.random.choice([-1, 1])
                U[i*m+j, r] = new_val
                # Check error
                prod = np.einsum('ijr,jkr,kir->ijk', U.reshape(n,m,R), V.reshape(m,p,R), W.reshape(p,n,R))
                new_error = np.sum(prod != 1)
                if new_error <= current_error:
                    current_error = new_error
                else:
                    # Revert
                    U[i*m+j, r] = old_val
                    
            elif mat_idx == 1:
                j, k = np.random.randint(0, m), np.random.randint(0, p)
                r = np.random.randint(0, R)
                old_val = V[j*p+k, r]
                new_val = old_val + np.random.choice([-1, 1])
                V[j*p+k, r] = new_val
                prod = np.einsum('ijr,jkr,kir->ijk', U.reshape(n,m,R), V.reshape(m,p,R), W.reshape(p,n,R))
                new_error = np.sum(prod != 1)
                if new_error <= current_error:
                    current_error = new_error
                else:
                    V[j*p+k, r] = old_val
                    
            else:
                k, i = np.random.randint(0, p), np.random.randint(0, n)
                r = np.random.randint(0, R)
                old_val = W[k*n+i, r]
                new_val = old_val + np.random.choice([-1, 1])
                W[k*n+i, r] = new_val
                prod = np.einsum('ijr,jkr,kir->ijk', U.reshape(n,m,R), V.reshape(m,p,R), W.reshape(p,n,R))
                new_error = np.sum(prod != 1)
                if new_error <= current_error:
                    current_error = new_error
                else:
                    W[k*n+i, r] = old_val
                    
            if current_error == 0:
                return True, U, V, W
                
        return False, U, V, W
        
    # Search loop
    # Try ranks from 28 upwards.
    # Limit the number of attempts per rank to avoid getting stuck.
    
    # We will use a time budget for each rank.
    # If time is running out, we return the best found so far (which is schoolbook).
    
    # Initialize with schoolbook
    schoolbook_rank = n * m * p
    U_sb = np.zeros((dim_u, schoolbook_rank), dtype=int)
    V_sb = np.zeros((dim_v, schoolbook_rank), dtype=int)
    W_sb = np.zeros((dim_w, schoolbook_rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
                
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = schoolbook_rank
    
    # Try to improve
    for R in range(28, schoolbook_rank + 1):
        if time.time() > deadline - 5:
            break
            
        # Allow a short time for each R
        # e.g., 100 attempts of 100 steps each?
        # Let's just run local search for a fixed number of iterations or time.
        
        found = False
        # Try a few random restarts for this R
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
                
            U, V, W = random_init(R)
            # Run local search
            success, U, V, W = local_search(U, V, W, R, steps=500)
            if success:
                best_rank = R
                best_U, best_V, best_W = U, V, W
                found = True
                break
                
        if found:
            # We found a solution for R. Can we find a smaller one?
            # The loop continues to R+1? No, we want the MINIMAL rank.
            # So we should break and return.
            break
            
    # Convert to lists of lists for output
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
