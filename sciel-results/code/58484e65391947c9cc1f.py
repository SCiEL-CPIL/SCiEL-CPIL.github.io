import numpy as np
import time
from itertools import combinations, product

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m + j, j*p + k, k*n + i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Schoolbook decomposition (Rank 48) as fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Non-zero indices of the tensor
    nz_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nz_indices.append((a, b, c))
    
    # Strategy: Try to find a lower rank decomposition by solving a system of equations
    # For rank R, we need to find U, V, W such that sum_r u_r ⊗ v_r ⊗ w_r = T
    
    # We'll try a structured approach:
    # 1. For each candidate rank R, set up the system of equations
    # 2. Use a combination of random initialization and gradient descent
    # 3. Focus on finding sparse solutions
    
    # The key insight: for matrix multiplication tensors, the optimal decomposition
    # often has a regular structure. We can exploit this by trying specific patterns.
    
    # Let's try ranks from 24 to 40 (below the schoolbook rank of 48)
    ranks_to_try = list(range(24, 41))
    ranks_to_try = [r for r in ranks_to_try if r < best_rank]
    
    for R in ranks_to_try:
        if time.time() > deadline - 15:
            break
        
        found = False
        
        # Try multiple random initializations
        for trial in range(50):
            if time.time() > deadline - 5:
                break
            
            # Use a fixed seed for reproducibility
            np.random.seed(12345 + R * 1000 + trial)
            
            # Initialize with small random values
            U = np.random.randn(n * m, R) * 0.1
            V = np.random.randn(m * p, R) * 0.1
            W = np.random.randn(p * n, R) * 0.1
            
            # Gradient descent with momentum
            lr = 0.01
            momentum = 0.9
            vU = np.zeros_like(U)
            vV = np.zeros_like(V)
            vW = np.zeros_like(W)
            
            for it in range(500):
                if time.time() > deadline - 3:
                    break
                
                # Compute the current reconstruction
                recon = np.zeros_like(T)
                for r in range(R):
                    recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                
                # Compute the error
                error = T - recon
                
                # Compute gradients
                # dL/dU[:,r] = sum over b,c of error[a,b,c] * V[b,r] * W[c,r]
                # dL/dV[:,r] = sum over a,c of error[a,b,c] * U[a,r] * W[c,r]
                # dL/dW[:,r] = sum over a,b of error[a,b,c] * U[a,r] * V[b,r]
                
                gU = np.zeros_like(U)
                gV = np.zeros_like(V)
                gW = np.zeros_like(W)
                
                for r in range(R):
                    # Gradient for U[:, r]
                    gU[:, r] = np.einsum('abc,br,cr->a', error, V[:, r], W[:, r])
                    # Gradient for V[:, r]
                    gV[:, r] = np.einsum('abc,ar,cr->b', error, U[:, r], W[:, r])
                    # Gradient for W[:, r]
                    gW[:, r] = np.einsum('abc,ar,br->c', error, U[:, r], V[:, r])
                
                # Update with momentum
                vU = momentum * vU + lr * gU
                vV = momentum * vV + lr * gV
                vW = momentum * vW + lr * gW
                
                U -= vU
                V -= vV
                W -= vW
                
                # Check convergence
                if it % 10 == 0:
                    recon = np.zeros_like(T)
                    for r in range(R):
                        recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                    err = np.linalg.norm(T - recon)
                    if err < 1e-8:
                        break
            
            # Final check
            recon = np.zeros_like(T)
            for r in range(R):
                recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            error = np.linalg.norm(T - recon)
            
            if error < 1e-8:
                # Try to round to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                recon_rounded = np.zeros_like(T)
                for r in range(R):
                    recon_rounded += np.einsum('a,b,c->abc', U_rounded[:, r], V_rounded[:, r], W_rounded[:, r])
                error_rounded = np.linalg.norm(T - recon_rounded)
                
                if error_rounded < 1e-6:
                    best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                    best_rank = R
                    found = True
                    break
                else:
                    # Use the unrounded version if it's exact
                    best_U, best_V, best_W = U, V, W
                    best_rank = R
                    found = True
                    break
        
        if found:
            break
    
    # If we haven't found a better solution, try a different approach:
    # Use the fact that the tensor has a regular structure and try to find
    # a decomposition with a specific sparsity pattern.
    
    if best_rank == rank_school and time.time() < deadline - 20:
        # Try a structured search for rank 24-30
        for R in range(24, 31):
            if time.time() > deadline - 10:
                break
            
            # Try to find a decomposition where each column of U, V, W
            # has a specific sparsity pattern
            
            # For simplicity, let's try to find a decomposition where
            # the non-zero pattern of each column is aligned with the
            # non-zero pattern of the tensor.
            
            # This is a heuristic approach that may not always work,
            # but it's worth trying given the time constraints.
            
            # We'll use a simulated annealing approach to search for
            # a valid decomposition.
            
            np.random.seed(99999 + R)
            
            # Initialize with a random sparse pattern
            U = np.zeros((n * m, R))
            V = np.zeros((m * p, R))
            W = np.zeros((p * n, R))
            
            # Set random non-zero entries
            for r in range(R):
                # Choose random positions for non-zero entries
                u_pos = np.random.choice(n * m, size=min(3, n * m), replace=False)
                v_pos = np.random.choice(m * p, size=min(3, m * p), replace=False)
                w_pos = np.random.choice(p * n, size=min(3, p * n), replace=False)
                
                for pos in u_pos:
                    U[pos, r] = np.random.choice([-1, 1])
                for pos in v_pos:
                    V[pos, r] = np.random.choice([-1, 1])
                for pos in w_pos:
                    W[pos, r] = np.random.choice([-1, 1])
            
            # Simulated annealing
            T_sa = 1.0
            cooling = 0.99
            min_T = 1e-6
            
            for sa_iter in range(1000):
                if time.time() > deadline - 5:
                    break
                
                if T_sa < min_T:
                    break
                
                # Compute current error
                recon = np.zeros_like(T)
                for r in range(R):
                    recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                current_error = np.sum((T - recon) ** 2)
                
                # Make a random move
                # Choose a random factor and a random entry to flip
                factor_choice = np.random.choice(3)  # 0=U, 1=V, 2=W
                if factor_choice == 0:
                    row = np.random.randint(n * m)
                    col = np.random.randint(R)
                    old_val = U[row, col]
                    U[row, col] = -U[row, col] if U[row, col] != 0 else np.random.choice([-1, 1])
                elif factor_choice == 1:
                    row = np.random.randint(m * p)
                    col = np.random.randint(R)
                    old_val = V[row, col]
                    V[row, col] = -V[row, col] if V[row, col] != 0 else np.random.choice([-1, 1])
                else:
                    row = np.random.randint(p * n)
                    col = np.random.randint(R)
                    old_val = W[row, col]
                    W[row, col] = -W[row, col] if W[row, col] != 0 else np.random.choice([-1, 1])
                
                # Compute new error
                recon = np.zeros_like(T)
                for r in range(R):
                    recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                new_error = np.sum((T - recon) ** 2)
                
                # Accept or reject
                delta = new_error - current_error
                if delta < 0 or np.random.random() < np.exp(-delta / T_sa):
                    # Accept
                    pass
                else:
                    # Reject
                    if factor_choice == 0:
                        U[row, col] = old_val
                    elif factor_choice == 1:
                        V[row, col] = old_val
                    else:
                        W[row, col] = old_val
                
                # Cool down
                T_sa *= cooling
                
                # Check if we found an exact solution
                if new_error < 1e-10:
                    break
            
            # Final check
            recon = np.zeros_like(T)
            for r in range(R):
                recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            error = np.linalg.norm(T - recon)
            
            if error < 1e-8:
                best_U, best_V, best_W = U, V, W
                best_rank = R
                break
    
    # Return the best found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
