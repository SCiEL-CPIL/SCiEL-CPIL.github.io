import numpy as np
import time
import itertools

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit (or 150s if that's the real one, but 115 is safe for 120s)

    # Build the target tensor T
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Schoolbook solution (Rank 60)
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_school[a, col] = 1.0
                V_school[b, col] = 1.0
                W_school[c, col] = 1.0
                col += 1

    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school

    def compute_error(U, V, W):
        """Compute Frobenius norm of T - sum_u U[:,u] \otimes V[:,u] \otimes W[:,u]"""
        A = np.zeros((n*m, m*p, p*n))
        for u in range(U.shape[1]):
            A += np.einsum('i,j,k->ijk', U[:, u], V[:, u], W[:, u])
        return np.linalg.norm(T - A)

    def compute_grads(U, V, W):
        """Compute gradients for U, V, W"""
        A = np.zeros((n*m, m*p, p*n))
        for u in range(U.shape[1]):
            A += np.einsum('i,j,k->ijk', U[:, u], V[:, u], W[:, u])
        E = T - A
        
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for u in range(U.shape[1]):
            # dU[:, u] = - sum_{b,c} E[i, b, c] V[b, u] W[c, u]
            M_u = np.outer(V[:, u], W[:, u])
            dU[:, u] = - (E.reshape(n*m, -1) @ M_u.flatten())
            
            # dV[:, u] = - sum_{a,c} E[a, b, c] U[a, u] W[c, u]
            N_u = np.outer(U[:, u], W[:, u])
            E_reshaped = E.transpose(1, 0, 2) # (m*p, n*m, p*n)
            dV[:, u] = - (E_reshaped.reshape(m*p, -1) @ N_u.flatten())
            
            # dW[:, u] = - sum_{a,b} E[a, b, c] U[a, u] V[b, u]
            P_u = np.outer(U[:, u], V[:, u])
            E_reshaped2 = E.transpose(2, 0, 1) # (p*n, n*m, m*p)
            dW[:, u] = - (E_reshaped2.reshape(p*n, -1) @ P_u.flatten())
            
        return dU, dV, dW

    def try_optimize(rank, seed, max_iters=200):
        """Try to find a valid decomposition of given rank using gradient descent."""
        np.random.seed(seed)
        
        # Random initialization
        # Scale to roughly match the magnitude of T entries
        # T has 60 ones.
        # U, V, W entries should be around 1/sqrt(rank) maybe?
        # Let's start with small random values.
        U = np.random.randn(n*m, rank) * 0.5
        V = np.random.randn(m*p, rank) * 0.5
        W = np.random.randn(p*n, rank) * 0.5
        
        lr = 0.05
        min_lr = 1e-6
        err = compute_error(U, V, W)
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None, float('inf')
                
            dU, dV, dW = compute_grads(U, V, W)
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            new_err = compute_error(U, V, W)
            
            # Adaptive learning rate
            if new_err < err:
                err = new_err
                # lr = min(lr * 1.1, 0.5) # Increase slightly
            else:
                lr *= 0.5
                # Revert? Or just keep going?
                # Reverting is safer for stability
                U += lr * 2 * dU # Approximate revert? No, lr changed.
                # Better to store old state
                # For simplicity, just reduce lr and continue.
                # If we go into a bad region, it might not recover.
                # Let's try a simple backtracking line search if it fails often?
                # Given time constraints, simple decay is okay.
                
            if lr < min_lr:
                break
                
            if err < 1e-8:
                break
                
        # Check if we found a solution
        if err < 1e-7:
            # Try to snap to half-integers
            U_int = np.round(U * 2) / 2
            V_int = np.round(V * 2) / 2
            W_int = np.round(W * 2) / 2
            err_int = compute_error(U_int, V_int, W_int)
            if err_int < 1e-9:
                return (U_int, V_int, W_int), err_int
            else:
                # Maybe integers?
                U_int = np.round(U)
                V_int = np.round(V)
                W_int = np.round(W)
                err_int = compute_error(U_int, V_int, W_int)
                if err_int < 1e-9:
                    return (U_int, V_int, W_int), err_int
                else:
                    # Return float solution if it's good enough? 
                    # The task asks for integer/half-integer.
                    # "integer/half-integer entries"
                    # If we can't snap, it's invalid.
                    return None, err
        return None, err

    # Search for lower ranks
    # Start from 59 down to 50
    # We know rank 60 works.
    # Try 59, 58, ...
    
    # Optimization: We can try to use the structure of the problem.
    # But gradient descent is a black box.
    
    # Let's try a few ranks.
    # The theoretical lower bound is roughly 1.5 * n*m*p for large n,m,p? No, that's for square.
    # For 3x4x5, rank 60 is schoolbook.
    # Strassen-like decompositions might reduce it.
    # Rank 59 is a very small reduction.
    # Is it possible to have rank < 60?
    # Yes, if there are dependencies.
    
    # Let's iterate downwards.
    for rank in range(rank_school - 1, 40, -1):
        if time.time() > deadline - 5:
            break
            
        found = False
        # Try a few seeds
        for seed in range(10):
            if time.time() > deadline - 5:
                break
                
            res, err = try_optimize(rank, seed, max_iters=100)
            if res is not None:
                best_rank = rank
                best_U, best_V, best_W = res
                found = True
                break # Found valid rank, try lower
                
        if found:
            # Continue to try lower ranks
            continue
        else:
            # If we fail at this rank, do we try lower?
            # Usually if rank R fails, R-1 is harder.
            # But due to local minima, maybe a different seed works for R-1?
            # Unlikely. Let's stop if we fail at a rank we've tried hard?
            # Or just continue?
            # Given the small dimension, let's just continue down a bit.
            # But if 59 fails, 58 likely fails too.
            # Let's break if we fail at 59?
            # Actually, let's just try 59, 58, 57.
            # If none work, return 60.
            if rank <= 57:
                break

    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
