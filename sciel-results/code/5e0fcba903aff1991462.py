import numpy as np
import time
import random

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # Initialize with schoolbook
    def get_schoolbook():
        rank = rank_schoolbook
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    U_school, V_school, W_school = get_schoolbook()
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_schoolbook

    start_time = time.time()
    deadline = 10.0 # 10 seconds for search
    
    # Try to find a lower rank decomposition
    # We will try ranks from 56 down to 63
    # Note: Finding exact integer solutions is hard.
    # We will use a heuristic: 
    # 1. Generate random U, V with small integers.
    # 2. Solve for W.
    # 3. Check if W is close to integers/half-integers.
    
    # The tensor T is defined by:
    # T[a,b,c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
    # T[a,b,c] = 0 otherwise
    
    # To solve for W given U, V:
    # We need sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
    # Let M[a,b,r] = U[a,r] V[b,r]
    # Then for each c, we have a system:
    # sum_r M[a,b,r] W[c,r] = T[a,b,c]
    # This is a linear system for the row W[c,:].
    # The matrix is M (size nm x mp x R).
    # We can stack all (a,b) pairs to form a big matrix A_big of size (nm*mp) x R.
    # A_big[idx, r] = U[a,r] V[b,r]
    # The RHS is T_flat of size nm*mp.
    # But we need W[c,r] for each c.
    # Actually, for a fixed c, the equation is:
    # sum_r (U[a,r] V[b,r]) W[c,r] = T[a,b,c]
    # This is a linear system A_c * W[c,:] = T_c
    # Where A_c is the matrix with entries A_c[a,b] = U[a,r] V[b,r]? No.
    # The unknowns are W[c, 0], ..., W[c, R-1].
    # The coefficients for W[c, r] in the equation for (a,b) are U[a,r] V[b,r].
    # So for a fixed c, we have a system of nm*mp equations:
    # sum_{r=0}^{R-1} (U[a,r] V[b,r]) W[c,r] = T[a,b,c]
    # Let X be the matrix of size (nm*mp) x R where X[a*mp+b, r] = U[a,r] V[b,r].
    # Then X * W[c,:]^T = T_flat[c]
    # Wait, T[a,b,c] depends on c.
    # So for each c, we have a different RHS vector T_c of length nm*mp.
    # And the same coefficient matrix X?
    # Yes, X depends only on U and V, not on c.
    # So we can solve X * W_c = T_c for each c.
    # This is an overdetermined system (nm*mp = 256 equations, R unknowns).
    # If R < 256, it might have no exact solution.
    # We can use least squares, but we need an EXACT solution.
    # So we need the residual to be 0.
    
    for R in range(56, rank_schoolbook):
        if time.time() - start_time > deadline:
            break
            
        for attempt in range(100): # 100 random trials per rank
            if time.time() - start_time > deadline:
                break
                
            # Generate random U, V
            # Use small integers to increase chance of exact solution
            # -1, 0, 1
            U_rand = np.random.randint(-1, 2, size=(n * m, R))
            V_rand = np.random.randint(-1, 2, size=(m * p, R))
            
            # Construct X
            # X shape: (n*m*m*p, R)
            # X[i] corresponds to a=i//p? No.
            # a is 0..nm-1, b is 0..mp-1.
            # X index = a * (m*p) + b
            X = np.zeros((n * m * m * p, R))
            for a in range(n * m):
                for b in range(m * p):
                    X[a * (m * p) + b, :] = U_rand[a, :] * V_rand[b, :]
            
            # Check rank of X
            # If rank(X) < R, we have free variables, might be easier to find integer solution?
            # Or if rank(X) < number of equations, it's overdetermined.
            
            # Solve for each c
            W_rand = np.zeros((p * n, R))
            valid = True
            for c in range(p * n):
                # RHS vector T_c
                T_c = np.zeros(n * m * m * p)
                # Fill T_c based on tensor definition
                for a in range(n * m):
                    for b in range(m * p):
                        # Check if (a,b,c) is a valid schoolbook term
                        i = a // m
                        j = a % m
                        j_b = b // p
                        k = b % p
                        i_c = c % n
                        k_c = c // n
                        
                        if i == i_c and j == j_b and k == k_c:
                            T_c[a * (m * p) + b] = 1
                        else:
                            T_c[a * (m * p) + b] = 0
                
                # Solve X * W_c = T_c
                # Use np.linalg.lstsq
                try:
                    W_c, residuals, rank_x, s = np.linalg.lstsq(X, T_c, rcond=None)
                    # Check if residual is close to 0
                    if np.linalg.norm(X @ W_c - T_c) > 1e-9:
                        valid = False
                        break
                    W_rand[c, :] = W_c
                except:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check if W_rand is close to integers or half-integers
            # Multiply by 2 and check closeness to integers
            W_check = W_rand * 2
            W_rounded = np.round(W_check)
            if np.linalg.norm(W_check - W_rounded) < 1e-6:
                # Found a valid decomposition!
                # Convert to exact rationals (integers / 2)
                # For output, we can just return the floats if they are exact, 
                # or integers if possible.
                # The problem says "integer/half-integer entries".
                # np.round(W_check) / 2 gives the values.
                U_out = U_rand.astype(float)
                V_out = V_rand.astype(float)
                W_out = W_rounded / 2.0
                
                # Verify one more time with exact arithmetic if possible?
                # For now, assume it's good.
                best_U = U_out
                best_V = V_out
                best_W = W_out
                best_rank = R
                break # Found a valid one for this R, try lower R? 
                # The loop continues to next R (lower)
        
        # If we found a solution for R, we can break if we want the minimum?
        # The loop goes from 56 to 63.
        # If we find 56, we break inner loop.
        # Should we break outer? 
        # "Find a rank-minimal... score is the rank".
        # So we want the lowest R.
        # If we find 56, we can stop searching for 57-63.
        # But we might find 55? We are looping 56..63.
        # We should loop from 56 upwards? No, we want minimal.
        # So loop R from 56 to 63.
        # If we find 56, that's better than 57.
        # So we should search 56 first.
        # If 56 fails, try 57, etc.
        # The current loop `for R in range(56, rank_schoolbook):` does exactly that.
        # If we find a solution for R=56, we update best.
        # We should probably break out of the R loop if we find a solution, 
        # because lower R is better.
        # Wait, if I find 56, I don't need to check 57.
        # So `break` after finding a solution for a specific R?
        # Yes.
        
        if best_rank == R:
            break

    # Convert to lists of lists for the interface
    # The interface expects lists of lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
