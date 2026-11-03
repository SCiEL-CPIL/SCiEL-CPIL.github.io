import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    num_a = n * m  # 16
    num_b = m * p  # 32
    num_c = p * n  # 32
    
    # Build the set of non-zero tensor entries: (a, b, c) where T[a,b,c] = 1
    # a = i*m + j, b = j'*p + k, c = k'*n + i'
    # T[a,b,c] = 1 iff i == i', k == k', j == j'
    nonzero_triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nonzero_triples.append((a, b, c))
    
    # For a given rank R, we need to find U (num_a x R), V (num_b x R), W (num_c x R)
    # such that for all (a,b,c): sum_t U[a,t]*V[b,t]*W[c,t] = T[a,b,c]
    #
    # Strategy: Fix W to be the identity-like basis (W[c,t] = 1 if c==t else 0) for t < num_c,
    # and 0 otherwise. But that forces R >= num_c = 32.
    # Actually, let's use a smarter approach.
    #
    # Alternative: We can fix one of the matrices to a simple form.
    # Let's fix W such that W[c, t] = delta_{c, t} for t = 0..num_c-1, and 0 for t >= num_c.
    # Then R must be at least num_c = 32.
    # With this fix, the constraint becomes:
    # For non-zero (a,b,c): sum_t U[a,t]*V[b,t]*delta_{c,t} = U[a,c]*V[b,c] = 1
    # For zero (a,b,c): U[a,c]*V[b,c] = 0
    #
    # So we need U[a,c]*V[b,c] = 1 for (a,b,c) in nonzero_triples, and 0 otherwise.
    # This means: for each c, U[:,c] and V[:,c] must satisfy:
    #   U[a,c]*V[b,c] = 1 if (a,b,c) is a non-zero triple
    #   U[a,c]*V[b,c] = 0 otherwise
    #
    # For a fixed c, let S_c = {(a,b) : (a,b,c) is non-zero}. Then we need U[a,c]*V[b,c] = 1 for (a,b) in S_c, and 0 otherwise.
    # This is a rank-1 constraint on the matrix M_c = U[:,c] * V[:,c]^T.
    # M_c must equal the matrix with 1s at positions S_c and 0s elsewhere.
    # But M_c is rank 1, so the matrix with 1s at S_c must be rank 1.
    #
    # Let's check: for c = k*n + i, S_c = {(i*m+j, j*p+k) : j = 0..m-1}
    # The matrix M_c has 1s at (i*m+j, j*p+k) for j=0..m-1.
    # This is a matrix with m ones, each in a different row and different column.
    # So M_c is a permutation matrix (partial), which has rank m = 4, not 1.
    # So this approach doesn't work with W being identity.
    #
    # Let's try a different approach. We'll use the fact that the tensor has structure.
    # The matrix multiplication tensor for (n,m,p) has rank at most n*m*p (schoolbook).
    # We want to find a lower rank decomposition.
    #
    # For 4x4x8, let's try to find a decomposition of rank R.
    # We'll set up the system of equations and solve it.
    #
    # The constraints are:
    # For each (a,b,c): sum_t U[a,t]*V[b,t]*W[c,t] = T[a,b,c]
    #
    # This is a system of num_a * num_b * num_c = 16*32*32 = 16384 equations
    # in num_a*R + num_b*R + num_c*R = (16+32+32)*R = 80R unknowns.
    #
    # For R = 48, we have 80*48 = 3840 unknowns and 16384 equations.
    # The system is overdetermined, but the equations are not independent.
    #
    # We can use a randomized approach: pick random U and V, then solve for W.
    # For each c, we have: sum_t U[a,t]*V[b,t]*W[c,t] = T[a,b,c] for all (a,b).
    # This is a linear system in W[c,:].
    #
    # Let's try this approach with exact rational arithmetic.
    
    start_time = time.time()
    timeout = 110
    
    # Try ranks from 48 to 128
    for R in range(48, 129):
        if time.time() - start_time > timeout:
            break
        
        num_trials = 50
        for trial in range(num_trials):
            if time.time() - start_time > timeout:
                break
            
            # Random U, V with entries in {-1, 0, 1}
            U = np.random.randint(-1, 2, size=(num_a, R))
            V = np.random.randint(-1, 2, size=(num_b, R))
            
            # For each c, solve for W[c,:]
            W = np.zeros((num_c, R), dtype=object)
            valid = True
            for c in range(num_c):
                # Build the linear system: A_c * W[c,:] = rhs_c
                # where A_c[a,b] = sum_t U[a,t]*V[b,t] ... no, that's not right.
                # The equation is: for each (a,b), sum_t U[a,t]*V[b,t]*W[c,t] = T[a,b,c]
                # Let x_t = W[c,t]. Then sum_t (U[a,t]*V[b,t]) * x_t = T[a,b,c]
                # So A_c is a matrix where A_c[(a,b), t] = U[a,t]*V[b,t]
                # and rhs_c[(a,b)] = T[a,b,c]
                
                # Number of (a,b) pairs is num_a * num_b = 16*32 = 512
                # But many are redundant. We only need to use the independent ones.
                # Actually, for a fixed c, T[a,b,c] is non-zero only for specific (a,b).
                # Let's build the system using all (a,b) pairs.
                
                A_c = np.zeros((num_a * num_b, R), dtype=object)
                rhs_c = np.zeros(num_a * num_b, dtype=object)
                
                for a in range(num_a):
                    for b in range(num_b):
                        idx = a * num_b + b
                        for t in range(R):
                            A_c[idx, t] = Fraction(U[a, t]) * Fraction(V[b, t])
                        
                        # Compute T[a,b,c]
                        i_in = a // m
                        j_in = a % m
                        j_in_prime = b // p
                        k_in = b % p
                        k_out = c // n
                        i_out = c % n
                        
                        if i_in == i_out and k_in == k_out and j_in == j_in_prime:
                            rhs_c[idx] = Fraction(1)
                        else:
                            rhs_c[idx] = Fraction(0)
                
                # Solve A_c * x = rhs_c using Gaussian elimination over rationals
                # This is a system of 512 equations in R unknowns.
                # We'll use least squares or find a particular solution.
                
                # Actually, since the system may be inconsistent, we should check.
                # Let's use a simple approach: if R < num_a * num_b, the system is overdetermined.
                # We'll try to find a solution using numpy lstsq with float, then verify exactly.
                
                A_float = np.array([[float(x) for x in row] for row in A_c], dtype=float)
                rhs_float = np.array([float(x) for x in rhs_c], dtype=float)
                
                try:
                    W_c, residuals, rank_ac, s = np.linalg.lstsq(A_float, rhs_float, rcond=None)
                    residual = np.linalg.norm(A_float @ W_c - rhs_float)
                    if residual > 1e-9:
                        valid = False
                        break
                    W[c, :] = [Fraction(x).limit_denominator(1000000) for x in W_c]
                except:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Verify the full decomposition exactly
            ok = True
            for a in range(num_a):
                for b in range(num_b):
                    for c in range(num_c):
                        val = sum(Fraction(U[a, t]) * Fraction(V[b, t]) * W[c, t] for t in range(R))
                        
                        i_in = a // m
                        j_in = a % m
                        j_in_prime = b // p
                        k_in = b % p
                        k_out = c // n
                        i_out = c % n
                        
                        expected = Fraction(1) if (i_in == i_out and k_in == k_out and j_in == j_in_prime) else Fraction(0)
                        
                        if val != expected:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                continue
            
            # Success! Convert to lists
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
        
        print(f"Rank {R}: not found")
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
