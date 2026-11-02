import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target tensor: T[a, b, c, d] = 1 if b == c else 0
    # a in [0, n), b in [0, m), c in [0, m), d in [0, p)
    # We need sum_r U[a*m+b, r] * V[c*p+d, r] * W[d*n+a, r] == T[a,b,c,d]
    
    # Precompute indices for vectorization
    # We want to compute E = sum_{a,b,c,d} (S[a,b,c,d] - T[a,b,c,d])^2
    # where S[a,b,c,d] = sum_r U[a*m+b, r] * V[c*p+d, r] * W[d*n+a, r]
    
    # To compute S efficiently, we can use einsum.
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    # Reshape U to (n, m, rank)
    # Reshape V to (m, p, rank)
    # Reshape W to (p, n, rank)
    
    # S[a,b,c,d] = sum_r U[a,b,r] V[c,d,r] W[d,a,r]
    # Note: W index is d*n + a. So W[d, a, r].
    
    # einsum: 'abr, cdr, dar -> abcd'
    # Wait, V index is c*p+d. So V[c,d,r].
    # W index is d*n+a. So W[d,a,r].
    
    # Let's check the einsum string:
    # U: a, b, r
    # V: c, d, r
    # W: d, a, r
    # Result: a, b, c, d
    # 'abr,cdr,dar->abcd'
    
    # Target T:
    T = np.zeros((n, m, m, p))
    for b in range(m):
        for c in range(m):
            if b == c:
                T[:, b, c, :] = 1.0
            # else 0
    
    # We will try ranks from 25 to 36
    # If we find a solution with rank < 36, we return it.
    # We start from low rank and go up.
    
    for rank in range(20, 37):
        # Try multiple random restarts
        for restart in range(100):
            # Initialize U, V, W
            # Use random sparse initialization
            # Density approx 1/3?
            
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # Sparsify: set 50% to 0
            U[np.random.rand(n * m, rank) < 0.5] = 0
            V[np.random.rand(m * p, rank) < 0.5] = 0
            W[np.random.rand(p * n, rank) < 0.5] = 0
            
            # Local search
            best_err = float('inf')
            U_curr, V_curr, W_curr = U, V, W
            
            for step in range(1000):
                # Calculate error
                U_r = U_curr.reshape(n, m, rank)
                V_r = V_curr.reshape(m, p, rank)
                W_r = W_curr.reshape(p, n, rank)
                
                # Compute S
                S = np.einsum('abr,cdr,dar->abcd', U_r, V_r, W_r)
                err = np.sum((S - T) ** 2)
                
                if err < 1e-8:
                    # Check if we can round to half-integers
                    U_round = np.round(U_curr * 2) / 2
                    V_round = np.round(V_curr * 2) / 2
                    W_round = np.round(W_curr * 2) / 2
                    
                    U_rr = U_round.reshape(n, m, rank)
                    V_rr = V_round.reshape(m, p, rank)
                    W_rr = W_round.reshape(p, n, rank)
                    
                    S_round = np.einsum('abr,cdr,dar->abcd', U_rr, V_rr, W_rr)
                    if np.allclose(S_round, T, atol=1e-6):
                        # Convert to lists
                        return {
                            "U": U_round.tolist(),
                            "V": V_round.tolist(),
                            "W": W_round.tolist()
                        }
                    
                    # If rounding failed, continue searching or accept if very close?
                    # The task requires exact decomposition with integer/half-integer entries.
                    # So we must find a solution that rounds correctly.
                    # If err is small, rounding should work.
                    # If not, we continue.
                    pass

                # Move: Pick a random factor and entry
                factor = np.random.choice(3)
                if factor == 0:
                    mat = U_curr
                    shape = (n * m, rank)
                elif factor == 1:
                    mat = V_curr
                    shape = (m * p, rank)
                else:
                    mat = W_curr
                    shape = (p * n, rank)
                
                i = np.random.randint(shape[0])
                j = np.random.randint(shape[1])
                
                old_val = mat[i, j]
                # Try changing value
                # Add random noise
                new_val = old_val + np.random.randn() * 0.5
                mat[i, j] = new_val
                
                # Calculate new error
                if factor == 0:
                    U_r = mat.reshape(n, m, rank)
                    V_r = V_curr.reshape(m, p, rank)
                    W_r = W_curr.reshape(p, n, rank)
                elif factor == 1:
                    U_r = U_curr.reshape(n, m, rank)
                    V_r = mat.reshape(m, p, rank)
                    W_r = W_curr.reshape(p, n, rank)
                else:
                    U_r = U_curr.reshape(n, m, rank)
                    V_r = V_curr.reshape(m, p, rank)
                    W_r = mat.reshape(p, n, rank)
                    
                S_new = np.einsum('abr,cdr,dar->abcd', U_r, V_r, W_r)
                new_err = np.sum((S_new - T) ** 2)
                
                # Accept if better or with probability exp(-delta/T)
                if new_err < err or np.random.rand() < np.exp(-(new_err - err) / 0.1):
                    # Keep the change
                    if factor == 0:
                        U_curr = mat
                    elif factor == 1:
                        V_curr = mat
                    else:
                        W_curr = mat
                    err = new_err
                else:
                    # Revert
                    mat[i, j] = old_val

            # End local search loop
            
    # If no solution found, return schoolbook
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
# EVOLVE-BLOCK-END
