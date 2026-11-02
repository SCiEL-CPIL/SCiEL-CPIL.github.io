import time
import numpy as np
import itertools

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140s limit to be safe within 150s
    
    # Schoolbook decomposition (Rank 105)
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_solution = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb

    # Search for lower rank
    # Theoretical lower bound
    lower_bound = max(n * m, m * p, p * n) # 35
    
    # We try ranks from lower_bound to best_rank - 1
    # However, for R < 5, 21R < 105, so underdetermined.
    # For R = 5, 21R = 105, square.
    # For R > 5, overdetermined.
    
    # We will search from R = 5 upwards.
    # If we find a solution, we return it.
    
    # Pre-compute equation indices
    # Equation index e corresponds to (i, j, k)
    # e = i * (m * p) + j * p + k
    # Unknown index u corresponds to (k', i', r)
    # u = (k' * n + i') * R + r
    
    # We only search up to a certain rank or time limit
    max_rank_to_search = min(best_rank - 1, 104)
    
    for R in range(5, max_rank_to_search + 1):
        if time.time() > deadline - 5:
            break
            
        # Number of trials for each R
        # If R is small, more trials. If R is large, fewer.
        trials = 100 if R < 10 else 20
        
        for t in range(trials):
            if time.time() > deadline - 5:
                break
                
            # Generate random U and V with entries in {-1, 0, 1}
            # Using integers to avoid floating point issues in construction
            U_rand = np.random.randint(-1, 2, size=(n * m, R))
            V_rand = np.random.randint(-1, 2, size=(m * p, R))
            
            # Construct linear system A w = b
            # A is (n*m*p) x (p*n*R)
            # b is (n*m*p)
            b = np.ones(n * m * p)
            A = np.zeros((n * m * p, p * n * R))
            
            # Fill A
            # Loop over equations (i, j, k)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * (m * p) + j * p + k
                        # For this equation, the sum is over r
                        # Term: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        # So A[eq_idx, (k*n+i)*R + r] = U[i*m+j, r] * V[j*p+k, r]
                        for r in range(R):
                            coeff = U_rand[i * m + j, r] * V_rand[j * p + k, r]
                            if coeff != 0:
                                unk_idx = (k * n + i) * R + r
                                A[eq_idx, unk_idx] = coeff
            
            # Solve A w = b
            try:
                # Use least squares to handle potential rank deficiency or inconsistency
                w_sol, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
                
                # Check if solution is valid (residual close to 0)
                # If the system is consistent, residual should be 0
                # Due to floating point, check against a small epsilon
                if len(residuals) > 0:
                    if residuals[0] > 1e-8:
                        continue
                else:
                    # If underdetermined, residuals might be empty?
                    # lstsq returns residuals only if M > N
                    # Here M = 105, N = 21R
                    # If R=5, N=105, M=N. residuals might be empty or [0]?
                    # Actually, if M <= N, residuals is empty.
                    # We need to check if A @ w_sol == b
                    if not np.allclose(A @ w_sol, b, atol=1e-8):
                        continue

                # Check if w_sol entries are integers or half-integers
                valid = True
                w_rounded = np.round(w_sol * 2) / 2
                if not np.allclose(w_sol, w_rounded, atol=1e-8):
                    valid = False
                
                if valid:
                    # Construct the solution matrices
                    U_list = U_rand.tolist()
                    V_list = V_rand.tolist()
                    W_list = [[0] * R for _ in range(p * n)]
                    for idx in range(len(w_sol)):
                        k_prime, i_prime, r_prime = idx // (n * R), (idx // R) % n, idx % R
                        # Wait, index mapping: unk_idx = (k*n + i) * R + r
                        # So k = unk_idx // (n*R)
                        # rem = unk_idx % (n*R)
                        # i = rem // R
                        # r = rem % R
                        k_val = idx // (n * R)
                        rem = idx % (n * R)
                        i_val = rem // R
                        r_val = rem % R
                        W_list[k_val * n + i_val][r_val] = w_rounded[idx]
                    
                    return {"U": U_list, "V": V_list, "W": W_list}
                        
            except Exception:
                continue
                
    return best_solution
# EVOLVE-BLOCK-END
