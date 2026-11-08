"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s limit
    
    # Try to find lower rank decompositions
    # The tensor has 16*28*28 = 12544 entries, but only 4*4*7=112 are non-zero
    
    # We need: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # And for all other (a,b,c), sum_r U[a,r]*V[b,r]*W[c,r] = 0
    
    # Let's try ranks from lower values up
    # Start with a reasonable guess based on known results
    # For 4x4x4, min rank is 49. For 4x4x7, let's try various ranks
    
    for rank in [100, 96, 92, 88, 84, 80, 76, 72, 68, 64, 60, 56, 52, 49]:
        if time.time() > deadline:
            break
            
        # Try to find a decomposition of this rank
        # Use random initialization and iterative refinement
        found = False
        for attempt in range(50):
            if time.time() > deadline:
                break
                
            np.random.seed(attempt * 1000 + rank)
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Iterate: fix two, solve for third
            for iteration in range(100):
                if time.time() > deadline:
                    break
                    
                # Solve for W given U, V
                # For each (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
                # This is linear in W for fixed U,V
                # Build system: for each (i,j,k), we have an equation
                # A_W * W_flat = b_W where A_W has rows for each (i,j,k)
                
                num_constraints = n * m * p  # 112 constraints for the =1 case
                # Plus we need the other constraints to be 0, but that's a lot
                # Let's just enforce the =1 constraints first and check others later
                
                # Actually, let's set up the full system
                # Number of (a,b,c) triples where we need sum = 0 or 1
                # Total possible: 16*28*28 = 12544
                # But most will be 0
                
                # For efficiency, let's only enforce the 112 =1 constraints initially
                # and check the 0 constraints separately
                
                # Build linear system for W
                # Each row corresponds to an (i,j,k) constraint
                A_W = np.zeros((num_constraints, p*n*rank))
                b_W = np.ones(num_constraints)
                
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i*m + j
                            b = j*p + k
                            c = k*n + i
                            # Coefficients for W[c, r]
                            for r in range(rank):
                                A_W[idx, c*rank + r] = U[a, r] * V[b, r]
                            idx += 1
                
                # Solve least squares
                W_flat, residuals, rank_sol, sv = np.linalg.lstsq(A_W, b_W, rcond=None)
                W = W_flat.reshape(p*n, rank)
                
                # Check how well this satisfies the =1 constraints
                # And check the =0 constraints
                error = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i*m + j
                            b = j*p + k
                            c = k*n + i
                            val = np.sum(U[a, :] * V[b, :] * W[c, :])
                            error += (val - 1)**2
                
                # Also check some random =0 constraints
                for _ in range(100):
                    a = np.random.randint(0, n*m)
                    b = np.random.randint(0, m*p)
                    c = np.random.randint(0, p*n)
                    # Check if this is a valid (i,j,k)
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k2 = b % p
                    k = c // n
                    i2 = c % n
                    if not (i == i2 and j == j2 and k == k2):
                        val = np.sum(U[a, :] * V[b, :] * W[c, :])
                        error += val**2
                
                if error < 1e-6:
                    # Verify fully
                    full_error = 0
                    for a in range(n*m):
                        for b in range(m*p):
                            i = a // m
                            j = a % m
                            j2 = b // p
                            k2 = b % p
                            if j != j2:
                                continue
                            for c in range(p*n):
                                k = c // n
                                i2 = c % n
                                if i == i2 and j == j2 and k == k2:
                                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                                    full_error += (val - 1)**2
                                else:
                                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                                    full_error += val**2
                    if full_error < 1e-6:
                        best_rank = rank
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        found = True
                        break
                
                # Update U given V, W
                A_U = np.zeros((num_constraints, n*m*rank))
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i*m + j
                            b = j*p + k
                            c = k*n + i
                            for r in range(rank):
                                A_U[idx, a*rank + r] = V[b, r] * W[c, r]
                            idx += 1
                U_flat, _, _, _ = np.linalg.lstsq(A_U, b_W, rcond=None)
                U = U_flat.reshape(n*m, rank)
                
                # Update V given U, W
                A_V = np.zeros((num_constraints, m*p*rank))
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i*m + j
                            b = j*p + k
                            c = k*n + i
                            for r in range(rank):
                                A_V[idx, b*rank + r] = U[a, r] * W[c, r]
                            idx += 1
                V_flat, _, _, _ = np.linalg.lstsq(A_V, b_W, rcond=None)
                V = V_flat.reshape(m*p, rank)
            
            if found:
                break
        
        if found:
            break
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
