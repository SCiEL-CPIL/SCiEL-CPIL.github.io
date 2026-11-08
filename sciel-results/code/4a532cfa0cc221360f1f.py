import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Build the multiplication tensor T[i,j,k] = 1 if output (i,k) depends on A[i,j]*B[j,k]
    # A is n×m, B is m×p, C is n×p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor index: (i, j, k) where i in [0,n), j in [0,m), k in [0,p)
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # W[k*n + i, r] follows trace convention
    
    # The tensor T is a 3D array of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # All other entries are 0
    
    # We need U @ V^T @ W^T to reconstruct T
    # Specifically: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i*m+j, j*p+k, k*n+i]
    
    # Start with schoolbook (rank 36)
    rank_schoolbook = n * m * p
    
    # Try to find lower rank
    best_rank = rank_schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # Initialize with schoolbook
    U = np.zeros((n*m, rank_schoolbook))
    V = np.zeros((m*p, rank_schoolbook))
    W = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try random search for lower ranks
    # For each candidate rank r from some minimum up to best_rank-1
    # Try random initialization and check if we can find an exact decomposition
    
    # The tensor has 1296 entries, but only 36 are non-zero (all 1s)
    # Actually, T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k, which is n*m*p = 36 entries
    
    # Let's verify the schoolbook works
    def verify(U, V, W, tol=1e-10):
        n, m, p = DIMENSIONS
        # Check all 36 non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1) > tol:
                        return False
        # Check some zero entries (sample)
        # There are many zero entries, check a sample
        np.random.seed(42)
        for _ in range(200):
            a = np.random.randint(0, n*m)
            b = np.random.randint(0, m*p)
            c = np.random.randint(0, p*n)
            # Check if this should be zero
            i, j = divmod(a, m)
            j2, k = divmod(b, p)
            k2, i2 = divmod(c, n)
            if not (j == j2 and i == i2 and k == k2):
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val) > tol:
                    return False
        return True
    
    # Verify schoolbook
    if not verify(best_U, best_V, best_W):
        print("Schoolbook verification failed!")
    
    # Now try to find lower rank decompositions
    # Use random search with exact rational arithmetic where possible
    
    # Strategy: try ranks from 20 up to best_rank-1
    # For each rank, do random restarts
    
    target_ranks = list(range(20, best_rank))
    target_ranks.reverse()  # Try lower ranks first
    
    for r in target_ranks:
        if time.time() > deadline:
            break
        
        # Try several random initializations
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            np.random.seed(attempt * 1000 + r)
            
            # Random initialization with small values
            U_r = np.random.randn(n*m, r) * 0.5
            V_r = np.random.randn(m*p, r) * 0.5
            W_r = np.random.randn(p*n, r) * 0.5
            
            # We need sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k
            # And = 0 for all other combinations
            
            # This is a system of equations. Let's set up the linear system.
            # For each (a,b,c) where a=i*m+j, b=j*p+k, c=k*n+i:
            # sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
            
            # This is nonlinear in U,V,W. But if we fix two of them, it's linear in the third.
            
            # Alternative: use the fact that for small rank, we can try to find integer solutions.
            
            # Let's try a different approach: use the known structure.
            # For M_{2,3,6}, try to exploit that 6 = 2*3.
            
            # Actually, let's just do a more systematic search.
            # Generate random U, V, and solve for W (or vice versa).
            
            # Fix U and V randomly, solve for W.
            # The constraint is: for each (a,b,c) with T[a,b,c]=1:
            # sum_r U[a,r]*V[b,r]*W[c,r] = 1
            # For each (a,b,c) with T[a,b,c]=0:
            # sum_r U[a,r]*V[b,r]*W[c,r] = 0
            
            # This is a linear system in W if U and V are fixed.
            
            # Build the linear system: A_sys * W_flat = b_sys
            # where W_flat is W.reshape(-1)
            
            # Number of constraints: n*m*p = 36 non-zero, plus many zeros
            # Let's use all 36 non-zero constraints plus a sample of zero constraints
            
            # Actually, let's use all constraints. The tensor has (n*m)*(m*p)*(p*n) = 6*18*12 = 1296 entries.
            # 36 are 1, 1260 are 0.
            
            # That's too many. Let's use the 36 non-zero ones and a random sample of zero ones.
            
            num_zero_constraints = 100
            zero_constraints = []
            np.random.seed(12345 + attempt)
            all_zero_indices = []
            for a in range(n*m):
                for b in range(m*p):
                    for c in range(p*n):
                        i, j = divmod(a, m)
                        j2, k = divmod(b, p)
                        k2, i2 = divmod(c, n)
                        if not (j == j2 and i == i2 and k == k2):
                            all_zero_indices.append((a, b, c))
            
            if len(all_zero_indices) > num_zero_constraints:
                zero_sample = np.random.choice(len(all_zero_indices), num_zero_constraints, replace=False)
                zero_constraints = [all_zero_indices[idx] for idx in zero_sample]
            else:
                zero_constraints = all_zero_indices
            
            # Build system
            num_constraints = 36 + len(zero_constraints)
            A_sys = np.zeros((num_constraints, p*n*r))
            b_sys = np.zeros(num_constraints)
            
            constraint_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i*m+j
                        b = j*p+k
                        c = k*n+i
                        # sum_r U[a,r]*V[b,r]*W[c,r] = 1
                        for r in range(r):
                            A_sys[constraint_idx, c*r+r] = U_r[a,r]*V_r[b,r]
                        b_sys[constraint_idx] = 1
                        constraint_idx += 1
            
            for (a, b, c) in zero_constraints:
                for r in range(r):
                    A_sys[constraint_idx, c*r+r] = U_r[a,r]*V_r[b,r]
                b_sys[constraint_idx] = 0
                constraint_idx += 1
            
            # Solve least squares
            try:
                W_flat, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                W_r = W_flat.reshape(p*n, r)
                
                # Verify
                if verify(U_r, V_r, W_r):
                    best_rank = r
                    best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                    print(f"Found rank {r} decomposition!")
                    break  # Move to next rank
            except:
                pass
    
    # Convert to lists
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
