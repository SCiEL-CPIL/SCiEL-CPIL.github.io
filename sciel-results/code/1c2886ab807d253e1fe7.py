"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]
import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Dimensions of U, V, W
    dim_u = n * m  # 10
    dim_v = m * p  # 30
    dim_w = p * n  # 12
    
    # Target tensor T flattened
    # T[(i*m+j), (j*p+k), (k*n+i)] = 1
    target = np.zeros((dim_u, dim_v, dim_w), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1

    # Generate candidate rank-1 tensors
    candidates = []
    
    # 1. Standard basis terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = np.zeros(dim_u, dtype=int)
                v = np.zeros(dim_v, dtype=int)
                w = np.zeros(dim_w, dtype=int)
                u[i*m+j] = 1
                v[j*p+k] = 1
                w[k*n+i] = 1
                candidates.append((u, v, w))

    # 2. Strassen-like terms for 2x2x2 sub-blocks
    # We need to select 2 rows from n (only 1 way for n=2), 2 cols from m, 2 cols from p.
    # n=2, so i is always {0, 1}.
    # j pairs from range(m)
    # k pairs from range(p)
    
    # Strassen coefficients for 2x2 matrices A, B -> C
    # M1 = (A11+A22)(B11+B22) -> u=e11+e22, v=e11+e22, w=e11+e22 (coeffs in C: C11+=1, C22+=1)
    # M2 = (A21+A22)B11 -> u=e21+e22, v=e11, w=e21 (coeffs in C: C21+=1)
    # M3 = A11(B12-B22) -> u=e11, v=e12-e22, w=e12 (coeffs in C: C12+=1)
    # M4 = A22(B21-B11) -> u=e22, v=e21-e11, w=e21 (coeffs in C: C21+=1, C11-=1) -> Wait, M4 contributes to C21 and C11?
    # Let's re-verify Strassen:
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M6 + M7
    
    # Term 1: M1. u=e11+e22, v=e11+e22. w: C11+=1, C22+=1. w=e11+e22.
    # Term 2: M2. u=e21+e22, v=e11. w: C21+=1. w=e21.
    # Term 3: M3. u=e11, v=e12-e22. w: C12+=1, C22-=1. w=e12-e22.
    # Term 4: M4. u=e22, v=e21-e11. w: C21+=1, C11-=1. w=e21-e11.
    # Term 5: M5. u=e11+e12, v=e22. w: C11+=1, C12+=1. w=e11+e12.
    # Term 6: M6. u=e21-e11, v=e11+e12. w: C22+=1. w=e22.
    # Term 7: M7. u=e12-e22, v=e21+e22. w: C11-=1, C22+=1. w=-e11+e22.
    
    # Local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Global mapping for a sub-block defined by j0, j1 (cols of M) and k0, k1 (cols of N)
    # i is always 0, 1.
    
    strassen_terms_local = [
        # u, v, w (local indices 0..3 for u/v, 0..3 for w? No, w is C which is 2x2, so 4 entries)
        # u is 2x2 (4 entries), v is 2x2 (4 entries), w is 2x2 (4 entries)
        (np.array([1,0,0,1]), np.array([1,0,0,1]), np.array([1,0,0,1])), # M1
        (np.array([0,0,1,1]), np.array([1,0,0,0]), np.array([0,0,1,0])), # M2
        (np.array([1,0,0,0]), np.array([0,1,0,-1]), np.array([0,1,0,-1])), # M3
        (np.array([0,0,0,1]), np.array([0,0,1,-1]), np.array([0,0,1,-1])), # M4 (w: C21=1, C11=-1 -> indices 2,0)
        (np.array([1,1,0,0]), np.array([0,0,0,1]), np.array([1,1,0,0])), # M5
        (np.array([0,0,1,-1]), np.array([1,1,0,0]), np.array([0,0,0,1])), # M6
        (np.array([0,1,0,-1]), np.array([0,0,1,1]), np.array([-1,0,0,1])), # M7
    ]
    
    # Add Strassen terms to candidates
    for j0 in range(m):
        for j1 in range(j0+1, m):
            for k0 in range(p):
                for k1 in range(k0+1, p):
                    # Map local indices to global
                    # u indices: (i, j) where i in {0,1}, j in {j0, j1}
                    # Local u index 0 -> (0, j0), 1 -> (0, j1), 2 -> (1, j0), 3 -> (1, j1)
                    # v indices: (j, k) where j in {j0, j1}, k in {k0, k1}
                    # Local v index 0 -> (j0, k0), 1 -> (j0, k1), 2 -> (j1, k0), 3 -> (j1, k1)
                    # w indices: (k, i) where k in {k0, k1}, i in {0,1}
                    # Local w index 0 -> (k0, 0), 1 -> (k0, 1), 2 -> (k1, 0), 3 -> (k1, 1)
                    
                    for u_loc, v_loc, w_loc in strassen_terms_local:
                        u = np.zeros(dim_u, dtype=int)
                        v = np.zeros(dim_v, dtype=int)
                        w = np.zeros(dim_w, dtype=int)
                        
                        # Fill u
                        # Local 0: i=0, j=j0
                        u[0*m + j0] += u_loc[0]
                        # Local 1: i=0, j=j1
                        u[0*m + j1] += u_loc[1]
                        # Local 2: i=1, j=j0
                        u[1*m + j0] += u_loc[2]
                        # Local 3: i=1, j=j1
                        u[1*m + j1] += u_loc[3]
                        
                        # Fill v
                        # Local 0: j=j0, k=k0
                        v[j0*p + k0] += v_loc[0]
                        # Local 1: j=j0, k=k1
                        v[j0*p + k1] += v_loc[1]
                        # Local 2: j=j1, k=k0
                        v[j1*p + k0] += v_loc[2]
                        # Local 3: j=j1, k=k1
                        v[j1*p + k1] += v_loc[3]
                        
                        # Fill w
                        # Local 0: k=k0, i=0
                        w[k0*n + 0] += w_loc[0]
                        # Local 1: k=k0, i=1
                        w[k0*n + 1] += w_loc[1]
                        # Local 2: k=k1, i=0
                        w[k1*n + 0] += w_loc[2]
                        # Local 3: k=k1, i=1
                        w[k1*n + 1] += w_loc[3]
                        
                        candidates.append((u, v, w))

    # Greedy Set Cover
    # We want to select a subset of candidates such that sum(u_r v_r w_r) = target
    # Since the number of candidates is large, we use a greedy approach.
    # We maintain the residual R = target.
    # At each step, find the candidate c that minimizes ||R - c|| (or maximizes overlap).
    # Since entries are small integers, we can use exact matching.
    
    # To speed up, we can pre-compute the "signature" or just use numpy.
    # Number of candidates ~ 1100.
    # Rank ~ 54.
    # 54 iterations.
    # Each iteration: check 1100 candidates.
    # 1100 * 54 * (10*30*12) operations ~ 200 million. Might be slow in Python.
    # We can optimize by only considering candidates that have non-zero overlap with R.
    
    R = target.copy()
    selected = []
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 100 # 100s limit
    
    while np.any(R != 0):
        if time.time() > deadline:
            break
            
        best_idx = -1
        best_norm = np.linalg.norm(R) # Current norm squared? Use sum of squares
        
        # Only check candidates that might reduce the norm
        # Approximation: check candidates with non-zero dot product with R
        for idx, (u, v, w) in enumerate(candidates):
            # Compute term = u[:, None, None] * v[None, :, None] * w[None, None, :]
            # This is expensive.
            # Instead, check if u, v, w have support where R is non-zero.
            # If not, skip.
            
            # Quick check: if u is zero, skip
            if not np.any(u): continue
            if not np.any(v): continue
            if not np.any(w): continue
            
            # Compute the tensor product
            # To avoid full tensor creation, we can compute the norm of R - term
            # ||R - term||^2 = ||R||^2 - 2<R, term> + ||term||^2
            # We want to minimize this.
            
            # <R, term> = sum(R * (u[:,:,None] * v[:,None,:] * w))
            # This is still heavy.
            
            # Optimization: Pre-compute flattened candidates?
            # Or just iterate.
            
            # Let's try a faster check:
            # If the candidate is a standard basis vector, it's fast.
            # For Strassen, it's sparse.
            
            # Compute inner product
            # term = np.einsum('i,j,k->ijk', u, v, w)
            # dot = np.sum(R * term)
            
            # We can approximate by checking if the candidate covers any non-zero part of R.
            # For now, let's just do it for all candidates, but break early if we find a perfect match.
            
            # Actually, we can just try to solve a linear system for a small subset?
            # No, greedy is safer.
            
            # To speed up, we can skip candidates that are "bad".
            # Let's just run it.
            
            term = np.einsum('i,j,k->ijk', u, v, w)
            diff = R - term
            norm = np.sum(diff * diff)
            
            if norm < best_norm:
                best_norm = norm
                best_idx = idx
                
    if best_idx != -1:
        u, v, w = candidates[best_idx]
        R -= np.einsum('i,j,k->ijk', u, v, w)
        selected.append((u, v, w))
    else:
        break

    if len(selected) < n*m*p:
        # Construct U, V, W matrices
        rank = len(selected)
        U = np.zeros((dim_u, rank), dtype=int)
        V = np.zeros((dim_v, rank), dtype=int)
        W = np.zeros((dim_w, rank), dtype=int)
        
        for r, (u, v, w) in enumerate(selected):
            U[:, r] = u
            V[:, r] = v
            W[:, r] = w
            
        # Verify
        # T_approx = sum(U[:,r,None,None] * V[:,r,None] * W[:,r])
        # T_approx = np.einsum('ir,jr,kr->ijk', U, V, W)
        # if np.allclose(T_approx, target):
        #     return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Since we used greedy, it might not be exact if it broke early.
        # Check if R is zero
        if np.all(R == 0):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(dim_u)]
    V = [[0] * rank for _ in range(dim_v)]
    W = [[0] * rank for _ in range(dim_w)]
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
