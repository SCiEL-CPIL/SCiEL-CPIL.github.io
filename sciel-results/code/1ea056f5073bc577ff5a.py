import numpy as np
import time
import random

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Shapes:
    # U: (n*m, R)
    # V: (m*p, R)
    # W: (p*n, R)
    
    # Tensor T is defined by T[i, j, k, l, k', l'] = delta[i, k'] delta[j, l'] delta[k, l]
    # Wait, the indices are:
    # Input A: i, j (i in 0..3, j in 0..3)
    # Input B: j', k (j' in 0..3, k in 0..4)
    # Output C: i', k' (i' in 0..3, k' in 0..4)
    # Product C[i', k'] = sum_{j, k} A[i, j] B[j, k] ... wait, standard matrix mult is C = A * B.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # Here A is 4x4, B is 4x5.
    # So C is 4x5.
    # Indices:
    # A: (i, j) with i in [0,3], j in [0,3]
    # B: (j, k) with j in [0,3], k in [0,4]
    # C: (i, k) with i in [0,3], k in [0,4]
    # Tensor T maps (A, B) to C.
    # T_{i, j, j', k, i', k'} = delta_{i, i'} delta_{j, j'} delta_{k, k'}
    
    # Flattened indices:
    # u index: i * m + j  (0..15)
    # v index: j * p + k  (0..19)
    # w index: i * p + k  (0..19) -- Wait, task says W[k*n+i, r].
    # Task says: "output factor follows the official trace convention W[k*n+i,r]"
    # So w index is k * n + i.
    # Let's check the shape: p*n = 5*4 = 20.
    # w index = k * 4 + i.
    
    n, m, p = 4, 4, 5
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Precompute the tensor T as a list of 3D slices or a 3D array
    # T[a, b, c] where a in [0, dim_u), b in [0, dim_v), c in [0, dim_w)
    # T[a, b, c] = 1 if the indices match, 0 otherwise.
    # a = i * m + j
    # b = j' * p + k
    # c = k' * n + i'
    # Match if i == i', j == j', k == k'
    
    T = np.zeros((dim_u, dim_v, dim_w), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k  # Note: B is indexed by j, k. j matches j from A.
                c = k * n + i  # Note: C is indexed by i, k. i matches i from A, k matches k from B.
                T[a, b, c] = 1

    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Search for minimal rank
    for R in range(27, 81):
        if time.time() > deadline:
            break
            
        # Try to find a decomposition of rank R
        # We will try random sparse U, V
        max_trials = 500 if R < 40 else 200
        found = False
        
        for trial in range(max_trials):
            if time.time() > deadline:
                break
                
            # Generate random sparse U and V
            # Use small integers to increase chance of integer W
            U = np.zeros((dim_u, R), dtype=int)
            V = np.zeros((dim_v, R), dtype=int)
            
            for r in range(R):
                # Pick 2-3 random non-zero entries for U
                nz_u = random.sample(range(dim_u), 3)
                for idx in nz_u:
                    U[idx, r] = random.choice([-1, 1])
                
                nz_v = random.sample(range(dim_v), 3)
                for idx in nz_v:
                    V[idx, r] = random.choice([-1, 1])
            
            # Construct the matrix M of size (dim_u * dim_v) x R
            # M[ (a * dim_v + b), r ] = U[a, r] * V[b, r]
            M = np.zeros((dim_u * dim_v, R), dtype=float)
            for a in range(dim_u):
                for b in range(dim_v):
                    for r in range(R):
                        M[a * dim_v + b, r] = U[a, r] * V[b, r]
            
            # We need to solve M * W_flat = T_flat
            # T_flat is of size (dim_u * dim_v * dim_w)
            # But the system is: for each c, M * W[:, c] = T[:, :, c]
            # So we can solve for each c independently, or stack them.
            
            # Let's solve for each c
            W = np.zeros((dim_w, R), dtype=float)
            valid = True
            for c in range(dim_w):
                t_c = T[:, :, c].flatten()
                # Solve M * w_c = t_c
                # Use least squares
                sol, residuals, rank, sv = np.linalg.lstsq(M, t_c, rcond=None)
                # Check residual
                if residuals.size > 0 and residuals[0] > 1e-9:
                    valid = False
                    break
                W[c, :] = sol
            
            if not valid:
                continue
                
            # Check if W can be rounded to integers or half-integers
            # And if the rounded W gives an exact solution
            W_rounded = np.zeros_like(W, dtype=int)
            exact = True
            for c in range(dim_w):
                for r in range(R):
                    val = W[c, r]
                    # Try rounding to nearest integer or half-integer
                    # val * 2 should be close to an integer
                    val2 = val * 2
                    nearest = round(val2)
                    if abs(val2 - nearest) > 1e-6:
                        exact = False
                        break
                    W_rounded[c, r] = nearest
                if not exact:
                    break
            
            if not exact:
                continue
                
            # Verify the solution with rounded W
            # Reconstruct T and compare
            error = 0
            for a in range(dim_u):
                for b in range(dim_v):
                    for c in range(dim_w):
                        recon = 0
                        for r in range(R):
                            recon += U[a, r] * V[b, r] * (W_rounded[c, r] / 2.0)
                        if abs(recon - T[a, b, c]) > 1e-9:
                            error += 1
                            break
                    if error > 0:
                        break
                if error > 0:
                    break
            if error > 0:
                continue
                
            # Found a valid decomposition
            # Convert to lists
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = [[w / 2.0 for w in row] for row in W_rounded.tolist()]
            
            return {"U": U_list, "V": V_list, "W": W_list}
            
        # If no solution found for this R, try next R
        
    # Fallback: Schoolbook
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
