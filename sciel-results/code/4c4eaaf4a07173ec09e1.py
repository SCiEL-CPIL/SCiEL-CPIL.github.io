import numpy as np
import time
import random

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin

    # Target tensor T[i, j, k, l, r, s] = 1 if i=r, j=l, k=s else 0
    # U: (n*m, R) -> index i*m + j
    # V: (m*p, R) -> index j*p + k
    # W: (p*n, R) -> index k*n + i
    # Tensor: T[i, j, k, l, r, s] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Note: The task description says W[k*n+i, r].
    # Standard MM: C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor indices: A is (i,j), B is (j,k), C is (i,k).
    # The prompt says: "output factor follows the official trace convention W[k*n+i,r]"
    # This implies the third index of W corresponds to the output index C.
    # Let's verify the index mapping.
    # U row: i*m + j (Input A index)
    # V row: j*p + k (Input B index)
    # W row: k*n + i (Output C index)
    
    # We need to construct the target tensor T of shape (n*m, m*p, p*n)
    # T[ai, bj, ck] = 1 if ai=i*m+j, bj=j*p+k, ck=k*n+i
    # i.e., T[ai, bj, ck] = 1 if (ai//m == ck%n) and (ai%m == bj//p) and (bj%p == ck//n)
    
    U_dim = n * m
    V_dim = m * p
    W_dim = p * n
    
    # Precompute target tensor
    target = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ai = i * m + j
                bj = j * p + k
                ck = k * n + i
                target[ai, bj, ck] = 1.0

    def compute_error(U, V, W):
        # U: (U_dim, R), V: (V_dim, R), W: (W_dim, R)
        # We want sum_r U[ai, r] V[bj, r] W[ck, r] == target[ai, bj, ck]
        # This is a tensor contraction.
        # Efficiently: 
        # temp = np.einsum('ir,kr->ikr', U, W) # (U_dim, W_dim, R)
        # res = np.einsum('kr,jr->jk', temp, V) # (U_dim, W_dim)
        # Wait, V is (V_dim, R). We need to contract r.
        # Let's do:
        # M = U @ V.T -> (U_dim, V_dim) is wrong.
        # We need T[ai, bj, ck] = sum_r U[ai, r] * V[bj, r] * W[ck, r]
        
        # Reshape to (U_dim, V_dim, W_dim)
        # U[ai, r] * V[bj, r] -> (U_dim, V_dim, R)
        # Then multiply by W[ck, r] and sum over r.
        
        # Optimize:
        # UV = np.einsum('ir,jr->ijr', U, V) # (U_dim, V_dim, R)
        # Res = np.einsum('ijr,kr->ijk', UV, W) # (U_dim, V_dim, W_dim)
        
        # This is 27*27*27*23 operations ~ 450k. Very fast.
        UV = np.einsum('ir,jr->ijr', U, V, optimize=True)
        Res = np.einsum('ijr,kr->ijk', UV, W, optimize=True)
        return np.linalg.norm(Res - target)

    # Discrete values allowed
    # The problem statement says "integer/half-integer entries".
    # Common decompositions use 0, 1, -1, 0.5, -0.5.
    # Let's use a set of candidates.
    candidates = [0.0, 0.5, -0.5, 1.0, -1.0]
    
    def random_init(R):
        U = np.random.choice(candidates, size=(U_dim, R))
        V = np.random.choice(candidates, size=(V_dim, R))
        W = np.random.choice(candidates, size=(W_dim, R))
        return U, V, W

    def local_search(U, V, W, R, max_iters=1000):
        current_err = compute_error(U, V, W)
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_err = current_err
        
        # Precompute gradients or just use step-wise updates?
        # Step-wise is safer for discrete.
        
        for _ in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Pick a random matrix and index
            mat_idx = np.random.randint(3)
            if mat_idx == 0:
                mat = U
                dim1, dim2 = U_dim, R
            elif mat_idx == 1:
                mat = V
                dim1, dim2 = V_dim, R
            else:
                mat = W
                dim1, dim2 = W_dim, R
                
            idx1 = np.random.randint(dim1)
            idx2 = np.random.randint(dim2)
            
            old_val = mat[idx1, idx2]
            
            # Try changing to a different candidate
            # Pick a random new value from candidates, excluding old_val
            possible = [c for c in candidates if c != old_val]
            if not possible:
                continue
            new_val = random.choice(possible)
            
            # Apply change
            mat[idx1, idx2] = new_val
            
            # Compute new error
            new_err = compute_error(U, V, W)
            
            # Accept if better, or with some probability (Simulated Annealing)
            # For simplicity, let's do strict hill climbing first, then add restarts.
            if new_err < best_err:
                best_err = new_err
                best_U[idx1, idx2] = new_val
                if mat_idx == 0:
                    pass # U is already updated in place, but we need to sync best_U
                    # Actually, we are modifying U, V, W in place.
                    # Let's just track the best state by copying when improvement is found.
                    # This is expensive.
                    # Better: Keep U, V, W as the current state.
                    # If new_err < current_err, accept.
                    # If new_err < best_err, update best.
                    pass
            else:
                # Revert
                mat[idx1, idx2] = old_val
                continue
            
            # If we accepted, update current_err
            current_err = new_err
            
            # Check for exact solution
            if current_err < 1e-9:
                return U, V, W, current_err

        return U, V, W, best_err

    # Strategy:
    # 1. Try to find rank 23 (known minimal for 3x3).
    # 2. Use multiple random restarts.
    
    best_solution = None
    best_rank = 27 # Fallback
    
    # We know rank 27 is the identity. 
    # Let's try to find rank 23, 22, 21...
    # Start with 23.
    
    for R in [23, 22, 21, 20, 19]:
        if time.time() > deadline - 30:
            break
            
        found = False
        max_restarts = 50 # Tune based on time
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
                
            U, V, W = random_init(R)
            
            # Initial local search
            U, V, W, err = local_search(U, V, W, R, max_iters=2000)
            
            if err < 1e-9:
                # Verify exactly
                # Re-calculate to be sure
                if compute_error(U, V, W) < 1e-9:
                    best_solution = (U, V, W)
                    best_rank = R
                    found = True
                    break
        
        if found:
            break
            
    if best_solution:
        U, V, W = best_solution
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback to rank 27
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
