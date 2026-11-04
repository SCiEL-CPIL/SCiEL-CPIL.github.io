import numpy as np
import random
import time
import sys

def construct():
    n, m, p = 2, 3, 7
    nm, mp, pn = n*m, m*p, p*n
    deadline = time.time() + 145  # Safety margin
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best = schoolbook()
    best_rank = n * m * p
    
    # Precompute target tensor indices
    # T[i, j, k] should be 1 if this is a valid term, but in decomposition,
    # we sum over c. The result for each (i,j,k) must be 1.
    # Wait, the tensor for matrix multiplication M_{n,m,p} has 1s at (i,j,k) for all i,j,k?
    # No. The tensor is defined such that C_{ik} = sum_{j} A_{ij} B_{jk}.
    # The bilinear form is sum_{j} A_{ij} B_{jk}.
    # In the decomposition, we have sum_{c} U_{ij,c} V_{jk,c} W_{ki,c}.
    # This sum must equal A_{ij} B_{jk} for the specific (i,j,k) term?
    # No, the tensor T is the multiplication tensor.
    # T_{i,j,k} = 1 if the term A_{ij} B_{jk} contributes to C_{ik}.
    # In the standard basis, the multiplication tensor has 1s at all (i,j,k) indices corresponding to the terms in the sum.
    # So for every i in [0,n), j in [0,m), k in [0,p), the coefficient of A_{ij} B_{jk} in C_{ik} is 1.
    # So yes, the target is 1 for all valid (i,j,k) tuples.
    
    # Indices for verification
    indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    def verify_exact(U, V, W, rank):
        for i, j, k in indices:
            s = 0
            u_idx = i*m + j
            v_idx = j*p + k
            w_idx = k*n + i
            for c in range(rank):
                s += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
            if s != 1:
                return False
        return True

    def verify_float(U, V, W, rank):
        # Convert to numpy arrays for speed
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Compute tensor product sum
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # We can reshape and use einsum or explicit loops if small.
        # Given small sizes, explicit loop with numpy dot might be faster if batched?
        # Let's just do a quick check.
        max_err = 0.0
        for i, j, k in indices:
            u_idx = i*m + j
            v_idx = j*p + k
            w_idx = k*n + i
            s = np.dot(U_np[u_idx], V_np[v_idx] * W_np[w_idx])
            err = abs(s - 1.0)
            if err > max_err:
                max_err = err
            if max_err > 1e-6:
                return False
        return True

    # Search for lower rank
    # Known lower bound is 21.
    # Try ranks from 21 to best_rank - 1
    for target_rank in range(21, best_rank):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random candidates
        # Use small integers
        max_attempts = 2000
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            # Values: -1, 0, 1
            U = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(nm)]
            V = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(mp)]
            W = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(pn)]
            
            # Quick float check
            if verify_float(U, V, W, target_rank):
                # Exact check
                if verify_exact(U, V, W, target_rank):
                    best = {"U": U, "V": V, "W": W}
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            continue

    return best
