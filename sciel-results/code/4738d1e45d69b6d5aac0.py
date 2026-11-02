import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def _verify_fast(U, V, W, n, m, p):
    """
    Fast vectorized verification of the tensor decomposition.
    Checks if sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] == 1 for all i,j,k.
    """
    R = len(U[0])
    # Reshape to 3D arrays
    U_3d = np.array(U).reshape(n, m, R)
    V_3d = np.array(V).reshape(m, p, R)
    W_3d = np.array(W).reshape(p, n, R)
    
    # We want T[i,j,k] = sum_r U_3d[i,j,r] * V_3d[j,k,r] * W_3d[k,i,r]
    # This is a bit tricky to do in one einsum because the indices are permuted.
    # Let's iterate over r to build the tensor, but use numpy for the inner product.
    # T = np.zeros((n, m, p))
    # for r in range(R):
    #    # U[:, :, r] is (n,m), V[:, :, r] is (m,p), W[:, :, r] is (p,n)
    #    # The term is outer(U[:, :, r], V[:, :, r])? No.
    #    # The term contributes to T[i,j,k] as U[i,j,r] * V[j,k,r] * W[k,i,r]
    #    # This is not a simple outer product of two matrices.
    #
    # Alternative:
    # For a fixed r, the tensor is T_r[i,j,k] = U_r[i,j] * V_r[j,k] * W_r[k,i]
    # We can compute this by:
    # 1. Multiply U_r[i,j] and V_r[j,k] to get M_r[i,k] = sum_j U_r[i,j] V_r[j,k]
    #    Wait, the sum in the definition is over r, not j.
    #    T[i,j,k] = sum_r (U_r[i,j] * V_r[j,k] * W_r[k,i])
    #
    # Let's just use a loop over r, but vectorize the (i,j,k) update.
    # For a fixed r:
    #   u = U_3d[:, :, r]  # shape (n, m)
    #   v = V_3d[:, :, r]  # shape (m, p)
    #   w = W_3d[:, :, r]  # shape (p, n)
    #
    #   We need to add u[i,j] * v[j,k] * w[k,i] to T[i,j,k].
    #   This can be done by:
    #   # First, compute M[i, k] = sum_j u[i,j] v[j,k]  -> This is matrix mult u @ v
    #   # But we need the term for specific j, not summed over j.
    #   # So we cannot just use matrix multiplication.
    #
    #   We can use broadcasting:
    #   # u[:, :, None] is (n, m, 1)
    #   # v[None, :, :] is (1, m, p)
    #   # w.T[None, :, :] is (1, n, p) ... wait w[k,i] so w.T[i,k]
    #   # Term = u[i,j] * v[j,k] * w[k,i]
    #   # Let's create a tensor of shape (n, m, p)
    #   # term = u[:, :, None] * v[None, :, :] * w.T[None, :, :]
    #   # w.T has shape (n, p). w.T[i, k] = w[k, i]. Correct.
    #   # So term = u[:, :, None] * v[None, :, :] * w.T[None, :, :]
    #   # This creates an (n, m, p) array.
    #
    T = np.zeros((n, m, p), dtype=float)
    for r in range(R):
        u = U_3d[:, :, r]
        v = V_3d[:, :, r]
        w = W_3d[:, :, r]
        # u: (n, m), v: (m, p), w: (p, n)
        # We want T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
        # w[k,i] is w.T[i,k]
        # Broadcasting:
        # u[:, :, None] -> (n, m, 1)
        # v[None, :, :] -> (1, m, p)
        # w.T[None, :, :] -> (1, n, p)
        # Product: (n, m, p)
        T += u[:, :, None] * v[None, :, :] * w.T[None, :, :]
        
    # Check if T is all ones
    return np.allclose(T, 1.0, atol=1e-9)

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 110.0  # Safety margin
    
    nA = n * m
    nB = m * p
    nC = p * n
    
    # 1. Schoolbook Solution (Rank 105)
    R_sb = n * m * p
    U_sb = [[0] * R_sb for _ in range(nA)]
    V_sb = [[0] * R_sb for _ in range(nB)]
    W_sb = [[0] * R_sb for _ in range(nC)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1.0
                V_sb[j * p + k][col] = 1.0
                W_sb[k * n + i][col] = 1.0
                col += 1
    
    best_U, best_V, best_W, best_R = U_sb, V_sb, W_sb, R_sb
    
    # Verify schoolbook (should be fast now)
    if _verify_fast(U_sb, V_sb, W_sb, n, m, p):
        # We have a valid baseline
        pass
    else:
        # This should never happen, but fallback to manual check if needed
        print("Warning: Schoolbook verification failed, but returning it anyway as it's mathematically sound.")

    # 2. Attempt Search for Lower Rank (Structural Change: Randomized Linear Search)
    # We try to find a rank R < 105 solution.
    # This is extremely hard. We will try a few random initializations for R=104.
    # If we find one, great. If not, we return schoolbook.
    
    # Given the difficulty, we limit the search to a very small number of attempts
    # to ensure we don't timeout.
    
    target_ranks = [104, 103]
    
    for R_target in target_ranks:
        if time.time() > deadline - 5.0:
            break
            
        # Try a few random initializations
        max_attempts = 5
        found = False
        for attempt in range(max_attempts):
            if time.time() > deadline - 2.0:
                break
                
            # Random initialization
            U_rand = np.random.randn(nA, R_target)
            V_rand = np.random.randn(nB, R_target)
            W_rand = np.random.randn(nC, R_target)
            
            # To make it more likely to work, we could try to satisfy the equations
            # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
            # This is a system of 105 equations.
            # Variables: 15*R + 35*R + 21*R = 71*R.
            # For R=104, 71*104 = 7384 variables.
            # 105 equations. Highly underdetermined.
            # We can fix U and V randomly, and solve for W?
            # W[k,i,r] is multiplied by U[i,j,r] V[j,k,r].
            # For a fixed r, the term is separable.
            # This is non-linear.
            
            # A simpler heuristic: 
            # If we set U and V to be identity-like or sparse, we might solve for W.
            # But with random U, V, the system for W is linear if we treat W entries as variables?
            # No, W[k,i,r] appears in equation (i,j,k) multiplied by U[i,j,r] V[j,k,r].
            # So for a fixed r, W[:, :, r] is a variable matrix (p x n).
            # The equation is: sum_r (U_r[i,j] V_r[j,k] W_r[k,i]) = 1.
            # This is linear in W_r if U_r, V_r are fixed.
            # We have 105 equations and 21*R unknowns (entries of W).
            # For R=104, 21*104 = 2184 unknowns.
            # We can solve the linear system!
            
            # Let's construct the linear system A * W_flat = b
            # b is vector of ones (length 105).
            # A is 105 x (21*R).
            
            A = np.zeros((n * m * p, nC * R_target))
            b = np.ones(n * m * p)
            
            # Fill A
            # Equation index e corresponds to (i, j, k)
            # e = i * m * p + j * p + k
            # Variable index v corresponds to W[k', i', r]
            # v = (k' * n + i') * R_target + r
            # Coefficient A[e, v] = U[i, j, r] * V[j, k, r] if k'=k, i'=i, else 0.
            
            # This is sparse. We can build it efficiently.
            # For each r:
            #   For each i, j, k:
            #     e = i * m * p + j * p + k
            #     v = (k * n + i) * R_target + r
            #     A[e, v] += U_rand[i * m + j, r] * V_rand[j * p + k, r]
            
            # This loop is 105 * 104 = ~11000 iterations. Very fast.
            for r in range(R_target):
                for i in range(n):
                    for j in range(m):
                        u_val = U_rand[i * m + j, r]
                        if u_val == 0: continue
                        for k in range(p):
                            v_val = V_rand[j * p + k, r]
                            if v_val == 0: continue
                            e = i * m * p + j * p + k
                            v = (k * n + i) * R_target + r
                            A[e, v] += u_val * v_val
            
            # Solve A * W_flat = b
            # Use least squares since it's underdetermined
            try:
                W_flat, residuals, rank, singular_values = np.linalg.lstsq(A, b, rcond=None)
                # Check if residual is small
                if residuals is not None and len(residuals) > 0 and residuals[0] < 1e-6:
                    # We found a solution for W given random U, V.
                    # We must verify the full decomposition because lstsq minimizes error,
                    # and we need EXACT integer/half-integer or at least exact float match.
                    # The problem requires "exactly decompose".
                    # lstsq gives float solutions. We need to check if they are close to 0.5 or 1.0 etc?
                    # The prompt says "integer/half-integer entries".
                    # Random U, V will produce non-integer W.
                    # This approach fails the "integer/half-integer" constraint unless we round.
                    # If we round, it's likely to be wrong.
                    
                    # So this randomized search for non-integer solutions is a dead end for the specific constraint.
                    # The schoolbook solution IS integer.
                    # Therefore, we should only accept integer/half-integer solutions.
                    # Random search is unlikely to find them.
                    
                    pass
            except Exception:
                pass
            
            found = False # We don't have a valid integer solution yet

    # Since random search for integer solutions is infeasible in this time,
    # we return the verified schoolbook solution.
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
