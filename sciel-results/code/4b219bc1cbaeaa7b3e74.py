"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # U dim: n*m = 12, V dim: m*p = 24, W dim: p*n = 18
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    total_dim = dim_u * dim_v * dim_w  # 5184
    
    # Build schoolbook decomposition
    rank_school = n * m * p  # 72
    terms_u = []
    terms_v = []
    terms_w = []
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0] * dim_u
                u[i * m + j] = 1
                v = [0] * dim_v
                v[j * p + k] = 1
                w = [0] * dim_w
                w[k * n + i] = 1
                terms_u.append(u)
                terms_v.append(v)
                terms_w.append(w)
    
    # Build the tensor as a flat vector
    tensor = np.zeros(total_dim)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # T[(i,j),(j,k),(k,i)] = 1
                idx = (i * m + j) * dim_v * dim_w + (j * p + k) * dim_w + (k * n + i)
                tensor[idx] = 1.0
    
    # Build matrix of all rank-1 term vectors
    # Each term c gives vector terms_u[c] ⊗ terms_v[c] ⊗ terms_w[c]
    A = np.zeros((total_dim, rank_school))
    for c in range(rank_school):
        vec = np.outer(np.outer(terms_u[c], terms_v[c]), terms_w[c]).flatten()
        A[:, c] = vec
    
    # Check if the tensor is in the span of all 72 terms (it should be, with all coefficients = 1)
    # Now try to find minimum subset
    
    # Try removing one term at a time
    # For each term to remove, check if tensor is in span of remaining 71 terms
    # This is a linear system: A_remaining @ x = tensor
    
    best_rank = rank_school
    best_coeffs = None
    best_removed = None
    
    # First, verify schoolbook works
    # A @ ones(72) should equal tensor
    
    # Try to find if we can remove 1 term
    for remove_idx in range(rank_school):
        if time.time() > deadline - 10:
            break
        remaining = [c for c in range(rank_school) if c != remove_idx]
        A_rem = A[:, remaining]
        try:
            # Solve A_rem @ x = tensor using least squares
            x, residuals, rank_rem, sv = np.linalg.lstsq(A_rem, tensor, rcond=None)
            # Check if reconstruction is exact
            recon = A_rem @ x
            if np.allclose(recon, tensor, atol=1e-10):
                best_rank = rank_school - 1
                best_coeffs = x
                best_removed = remove_idx
                break
        except Exception:
            continue
    
    if best_rank < rank_school:
        # We found a rank-71 decomposition
        remaining = [c for c in range(rank_school) if c != best_removed]
        U = np.zeros((dim_u, best_rank))
        V = np.zeros((dim_v, best_rank))
        W = np.zeros((dim_w, best_rank))
        for new_c, old_c in enumerate(remaining):
            U[:, new_c] = terms_u[old_c]
            V[:, new_c] = terms_v[old_c]
            W[:, new_c] = terms_w[old_c]
        # Apply coefficients
        for c in range(best_rank):
            U[:, c] *= best_coeffs[c]
            V[:, c] *= best_coeffs[c]
            W[:, c] *= best_coeffs[c]
        
        # Convert to list of lists
        U_list = [[int(U[i, c]) if abs(U[i, c] - round(U[i, c])) < 1e-9 else U[i, c] for c in range(best_rank)] for i in range(dim_u)]
        V_list = [[int(V[i, c]) if abs(V[i, c] - round(V[i, c])) < 1e-9 else V[i, c] for c in range(best_rank)] for i in range(dim_v)]
        W_list = [[int(W[i, c]) if abs(W[i, c] - round(W[i, c])) < 1e-9 else W[i, c] for c in range(best_rank)] for i in range(dim_w)]
        
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
    U = [[0] * rank_school for _ in range(dim_u)]
    V = [[0] * rank_school for _ in range(dim_v)]
    W = [[0] * rank_school for _ in range(dim_w)]
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
