"""Matrix multiplication 4x4x6 tensor decomposition."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Schoolbook as fallback
    rank_school = n * m * p  # 96
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Strassen 2x2x2 decomposition terms
    strassen_terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
        ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(0,1):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):-1}),
    ]
    
    def verify(U, V, W, rank, n, m, p):
        """Verify that the decomposition is correct."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if total != 1:
                        return False
        return True
    
    # Block Strassen - 12 independent 2x2x2 Strassen products
    rank1 = 12 * 7  # 84
    U1 = [[0] * rank1 for _ in range(n * m)]  # 16 x 84
    V1 = [[0] * rank1 for _ in range(m * p)]  # 24 x 84
    W1 = [[0] * rank1 for _ in range(p * n)]  # 24 x 84
    
    col = 0
    for a in range(2):       # row block of A/C
        for c in range(2):   # col block of A / row block of B
            for b in range(3):  # col block of B/C
                for (a_coefs, b_coefs, c_coefs) in strassen_terms:
                    for (ar, ac), coef in a_coefs.items():
                        gi = 2*a + ar
                        gj = 2*c + ac
                        U1[gi * m + gj][col] += coef
                    for (br, bc), coef in b_coefs.items():
                        gi = 2*c + br
                        gj = 2*b + bc
                        V1[gi * p + gj][col] += coef
                    for (cr, cc), coef in c_coefs.items():
                        gi = 2*b + cc
                        gj = 2*a + cr
                        W1[gi * n + gj][col] += coef
                    col += 1
    
    assert col == rank1, f"col={col}, rank1={rank1}"
    
    if verify(U1, V1, W1, rank1, n, m, p):
        best_rank = rank1
        best_U = U1
        best_V = V1
        best_W = W1
    
    if time.time() > deadline:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Try to reduce rank by finding linear dependencies
    # Convert to numpy for faster computation
    U1_np = np.array(U1, dtype=float)
    V1_np = np.array(V1, dtype=float)
    W1_np = np.array(W1, dtype=float)
    
    # For each rank-1 term, check if it can be removed
    # A rank-1 term r can be removed if the tensor contribution from term r
    # can be expressed as a combination of the other terms.
    # This is equivalent to checking if the vector T_r is in the span of {T_s : s != r}
    # where T_r[i,j,k] = U1[i*m+j,r] * V1[j*p+k,r] * W1[k*n+i,r]
    
    def tensor_vector(U_col, V_col, W_col, n, m, p):
        """Flatten a rank-1 tensor into a vector of length n*m*p."""
        vec = np.zeros(n * m * p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    vec[i * m * p + j * p + k] = U_col[i * m + j] * V_col[j * p + k] * W_col[k * n + i]
        return vec
    
    # Try removing one term at a time
    for r_to_remove in range(rank1):
        if time.time() > deadline:
            break
        
        # Build the tensor vector for term r_to_remove
        target = tensor_vector(U1_np[:, r_to_remove], V1_np[:, r_to_remove], W1_np[:, r_to_remove], n, m, p)
        
        # Build the matrix of other terms
        other_indices = [i for i in range(rank1) if i != r_to_remove]
        if len(other_indices) == 0:
            continue
            
        # For efficiency, only check if the target is in the span using least squares
        # Build matrix M where M[:, s] = tensor_vector(term s) for s in other_indices
        # This is expensive, so let's be selective
        
        # Only try this for a subset of terms to save time
        if r_to_remove % 10 != 0:
            continue
        
        M = np.zeros((n * m * p, len(other_indices)))
        for idx, s in enumerate(other_indices):
            M[:, idx] = tensor_vector(U1_np[:, s], V1_np[:, s], W1_np[:, s], n, m, p)
        
        # Check if target is in the column space of M
        # Use rank check: rank([M | target]) == rank(M)
        M_aug = np.column_stack([M, target])
        rank_M = np.linalg.matrix_rank(M)
        rank_M_aug = np.linalg.matrix_rank(M_aug)
        
        if rank_M == rank_M_aug:
            # Term r_to_remove is redundant!
            # Remove it
            keep_indices = other_indices
            U_new = U1_np[:, keep_indices]
            V_new = V1_np[:, keep_indices]
            W_new = W1_np[:, keep_indices]
            
            # Convert back to list of lists
            U_new_list = U_new.tolist()
            V_new_list = V_new.tolist()
            W_new_list = W_new.tolist()
            
            new_rank = len(keep_indices)
            if verify(U_new_list, V_new_list, W_new_list, new_rank, n, m, p):
                if new_rank < best_rank:
                    best_rank = new_rank
                    best_U = U_new_list
                    best_V = V_new_list
                    best_W = W_new_list
    
    if time.time() > deadline:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Try a different approach: use a 4x4x4 rank-49 decomposition
    # For now, return the best we have
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
