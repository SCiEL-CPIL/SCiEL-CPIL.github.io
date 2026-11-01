"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Helper to validate a decomposition
    def is_valid_decomposition(U, V, W, rank):
        # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # We check against the standard matrix multiplication tensor T[i, j, k] = 1 if (i,j,k) is a valid path? 
        # No, the tensor for matmul A(n x m) * B(m x p) = C(n x p) is defined by:
        # C[i, k] = sum_j A[i, j] B[j, k]
        # The tensor T has indices (i, j, k) and entries T[i, j, k] = 1 if the indices are valid, 
        # but the structure is: 
        # Product of three matrices U, V, W reconstructs the multiplication.
        # Specifically, for any A, B:
        # C = A B
        # This corresponds to the tensor T where T[i, j, k] contributes to C[i, k] from A[i, j] and B[j, k].
        # The decomposition requires:
        # sum_c U[i, j, c] V[j, k, c] W[?, c] ... wait.
        # Standard Strassen-like formulation:
        # We define tensors U, V, W of shapes (n*m, r), (m*p, r), (p*n, r) is NOT the standard convention for the third matrix in the product sum.
        
        # Let's re-read the task contract carefully:
        # "output factor follows the official trace convention W[k*n+i,r]"
        # "U (n*m, rank), V (m*p, rank), W (p*n, rank)"
        
        # The condition for matrix multiplication A (n x m) * B (m x p) = C (n x p) is:
        # C[i, k] = sum_j A[i, j] B[j, k]
        
        # The tensor decomposition usually looks like:
        # T[i, j, k] = sum_c U[i, j, c] * V[j, k, c] * W[?, c] is not quite right.
        # Usually, it's:
        # A[i, j] B[j, k] -> C[i, k]
        # We want to write the multiplication operation as a sum of rank-1 tensors.
        # A rank-1 tensor in the space of (A, B, C) is given by three vectors u (index A), v (index B), w (index C).
        # The contribution to C[i, k] is u[i, j] * v[j, k] * w[i, k]? No.
        
        # Let's look at the indices in the prompt:
        # U[i*m+j, c] -> indices (i, j) of A
        # V[j*p+k, c] -> indices (j, k) of B
        # W[k*n+i, c] -> indices (k, i) of C? Note: C is n x p. Indices are (i, k). 
        # The prompt says W[k*n+i, c]. This maps index (k, i) to row k*n+i.
        # Since C has row index i (0..n-1) and col index k (0..p-1), the pair is (i, k).
        # The prompt uses k*n+i. This is effectively (col, row) if we consider C^T, or just a specific flattening.
        # Let's verify the "trace convention" or standard convention for these problems.
        # Usually, the third factor corresponds to the output matrix.
        # If W index is k*n+i, and we sum over c, the term is U[i,j,c] * V[j,k,c] * W[k,i,c] (assuming W index maps to i,k).
        # Wait, if W is indexed by k*n+i, then W[row, c] where row = k*n + i.
        # The term in the sum for C[i, k] would be:
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # This term depends on i, j, k.
        # But C[i, k] only depends on i, k. The sum is over j.
        # So, for the decomposition to represent A*B, we must have:
        # C[i, k] = sum_j ( sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] ) ???
        # No, the decomposition is of the *operation*.
        # The operation is a tensor T in A* * B* * C*.
        # T[a, b, c] is 1 if a, b, c form a valid multiplication step?
        # Actually, the tensor T is defined such that <T, A \otimes B \otimes C> = Tr(A B C^T)?
        # Or simply: The multiplication map is a trilinear form.
        # T_{i, j, k} = 1 if (i, j, k) is a triple that appears in the sum.
        # For standard matmul: C_{i, k} = sum_j A_{i, j} B_{j, k}.
        # The trilinear form is L(A, B, C) = sum_{i, j, k} A_{i, j} B_{j, k} C_{i, k}.
        # So the tensor T has entries T[i, j, k] = 1 if the indices are compatible?
        # Yes, T[i, j, k] = 1 for all valid i, j, k.
        
        # So the condition is:
        # For all i, j, k:
        # 1 = sum_c U[i, j, c] * V[j, k, c] * W[k, i, c]  (Note: W index is k*n+i)
        
        # Let's check the schoolbook construction from the seed:
        # U[i*m+j, c] = 1 if c corresponds to (i,j,k)
        # V[j*p+k, c] = 1 if c corresponds to (i,j,k)
        # W[k*n+i, c] = 1 if c corresponds to (i,j,k)
        # Sum over c: Only one c is non-zero for a given (i,j,k) in the sum?
        # No, for a fixed (i,j,k), the sum is over c.
        # In schoolbook, for a specific (i,j,k), there is exactly one column c* where U[i,j,c*]=1, V[j,k,c*]=1, W[k,i,c*]=1.
        # All other columns are 0 for at least one of the three.
        # So sum is 1. Correct.

        # So validation:
        # For all i in 0..n-1, j in 0..m-1, k in 0..p-1:
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
        
        U_mat = np.array(U)
        V_mat = np.array(V)
        W_mat = np.array(W)
        
        # We can vectorize this check.
        # But looping is fine for n=2, m=5, p=6 (60 checks).
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_mat[i*m+j, :]
                    v_val = V_mat[j*p+k, :]
                    w_val = W_mat[k*n+i, :]
                    total = np.sum(u_val * v_val * w_val)
                    if not np.isclose(total, 1.0):
                        return False
        return True

    # 1. Establish Baseline (Schoolbook)
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
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb, "rank": rank_sb}
    
    # 2. Search for lower rank
    # We attempt to find an integer decomposition with rank < rank_sb.
    # Given the complexity, we try a few random restarts for ranks slightly lower.
    # If not found, we return the schoolbook.
    
    start_time = time.time()
    time_limit = 110.0 # Safety margin
    
    # Heuristic lower bound: rank >= n*p? No, rank >= max(n*m, m*p, n*p) is a lower bound for bilinear, 
    # but for trilinear tensor rank, it's different. 
    # Known lower bounds for 2x5x6 are around 50-60.
    # We'll try to search downwards from rank_sb - 1 to a reasonable limit, or just random searches.
    
    # To be efficient, we can try to fix the "structure" and just randomize coefficients?
    # No, random integers are sparse.
    
    # Strategy: Randomly initialize U, V, W with small integers (-1, 0, 1) for a target rank r.
    # Check if it works.
    
    min_search_rank = max(1, rank_sb - 20) # Only search a small window to be safe
    found_better = False
    
    for r in range(rank_sb - 1, min_search_rank - 1, -1):
        if time.time() - start_time > time_limit:
            break
            
        attempts = 100 # Limit attempts per rank to avoid hanging
        for _ in range(attempts):
            if time.time() - start_time > time_limit:
                break
            
            # Generate random integer matrices
            # To increase chance, maybe use a mix of 0, 1, -1
            U_rand = [[random.randint(-1, 1) for _ in range(r)] for _ in range(n * m)]
            V_rand = [[random.randint(-1, 1) for _ in range(r)] for _ in range(m * p)]
            W_rand = [[random.randint(-1, 1) for _ in range(r)] for _ in range(p * n)]
            
            if is_valid_decomposition(U_rand, V_rand, W_rand, r):
                best_result = {"U": U_rand, "V": V_rand, "W": W_rand, "rank": r}
                found_better = True
                break
        
        if found_better:
            break
        if time.time() - start_time > time_limit:
            break
            
    return {"U": best_result["U"], "V": best_result["V"], "W": best_result["W"]}
# EVOLVE-BLOCK-END
