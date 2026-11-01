import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin for 150s limit
    
    n, m, p = 2, 2, 11
    N, M, P = n * m, m * p, p * n  # 4, 22, 22
    
    # 1. Initialize with Schoolbook Decomposition (Rank 44)
    rank = N * M * P
    U = np.zeros((N, rank))
    V = np.zeros((M, rank))
    W = np.zeros((P, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    def to_list(mat):
        # Convert numpy array to list of lists of floats/ints
        # The interface expects integer/half-integer entries.
        # We will round to nearest 0.5 if close, else keep float.
        # However, the problem says "integer/half-integer entries".
        # We should ensure our search produces these.
        # For now, return floats. The evaluator likely checks exactness.
        # If we find a solution, it will be exact rational/integer.
        return mat.tolist()
        
    def check_validity(U, V, W, rank):
        # Verify T_UVW = sum r (U_r outer V_r outer W_r) matches Target
        # Target T[i, j, k] = 1 if i,j,k correspond to matmul
        # T[i, j, k] = 1 if i = i1+m*i2, j = m*i2+k, k = k*n+i1? 
        # Wait, indices:
        # A is n x m. B is m x p. C is n x p.
        # C[i, k] = sum_j A[i, j] B[j, k]
        # Tensor indices: i (row of A/C), j (col of A/row of B), k (col of B/C)
        # U shape (n*m, rank). Index i*m + j? No, usually U is (n*m, rank) for A's entries.
        # V shape (m*p, rank) for B's entries.
        # W shape (p*n, rank) for C's entries.
        # Contract: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # We can verify by checking a few random points or the full tensor.
        # Full tensor is 4*22*22 = 1936 entries.
        # Rank is ~40.
        # 1936 * 40 * 3 ops ~ 200k ops. Very fast.
        
        T_target = np.zeros((N, M, P))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i*m+j, j*p+k, k*n+i] = 1.0
                    
        # Compute T_decomp
        # T_decomp[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
        # This is a 3-way contraction.
        # Can do: (U * V[:, None, :]) @ W[None, :, :] ? No.
        # Easier: for each r, add np.outer(np.outer(U[:,r], V[:,r]), W[:,r])
        # Or use einsum: np.einsum('ir,jr,kr->ijk', U, V, W)
        
        T_decomp = np.einsum('ir,jr,kr->ijk', U, V, W)
        
        # Check exactness (with small tolerance for float errors)
        return np.allclose(T_decomp, T_target, atol=1e-9)

    # Current best
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # 2. Iterative Rank Reduction Search
    # Strategy: Try to remove one column at a time.
    # If we remove column r, we need to adjust remaining columns to compensate.
    # This is equivalent to finding a vector x such that:
    # sum_{s != r} x_s (U_s outer V_s outer W_s) = U_r outer V_r outer W_r
    # This is a linear system in x for the entries of the tensor.
    
    # To make it faster, we can try "merging" two columns into one?
    # Or simply try to drop a column and solve for the rest?
    # Dropping a column reduces rank by 1.
    # If we have rank R, we have R vectors.
    # If we drop one, we have R-1 vectors.
    # The target tensor T is fixed.
    # We want T = sum_{s=1}^{R-1} c_s (U_s outer V_s outer W_s)
    # This is a linear system A x = b, where A is the matrix of flattened tensors of the remaining terms.
    # Size of A: (N*M*P) x (R-1).
    # N*M*P = 1936. R ~ 40.
    # Solving a 1936x39 linear system is fast.
    
    # However, the U, V, W themselves might need to change?
    # No, if we fix the *structure* (the U, V, W vectors) and only change coefficients,
    # we are looking for a linear dependency in the span of the current terms.
    # If the current set of R terms is linearly independent in the tensor space,
    # we cannot drop one without changing the U, V, W vectors themselves.
    
    # So, simple "drop and solve" only works if the current decomposition has linear dependencies.
    # Schoolbook has no linear dependencies (each term corresponds to a unique (i,j,k) entry).
    # So we need to *modify* the U, V, W vectors to create dependencies, or use a different search.
    
    # Alternative: Randomized Search / Local Optimization
    # 1. Start with a valid decomposition (e.g., Strassen-based or Schoolbook).
    # 2. Pick two random columns r1, r2.
    # 3. Try to replace them with a single column r_new (rank - 1) or modify them to allow dropping one.
    
    # Let's try a "Perturbation and Projection" approach.
    # 1. Start with Schoolbook (Rank 44).
    # 2. Randomly perturb U, V, W slightly.
    # 3. Project back to the manifold of decompositions? No, that's hard.
    
    # Let's try the "Linear System for Merging" approach more directly.
    # Suppose we want to merge term r1 and r2 into a new term r_new.
    # U_r1 outer V_r1 outer W_r1 + U_r2 outer V_r2 outer W_r2 = U_new outer V_new outer W_new
    # This is generally not possible for arbitrary terms.
    
    # Better approach: **Greedy Linear Dependency Search**
    # 1. Generate a large set of candidate rank-1 tensors (e.g., from Strassen, Winograd, or random).
    # 2. Use Gaussian Elimination to find the minimum number of these tensors that span the target tensor.
    # This is the "Dictionary Learning" or "Sparse Coding" approach for tensor rank.
    
    # Step 2.1: Build a dictionary of candidate rank-1 tensors.
    # Sources:
    # a) Schoolbook terms (44 of them).
    # b) Strassen terms (if we can construct them for this shape).
    # c) Random rank-1 tensors with small integer/half-integer entries.
    
    # Since p=11 is large, Strassen is tricky to apply directly without padding.
    # But we can use random terms.
    
    # Let's generate a dictionary of size D.
    # D should be large enough to likely contain the minimal basis.
    # Rank is likely around 30-40.
    # Let's generate D = 200 random terms.
    
    def generate_random_term(seed):
        rng = np.random.RandomState(seed)
        # Entries in {-1, 0, 1} or {-0.5, 0, 0.5, 1}
        # Let's use integers for simplicity, scale later if needed.
        u = rng.choice([-1, 0, 1], size=N)
        v = rng.choice([-1, 0, 1], size=M)
        w = rng.choice([-1, 0, 1], size=P)
        # Flatten
        return u, v, w

    # Build Dictionary
    dict_size = 300
    U_dict = np.zeros((N, dict_size))
    V_dict = np.zeros((M, dict_size))
    W_dict = np.zeros((P, dict_size))
    
    # Add Schoolbook terms
    for col in range(N * M * P):
        if col >= dict_size: break
        U_dict[:, col] = best_U[:, col]
        V_dict[:, col] = best_V[:, col]
        W_dict[:, col] = best_W[:, col]
        
    # Add Random terms
    for i in range(N * M * P, dict_size):
        u, v, w = generate_random_term(i)
        U_dict[:, i] = u
        V_dict[:, i] = v
        W_dict[:, i] = w
        
    # Flatten the dictionary terms into a matrix A
    # A has shape (N*M*P, dict_size)
    # A[:, r] = flatten(U_dict[:, r] outer V_dict[:, r] outer W_dict[:, r])
    
    # Efficient construction:
    # T_r[i, j, k] = U_dict[i, r] * V_dict[j, r] * W_dict[k, r]
    # We can use einsum to get a 4D array then reshape, or loop.
    # Loop is fine for 300 terms.
    
    A = np.zeros((N * M * P, dict_size))
    for r in range(dict_size):
        # Create outer product
        # u outer v outer w
        # Shape (N, M, P)
        t = np.einsum('i,j,k->ijk', U_dict[:, r], V_dict[:, r], W_dict[:, r])
        A[:, r] = t.flatten()
        
    # Target tensor T
    T_target = np.zeros((N, M, P))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0
    b = T_target.flatten()
    
    # Solve A x = b for sparse x
    # We want the minimum number of non-zero entries in x.
    # This is a sparse recovery problem.
    # We can use Greedy Forward Selection (Orthogonal Matching Pursuit style) or L1 minimization.
    # Given time constraints, Greedy Forward Selection is robust.
    
    # Greedy Forward Selection:
    # 1. Start with empty set S.
    # 2. At each step, find the column a_j in A that best correlates with the residual r.
    # 3. Add j to S.
    # 4. Solve least squares for coefficients in S.
    # 5. Update residual.
    # 6. Stop when residual is zero (or below tol).
    
    S = []
    residual = b.copy()
    max_iter = 50  # We expect rank < 50
    
    for _ in range(max_iter):
        if time.time() > deadline - 10:
            break
            
        if np.allclose(residual, 0, atol=1e-9):
            break
            
        # Find best column
        # corr = A.T @ residual
        # But we need to account for the fact that we will re-solve coefficients.
        # Simple heuristic: Max absolute correlation.
        corr = np.abs(A.T @ residual)
        
        # Exclude already selected columns
        for j in S:
            corr[j] = 0
            
        best_j = np.argmax(corr)
        
        if corr[best_j] < 1e-12:
            # No good column found
            break
            
        S.append(best_j)
        
        # Solve least squares for A[:, S] x = b
        A_S = A[:, S]
        try:
            x_S, residuals, rank_ls, sv = np.linalg.lstsq(A_S, b, rcond=None)
        except:
            break
            
        residual = b - A_S @ x_S
        
        # Check if exact
        if np.allclose(residual, 0, atol=1e-9):
            # We found a decomposition of rank len(S)
            # Construct U, V, W from the selected columns and coefficients
            # Note: The coefficients x_S are scalars.
            # We can absorb them into U, V, or W.
            # Let's absorb into U.
            
            new_rank = len(S)
            U_new = np.zeros((N, new_rank))
            V_new = np.zeros((M, new_rank))
            W_new = np.zeros((P, new_rank))
            
            for idx, j in enumerate(S):
                U_new[:, idx] = U_dict[:, j] * x_S[idx]
                V_new[:, idx] = V_dict[:, j]
                W_new[:, idx] = W_dict[:, j]
                
            # Verify
            if check_validity(U_new, V_new, W_new, new_rank):
                # Update best
                if new_rank < best_rank:
                    best_rank = new_rank
                    best_U, best_V, best_W = U_new, V_new, W_new
                return {
                    "U": to_list(U_new),
                    "V": to_list(V_new),
                    "W": to_list(W_new)
                }
                
    # If we didn't find an exact solution via greedy, 
    # maybe the dictionary wasn't rich enough or greedy got stuck.
    # Fallback to the initial schoolbook (or best found so far)
    
    # If we haven't returned, we return the best known valid decomposition.
    # The schoolbook is always valid.
    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
