import numpy as np
import time
from itertools import product, combinations

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Tensor: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1

    # Strategy: Use the known rank-50 decomposition (6 Strassen 2x2x2 blocks + 2 schoolbook 2x2x1 blocks)
    # Then attempt to reduce rank by finding linear dependencies or merging terms.
    
    # First, build the rank-50 solution
    rank_50 = 50
    U50 = np.zeros((n * m, rank_50), dtype=np.int64)
    V50 = np.zeros((m * p, rank_50), dtype=np.int64)
    W50 = np.zeros((p * n, rank_50), dtype=np.int64)

    col = 0

    def add_strassen_block(i0, j0, k0):
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)

        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)

        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)

        U50[a_idx, col] += 1; U50[d_idx, col] += 1
        V50[e_idx, col] += 1; V50[h_idx, col] += 1
        W50[c00_idx, col] += 1; W50[c11_idx, col] += 1
        col += 1

        U50[c_idx, col] += 1; U50[d_idx, col] += 1
        V50[e_idx, col] += 1
        W50[c10_idx, col] += 1; W50[c11_idx, col] -= 1
        col += 1

        U50[a_idx, col] += 1
        V50[f_idx, col] += 1; V50[h_idx, col] -= 1
        W50[c01_idx, col] += 1; W50[c11_idx, col] += 1
        col += 1

        U50[d_idx, col] += 1
        V50[g_idx, col] += 1; V50[e_idx, col] -= 1
        W50[c00_idx, col] += 1; W50[c10_idx, col] += 1
        col += 1

        U50[a_idx, col] += 1; U50[b_idx, col] += 1
        V50[h_idx, col] += 1
        W50[c00_idx, col] -= 1; W50[c01_idx, col] += 1
        col += 1

        U50[c_idx, col] += 1; U50[a_idx, col] -= 1
        V50[e_idx, col] += 1; V50[f_idx, col] += 1
        W50[c11_idx, col] += 1
        col += 1

        U50[b_idx, col] += 1; U50[d_idx, col] -= 1
        V50[g_idx, col] += 1; V50[h_idx, col] += 1
        W50[c00_idx, col] += 1
        col += 1

    def add_schoolbook_221_block(i0, j0, k0):
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)

        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0

        c00_idx = k0 * n + i0
        c10_idx = k0 * n + (i0 + 1)

        U50[a_idx, col] += 1; V50[e_idx, col] += 1; W50[c00_idx, col] += 1; col += 1
        U50[b_idx, col] += 1; V50[g_idx, col] += 1; W50[c00_idx, col] += 1; col += 1
        U50[c_idx, col] += 1; V50[e_idx, col] += 1; W50[c10_idx, col] += 1; col += 1
        U50[d_idx, col] += 1; V50[g_idx, col] += 1; W50[c10_idx, col] += 1; col += 1

    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            add_strassen_block(0, j_start, k_start)
        add_schoolbook_221_block(0, j_start, 6)

    assert col == rank_50, f"Expected rank 50, got {col}"

    # Verify rank-50 solution
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank_50):
                    s += U50[i*m+j, r] * V50[j*p+k, r] * W50[k*n+i, r]
                if s != 1:
                    raise ValueError(f"Rank-50 decomposition failed at ({i},{j},{k}): got {s}")

    # Now attempt to reduce rank by finding a linear combination of rank-1 terms
    # that still decomposes the tensor. This is equivalent to finding a non-trivial
    # element in the kernel of the linear map from R^rank to the tensor space.
    
    # The tensor space has dimension n*m*p = 56.
    # We have 50 rank-1 terms. If there's a linear dependency among them that
    # doesn't change the tensor decomposition, we can eliminate one term.
    
    # A linear combination sum_r alpha_r * (U[:,r] ⊗ V[:,r] ⊗ W[:,r]) = 0
    # means we can subtract this from any term and still have a valid decomposition.
    
    # But we need to eliminate an entire term, not just modify coefficients.
    # If sum_{r != r0} beta_r * T_r = T_{r0}, then we can replace T_{r0} with the sum.
    # But that doesn't reduce rank unless we can express one term as a linear combination
    # of others in a way that allows us to remove it.
    
    # Actually, the rank of the tensor is the minimum number of rank-1 tensors that sum to it.
    # The 50 terms give one such decomposition. To find a lower-rank decomposition,
    # we need to find a different set of fewer rank-1 tensors.
    
    # This is a hard problem. Let's try a different approach:
    # Use the fact that the tensor is a direct sum of blocks and look for cross-block mergers.
    
    # The tensor M_{2,4,7} can be seen as:
    # For j in {0,1}: M_{2,2,7} block (rank 25 via Strassen + schoolbook)
    # For j in {2,3}: M_{2,2,7} block (rank 25 via Strassen + schoolbook)
    # Total rank: 50.
    
    # To reduce rank, we need to find a decomposition that mixes the two j-blocks.
    # This is equivalent to finding a rank-<50 decomposition of M_{2,4,7}.
    
    # Known lower bound: R(M_{n,m,p}) >= max(n*m, m*p, p*n, (n*p)/(max(n,m,p)))... 
    # For n=2, m=4, p=7: max(8, 28, 14, 14/7=2) = 28.
    # So rank could be as low as 28 in theory.
    
    # Let's try to construct a rank-<50 decomposition by allowing the U, V, W matrices
    # to have entries that are not just 0, ±1.
    
    # Approach: Start with the rank-50 solution and try to eliminate one term by
    # expressing it as a linear combination of the others.
    
    # The tensor equation is:
    # For all (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # If we have a linear dependency: sum_r alpha_r * (U[:,r] ⊗ V[:,r] ⊗ W[:,r]) = 0
    # in the tensor space, then we can modify the decomposition by adding this zero tensor.
    # But to reduce rank, we need one of the alpha_r to be nonzero and then rescale.
    
    # Let's check if the 50 rank-1 tensors are linearly independent in the tensor space.
    # The tensor space has dimension 56, so at most 56 are linearly independent.
    # We have 50, so they could be independent.
    
    # Let's try a random search: pick a random linear combination of the 50 terms,
    # and check if it's zero. If yes, we can use it to eliminate one term.
    
    # Actually, let's try a more systematic approach:
    # Construct the tensor as a matrix of size (n*m) * (m*p) * (p*n) and check
    # the rank of the Gram matrix of the rank-1 tensors.
    
    # But this is getting complex. Let's try a simpler heuristic:
    # Remove one term and see if the remaining 49 can still decompose the tensor
    # by adjusting the other coefficients.
    
    # For each term r0, try to find coefficients for the other 49 terms such that
    # the tensor equation is satisfied. This is a system of 56 equations with 49 unknowns,
    # which is overdetermined. So it's unlikely to have a solution.
    
    # Let's try a different approach: use the Strassen algorithm more cleverly.
    # The current decomposition uses 6 Strassen 2x2x2 blocks (7 ops each) and 2 schoolbook 2x2x1 blocks (4 ops each).
    # Total: 6*7 + 2*4 = 42 + 8 = 50.
    
    # Can we do better by using a different grouping?
    # For example, group the 4 columns into two 2-column blocks, and the 7 rows into blocks of 2,2,2,1.
    # That's what we're doing now.
    
    # Alternative: Use a 2x4x7 decomposition directly.
    # The tensor M_{2,4,7} has rank R. We know R <= 50.
    # Can we find R < 50?
    
    # Let's try to use the fact that M_{2,4,7} can be decomposed as:
    # M_{2,4,7} = M_{2,4,4} ⊕ M_{2,4,3}
    # where ⊕ denotes direct sum.
    # R(M_{2,4,4}) <= ? and R(M_{2,4,3}) <= ?
    
    # For M_{2,4,4}: group j into two 2-blocks, k into two 2-blocks.
    # Each 2x2x2 block has rank 7 via Strassen.
    # Total: 2*2*7 = 28.
    
    # For M_{2,4,3}: group j into two 2-blocks, k into 2+1 blocks.
    # 2x2x2 blocks: 2*1*7 = 14 (for the 2x2x2 part)
    # 2x2x1 blocks: 2*1*4 = 8 (for the 2x2x1 part)
    # Total: 22.
    
    # Grand total: 28 + 22 = 50. Same as before.
    
    # So the block decomposition always gives 50. To do better, we need to find
    # cross-block mergers.
    
    # Let's try a random optimization approach:
    # Start with the rank-50 solution and try to eliminate one term by finding
    # a linear combination of the other 49 that approximates the removed term.
    
    # If we can find such a combination exactly, we've reduced the rank.
    
    # Let's try this for each term.
    
    U = U50.copy()
    V = V50.copy()
    W = W50.copy()
    
    best_rank = rank_50
    best_U, best_V, best_W = U50.copy(), V50.copy(), W50.copy()
    
    # Try to eliminate each term
    for r0 in range(rank_50):
        if time.time() > deadline - 20:
            break
        
        # Remove term r0
        U_r0 = U50[:, r0].copy()
        V_r0 = V50[:, r0].copy()
        W_r0 = W50[:, r0].copy()
        
        # Check if U_r0, V_r0, W_r0 are all zero
        if np.all(U_r0 == 0) or np.all(V_r0 == 0) or np.all(W_r0 == 0):
            # This term is redundant
            U_new = np.delete(U50, r0, axis=1)
            V_new = np.delete(V50, r0, axis=1)
            W_new = np.delete(W50, r0, axis=1)
            best_rank = rank_50 - 1
            best_U, best_V, best_W = U_new, V_new, W_new
            break
        
        # Try to express U_r0 ⊗ V_r0 ⊗ W_r0 as a linear combination of the other 49 terms
        # This is equivalent to solving for alpha in R^49 such that:
        # sum_{r != r0} alpha_r * (U[:,r] ⊗ V[:,r] ⊗ W[:,r]) = U_r0 ⊗ V_r0 ⊗ W_r0
        
        # This is a system of 56 equations with 49 unknowns.
        # Let's set it up and solve it.
        
        # Build the tensor for the right-hand side
        rhs = np.zeros((n*m, m*p, p*n))
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    rhs[a, b, c] = U_r0[a] * V_r0[b] * W_r0[c]
        
        # Build the matrix A where A[:, r] is the vectorized form of U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        A = np.zeros((n*m * m*p * p*n, rank_50 - 1))
        idx = 0
        for r in range(rank_50):
            if r == r0:
                continue
            for a in range(n*m):
                for b in range(m*p):
                    for c in range(p*n):
                        A[idx, idx // (n*m * m*p * p*n)] = U50[a, r] * V50[b, r] * W50[c, r]
                        idx += 1
        
        # Actually, let's vectorize this more efficiently
        A = np.zeros((n*m * m*p * p*n, rank_50 - 1))
        col_idx = 0
        for r in range(rank_50):
            if r == r0:
                continue
            tensor_r = np.outer(np.outer(U50[:, r], V50[:, r]), W50[:, r])
            A[:, col_idx] = tensor_r.flatten()
            col_idx += 1
        
        rhs_vec = rhs.flatten()
        
        # Solve A * alpha = rhs_vec using least squares
        try:
            alpha, residuals, rank_A, singular_values = np.linalg.lstsq(A, rhs_vec, rcond=None)
            
            # Check if the solution is exact
            if np.allclose(A @ alpha, rhs_vec, atol=1e-10):
                # We can eliminate term r0
                U_new = np.delete(U50, r0, axis=1)
                V_new = np.delete(V50, r0, axis=1)
                W_new = np.delete(W50, r0, axis=1)
                
                # Verify the new decomposition
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for r in range(rank_50 - 1):
                                s += U_new[i*m+j, r] * V_new[j*p+k, r] * W_new[k*n+i, r]
                            if s != 1:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    best_rank = rank_50 - 1
                    best_U, best_V, best_W = U_new, V_new, W_new
                    break
        except np.linalg.LinAlgError:
            pass
    
    # Verify the best solution
    if best_rank < rank_50:
        U, V, W = best_U, best_V, best_W
    else:
        U, V, W = U50, V50, W50
    
    # Final verification
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(U.shape[1]):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if s != 1:
                    raise ValueError(f"Decomposition failed at ({i},{j},{k}): got {s}")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
