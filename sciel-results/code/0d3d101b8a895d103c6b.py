import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = 3, 4, 6
    
    def build_tensor_vector(U_row, V_row, W_row):
        # Returns the vectorized tensor for a single rank-1 term
        # U_row shape: (n*m,), V_row shape: (m*p,), W_row shape: (p*n,)
        # Tensor T[i, k, l] = U[i] * V[k] * W[l]
        # We need to return a flat vector of size (n*m * m*p * p*n)
        # But we only care about the non-zero entries?
        # No, for linear algebra we need the full vector or a consistent subset.
        # The target tensor has non-zeros only at consistent indices.
        # However, the decomposition must match the target EXACTLY.
        # The target T has 1s at (i, k, l) where i%m == k//p, etc.
        # If we use the full vector space, the rank might be higher due to "ghost" terms?
        # No, if U, V, W are sparse in the right places, the product will be zero elsewhere.
        # In the Strassen/Schoolbook constructions, the entries are 0 or 1 at specific indices.
        # So the product will be 0 at inconsistent indices.
        # So we can just use the full vector.
        
        T = np.einsum('i,k,l->ikl', U_row, V_row, W_row)
        return T.flatten()

    def get_strassen_66():
        # Reconstruct the 66 rank solution from the seed
        rank = 66
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        col = 0

        def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
            nonlocal col
            A00 = a_r * m + a_c
            A01 = a_r * m + a_c + 1
            A10 = (a_r + 1) * m + a_c
            A11 = (a_r + 1) * m + a_c + 1

            B00 = b_r * p + b_c
            B01 = b_r * p + b_c + 1
            B10 = (b_r + 1) * p + b_c
            B11 = (b_r + 1) * p + b_c + 1

            C00 = c_c * n + c_r
            C01 = (c_c + 1) * n + c_r
            C10 = c_c * n + (c_r + 1)
            C11 = (c_c + 1) * n + (c_r + 1)

            U[A00][col] += 1; U[A11][col] += 1
            V[B00][col] += 1; V[B11][col] += 1
            W[C00][col] += 1; W[C11][col] += 1
            col += 1
            U[A10][col] += 1; U[A11][col] += 1
            V[B00][col] += 1
            W[C10][col] += 1; W[C11][col] -= 1
            col += 1
            U[A00][col] += 1
            V[B01][col] += 1; V[B11][col] -= 1
            W[C01][col] += 1; W[C11][col] += 1
            col += 1
            U[A11][col] += 1
            V[B10][col] += 1; V[B00][col] -= 1
            W[C00][col] += 1; W[C10][col] += 1
            col += 1
            U[A00][col] += 1; U[A01][col] += 1
            V[B11][col] += 1
            W[C00][col] -= 1; W[C01][col] += 1
            col += 1
            U[A10][col] += 1; U[A00][col] -= 1
            V[B00][col] += 1; V[B01][col] += 1
            W[C11][col] += 1
            col += 1
            U[A01][col] += 1; U[A11][col] -= 1
            V[B10][col] += 1; V[B11][col] += 1
            W[C00][col] += 1
            col += 1

        def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
            nonlocal col
            A0 = a_r * m + a_c
            A1 = a_r * m + a_c + 1

            B00 = b_r * p + b_c
            B01 = b_r * p + b_c + 1
            B10 = (b_r + 1) * p + b_c
            B11 = (b_r + 1) * p + b_c + 1

            C0 = c_c * n + c_r
            C1 = (c_c + 1) * n + c_r

            U[A0][col] += 1; V[B00][col] += 1; W[C0][col] += 1
            col += 1
            U[A1][col] += 1; V[B10][col] += 1; W[C0][col] += 1
            col += 1
            U[A0][col] += 1; V[B01][col] += 1; W[C1][col] += 1
            col += 1
            U[A1][col] += 1; V[B11][col] += 1; W[C1][col] += 1
            col += 1

        add_1x2_2x2(0, 0, 0, 0, 0, 0)
        add_1x2_2x2(0, 2, 2, 0, 0, 0)
        add_1x2_2x2(0, 0, 0, 2, 0, 2)
        add_1x2_2x2(0, 2, 2, 2, 0, 2)
        add_1x2_2x2(0, 0, 0, 4, 0, 4)
        add_1x2_2x2(0, 2, 2, 4, 0, 4)

        add_strassen(1, 0, 0, 0, 1, 0)
        add_strassen(1, 2, 2, 0, 1, 0)
        add_strassen(1, 0, 0, 2, 1, 2)
        add_strassen(1, 2, 2, 2, 1, 2)
        add_strassen(1, 0, 0, 4, 1, 4)
        add_strassen(1, 2, 2, 4, 1, 4)
        
        return U, V, W

    def verify(U, V, W):
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        
        # Compute reconstructed tensor
        R = np.einsum('ic,kc,lc->ikl', Ua, Va, Wa)
        
        # Target tensor
        target = np.zeros((n*m, m*p, p*n), dtype=float)
        for c_r in range(n):
            for b in range(m):
                for c_c in range(p):
                    i = c_r * m + b
                    k = b * p + c_c
                    l = c_c * n + c_r
                    target[i, k, l] = 1.0
                    
        if not np.allclose(R, target, atol=1e-9):
            return False
        return True

    def prune_decomposition(U, V, W, max_iter=50):
        rank = len(U[0])
        if rank == 0:
            return U, V, W, 0
        
        current_U, current_V, current_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
        current_rank = rank
        
        # Precompute tensor vectors for all terms
        # T_c is a vector of size (n*m * m*p * p*n)
        # We only need to check consistency on the "support" of the target tensor?
        # No, the decomposition must be exact.
        # However, if we restrict to the support of the target tensor, we might find
        # dependencies that don't hold globally?
        # If the factors are structured such that they are zero outside the support,
        # then checking on the support is sufficient.
        # In our constructions, U, V, W have 0s and 1s.
        # The product will be 0 outside the "consistent" indices.
        # So we can project onto the consistent indices to reduce dimensionality.
        
        support_indices = []
        for c_r in range(n):
            for b in range(m):
                for c_c in range(p):
                    i = c_r * m + b
                    k = b * p + c_c
                    l = c_c * n + c_r
                    support_indices.append((i, k, l))
        
        num_support = len(support_indices)
        
        def get_projected_vector(U_row, V_row, W_row):
            vec = np.zeros(num_support)
            for idx, (i, k, l) in enumerate(support_indices):
                vec[idx] = U_row[i] * V_row[k] * W_row[l]
            return vec

        # Store projected vectors
        vectors = []
        for c in range(current_rank):
            vectors.append(get_projected_vector(current_U[:, c], current_V[:, c], current_W[:, c]))
        
        M = np.array(vectors) # Shape (current_rank, num_support)
        
        for _ in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            if current_rank <= 0:
                break
                
            # Compute rank of M
            try:
                s = np.linalg.svd(M, compute_uv=False)
            except np.linalg.LinAlgError:
                break
                
            if len(s) == 0:
                break
                
            tol = 1e-9 * max(s[0], 1.0)
            num_rank = int(np.sum(s > tol))
            
            if num_rank >= current_rank:
                break
                
            # Try to remove a row
            removed_any = False
            candidates = list(range(current_rank))
            random.shuffle(candidates)
            
            for rem_idx in candidates:
                if time.time() > deadline - 5.0:
                    break
                    
                # Remove row rem_idx
                # Check if rank of M without rem_idx is still num_rank
                # M_new = M without row rem_idx
                # This is expensive to rebuild SVD every time.
                # Optimization: Use the fact that we just need to check if row rem_idx
                # is in the span of the others.
                # This is equivalent to checking if the coefficient in the RREF is non-zero?
                # Or simpler: just try it.
                
                M_new = np.delete(M, rem_idx, axis=0)
                
                if M_new.shape[0] == 0:
                    if num_rank == 0:
                        # Remove the term
                        current_U.pop(rem_idx)
                        current_V.pop(rem_idx)
                        current_W.pop(rem_idx)
                        current_rank -= 1
                        M = np.array(vectors[:rem_idx] + vectors[rem_idx+1:])
                        removed_any = True
                        break
                    else:
                        continue

                try:
                    s_new = np.linalg.svd(M_new, compute_uv=False)
                    tol_new = 1e-9 * max(s_new[0], 1.0) if len(s_new) > 0 else 0
                    num_rank_new = int(np.sum(s_new > tol_new))
                except:
                    continue
                    
                if num_rank_new == num_rank:
                    # Row rem_idx was redundant
                    current_U.pop(rem_idx)
                    current_V.pop(rem_idx)
                    current_W.pop(rem_idx)
                    current_rank -= 1
                    vectors.pop(rem_idx)
                    M = np.array(vectors) if vectors else np.zeros((0, num_support))
                    removed_any = True
                    break
                    
            if not removed_any:
                break
                
        return current_U, current_V, current_W, current_rank

    best_U, best_V, best_W, best_rank = None, None, None, float('inf')
    
    # Generate candidates
    candidates = []
    
    # 1. The 66-rank solution
    U66, V66, W66 = get_strassen_66()
    candidates.append((U66, V66, W66))
    
    # Prune the candidate
    for U, V, W in candidates:
        if time.time() > deadline - 10.0:
            break
            
        U_p, V_p, W_p, r_p = prune_decomposition(U, V, W)
        
        # Verify the pruned decomposition
        if verify(U_p, V_p, W_p):
            if r_p < best_rank:
                best_rank = r_p
                best_U, best_V, best_W = U_p, V_p, W_p
        else:
            # If pruning broke it, fall back to original
            if verify(U, V, W):
                r_orig = len(U[0])
                if r_orig < best_rank:
                    best_rank = r_orig
                    best_U, best_V, best_W = U, V, W

    # If we didn't find anything better than 66, return 66
    if best_U is None:
        best_U, best_V, best_W = get_strassen_66()
        best_rank = 66

    return {"U": best_U, "V": best_V, "W": best_W}
