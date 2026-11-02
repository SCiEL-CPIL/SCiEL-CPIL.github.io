import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin for 150s limit
    
    n, m, p = 2, 3, 9
    
    # Build the rank-50 decomposition
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    u_a, u_b, u_c, u_d = 0, 1, 3, 4
    
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        v_e = k0
        v_f = k1
        v_g = p + k0
        v_h = p + k1
        
        U[u_a][col] += 1
        U[u_d][col] += 1
        V[v_e][col] += 1
        V[v_h][col] += 1
        
        U[u_c][col+1] += 1
        U[u_d][col+1] += 1
        V[v_e][col+1] += 1
        
        U[u_a][col+2] += 1
        V[v_f][col+2] += 1
        V[v_h][col+2] -= 1
        
        U[u_d][col+3] += 1
        V[v_g][col+3] += 1
        V[v_e][col+3] -= 1
        
        U[u_a][col+4] += 1
        U[u_b][col+4] += 1
        V[v_h][col+4] += 1
        
        U[u_c][col+5] += 1
        U[u_a][col+5] -= 1
        V[v_e][col+5] += 1
        V[v_f][col+5] += 1
        
        U[u_b][col+6] += 1
        U[u_d][col+6] -= 1
        V[v_g][col+6] += 1
        V[v_h][col+6] += 1
        
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        W[w_00][col] += 1
        W[w_00][col+3] += 1
        W[w_00][col+4] -= 1
        W[w_00][col+6] += 1
        
        W[w_01][col+2] += 1
        W[w_01][col+4] += 1
        
        W[w_10][col+1] += 1
        W[w_10][col+3] += 1
        
        W[w_11][col] += 1
        W[w_11][col+1] -= 1
        W[w_11][col+2] += 1
        W[w_11][col+5] += 1
        
        col += 7
    
    k8 = 8
    v_e = k8
    v_g = p + k8
    
    U[u_a][col] += 1
    V[v_e][col] += 1
    U[u_b][col+1] += 1
    V[v_g][col+1] += 1
    U[u_c][col+2] += 1
    V[v_e][col+2] += 1
    U[u_d][col+3] += 1
    V[v_g][col+3] += 1
    
    w_08 = k8 * n + 0
    w_18 = k8 * n + 1
    
    W[w_08][col] += 1
    W[w_08][col+1] += 1
    W[w_18][col+2] += 1
    W[w_18][col+3] += 1
    
    col += 4
    
    u_ar0, u_ar1 = 2, 5
    for k in range(9):
        v_b = 2 * p + k
        U[u_ar0][col] += 1
        V[v_b][col] += 1
        W[k * n + 0][col] += 1
        U[u_ar1][col+1] += 1
        V[v_b][col+1] += 1
        W[k * n + 1][col+1] += 1
        col += 2
    
    # Convert to numpy
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build the target tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n*m):
        ai = i // m
        am = i % m
        for j in range(m*p):
            bj = j // p
            bm = j % p
            if am == bj:
                k = bm * n + ai
                T[i, j, k] = 1.0
    
    # Compute the full decomposition tensor efficiently
    decomp_full = np.einsum('ir,jr,kr->ijk', U_np, V_np, W_np)
    
    # Verify the full decomposition is correct
    if not np.allclose(decomp_full, T, atol=1e-10):
        return {"U": U, "V": V, "W": W}
    
    # Strategy 1: Try the (3,3,3) Strassen embedding approach
    # Embed 2x3 matrix A into 3x3 matrix A_ext where row3 = row1 + row2
    # Embed 3x9 matrix B as three 3x3 blocks
    # The product A_ext * B_ext gives us info about A*B
    # Strassen rank-7 for 3x3x3 might give us a lower-rank decomposition
    
    # Actually, let's try a more direct approach: systematic null-space search
    # with better pruning using incremental residual computation
    
    best_rank = rank
    best_U = U_np.copy()
    best_V = V_np.copy()
    best_W = W_np.copy()
    
    try:
        # Build the coefficient matrix for null space analysis
        # Each column r contributes U[:,r] ⊗ V[:,r] ⊗ W[:,r] to the tensor
        # We can use a smaller representation: for each (i,j,k) entry,
        # the contribution is U[i,r]*V[j,r]*W[k,r]
        
        # Instead of building the full 2916x50 matrix, use a smarter approach:
        # For each column r, compute its "fingerprint" as the set of nonzero
        # (i,j,k) entries and their values.
        
        # Build sparse representation: for each r, store which (i,j,k) are nonzero
        # and their values
        nonzero_entries = []  # list of (r, [(i,j,k, val), ...])
        for r in range(rank):
            u_r = U_np[:, r]
            v_r = V_np[:, r]
            w_r = W_np[:, r]
            nz = []
            for i in range(n*m):
                if u_r[i] == 0:
                    continue
                for j in range(m*p):
                    if v_r[j] == 0:
                        continue
                    for k in range(p*n):
                        if w_r[k] == 0:
                            continue
                        val = u_r[i] * v_r[j] * w_r[k]
                        if val != 0:
                            nz.append((i, j, k, val))
            nonzero_entries.append(nz)
        
        # Now try to find columns that can be removed
        # A column r can be removed if its contribution is a linear combination
        # of other columns' contributions
        
        # Use a greedy approach: try removing each column and check if
        # the remaining columns still span the tensor
        
        # More efficient: compute the rank of the full set, then try
        # removing columns one at a time
        
        # Build a matrix where each row is an (i,j,k) entry and each column
        # is a term r. The entry is U[i,r]*V[j,r]*W[k,r].
        # But this is 2916 x 50 which is manageable.
        
        # Actually, let's be smarter. The tensor T has a specific structure:
        # T[i,j,k] = 1 iff i%m == j%p and k = (j%p)*n + (i//m)
        
        # Let's enumerate the support of T
        T_support = []
        for i in range(n*m):
            ai = i // m
            am = i % m
            for j in range(m*p):
                bj = j // p
                bm = j % p
                if am == bj:
                    k = bm * n + ai
                    T_support.append((i, j, k))
        
        # For each column r, compute its projection onto the T support
        # proj_r[i,j,k] = U[i,r]*V[j,r]*W[k,r] for (i,j,k) in T_support
        
        # Build matrix M where M[t, r] = U[i_t,r]*V[j_t,r]*W[k_t,r]
        # for the t-th support element (i_t, j_t, k_t)
        n_support = len(T_support)
        M = np.zeros((n_support, rank))
        for t, (i, j, k) in enumerate(T_support):
            for r in range(rank):
                M[t, r] = U_np[i, r] * V_np[j, r] * W_np[k, r]
        
        # Now we need to find the minimum subset of columns that still
        # spans the same space. This is equivalent to finding the rank
        # of M and then finding a basis.
        
        # Compute SVD of M
        _, s, Vt = np.linalg.svd(M, full_matrices=False)
        
        # Numerical rank
        tol = s[0] * rank * 1e-10
        num_rank = np.sum(s > tol)
        
        if num_rank < rank:
            # There are dependencies. The null space has dimension rank - num_rank
            null_dim = rank - num_rank
            
            # Get null space vectors
            null_vecs = Vt[num_rank:]  # shape (null_dim, rank)
            
            # For each null vector, find columns with large coefficients
            # and try removing them
            
            best_residual = float('inf')
            best_mask = np.ones(rank, dtype=bool)
            
            for null_idx in range(null_dim):
                if time.time() > deadline - 30:
                    break
                
                null_vec = null_vecs[null_idx]
                abs_null = np.abs(null_vec)
                
                # Try removing the column with the largest null space coefficient
                # and also try combinations of top-2, top-3 columns
                
                # Single column removal
                candidates = np.argsort(abs_null)[::-1]
                
                for drop_col in candidates[:5]:  # Try top 5
                    if time.time() > deadline - 20:
                        break
                    
                    mask = np.ones(rank, dtype=bool)
                    mask[drop_col] = False
                    U_new = U_np[:, mask]
                    V_new = V_np[:, mask]
                    W_new = W_np[:, mask]
                    
                    decomp_new = np.einsum('ir,jr,kr->ijk', U_new, V_new, W_new)
                    residual = np.sum((decomp_new - T)**2)
                    
                    if residual < best_residual:
                        best_residual = residual
                        best_mask = mask.copy()
                    
                    if residual < 1e-8:
                        break
                
                if best_residual < 1e-8:
                    break
            
            # Try pairs of columns
            if best_residual >= 1e-8:
                for null_idx in range(null_dim):
                    if time.time() > deadline - 15:
                        break
                    
                    null_vec = null_vecs[null_idx]
                    abs_null = np.abs(null_vec)
                    candidates = np.argsort(abs_null)[::-1]
                    
                    for i in range(min(4, len(candidates))):
                        for j in range(i+1, min(4, len(candidates))):
                            if time.time() > deadline - 10:
                                break
                            
                            drop1 = candidates[i]
                            drop2 = candidates[j]
                            
                            mask = np.ones(rank, dtype=bool)
                            mask[drop1] = False
                            mask[drop2] = False
                            U_new = U_np[:, mask]
                            V_new = V_np[:, mask]
                            W_new = W_np[:, mask]
                            
                            decomp_new = np.einsum('ir,jr,kr->ijk', U_new, V_new, W_new)
                            residual = np.sum((decomp_new - T)**2)
                            
                            if residual < best_residual:
                                best_residual = residual
                                best_mask = mask.copy()
                            
                            if residual < 1e-8:
                                break
                        
                        if best_residual < 1e-8:
                            break
                    
                    if best_residual < 1e-8:
                        break
            
            if best_residual < 1e-8 and best_mask.sum() < best_rank:
                best_rank = int(best_mask.sum())
                best_U = U_np[:, best_mask]
                best_V = V_np[:, best_mask]
                best_W = W_np[:, best_mask]
    except Exception as e:
        pass
    
    # Strategy 2: Try a direct (3,3,3) Strassen-based construction
    # This is a different approach: embed into 3x3x3 and use Strassen rank 7
    # per block, but with sharing due to the linear constraint
    
    try:
        if time.time() < deadline - 20:
            # The 2x3 matrix A can be written as:
            # A = [a11 a12 a13; a21 a22 a23]
            # Embed into 3x3: A_ext = [a11 a12 a13; a21 a22 a23; a11+a21 a12+a22 a13+a23]
            
            # The 3x9 matrix B has 3 rows, 9 columns. Think of it as 3 blocks of 3x3:
            # B = [B0; B1; B2] where each Bi is 1x3? No, B is 3x9.
            # Actually B is 3 rows, 9 columns. Let's think of it as 3 column-blocks of 3x3.
            
            # The product C = A*B is 2x9.
            # C[i,k] = sum_m A[i,m] * B[m,k]
            
            # Using the embedding: A_ext is 3x3, and we can form a 3x9 matrix
            # B_ext by repeating or combining B.
            
            # Actually, a cleaner approach: think of the (2,3,9) tensor as
            # three (2,3,3) tensors (one for each group of 3 output columns).
            # Each (2,3,3) tensor has rank at most 6 (trivially) or less.
            
            # For a 2x3x3 tensor, the trivial rank is 3 (one per output column
            # with 3 terms each... no). Let me think again.
            
            # T[i,j,k] for 2x3x3: i in 0..5, j in 0..8, k in 0..5
            # T[i,j,k] = 1 if i%3 == j%3 and k = (j%3)*2 + (i//3)
            
            # This is a 2x3x3 matrix multiplication tensor.
            # Its rank is known to be... let's compute it.
            
            # For 2x3x3, the trivial rank is 3 (sum over m=0,1,2 of A[i,m]*B[m,k])
            # But we can do better with bilinear algorithms.
            
            # Actually, for multiplying 2x3 by 3x3, the rank is 7 (I believe).
            # Let me try to construct it.
            
            # Hmm, this is getting complex. Let's stick with the null-space approach
            # but improve it.
            pass
    except Exception:
        pass
    
    # Return the best result found
    if best_rank < rank:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    
    # Fall back to rank-50
    return {"U": U, "V": V, "W": W}
