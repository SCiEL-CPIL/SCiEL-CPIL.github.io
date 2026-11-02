# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # Leave 5s safety margin
    
    n, m, p = 2, 2, 7
    
    # Target rank to search for
    target_rank = 24
    
    # Build the target tensor T[i, j, k]
    # i: 0..3 (A index), j: 0..13 (B index), k: 0..13 (C index)
    # T[i, j, k] = 1 if A[i] * B[j] contributes to C[k], else 0
    # A is 2x2, indices (r_a, c_a) -> i = r_a * 2 + c_a
    # B is 2x7, indices (r_b, c_b) -> j = r_b * 7 + c_b
    # C is 2x7, indices (r_c, c_c) -> k = r_c * 7 + c_c  (Wait, convention W[k*n+i,r] means C is p x n? No, C is n x p? 
    # Task says: "output factor follows the official trace convention W[k*n+i,r]"
    # Usually C is n x p. Here n=2, p=7. So C is 2x7.
    # k index in W corresponds to C index.
    # C[r_c, c_c] = sum_t A[r_c, t] B[t, c_c].
    # i corresponds to A[r_a, c_a]. j corresponds to B[r_b, c_b].
    # Term exists if r_a == r_c, c_a == r_b, c_b == c_c.
    # So T[i, j, k] = 1 if:
    # r_a = i // 2, c_a = i % 2
    # r_b = j // 7, c_b = j % 7
    # r_c = k // 7, c_c = k % 7
    # r_a == r_c AND c_a == r_b AND c_b == c_c
    
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n*m):
        r_a, c_a = divmod(i, m)
        for j in range(m*p):
            r_b, c_b = divmod(j, p)
            if c_a == r_b:
                r_c = r_a
                c_c = c_b
                k = r_c * n + c_c # Wait, W index is k*n+i? No, W is (p*n, rank).
                # The task says W[k*n+i, r]. This implies the first index of W is k*n+i.
                # But usually the third index of the tensor corresponds to the output index.
                # If W is (p*n, rank), the index is 0..13.
                # Let's check the convention again.
                # "output factor follows the official trace convention W[k*n+i,r]"
                # This likely means the row index of W is `idx = k*n + i`.
                # But which k and i?
                # In tensor decomposition $T_{ijk} = \sum U_{ir} V_{jr} W_{kr}$.
                # Here the third mode size is p*n = 14.
                # So k in the tensor notation is 0..13.
                # The mapping from matrix indices (r_c, c_c) to tensor index k_t is:
                # If C is 2x7 (n x p), then k_t = r_c * p + c_c.
                # The text "W[k*n+i,r]" might be a specific notation for the row index of W.
                # Let's assume standard flattening: C is 2x7.
                # If C is row-major, index = r_c * 7 + c_c.
                # If the text implies W's row index is constructed as k*n+i, maybe k is column and i is row?
                # Let's look at the seed program's verification logic if available or infer.
                # Seed program:
                # w_idx = (2 * b) * n + 0  -> col 0, row 0?
                # w_idx = (2 * b + 1) * n + 0 -> col 1, row 0?
                # Here n=2.
                # If b=0, col 0. w_idx = 0*2+0=0 (row 0, col 0?), 0*2+1=1 (row 1, col 0?).
                # This suggests index = col * n + row.
                # So C is stored column-major? Or the index formula is col * n + row.
                # Let's verify with the seed's V mapping.
                # v_idx = 0 * p + (2 * b) -> row 0, col 2b.
                # v_idx = 1 * p + (2 * b) -> row 1, col 2b.
                # So V index = row * p + col. Standard row-major for B (2x7).
                # U index: i in 0..3.
                # Seed U: U[i][col]. i is 0..3.
                # A is 2x2.
                # If U index is row-major: i = r_a * 2 + c_a.
                
                # Let's re-read the W index in seed:
                # w_idx = (2 * b) * n + 0. Here n=2.
                # If b=0 (col 0), w_idx = 0.
                # If b=1 (col 1), w_idx = 1*2 + 0 = 2.
                # If b=0, row 1: w_idx = 0*2 + 1 = 1.
                # So W index = col * n + row.
                # This means the third mode of the tensor is indexed by (c_c, r_c) such that index = c_c * n + r_c.
                # So k_t = c_c * n + r_c.
                
                # Let's verify consistency.
                # T[i, j, k_t] = 1 if match.
                # i = r_a * m + c_a = r_a * 2 + c_a.
                # j = r_b * p + c_b = r_b * 7 + c_b.
                # k_t = c_c * n + r_c = c_c * 2 + r_c.
                
                # Condition: r_a == r_c, c_a == r_b, c_b == c_c.
                # So:
                # r_a = i // 2
                # c_a = i % 2
                # r_b = j // 7
                # c_b = j % 7
                # c_c = k_t // 2
                # r_c = k_t % 2
                
                # Check: r_a == r_c, c_a == r_b, c_b == c_c.
                
                # Let's build T based on this.
                pass

    # Re-build T correctly
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n*m):
        r_a, c_a = divmod(i, m)
        for j in range(m*p):
            r_b, c_b = divmod(j, p)
            if c_a == r_b:
                r_c = r_a
                c_c = c_b
                k_t = c_c * n + r_c
                T[i, j, k_t] = 1.0

    # ALS Search for Rank 24
    def run_als(rank, max_iters=50, tol=1e-10, seed=0):
        rng = np.random.default_rng(seed)
        U = rng.standard_normal((n*m, rank))
        V = rng.standard_normal((m*p, rank))
        W = np.zeros((p*n, rank)) # Initialize W to zero or random? Random is better.
        W = rng.standard_normal((p*n, rank))
        
        best_err = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None, float('inf')
            
            # Update U
            # For each i, solve M_i * u_i = vec(T[i, :, :])
            # M_i is (p*n * m*p) x rank? No.
            # T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
            # For fixed i, we want to find u = U[i, :].
            # vec(T[i, :, :]) = sum_r u_r * vec(V[:, r] * W[:, r]^T) ? No.
            # The term is V[j, r] * W[k, r]. This is the outer product of V[:, r] and W[:, r].
            # So vec(V[:, r] W[:, r]^T) is a vector of length (m*p)*(p*n) = 14*14=196.
            # Let X_r = np.kron(W[:, r], V[:, r])? 
            # (V W^T)_{jk} = V_j W_k.
            # vec(V W^T) = np.kron(W, V) if V is column vector?
            # Yes, vec(A B^T) = np.kron(B, A).
            # So we form a matrix M of shape (196, rank) where column r is np.kron(W[:, r], V[:, r]).
            # Then for each i, we solve M * u_i = vec(T[i, :, :]).
            
            M = np.zeros((m*p * p*n, rank))
            for r in range(rank):
                M[:, r] = np.kron(W[:, r], V[:, r])
            
            for i in range(n*m):
                target = T[i, :, :].flatten()
                # Solve least squares
                u_i, residuals, rank_ls, sv = np.linalg.lstsq(M, target, rcond=None)
                U[i, :] = u_i
            
            # Update V
            # For each j, solve N_j * v_j = vec(T[:, j, :])?
            # T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
            # For fixed j, we want v = V[j, :].
            # vec(T[:, j, :]) is length (n*m)*(p*n) = 4*14=56.
            # Term: U[i, r] * W[k, r].
            # vec(U[:, r] W[:, r]^T) = np.kron(W[:, r], U[:, r]).
            # So N is (56, rank) where column r is np.kron(W[:, r], U[:, r]).
            
            N = np.zeros((n*m * p*n, rank))
            for r in range(rank):
                N[:, r] = np.kron(W[:, r], U[:, r])
                
            for j in range(m*p):
                target = T[:, j, :].flatten()
                v_j, _, _, _ = np.linalg.lstsq(N, target, rcond=None)
                V[j, :] = v_j
                
            # Update W
            # For each k, solve P_k * w_k = vec(T[:, :, k])
            # T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
            # For fixed k, we want w = W[k, :].
            # vec(T[:, :, k]) is length (n*m)*(m*p) = 4*14=56.
            # Term: U[i, r] * V[j, r].
            # vec(U[:, r] V[:, r]^T) = np.kron(V[:, r], U[:, r]).
            # So P is (56, rank) where column r is np.kron(V[:, r], U[:, r]).
            
            P = np.zeros((n*m * m*p, rank))
            for r in range(rank):
                P[:, r] = np.kron(V[:, r], U[:, r])
                
            for k in range(p*n):
                target = T[:, :, k].flatten()
                w_k, _, _, _ = np.linalg.lstsq(P, target, rcond=None)
                W[k, :] = w_k
                
            # Check error
            # Compute reconstructed tensor
            # T_recon[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
            # This is a 4x14x14 tensor.
            # We can compute it efficiently.
            # T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            err = np.linalg.norm(T_recon - T)
            
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                
            if err < tol:
                break
                
        if best_err < 1e-6:
            return (best_U, best_V, best_W), best_err
        else:
            return None, best_err

    # Try to find rank 24
    found = False
    best_solution = None
    
    # Try a few seeds
    for seed in range(10):
        if time.time() > deadline - 5:
            break
        res, err = run_als(target_rank, max_iters=100, tol=1e-12, seed=seed)
        if res is not None:
            found = True
            best_solution = res
            break
            
    if found:
        U, V, W = best_solution
        # Convert to list of lists for return?
        # The interface expects {'U', 'V', 'W'} with shapes.
        # The seed returned lists. Let's return lists to be safe, or arrays?
        # "factor matrices of shapes ... with integer/half-integer entries"
        # The seed returned lists of lists.
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback to rank 25
        rank = 25
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        # Strassen decomposition for 2x2x2
        # Using the specific coefficients from the seed
        strassen_U = [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 1, 0, -1]
        ]
        
        # Note: The seed's Strassen V and W were local dicts.
        # We need to map them correctly.
        # The seed used:
        # strassen_W_local = [
        #     {0: 1, 3: 1},
        #     {2: 1, 3: -1},
        #     {1: 1, 3: 1},
        #     {0: 1, 2: 1},
        #     {0: -1, 1: 1},
        #     {3: 1},
        #     {0: 1}
        # ]
        # This corresponds to the 7 products.
        # Let's just copy the seed's logic for the fallback.
        
        strassen_W_local = [
            {0: 1, 3: 1},
            {2: 1, 3: -1},
            {1: 1, 3: 1},
            {0: 1, 2: 1},
            {0: -1, 1: 1},
            {3: 1},
            {0: 1}
        ]
        
        strassen_V_local = [
            {0: 1, 3: 1},
            {0: 1},
            {1: 1, 3: -1},
            {2: 1, 0: -1},
            {3: 1},
            {0: 1, 1: 1},
            {2: 1, 3: 1}
        ]
        
        for b in range(3):
            for r in range(7):
                for i in range(4):
                    U[i][col] += strassen_U[r][i]
                for loc_idx, coeff in strassen_V_local[r].items():
                    if loc_idx == 0: v_idx = 0 * p + (2 * b)
                    elif loc_idx == 1: v_idx = 0 * p + (2 * b + 1)
                    elif loc_idx == 2: v_idx = 1 * p + (2 * b)
                    elif loc_idx == 3: v_idx = 1 * p + (2 * b + 1)
                    V[v_idx][col] += coeff
                for loc_idx, coeff in strassen_W_local[r].items():
                    if loc_idx == 0: w_idx = (2 * b) * n + 0
                    elif loc_idx == 1: w_idx = (2 * b + 1) * n + 0
                    elif loc_idx == 2: w_idx = (2 * b) * n + 1
                    elif loc_idx == 3: w_idx = (2 * b + 1) * n + 1
                    W[w_idx][col] += coeff
                col += 1
                
        # Schoolbook for the last column (2x2x1)
        schoolbook_U = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
        
        schoolbook_V_local = [
            {0: 1},
            {1: 1},
            {0: 1},
            {1: 1}
        ]
        
        schoolbook_W_local = [
            {0: 1},
            {0: 1},
            {1: 1},
            {1: 1}
        ]
        
        for r in range(4):
            for i in range(4):
                U[i][col] += schoolbook_U[r][i]
            for loc_idx, coeff in schoolbook_V_local[r].items():
                if loc_idx == 0: v_idx = 0 * p + 6
                elif loc_idx == 1: v_idx = 1 * p + 6
                V[v_idx][col] += coeff
            for loc_idx, coeff in schoolbook_W_local[r].items():
                if loc_idx == 0: w_idx = 6 * n + 0
                elif loc_idx == 1: w_idx = 6 * n + 1
                W[w_idx][col] += coeff
            col += 1
            
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
