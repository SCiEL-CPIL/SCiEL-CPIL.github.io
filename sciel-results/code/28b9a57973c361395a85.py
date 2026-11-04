import time
import numpy as np

def _strassen_28():
    """Return (U, V, W) as lists of lists, rank 28, 4 independent Strassen blocks."""
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_block(col_offset, X_rows, Y_rows, Z_rows):
        U[X_rows[0]][col_offset + 0] += 1
        U[X_rows[3]][col_offset + 0] += 1
        V[Y_rows[0]][col_offset + 0] += 1
        V[Y_rows[3]][col_offset + 0] += 1
        W[Z_rows[0]][col_offset + 0] += 1
        W[Z_rows[3]][col_offset + 0] += 1

        U[X_rows[2]][col_offset + 1] += 1
        U[X_rows[3]][col_offset + 1] += 1
        V[Y_rows[0]][col_offset + 1] += 1
        W[Z_rows[2]][col_offset + 1] += 1
        W[Z_rows[3]][col_offset + 1] -= 1

        U[X_rows[0]][col_offset + 2] += 1
        V[Y_rows[1]][col_offset + 2] += 1
        V[Y_rows[3]][col_offset + 2] -= 1
        W[Z_rows[1]][col_offset + 2] += 1
        W[Z_rows[3]][col_offset + 2] += 1

        U[X_rows[3]][col_offset + 3] += 1
        V[Y_rows[2]][col_offset + 3] += 1
        V[Y_rows[0]][col_offset + 3] -= 1
        W[Z_rows[0]][col_offset + 3] += 1
        W[Z_rows[2]][col_offset + 3] += 1

        U[X_rows[0]][col_offset + 4] += 1
        U[X_rows[1]][col_offset + 4] += 1
        V[Y_rows[3]][col_offset + 4] += 1
        W[Z_rows[0]][col_offset + 4] -= 1
        W[Z_rows[1]][col_offset + 4] += 1

        U[X_rows[2]][col_offset + 5] += 1
        U[X_rows[0]][col_offset + 5] -= 1
        V[Y_rows[0]][col_offset + 5] += 1
        V[Y_rows[1]][col_offset + 5] += 1
        W[Z_rows[3]][col_offset + 5] += 1

        U[X_rows[1]][col_offset + 6] += 1
        U[X_rows[3]][col_offset + 6] -= 1
        V[Y_rows[2]][col_offset + 6] += 1
        V[Y_rows[3]][col_offset + 6] += 1
        W[Z_rows[0]][col_offset + 6] += 1

    X1 = [0, 1, 4, 5];  Y1 = [0, 1, 4, 5];   Z1 = [0, 2, 1, 3]
    X2 = [2, 3, 6, 7];  Y2 = [8, 9, 12, 13]; Z2 = [0, 2, 1, 3]
    X3 = [0, 1, 4, 5];  Y3 = [2, 3, 6, 7];   Z3 = [4, 6, 5, 7]
    X4 = [2, 3, 6, 7];  Y4 = [10, 11, 14, 15]; Z4 = [4, 6, 5, 7]

    add_block(0,  X1, Y1, Z1)
    add_block(7,  X2, Y2, Z2)
    add_block(14, X3, Y3, Z3)
    add_block(21, X4, Y4, Z4)
    return U, V, W

def _verify_decomp(U, V, W, n=2, m=4, p=4):
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Construct target tensor
    # T[i, j, k] = 1 if i%4 == j//4 and j%4 == k//2 and k%2 == i//4
    # i in 0..7, j in 0..15, k in 0..7
    # i = 4*row_A + col_A? No.
    # A is 2x4. Index i: 0..7. i = 4*r + c? 
    # Standard row-major: A[r, c] is at index r*4 + c.
    # B is 4x4. Index j: 0..15. j = 4*r + c.
    # C is 4x4? No, C is 2x4. Index k: 0..7. k = 4*r + c.
    # C[r, c] = sum_k A[r, k] B[k, c].
    # A[r, k] is at i = 4*r + k.
    # B[k, c] is at j = 4*k + c.
    # C[r, c] is at k_idx = 4*r + c.
    # So T[i, j, k_idx] = 1 if:
    # i = 4*r + k
    # j = 4*k + c
    # k_idx = 4*r + c
    # Check consistency:
    # i % 4 == k
    # j // 4 == k
    # j % 4 == c
    # k_idx // 4 == r
    # i // 4 == r
    
    T = np.zeros((8, 16, 8), dtype=float)
    for r in range(2):
        for k in range(4):
            for c in range(4):
                i = 4 * r + k
                j = 4 * k + c
                k_idx = 4 * r + c
                T[i, j, k_idx] = 1.0

    # Compute decomposition
    # Result[i, j, k_idx] = sum_r U[i, r] V[j, r] W[k_idx, r]
    # We can compute this efficiently using matrix multiplication tricks or loops.
    # Given small size, loops are fine.
    
    for i in range(8):
        for j in range(16):
            # Compute vector of size R: U[i, :] * V[j, :]
            prod = U[i] * V[j]
            # Now for each k_idx, sum(prod * W[k_idx, :])
            for k_idx in range(8):
                s = np.dot(prod, W[k_idx])
                if not np.isclose(s, T[i, j, k_idx], atol=1e-9):
                    return False
    return True

def _try_rank_search(deadline):
    n, m, p = 2, 4, 4
    nm, mp, pn = n * m, m * p, p * n
    
    # Target tensor T
    T = np.zeros((nm, mp, pn), dtype=float)
    for r in range(n):
        for k in range(m):
            for c in range(p):
                i = m * r + k
                j = m * k + c
                k_idx = p * r + c # Wait, C is n x p. Index is r*p + c.
                # C is 2x4. r in 0..1, c in 0..3. Index = 4*r + c.
                # My previous loop: k_idx = 4*r + c. Correct.
                T[i, j, k_idx] = 1.0

    # Pre-flatten T for each k_idx
    T_flat = [T[:, :, k].flatten() for k in range(pn)]
    
    best_rank = 28
    best_decomp = None
    
    # Search for ranks 24 to 27
    for R in range(24, 28):
        if time.time() > deadline - 10:
            break
            
        # Number of trials per rank
        # Reduce trials if time is tight
        max_trials = 500
        if time.time() > deadline - 30:
            max_trials = 100
            
        for trial in range(max_trials):
            if time.time() > deadline - 5:
                break
                
            # Random initialization
            # Use small integers/half-integers to increase chance of exact solution
            # Entries in {-1, 0, 1, 0.5, -0.5}
            # Or just random integers -2..2
            U_cand = np.random.choice([-1, 0, 1, 0.5, -0.5], size=(nm, R))
            V_cand = np.random.choice([-1, 0, 1, 0.5, -0.5], size=(mp, R))
            
            # Check if T slices are in the span of U_r V_r^T
            valid = True
            W_cand = np.zeros((pn, R))
            
            for k_idx in range(pn):
                # Construct matrix A_k of size (nm*mp) x R
                # A_k[:, r] = U_cand[:, r] * V_cand[:, r] (element-wise product, flattened)
                # We want A_k @ W_k = T_flat[k_idx]
                
                # Efficient check:
                # Compute projection of T_flat[k_idx] onto the column space of A_k
                # If residual is small, it's in the span.
                
                # Build A_k
                # A_k is (128, R)
                A_k = np.zeros((nm * mp, R))
                for r in range(R):
                    A_k[:, r] = (U_cand[:, r] * V_cand[:, r]).flatten()
                
                # Solve A_k W_k = T_flat[k_idx]
                # Use least squares
                try:
                    W_k, residuals, rank_k, sv = np.linalg.lstsq(A_k, T_flat[k_idx], rcond=None)
                except np.linalg.LinAlgError:
                    valid = False
                    break
                
                # Check residual
                if residuals.size > 0 and residuals[0] > 1e-9:
                    valid = False
                    break
                else:
                    W_cand[k_idx] = W_k
            
            if valid:
                # Check if W_cand entries are "nice" (close to integer or half-integer)
                # And U, V are nice (they were generated from a set, so they are)
                # We need to verify the full decomposition
                # Since we found W via lstsq, it should be exact if residual was 0.
                # But we must ensure the solution is consistent and exact.
                
                # Verify
                if _verify_decomp(U_cand.tolist(), V_cand.tolist(), W_cand.tolist()):
                    best_rank = R
                    best_decomp = (U_cand.tolist(), V_cand.tolist(), W_cand.tolist())
                    return best_rank, best_decomp
    
    return best_rank, best_decomp

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # 10s margin
    
    try:
        rank_found, decomp_found = _try_rank_search(deadline)
        if decomp_found is not None:
            U, V, W = decomp_found
            # Final verification
            if _verify_decomp(U, V, W):
                return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback to rank 28
    U, V, W = _strassen_28()
    return {"U": U, "V": V, "W": W}
