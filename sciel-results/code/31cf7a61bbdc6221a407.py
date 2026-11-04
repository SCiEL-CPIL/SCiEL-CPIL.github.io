import time
import numpy as np

def construct():
    n, m, p = 3, 4, 4
    dim_u = n * m  # 12
    dim_v = m * p  # 16
    dim_w = p * n  # 12

    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    def build_rank44():
        rank = 44
        U = [[0] * rank for _ in range(dim_u)]
        V = [[0] * rank for _ in range(dim_v)]
        W = [[0] * rank for _ in range(dim_w)]

        col = 0

        def add_strassen(X_u, Y_v, Z_w, col):
            U[X_u[0]][col] += 1; U[X_u[3]][col] += 1
            V[Y_v[0]][col] += 1; V[Y_v[3]][col] += 1
            W[Z_w[0]][col] += 1; W[Z_w[3]][col] += 1
            col += 1
            U[X_u[2]][col] += 1; U[X_u[3]][col] += 1
            V[Y_v[0]][col] += 1
            W[Z_w[2]][col] += 1; W[Z_w[3]][col] -= 1
            col += 1
            U[X_u[0]][col] += 1
            V[Y_v[1]][col] += 1; V[Y_v[3]][col] -= 1
            W[Z_w[1]][col] += 1; W[Z_w[3]][col] += 1
            col += 1
            U[X_u[3]][col] += 1
            V[Y_v[2]][col] += 1; V[Y_v[0]][col] -= 1
            W[Z_w[0]][col] += 1; W[Z_w[2]][col] += 1
            col += 1
            U[X_u[0]][col] += 1; U[X_u[1]][col] += 1
            V[Y_v[3]][col] += 1
            W[Z_w[0]][col] -= 1; W[Z_w[1]][col] += 1
            col += 1
            U[X_u[2]][col] += 1; U[X_u[0]][col] -= 1
            V[Y_v[0]][col] += 1; V[Y_v[1]][col] += 1
            W[Z_w[3]][col] += 1
            col += 1
            U[X_u[1]][col] += 1; U[X_u[3]][col] -= 1
            V[Y_v[2]][col] += 1; V[Y_v[3]][col] += 1
            W[Z_w[0]][col] += 1
            col += 1
            return col

        col = add_strassen([0, 1, 4, 5], [0, 1, 4, 5], [0, 3, 1, 4], col)
        col = add_strassen([2, 3, 6, 7], [8, 9, 12, 13], [0, 3, 1, 4], col)
        col = add_strassen([0, 1, 4, 5], [2, 3, 6, 7], [6, 9, 7, 10], col)
        col = add_strassen([2, 3, 6, 7], [10, 11, 14, 15], [6, 9, 7, 10], col)

        for j in range(4):
            U[8 + j][col] += 1; V[j * 4 + 0][col] += 1; W[2][col] += 1
            col += 1
            U[8 + j][col] += 1; V[j * 4 + 1][col] += 1; W[5][col] += 1
            col += 1
        for j in range(4):
            U[8 + j][col] += 1; V[j * 4 + 2][col] += 1; W[8][col] += 1
            col += 1
            U[8 + j][col] += 1; V[j * 4 + 3][col] += 1; W[11][col] += 1
            col += 1

        return U, V, W

    def verify_decomp(U, V, W, n, m, p):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    coeff = 0.0
                    for r in range(rank):
                        coeff += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(coeff - 1.0) > 1e-9:
                        return False
        return True

    U, V, W = build_rank44()
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    if not verify_decomp(U_np, V_np, W_np, n, m, p):
        rank = 48
        U = [[0] * rank for _ in range(dim_u)]
        V = [[0] * rank for _ in range(dim_v)]
        W = [[0] * rank for _ in range(dim_w)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Build contribution matrix: contrib[ijk, r] = U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    rank = 44
    contrib = np.zeros((n * m * p, rank))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(rank):
                    contrib[idx, r] = U_np[i * m + j, r] * V_np[j * p + k, r] * W_np[k * n + i, r]
                idx += 1

    target = np.ones(n * m * p)

    # Try to find redundant terms: check if removing a subset still sums to target
    best_indices = list(range(rank))

    # Try removing 1 term
    for r_remove in range(rank):
        if time.time() > deadline - 5:
            break
        remaining = [r for r in range(rank) if r != r_remove]
        col_sums = np.sum(contrib[:, remaining], axis=1)
        if np.allclose(col_sums, target, atol=1e-9):
            best_indices = remaining
            break

    # Try removing 2 terms (only if we didn't already find a reduction)
    if len(best_indices) == rank and time.time() < deadline - 10:
        for r1 in range(rank):
            if time.time() > deadline - 5:
                break
            for r2 in range(r1 + 1, rank):
                remaining = [r for r in range(rank) if r != r1 and r != r2]
                col_sums = np.sum(contrib[:, remaining], axis=1)
                if np.allclose(col_sums, target, atol=1e-9):
                    best_indices = remaining
                    break
            if len(best_indices) < rank:
                break

    if len(best_indices) < rank:
        new_rank = len(best_indices)
        U_new = [[0] * new_rank for _ in range(dim_u)]
        V_new = [[0] * new_rank for _ in range(dim_v)]
        W_new = [[0] * new_rank for _ in range(dim_w)]
        for new_r, old_r in enumerate(best_indices):
            for i in range(dim_u):
                U_new[i][new_r] = U_np[i, old_r]
            for i in range(dim_v):
                V_new[i][new_r] = V_np[i, old_r]
            for i in range(dim_w):
                W_new[i][new_r] = W_np[i, old_r]

        U_chk = np.array(U_new, dtype=float)
        V_chk = np.array(V_new, dtype=float)
        W_chk = np.array(W_new, dtype=float)
        if verify_decomp(U_chk, V_chk, W_chk, n, m, p):
            return {"U": U_new, "V": V_new, "W": W_new}

    # Try to find a rank-43 decomposition by modifying the bottom-row terms
    # The bottom 8 terms (indices 28-43) are schoolbook for A_bottom * B
    # Try to replace some with Strassen-like combinations
    if time.time() < deadline - 15:
        # The bottom-left terms: for j=0,1,2,3: U[8+j]*V[4j+0]*W[2] and U[8+j]*V[4j+1]*W[5]
        # The bottom-right terms: for j=0,1,2,3: U[8+j]*V[4j+2]*W[8] and U[8+j]*V[4j+3]*W[11]
        
        # Try a different factorization of the bottom block
        # A_bottom (1x4) * B (4x4) = C_bottom (1x4)
        # This is a (1,4,4) tensor with rank 4 (minimum)
        # But we can try to share structure with the top blocks
        
        # Alternative: try to use a rank-3 decomposition for a (1,3,3) subproblem
        # embedded in the (1,4,4) by grouping columns
        
        # Actually, the (1,4,4) tensor has rank exactly 4 (it's a "slice" tensor)
        # So we can't reduce below 4 for each of the 2 bottom output rows
        # Total bottom: 8 terms, which is already minimal
        
        # Try a different approach: can we merge some Strassen terms with bottom terms?
        # This requires a more sophisticated search
        
        # For now, try a random search: perturb the bottom terms slightly and see if we can
        # find a valid rank-43 decomposition
        # This is unlikely to work quickly, so limit the search
        
        # Try: replace the 8 bottom schoolbook terms with 7 terms using a different factorization
        # The (1,4,4) tensor: T[i,j,k] = delta_{i,2} * delta_{j,k} for the relevant indices
        # Actually it's: for output row 2, C[2,k] = sum_j A[2,j] * B[j,k]
        # This is a rank-4 tensor (each output element needs its own term)
        
        # Let's try a completely different approach: use the fact that we have 4 Strassen blocks
        # and see if any Strassen terms can be shared or eliminated
        
        # Check if the 44-term decomposition has any structure that allows reduction
        # by examining the tensor's border rank or other invariants
        
        # For a quick check: try to see if we can express one of the bottom terms
        # as a combination of Strassen terms
        # This would require solving a linear system, which is complex
        
        # Given time constraints, return the best we have
        pass

    return {"U": U, "V": V, "W": W}
