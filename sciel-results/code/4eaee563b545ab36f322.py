"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    # Build the rank-56 Strassen decomposition
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]), # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]), # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]), # M7
    ]
    
    def build_rank56():
        rank = 56
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        current_rank = 0
        for q in range(4):
            for s in range(7):
                u_loc, v_loc, w_loc = strassen_terms[s]
                r = current_rank
                for i in range(2):
                    for j_loc in range(2):
                        U[i * m + j_loc][r] = u_loc[i * 2 + j_loc]
                for j_loc in range(2):
                    for k_loc in range(2):
                        V[j_loc * p + (2 * q + k_loc)][r] = v_loc[j_loc * 2 + k_loc]
                for i in range(2):
                    for k_loc in range(2):
                        W[(2 * q + k_loc) * n + i][r] = w_loc[k_loc * 2 + i]
                current_rank += 1
        for q in range(4):
            for s in range(7):
                u_loc, v_loc, w_loc = strassen_terms[s]
                r = current_rank
                for i in range(2):
                    for j_loc in range(2):
                        U[i * m + (j_loc + 2)][r] = u_loc[i * 2 + j_loc]
                for j_loc in range(2):
                    for k_loc in range(2):
                        V[(j_loc + 2) * p + (2 * q + k_loc)][r] = v_loc[j_loc * 2 + k_loc]
                for i in range(2):
                    for k_loc in range(2):
                        W[(2 * q + k_loc) * n + i][r] = w_loc[k_loc * 2 + i]
                current_rank += 1
        return U, V, W, rank
    
    def verify_full(U, V, W, rank, tol=1e-9):
        """Verify the decomposition exactly: sum_r U[i,r]*V[j,r]*W[k,r] = T[i,j,k]"""
        for i in range(n * m):
            i_row, i_col = divmod(i, m)
            for j in range(m * p):
                j_row, j_col = divmod(j, p)
                if i_col != j_row:
                    continue
                for k in range(p * n):
                    k_row, k_col = divmod(k, n)
                    expected = 1.0 if j_col == k_row else 0.0
                    actual = np.sum(U[i, :] * V[j, :] * W[k, :])
                    if abs(actual - expected) > tol:
                        return False
        return True
    
    # Start with rank-56 solution
    U, V, W, rank = build_rank56()
    
    # Verify the base decomposition
    if not verify_full(U, V, W, rank):
        # Fallback: return as-is
        U_list = [[int(U[i, r]) for r in range(rank)] for i in range(n * m)]
        V_list = [[int(V[i, r]) for r in range(rank)] for i in range(m * p)]
        W_list = [[int(W[i, r]) for r in range(rank)] for i in range(p * n)]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Strategy: try to find rank reductions by looking for linear dependencies
    # among the 56 rank-1 terms. We'll use a greedy approach:
    # For each subset of columns to try removing, check if the remaining
    # columns can still span the tensor.
    
    # More efficient: use the structure. The 56 terms come from 8 blocks of 7.
    # Within each block, the 7 Strassen terms are independent.
    # Across blocks, there may be dependencies due to the small matrix sizes.
    
    # Let's try a systematic search: try removing 1, 2, 3, ... columns
    # and check if the decomposition still works.
    
    # For efficiency, we'll use a random sampling approach for larger removals.
    
    # First, try removing single columns
    removed_cols = []
    for r_to_remove in range(rank):
        if time.time() > deadline:
            break
        
        mask = np.ones(rank, dtype=bool)
        mask[r_to_remove] = False
        
        U_test = U[:, mask]
        V_test = V[:, mask]
        W_test = W[:, mask]
        rank_test = rank - 1
        
        if verify_full(U_test, V_test, W_test, rank_test):
            best_rank = rank_test
            best_U, best_V, best_W = U_test, V_test, W_test
            U, V, W = U_test, V_test, W_test
            rank = rank_test
            removed_cols.append(r_to_remove)
    
    # Try removing pairs of columns
    if time.time() < deadline - 30:
        for r1 in range(rank):
            if time.time() > deadline:
                break
            for r2 in range(r1 + 1, rank):
                mask = np.ones(rank, dtype=bool)
                mask[r1] = False
                mask[r2] = False
                
                U_test = U[:, mask]
                V_test = V[:, mask]
                W_test = W[:, mask]
                rank_test = rank - 2
                
                if verify_full(U_test, V_test, W_test, rank_test):
                    best_rank = rank_test
                    best_U, best_V, best_W = U_test, V_test, W_test
                    U, V, W = U_test, V_test, W_test
                    rank = rank_test
                    break
            if time.time() > deadline:
                break
    
    # Convert to lists with proper integer/half-integer handling
    def to_list(mat, rank_val):
        result = []
        for i in range(mat.shape[0]):
            row = []
            for r in range(rank_val):
                val = mat[i, r]
                # Round to nearest 0.5
                val_rounded = round(val * 2) / 2
                # Check if it's an integer or half-integer
                if abs(val_rounded * 2 - round(val_rounded * 2)) < 1e-6:
                    if abs(val_rounded - round(val_rounded)) < 1e-6:
                        row.append(int(round(val_rounded)))
                    else:
                        row.append(round(val_rounded * 2) / 2)
                else:
                    row.append(round(val_rounded * 2) / 2)
            result.append(row)
        return result
    
    U_list = to_list(best_U, best_rank)
    V_list = to_list(best_V, best_rank)
    W_list = to_list(best_W, best_rank)
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
