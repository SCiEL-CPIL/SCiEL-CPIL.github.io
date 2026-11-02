"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    def validate(U, V, W):
        """Check if the decomposition is exact using vectorized numpy."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        T = np.einsum('ijt,jkt,kit->ijk', U3, V3, W3)
        return np.allclose(T, 1.0, atol=1e-9)
    
    def build_solution(blocks, schoolbook_terms):
        """Build U, V, W from Strassen blocks and schoolbook terms."""
        total_rank = len(blocks) * 7 + len(schoolbook_terms)
        U = [[0.0] * total_rank for _ in range(n * m)]
        V = [[0.0] * total_rank for _ in range(m * p)]
        W = [[0.0] * total_rank for _ in range(p * n)]
        
        col = 0
        for (i0, j0, k0) in blocks:
            # M1
            c = col
            U[i0*m + j0][c] += 1; U[(i0+1)*m + (j0+1)][c] += 1
            V[j0*p + k0][c] += 1; V[(j0+1)*p + (k0+1)][c] += 1
            W[k0*n + i0][c] += 1; W[(k0+1)*n + (i0+1)][c] += 1
            col += 1
            # M2
            c = col
            U[(i0+1)*m + j0][c] += 1; U[(i0+1)*m + (j0+1)][c] += 1
            V[j0*p + k0][c] += 1
            W[k0*n + (i0+1)][c] += 1; W[(k0+1)*n + (i0+1)][c] += -1
            col += 1
            # M3
            c = col
            U[i0*m + j0][c] += 1
            V[j0*p + (k0+1)][c] += 1; V[(j0+1)*p + (k0+1)][c] += -1
            W[(k0+1)*n + i0][c] += 1; W[(k0+1)*n + (i0+1)][c] += 1
            col += 1
            # M4
            c = col
            U[(i0+1)*m + (j0+1)][c] += 1
            V[(j0+1)*p + k0][c] += 1; V[j0*p + k0][c] += -1
            W[k0*n + i0][c] += 1; W[k0*n + (i0+1)][c] += 1
            col += 1
            # M5
            c = col
            U[i0*m + j0][c] += 1; U[i0*m + (j0+1)][c] += 1
            V[(j0+1)*p + (k0+1)][c] += 1
            W[k0*n + i0][c] += -1; W[(k0+1)*n + i0][c] += 1
            col += 1
            # M6
            c = col
            U[(i0+1)*m + j0][c] += 1; U[i0*m + j0][c] += -1
            V[j0*p + k0][c] += 1; V[j0*p + (k0+1)][c] += 1
            W[(k0+1)*n + (i0+1)][c] += 1
            col += 1
            # M7
            c = col
            U[i0*m + (j0+1)][c] += 1; U[(i0+1)*m + (j0+1)][c] += -1
            V[(j0+1)*p + k0][c] += 1; V[(j0+1)*p + (k0+1)][c] += 1
            W[k0*n + i0][c] += 1
            col += 1
        
        for (i, j, k) in schoolbook_terms:
            c = col
            U[i*m + j][c] += 1
            V[j*p + k][c] += 1
            W[k*n + i][c] += 1
            col += 1
        
        return U, V, W
    
    def get_schoolbook_for_blocks(blocks):
        """Get schoolbook terms not covered by blocks."""
        covered = set()
        for (i0, j0, k0) in blocks:
            for i in (i0, i0+1):
                for j in (j0, j0+1):
                    for k in (k0, k0+1):
                        covered.add((i, j, k))
        return [(i, j, k) for i in range(n) for j in range(m) for k in range(p) if (i, j, k) not in covered]
        
    def try_reduce_rank(U, V, W):
        """Attempt to reduce rank by finding linear dependencies."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        current_rank = U_arr.shape[1]
        
        def get_column_tensor(U_col, V_col, W_col):
            U3 = U_col.reshape(n, m)
            V3 = V_col.reshape(m, p)
            W3 = W_col.reshape(p, n)
            tensor = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        tensor[i, j, k] = U3[i, j] * V3[j, k] * W3[k, i]
            return tensor.flatten()
        
        tensor_cols = []
        for t in range(current_rank):
            col_tensor = get_column_tensor(
                U_arr[:, t], V_arr[:, t], W_arr[:, t]
            )
            tensor_cols.append(col_tensor)
        
        tensor_matrix = np.array(tensor_cols).T
        
        # Check for dependencies
        # We try to remove columns one by one if they are in the span of the rest
        # This is a greedy approach
        for _ in range(current_rank):
            if time.time() > deadline - 5:
                break
            removed = False
            for t_to_remove in range(current_rank - 1, -1, -1):
                keep_cols = [c for c in range(current_rank) if c != t_to_remove]
                if len(keep_cols) == 0:
                    continue
                remaining_matrix = tensor_matrix[:, keep_cols]
                target = tensor_matrix[:, t_to_remove]
                
                try:
                    alpha, residuals, rank_lsq, sv = np.linalg.lstsq(
                        remaining_matrix, target, rcond=1e-10
                    )
                    fitted = remaining_matrix @ alpha
                    error = np.linalg.norm(target - fitted)
                    
                    if error < 1e-8:
                        new_U = U_arr[:, keep_cols]
                        new_V = V_arr[:, keep_cols]
                        new_W = W_arr[:, keep_cols]
                        
                        if validate(new_U.tolist(), new_V.tolist(), new_W.tolist()):
                            U_arr = new_U
                            V_arr = new_V
                            W_arr = new_W
                            current_rank = len(keep_cols)
                            
                            # Rebuild tensor matrix
                            tensor_cols = []
                            for t in range(current_rank):
                                col_tensor = get_column_tensor(
                                    U_arr[:, t], V_arr[:, t], W_arr[:, t]
                                )
                                tensor_cols.append(col_tensor)
                            tensor_matrix = np.array(tensor_cols).T
                            removed = True
                            break
                except np.linalg.LinAlgError:
                    continue
            if not removed:
                break
        
        return U_arr, V_arr, W_arr

    # Define valid disjoint block configurations
    # i0 can be 0 or 1
    # j0 pairs must be disjoint intervals of length 2 in [0,4]
    # Valid pairs: (0,2), (0,3), (1,3)
    # k0 pairs same
    
    i_options = [0, 1]
    j_pairs = [(0, 2), (0, 3), (1, 3)]
    k_pairs = [(0, 2), (0, 3), (1, 3)]
    
    best_rank = 75
    best_solution = None
    
    for i0 in i_options:
        for j_pair in j_pairs:
            for k_pair in k_pairs:
                blocks = []
                for j0 in j_pair:
                    for k0 in k_pair:
                        blocks.append((i0, j0, k0))
                
                sb = get_schoolbook_for_blocks(blocks)
                rank = len(blocks) * 7 + len(sb)
                
                if rank >= best_rank:
                    continue
                
                U, V, W = build_solution(blocks, sb)
                if validate(U, V, W):
                    U_red, V_red, W_red = try_reduce_rank(U, V, W)
                    new_rank = U_red.shape[1]
                    if new_rank < best_rank:
                        best_rank = new_rank
                        best_solution = (U_red.tolist(), V_red.tolist(), W_red.tolist())
                
                if time.time() > deadline - 20:
                    break
        if time.time() > deadline - 20:
            break
        if time.time() > deadline - 20:
            break
    
    if best_solution is not None:
        return {"U": best_solution[0], "V": best_solution[1], "W": best_solution[2]}
    
    # Fallback to initial known good solution
    initial_blocks = [(0, 0, 0), (0, 2, 0), (0, 0, 2), (0, 2, 2)]
    U, V, W = build_solution(initial_blocks, get_schoolbook_for_blocks(initial_blocks))
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
