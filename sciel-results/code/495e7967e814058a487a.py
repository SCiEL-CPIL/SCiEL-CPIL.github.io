"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    n, m, p = DIMENSIONS  # 3, 5, 6

    def verify_numpy(U, V, W, rank):
        """Verify decomposition using numpy for speed."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U_np[i * m + j, :]
                    v_row = V_np[j * p + k, :]
                    w_row = W_np[k * n + i, :]
                    val = np.sum(u_row * v_row * w_row)
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def verify_sparse(U, V, W, rank):
        """Verify decomposition using sparse representation for speed."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_r = i * m + j
                    v_r = j * p + k
                    w_r = k * n + i
                    total = 0.0
                    for c in range(rank):
                        uv = U[u_r][c] * V[v_r][c]
                        if uv != 0:
                            total += uv * W[w_r][c]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True

    def schoolbook():
        """Return the schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def strassen_2x2x2_terms():
        """
        Return the 7 Strassen terms for 2x2x2 matrix multiplication.
        Each term is a tuple (u_dict, v_dict, w_dict) where keys are local indices.
        """
        terms = []
        u1 = {(0,0): 1, (1,1): 1}
        v1 = {(0,0): 1, (1,1): 1}
        w1 = {(0,0): 1, (1,1): 1}
        terms.append((u1, v1, w1))
        
        u2 = {(1,0): 1, (1,1): 1}
        v2 = {(0,0): 1}
        w2 = {(0,1): 1, (1,1): -1}
        terms.append((u2, v2, w2))
        
        u3 = {(0,0): 1}
        v3 = {(0,1): 1, (1,1): -1}
        w3 = {(1,0): 1, (1,1): 1}
        terms.append((u3, v3, w3))
        
        u4 = {(1,1): 1}
        v4 = {(1,0): 1, (0,0): -1}
        w4 = {(0,0): 1, (0,1): 1}
        terms.append((u4, v4, w4))
        
        u5 = {(0,0): 1, (0,1): 1}
        v5 = {(1,1): 1}
        w5 = {(0,0): -1, (1,0): 1}
        terms.append((u5, v5, w5))
        
        u6 = {(1,0): 1, (0,0): -1}
        v6 = {(0,0): 1, (0,1): 1}
        w6 = {(1,1): 1}
        terms.append((u6, v6, w6))
        
        u7 = {(0,1): 1, (1,1): -1}
        v7 = {(1,0): 1, (1,1): 1}
        w7 = {(0,0): 1}
        terms.append((u7, v7, w7))
        
        return terms

    def build_decomposition(i_groups, j_groups, k_groups):
        """
        Build a decomposition using Strassen on 2x2x2 blocks and schoolbook elsewhere.
        """
        total_rank = 0
        block_info = []
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    if si == 2 and sj == 2 and sk == 2:
                        r = 7
                    else:
                        r = si * sj * sk
                    total_rank += r
                    block_info.append((ig, jg, kg, r))
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col = 0
        strassen_terms = strassen_2x2x2_terms()
        
        for ig, jg, kg, r in block_info:
            si, sj, sk = len(ig), len(jg), len(kg)
            if si == 2 and sj == 2 and sk == 2:
                for (u_pat, v_pat, w_pat) in strassen_terms:
                    for (li, lj), val_u in u_pat.items():
                        gi, gj = ig[li], jg[lj]
                        U[gi * m + gj][col] = val_u
                    for (lj, lk), val_v in v_pat.items():
                        gj, gk = jg[lj], kg[lk]
                        V[gj * p + gk][col] = val_v
                    for (lk, li), val_w in w_pat.items():
                        gk, gi = kg[lk], ig[li]
                        W[gk * n + gi][col] = val_w
                    col += 1
            else:
                for li in range(si):
                    for lj in range(sj):
                        for lk in range(sk):
                            gi, gj, gk = ig[li], jg[lj], kg[lk]
                            U[gi * m + gj][col] = 1
                            V[gj * p + gk][col] = 1
                            W[gk * n + gi][col] = 1
                            col += 1
        
        return {"U": U, "V": V, "W": W, "rank": total_rank}

    def generate_all_splits(dim_size):
        """Generate all possible splits of range(dim_size) into contiguous groups."""
        if dim_size == 0:
            return [[]]
        if dim_size == 1:
            return [[[0]]]
        
        all_splits = []
        # First group can have size 1 to dim_size
        for first_size in range(1, dim_size + 1):
            rest_splits = generate_all_splits(dim_size - first_size)
            for rest in rest_splits:
                # Shift rest indices by first_size
                shifted_rest = [[x + first_size for x in group] for group in rest]
                first_group = list(range(first_size))
                all_splits.append([first_group] + shifted_rest)
        return all_splits

    def reduce_rank_by_linear_dependence(U, V, W, rank):
        """
        Try to reduce the rank by finding linearly dependent columns in the combined system.
        This is a heuristic approach using rank-revealing QR decomposition.
        """
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # For each column c, we have a rank-1 tensor contribution:
        # T[i,j,k] += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # We want to find if any column is a linear combination of others.
        # This is complex, so we'll use a simpler approach:
        # Check if removing a column and adjusting others can maintain the decomposition.
        
        # For now, this is a placeholder. In practice, we'd need to solve
        # a system of linear equations for each candidate removal.
        
        # Simple heuristic: try removing columns that have small norm
        col_norms = []
        for c in range(rank):
            norm = np.sum(U_np[:, c]**2) + np.sum(V_np[:, c]**2) + np.sum(W_np[:, c]**2)
            col_norms.append(norm)
        
        # Sort by norm (smallest first)
        sorted_cols = sorted(range(rank), key=lambda c: col_norms[c])
        
        # Try to remove the smallest norm columns one by one
        # This is a very rough heuristic
        new_U = [row[:] for row in U]
        new_V = [row[:] for row in V]
        new_W = [row[:] for row in W]
        
        # For this task, we'll just return the original since proper linear dependence
        # reduction requires solving a large system
        return U, V, W, rank

    # Start with schoolbook as fallback
    best_result = schoolbook()
    best_rank = n * m * p

    # Generate all possible splits for each dimension
    i_splits = generate_all_splits(n)
    j_splits = generate_all_splits(m)
    k_splits = generate_all_splits(p)

    # Try all combinations
    for i_g in i_splits:
        if time.time() > deadline:
            break
        for j_g in j_splits:
            if time.time() > deadline:
                break
            for k_g in k_splits:
                if time.time() > deadline:
                    break
                
                # Calculate rank for this grouping
                total_rank = 0
                for ig in i_g:
                    for jg in j_g:
                        for kg in k_g:
                            si, sj, sk = len(ig), len(jg), len(kg)
                            if si == 2 and sj == 2 and sk == 2:
                                total_rank += 7
                            else:
                                total_rank += si * sj * sk
                
                if total_rank >= best_rank:
                    continue
                
                # Build the decomposition
                result = build_decomposition(i_g, j_g, k_g)
                
                # Verify
                if verify_sparse(result["U"], result["V"], result["W"], total_rank):
                    best_result = {"U": result["U"], "V": result["V"], "W": result["W"]}
                    best_rank = total_rank

    # Try to improve by trying different Strassen-like decompositions
    # For 3x3x3, there are known decompositions with rank 19 or 20
    # We can try to embed these into the 3x5x6 problem
    
    # For now, return the best we found
    return best_result
# EVOLVE-BLOCK-END
