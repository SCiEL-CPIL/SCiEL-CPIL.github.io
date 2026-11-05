"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

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

    def try_polynomial_interpolation():
        """
        Try to find a lower-rank decomposition using polynomial interpolation techniques.
        This exploits the structure of the tensor to find linear dependencies.
        """
        # For 3x5x6, we can try to find a decomposition with rank < 90
        # by exploiting the non-square structure
        
        # Strategy: Use the fact that the tensor has a specific structure
        # and try to find linear combinations that reduce the rank
        
        # First, let's try a systematic search for small perturbations of schoolbook
        # that might reduce the rank
        
        # The key insight: if we can find vectors u, v, w such that
        # sum_c u[i,j,c] * v[j,k,c] * w[k,i,c] = delta[i,j,k]
        # with fewer terms, we have a better decomposition.
        
        # For now, let's try a few hand-crafted patterns
        
        # Pattern 1: Try to merge some schoolbook terms
        # This is a placeholder for a more sophisticated search
        
        return None

    # Start with schoolbook as fallback
    best_result = schoolbook()
    best_rank = n * m * p

    # Try various groupings to minimize rank
    # For n=3, m=5, p=6:
    # i can be split as [0,1],[2] or [0],[1,2]
    # j can be split in various ways
    # k can be split in various ways
    
    i_splits = [
        [[0, 1], [2]],
        [[0], [1, 2]],
        [[0], [1], [2]],
    ]
    j_splits = [
        [[0, 1], [2, 3], [4]],
        [[0], [1, 2], [3, 4]],
        [[0, 1], [2], [3, 4]],
        [[0], [1], [2, 3, 4]],
        [[0, 1], [2, 3, 4]],
        [[0], [1, 2, 3], [4]],
        [[0, 1, 2], [3, 4]],
        [[0], [1], [2], [3, 4]],
        [[0, 1], [2], [3], [4]],
        [[0], [1, 2], [3], [4]],
        [[0, 1], [2, 3, 4]],
        [[0], [1, 2, 3], [4]],
        [[0, 1, 2], [3], [4]],
        [[0], [1], [2, 3], [4]],
        [[0], [1], [2], [3], [4]],
    ]
    k_splits = [
        [[0, 1], [2, 3], [4, 5]],
        [[0], [1, 2], [3, 4, 5]],
        [[0, 1], [2], [3, 4, 5]],
        [[0], [1], [2, 3], [4, 5]],
        [[0, 1], [2, 3], [4], [5]],
        [[0], [1, 2], [3], [4, 5]],
        [[0, 1], [2], [3], [4, 5]],
        [[0], [1], [2], [3], [4, 5]],
        [[0, 1, 2], [3, 4, 5]],
        [[0], [1, 2, 3], [4, 5]],
        [[0], [1], [2, 3, 4], [5]],
        [[0, 1], [2, 3, 4], [5]],
        [[0], [1, 2], [3, 4], [5]],
        [[0, 1], [2], [3, 4], [5]],
        [[0], [1], [2], [3, 4], [5]],
        [[0], [1, 2], [3], [4], [5]],
        [[0, 1], [2], [3], [4], [5]],
        [[0], [1], [2], [3], [4], [5]],
        [[0, 1, 2], [3], [4], [5]],
        [[0], [1, 2, 3], [4], [5]],
        [[0], [1], [2, 3], [4], [5]],
        [[0], [1], [2], [3, 4], [5]],
        [[0, 1], [2, 3], [4], [5]],
        [[0], [1, 2], [3, 4], [5]],
    ]
    
    # Try all combinations
    for i_g in i_splits:
        for j_g in j_splits:
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

    # Try polynomial interpolation approach if we have time
    if time.time() < deadline - 30:
        poly_result = try_polynomial_interpolation()
        if poly_result is not None:
            poly_rank = len(poly_result["U"][0])
            if poly_rank < best_rank:
                if verify_sparse(poly_result["U"], poly_result["V"], poly_result["W"], poly_rank):
                    best_result = poly_result
                    best_rank = poly_rank

    return best_result
# EVOLVE-BLOCK-END
