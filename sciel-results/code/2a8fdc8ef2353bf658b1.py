"""Tensor decomposition for 3x5x6 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s limit

    n, m, p = DIMENSIONS  # 3, 5, 6

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

    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = 0.0
                    for c in range(rank):
                        actual += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(actual - 1.0) > 1e-9:
                        return False
        return True

    def strassen_2x2x2_terms():
        """
        Return the 7 Strassen terms for 2x2x2 matrix multiplication.
        Each term is (U_pattern, V_pattern, W_pattern).
        """
        terms = []
        # M1 = (A11+A22)(B11+B22)
        u1 = {(0,0): 1, (1,1): 1}
        v1 = {(0,0): 1, (1,1): 1}
        w1 = {(0,0): 1, (1,1): 1}
        terms.append((u1, v1, w1))
        
        # M2 = (A21+A22)B11
        u2 = {(1,0): 1, (1,1): 1}
        v2 = {(0,0): 1}
        w2 = {(0,1): 1, (1,1): -1}
        terms.append((u2, v2, w2))
        
        # M3 = A11(B21-B11)
        u3 = {(0,0): 1}
        v3 = {(0,1): 1, (1,1): -1}
        w3 = {(1,0): 1, (1,1): 1}
        terms.append((u3, v3, w3))
        
        # M4 = A22(B12-B22)
        u4 = {(1,1): 1}
        v4 = {(1,0): 1, (0,0): -1}
        w4 = {(0,0): 1, (0,1): 1}
        terms.append((u4, v4, w4))
        
        # M5 = (A11+A12)B22
        u5 = {(0,0): 1, (0,1): 1}
        v5 = {(1,1): 1}
        w5 = {(0,0): -1, (1,0): 1}
        terms.append((u5, v5, w5))
        
        # M6 = (A21-A11)(B11+B12)
        u6 = {(1,0): 1, (0,0): -1}
        v6 = {(0,0): 1, (0,1): 1}
        w6 = {(1,1): 1}
        terms.append((u6, v6, w6))
        
        # M7 = (A12-A22)(B21+B22)
        u7 = {(0,1): 1, (1,1): -1}
        v7 = {(1,0): 1, (1,1): 1}
        w7 = {(0,0): 1}
        terms.append((u7, v7, w7))
        
        return terms

    def build_recursive_strassen():
        """
        Build a decomposition using recursive Strassen where all three dims are 2,
        and schoolbook otherwise.
        """
        i_groups = [[0, 1], [2]]
        j_groups = [[0, 1], [2, 3], [4]]
        k_groups = [[0, 1], [2, 3], [4, 5]]
        
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
        
        return {"U": U, "V": V, "W": W}

    def build_extended_strassen():
        """
        Try more aggressive Strassen-like partitions.
        For 3x5x6, we can try:
        - i: [0,1], [2] (sizes 2,1)
        - j: [0,1,2], [3,4] (sizes 3,2) 
        - k: [0,1,2], [3,4,5] (sizes 3,3)
        
        Or try to use rank-10 for 2x3x3 blocks (if known), etc.
        For now, use schoolbook for non-2x2x2 blocks but try different groupings.
        """
        # Try several partition strategies
        best_rank = n * m * p
        best_result = None
        
        # Strategy 1: Current one
        partitions = [
            ([[0, 1], [2]], [[0, 1], [2, 3], [4]], [[0, 1], [2, 3], [4, 5]]),
            # Strategy 2: Different j grouping
            ([[0, 1], [2]], [[0, 1, 2], [3, 4]], [[0, 1], [2, 3], [4, 5]]),
            # Strategy 3: Different k grouping
            ([[0, 1], [2]], [[0, 1], [2, 3], [4]], [[0, 1, 2], [3, 4, 5]]),
            # Strategy 4: Try to get more 2x2x2 blocks
            ([[0, 1], [2]], [[0, 1], [2, 3], [4]], [[0, 1], [2, 3], [4, 5]]),
        ]
        
        for i_groups, j_groups, k_groups in partitions:
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
            
            if total_rank < best_rank:
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
                
                if verify(U, V, W, total_rank):
                    best_rank = total_rank
                    best_result = {"U": U, "V": V, "W": W}
        
        return best_result, best_rank

    # Start with schoolbook as fallback
    best_result = schoolbook()
    best_rank = n * m * p

    # Try recursive Strassen decomposition
    if time.time() < deadline:
        improved = build_recursive_strassen()
        improved_rank = len(improved["U"][0])
        if improved_rank < best_rank:
            if verify(improved["U"], improved["V"], improved["W"], improved_rank):
                best_result = improved
                best_rank = improved_rank

    # Try extended Strassen with different partitions
    if time.time() < deadline:
        ext_result, ext_rank = build_extended_strassen()
        if ext_result is not None and ext_rank < best_rank:
            best_result = ext_result
            best_rank = ext_rank

    # Try gradient-based optimization to find lower rank decompositions
    if time.time() < deadline:
        def tensor_residual(U_mat, V_mat, W_mat):
            """Compute the residual tensor for a given decomposition."""
            # U_mat: (n*m, r), V_mat: (m*p, r), W_mat: (p*n, r)
            r = U_mat.shape[1]
            residual = np.zeros((n, m, p))
            for c in range(r):
                u_col = U_mat[:, c].reshape(n, m)
                v_col = V_mat[:, c].reshape(m, p)
                w_col = W_mat[:, c].reshape(p, n)
                # The tensor entry (i,j,k) gets contribution U[i,j]*V[j,k]*W[k,i]
                # We need to compute sum over c of U[i,j,c]*V[j,k,c]*W[k,i,c]
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            residual[i,j,k] += u_col[i,j] * v_col[j,k] * w_col[k,i]
            return residual - 1.0  # Target is all ones

        def frob_norm(residual):
            return np.sqrt(np.sum(residual**2))

        # Try gradient descent for various target ranks
        rng = np.random.RandomState(42)
        for target_rank in range(70, best_rank):
            if time.time() > deadline - 10:
                break
            
            found = False
            for trial in range(3):
                if time.time() > deadline - 10:
                    break
                
                U_mat = rng.randn(n * m, target_rank) * 0.1
                V_mat = rng.randn(m * p, target_rank) * 0.1
                W_mat = rng.randn(p * n, target_rank) * 0.1
                
                # Initialize with schoolbook-like structure for better starting point
                for c in range(min(target_rank, n*m*p)):
                    i = (c // (m*p)) % n
                    j = (c // p) % m
                    k = c % p
                    U_mat[i*m+j, c] = 1.0
                    V_mat[j*p+k, c] = 1.0
                    W_mat[k*n+i, c] = 1.0
                
                lr = 0.01
                prev_loss = frob_norm(tensor_residual(U_mat, V_mat, W_mat))
                
                for step in range(200):
                    if time.time() > deadline - 10:
                        break
                    
                    residual = tensor_residual(U_mat, V_mat, W_mat)
                    loss = frob_norm(residual)
                    
                    if loss < 1e-8:
                        # Found exact decomposition!
                        U_list = U_mat.tolist()
                        V_list = V_mat.tolist()
                        W_list = W_mat.tolist()
                        # Round to nearest rational if close
                        for row in U_list:
                            for idx in range(len(row)):
                                row[idx] = round(row[idx], 10)
                        for row in V_list:
                            for idx in range(len(row)):
                                row[idx] = round(row[idx], 10)
                        for row in W_list:
                            for idx in range(len(row)):
                                row[idx] = round(row[idx], 10)
                        
                        if verify(U_list, V_list, W_list, target_rank):
                            best_result = {"U": U_list, "V": V_list, "W": W_list}
                            best_rank = target_rank
                            found = True
                            break
                    
                    # Compute gradients numerically
                    eps = 1e-5
                    U_grad = np.zeros_like(U_mat)
                    V_grad = np.zeros_like(V_mat)
                    W_grad = np.zeros_like(W_mat)
                    
                    # Only compute a subset of gradients for speed
                    if step % 5 == 0:
                        # Full gradient computation
                        for idx in range(n * m):
                            for c in range(target_rank):
                                orig = U_mat[idx, c]
                                U_mat[idx, c] = orig + eps
                                loss_plus = frob_norm(tensor_residual(U_mat, V_mat, W_mat))
                                U_mat[idx, c] = orig
                                U_grad[idx, c] = (loss_plus - prev_loss) / eps
                        for idx in range(m * p):
                            for c in range(target_rank):
                                orig = V_mat[idx, c]
                                V_mat[idx, c] = orig + eps
                                loss_plus = frob_norm(tensor_residual(U_mat, V_mat, W_mat))
                                V_mat[idx, c] = orig
                                V_grad[idx, c] = (loss_plus - prev_loss) / eps
                        for idx in range(p * n):
                            for c in range(target_rank):
                                orig = W_mat[idx, c]
                                W_mat[idx, c] = orig + eps
                                loss_plus = frob_norm(tensor_residual(U_mat, V_mat, W_mat))
                                W_mat[idx, c] = orig
                                W_grad[idx, c] = (loss_plus - prev_loss) / eps
                    
                    U_mat -= lr * U_grad
                    V_mat -= lr * V_grad
                    W_mat -= lr * W_grad
                    
                    prev_loss = loss
                    lr *= 0.995  # Slow decay
                
                if found:
                    break
        
        if found:
            break

    return best_result
# EVOLVE-BLOCK-END
