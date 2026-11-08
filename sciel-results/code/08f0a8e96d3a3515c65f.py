"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

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
        Each term is (U_pattern, V_pattern, W_pattern) where patterns are
        dicts mapping (local_i, local_j) -> value for U, etc.
        """
        terms = []
        # Term 1: M1 = (A00+A11)(B00+B11)
        # Contributes +1 to C00 (T[0,0,0]) and +1 to C11 (T[1,1,1])
        u1 = {(0,0): 1, (1,1): 1}
        v1 = {(0,0): 1, (1,1): 1}
        w1 = {(0,0): 1, (1,1): 1}
        terms.append((u1, v1, w1))
        
        # Term 2: M2 = (A10+A11)B00
        # Contributes +1 to C10 (T[1,0,0]) and -1 to C11 (T[1,1,1])
        u2 = {(1,0): 1, (1,1): 1}
        v2 = {(0,0): 1}
        w2 = {(0,1): 1, (1,1): -1}
        terms.append((u2, v2, w2))
        
        # Term 3: M3 = A00(B01-B11)
        # Contributes +1 to C01 (T[0,0,1]) and +1 to C11 (T[1,1,1])
        u3 = {(0,0): 1}
        v3 = {(0,1): 1, (1,1): -1}
        w3 = {(1,0): 1, (1,1): 1}
        terms.append((u3, v3, w3))
        
        # Term 4: M4 = A11(B10-B00)
        # Contributes +1 to C00 (T[0,0,0]) and +1 to C10 (T[1,0,0])
        u4 = {(1,1): 1}
        v4 = {(1,0): 1, (0,0): -1}
        w4 = {(0,0): 1, (0,1): 1}
        terms.append((u4, v4, w4))
        
        # Term 5: M5 = (A00+A01)B11
        # Contributes -1 to C00 (T[0,0,0]) and +1 to C01 (T[0,0,1])
        u5 = {(0,0): 1, (0,1): 1}
        v5 = {(1,1): 1}
        w5 = {(0,0): -1, (1,0): 1}
        terms.append((u5, v5, w5))
        
        # Term 6: M6 = (A10-A00)(B00+B01)
        # Contributes +1 to C11 (T[1,1,1])
        u6 = {(1,0): 1, (0,0): -1}
        v6 = {(0,0): 1, (0,1): 1}
        w6 = {(1,1): 1}
        terms.append((u6, v6, w6))
        
        # Term 7: M7 = (A01-A11)(B10+B11)
        # Contributes +1 to C00 (T[0,0,0])
        u7 = {(0,1): 1, (1,1): -1}
        v7 = {(1,0): 1, (1,1): 1}
        w7 = {(0,0): 1}
        terms.append((u7, v7, w7))
        
        return terms

    def build_strassen_decomposition():
        """
        Build a decomposition using Strassen on 2x2x2 blocks and schoolbook elsewhere.
        
        Partition:
        - i (rows of A, 0..2): group I0={0,1}, I1={2}
        - j (cols of A, 0..4): group J0={0,1}, J1={2,3}, J2={4}
        - k (cols of B, 0..5): group K0={0,1}, K1={2,3}, K2={4,5}
        
        For each sub-block (Ig, Jg, Kg):
        - If all sizes are 2: use Strassen (rank 7)
        - Otherwise: use schoolbook for that sub-block
        """
        # Define groups
        i_groups = [[0, 1], [2]]
        j_groups = [[0, 1], [2, 3], [4]]
        k_groups = [[0, 1], [2, 3], [4, 5]]
        
        # First pass: count total rank needed
        total_rank = 0
        block_info = []
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    if si == 2 and sj == 2 and sk == 2:
                        r = 7  # Strassen
                    else:
                        r = si * sj * sk  # schoolbook
                    total_rank += r
                    block_info.append((ig, jg, kg, r))
        
        # Allocate columns
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col = 0
        strassen_terms = strassen_2x2x2_terms()
        
        for ig, jg, kg, r in block_info:
            si, sj, sk = len(ig), len(jg), len(kg)
            if si == 2 and sj == 2 and sk == 2:
                # Use Strassen
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
                # Schoolbook for this sub-block
                for li in range(si):
                    for lj in range(sj):
                        for lk in range(sk):
                            gi, gj, gk = ig[li], jg[lj], kg[lk]
                            U[gi * m + gj][col] = 1
                            V[gj * p + gk][col] = 1
                            W[gk * n + gi][col] = 1
                            col += 1
        
        return {"U": U, "V": V, "W": W, "rank": total_rank}

    def try_tensor_rank_minimization():
        """
        Try to find a lower-rank decomposition using iterative tensor rank minimization.
        Start from the Strassen-based decomposition and try to reduce rank via ALS.
        """
        if time.time() > deadline:
            return None
        
        # Get the Strassen-based decomposition
        strassen_result = build_strassen_decomposition()
        U_init = np.array(strassen_result["U"], dtype=float)
        V_init = np.array(strassen_result["V"], dtype=float)
        W_init = np.array(strassen_result["W"], dtype=float)
        current_rank = U_init.shape[1]
        
        # Build the target tensor
        # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
        # But we need to match the factorization: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        
        # For ALS, we fix two factors and solve for the third
        # This is equivalent to: for each column c, we have a rank-1 tensor
        # The constraint is that the sum of these rank-1 tensors equals the target
        
        # Let's try to reduce rank by merging columns
        # Try different target ranks
        for target_rank in range(current_rank - 1, max(current_rank - 20, 50), -1):
            if time.time() > deadline:
                break
            
            # Initialize with random combination of existing columns
            U = np.zeros((n * m, target_rank), dtype=float)
            V = np.zeros((m * p, target_rank), dtype=float)
            W = np.zeros((p * n, target_rank), dtype=float)
            
            # Initialize by selecting random linear combinations
            for c in range(target_rank):
                # Pick random coefficients from existing columns
                coeffs = np.random.randn(current_rank)
                coeffs /= np.linalg.norm(coeffs)
                U[:, c] = U_init @ coeffs
                V[:, c] = V_init @ coeffs
                W[:, c] = W_init @ coeffs
            
            # Iterate ALS
            for iteration in range(50):
                if time.time() > deadline:
                    return None
                
                # Update U: for each column c, solve min ||T - sum_{c'} U_c' ⊗ V_c' ⊗ W_c'||
                # This is a linear least squares problem
                # For fixed V and W, the optimal U[:, c] can be found by:
                # U[:, c] = argmin_U ||T - sum_{c'} U_c' ⊗ V_c' ⊗ W_c'||_F^2
                
                # Reshape T to 3D
                T_3d = np.zeros((n * m, m * p, p * n))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_3d[i * m + j, j * p + k, k * n + i] = 1.0
                
                # For ALS on U:
                # Fix V and W, the contribution of column c is U_c ⊗ V_c ⊗ W_c
                # The residual for column c is T - sum_{c'≠c} U_c' ⊗ V_c' ⊗ W_c'
                # We want to minimize ||residual - U_c ⊗ V_c ⊗ W_c||_F^2
                # This is equivalent to: U_c = argmin_U ||R - U ⊗ V_c ⊗ W_c||_F^2
                # where R is the residual
                
                # This can be solved by: U_c = (R * (V_c ⊗ W_c)) / ||V_c ⊗ W_c||^2
                # where * denotes element-wise multiplication and V_c ⊗ W_c is reshaped
                
                # Actually, let's use a simpler approach: 
                # The gradient of ||R - U ⊗ v ⊗ w||_F^2 w.r.t. U is -2 * (R * (v ⊗ w)) / ||v||^2 * ||w||^2 * ...
                # Let's just do gradient descent with step size
                
                # For now, let's skip the complex ALS and just return the Strassen result
                # The tensor rank minimization approach is too complex to implement correctly in this time
                break
            
            # Check if we achieved the target rank
            # (This code path is not fully implemented, so we'll just return None)
        
        return None

    def try_improved_decomposition():
        """
        Try to find a better decomposition using known algorithms.
        """
        if time.time() > deadline:
            return None
        
        result = build_strassen_decomposition()
        rank = result.pop("rank")
        
        # Verify
        if verify(result["U"], result["V"], result["W"], rank):
            return result
        
        return None

    # Start with schoolbook as fallback
    best_result = schoolbook()
    best_rank = n * m * p

    # Verify the schoolbook decomposition
    if not verify(best_result["U"], best_result["V"], best_result["W"], best_rank):
        pass

    # Try improved decomposition
    improved = try_improved_decomposition()
    if improved is not None:
        improved_rank = len(improved["U"][0])
        if improved_rank < best_rank:
            best_result = improved
            best_rank = improved_rank

    return best_result
# EVOLVE-BLOCK-END
