"""Improved rank-minimal exact tensor decomposition for 2x4 by 4x6 matrix multiplication."""

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Build the rank-42 Strassen decomposition
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    def add_strassen_block(a_idx, b_idx, w_idx, start_col):
        c = start_col
        # M1 = (a00+a11)(b00+b11)
        U[a_idx[0]][c] += 1; U[a_idx[3]][c] += 1
        V[b_idx[0]][c] += 1; V[b_idx[3]][c] += 1
        W[w_idx[0]][c] += 1; W[w_idx[3]][c] += 1
        # M2 = (a10+a11)b00
        U[a_idx[2]][c+1] += 1; U[a_idx[3]][c+1] += 1
        V[b_idx[0]][c+1] += 1
        W[w_idx[2]][c+1] += 1; W[w_idx[3]][c+1] -= 1
        # M3 = a00(b01-b11)
        U[a_idx[0]][c+2] += 1
        V[b_idx[1]][c+2] += 1; V[b_idx[3]][c+2] -= 1
        W[w_idx[1]][c+2] += 1; W[w_idx[3]][c+2] += 1
        # M4 = a11(b10-b00)
        U[a_idx[3]][c+3] += 1
        V[b_idx[2]][c+3] += 1; V[b_idx[0]][c+3] -= 1
        W[w_idx[0]][c+3] += 1; W[w_idx[2]][c+3] += 1
        # M5 = (a00+a01)b11
        U[a_idx[0]][c+4] += 1; U[a_idx[1]][c+4] += 1
        V[b_idx[3]][c+4] += 1
        W[w_idx[0]][c+4] -= 1; W[w_idx[1]][c+4] += 1
        # M6 = (a10-a00)(b00+b01)
        U[a_idx[2]][c+5] += 1; U[a_idx[0]][c+5] -= 1
        V[b_idx[0]][c+5] += 1; V[b_idx[1]][c+5] += 1
        W[w_idx[3]][c+5] += 1
        # M7 = (a01-a11)(b10+b11)
        U[a_idx[1]][c+6] += 1; U[a_idx[3]][c+6] -= 1
        V[b_idx[2]][c+6] += 1; V[b_idx[3]][c+6] += 1
        W[w_idx[0]][c+6] += 1

    a_L = [0, 1, 4, 5]
    a_R = [2, 3, 6, 7]
    
    b_1_T = [0, 1, 6, 7]
    b_1_B = [12, 13, 18, 19]
    w_1 = [0, 2, 1, 3]
    
    add_strassen_block(a_L, b_1_T, w_1, 0)
    add_strassen_block(a_R, b_1_B, w_1, 7)
    
    b_2_T = [2, 3, 8, 9]
    b_2_B = [14, 15, 20, 21]
    w_2 = [4, 6, 5, 7]
    
    add_strassen_block(a_L, b_2_T, w_2, 14)
    add_strassen_block(a_R, b_2_B, w_2, 21)
    
    b_3_T = [4, 5, 10, 11]
    b_3_B = [16, 17, 22, 23]
    w_3 = [8, 10, 9, 11]
    
    add_strassen_block(a_L, b_3_T, w_3, 28)
    add_strassen_block(a_R, b_3_B, w_3, 35)
    
    # Verify the decomposition
    def verify_decomp(U_mat, V_mat, W_mat, rk):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rk):
                        total += U_mat[i*m+j][r] * V_mat[j*p+k][r] * W_mat[k*n+i][r]
                    if total != 1:
                        return False
        return True
    
    if not verify_decomp(U, V, W, rank):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i*m+j][r] = 1
                    V[j*p+k][r] = 1
                    W[k*n+i][r] = 1
        return {"U": U, "V": V, "W": W}
    
    # Convert to float arrays
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    def to_list(arr):
        result = []
        for row in arr:
            result.append([])
            for val in row:
                frac_val = round(val * 2) / 2.0
                if abs(frac_val - round(frac_val)) < 1e-10:
                    frac_val = int(round(frac_val))
                result[-1].append(frac_val)
        return result
    
    def verify_float(final_U, final_V, final_W, new_rank, tol=1e-7):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(new_rank):
                        total += final_U[i*m+j, r] * final_V[j*p+k, r] * final_W[k*n+i, r]
                    if abs(total - 1.0) > tol:
                        return False
        return True
    
    # Build the tensor matrix
    flat_terms = []
    for r in range(rank):
        term = np.outer(np.outer(U_np[:, r], V_np[:, r]), W_np[:, r])
        flat_terms.append(term.flatten())
    M = np.array(flat_terms).T  # shape (2304, 42)
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank
    
    # Strategy 1: Correct rank reduction via null space with span checks
    try:
        U_svd, s_vals, Vt_svd = np.linalg.svd(M, full_matrices=False)
        tol = 1e-9
        effective_rank = int(np.sum(s_vals > tol))
        
        if effective_rank < rank and time.time() < deadline - 30:
            null_space = Vt_svd[effective_rank:, :]
            
            for null_vec in null_space:
                if time.time() > deadline - 25:
                    break
                
                # Find the column with the largest |coefficient| to eliminate
                max_idx = int(np.argmax(np.abs(null_vec)))
                max_val = abs(null_vec[max_idx])
                if max_val < 1e-8:
                    continue
                
                keep = [c for c in range(rank) if c != max_idx]
                
                # Check if u_max is in span of {u_c : c in keep}
                U_keep = U_np[:, keep]
                try:
                    U_coeffs, _, _, _ = np.linalg.lstsq(U_keep, U_np[:, max_idx], rcond=1e-10)
                    U_residual = np.linalg.norm(U_keep @ U_coeffs - U_np[:, max_idx])
                except Exception:
                    U_residual = 1e10
                
                # Check if v_max is in span of {v_c : c in keep}
                V_keep = V_np[:, keep]
                try:
                    V_coeffs, _, _, _ = np.linalg.lstsq(V_keep, V_np[:, max_idx], rcond=1e-10)
                    V_residual = np.linalg.norm(V_keep @ V_coeffs - V_np[:, max_idx])
                except Exception:
                    V_residual = 1e10
                
                # Check if w_max is in span of {w_c : c in keep}
                W_keep = W_np[:, keep]
                try:
                    W_coeffs, _, _, _ = np.linalg.lstsq(W_keep, W_np[:, max_idx], rcond=1e-10)
                    W_residual = np.linalg.norm(W_keep @ W_coeffs - W_np[:, max_idx])
                except Exception:
                    W_residual = 1e10
                
                if U_residual < 1e-8 and V_residual < 1e-8 and W_residual < 1e-8:
                    # All three vectors are in the span, so we can eliminate
                    new_rank = len(keep)
                    final_U = U_np[:, keep]
                    final_V = V_np[:, keep]
                    final_W = W_np[:, keep]
                    
                    if verify_float(final_U, final_V, final_W, new_rank) and new_rank < best_rank:
                        best_rank = new_rank
                        best_result = {
                            "U": to_list(final_U),
                            "V": to_list(final_V),
                            "W": to_list(final_W)
                        }
    except Exception:
        pass
    
    # Strategy 2: Use structured decomposition exploiting rectangular geometry
    # The key insight: m=4 is small, and we can view this as multiplying
    # A (2x4) by B (4x6). We can split B into 3 blocks of 2 columns each.
    # For each 2x4 * 4x2 product, we can use a rank-6 decomposition
    # (better than 2*7=14 from two independent 2x2 Strassen blocks)
    # Total: 3 blocks * 6 = 18? No, that's too optimistic.
    # Let's try a direct approach: find a rank-36 decomposition
    
    if time.time() < deadline - 40:
        try:
            # Construct a rank-36 decomposition using the following strategy:
            # View the multiplication as C = A * B where A is 2x4, B is 4x6
            # Split B into 3 blocks: B1 (cols 0-1), B2 (cols 2-3), B3 (cols 4-5)
            # Each C_block = A * B_block is 2x2
            # For each 2x4 * 4x2, we can potentially use 6 multiplications
            # instead of 14 (two separate 2x2 Strassen)
            
            # Actually, let's try a different approach: use the fact that
            # the middle dimension is small. We can write:
            # C(i,k) = sum_j A(i,j) * B(j,k)
            # = sum_j A(i,j) * B(j,k)
            
            # Let's try to find a rank-36 solution by solving a linear system
            # with structured constraints.
            
            # Alternative: Try to reduce the rank by finding linear dependencies
            # in the Strassen decomposition that exploit the rectangular shape.
            
            # For now, let's try a more aggressive null-space search
            # with multiple null vectors applied simultaneously.
            
            current_U = U_np.copy()
            current_V = V_np.copy()
            current_W = W_np.copy()
            current_rank = rank
            eliminated = set()
            
            # Iteratively try to eliminate columns
            max_iterations = 10
            for iteration in range(max_iterations):
                if time.time() > deadline - 35:
                    break
                
                remaining = [c for c in range(rank) if c not in eliminated]
                if len(remaining) <= 30:
                    break
                
                # Build the matrix for remaining terms
                M_sub = np.zeros((n*m*p, len(remaining)))
                for idx, col in enumerate(remaining):
                    term = np.outer(np.outer(current_U[:, col], current_V[:, col]), current_W[:, col])
                    M_sub[:, idx] = term.flatten()
                
                try:
                    U_sub, s_sub, Vt_sub = np.linalg.svd(M_sub, full_matrices=False)
                except np.linalg.LinAlgError:
                    break
                
                tol_sub = 1e-9
                eff_rank_sub = int(np.sum(s_sub > tol_sub))
                
                if eff_rank_sub >= len(remaining):
                    break
                
                null_sub = Vt_sub[eff_rank_sub:, :]
                
                if null_sub.shape[0] == 0:
                    break
                
                # Try to eliminate multiple columns at once
                found_elimination = False
                for nv_idx in range(null_sub.shape[0]):
                    if time.time() > deadline - 33:
                        break
                    
                    nv = null_sub[nv_idx]
                    
                    # Try each possible column to eliminate
                    for local_idx in range(len(remaining)):
                        if abs(nv[local_idx]) < 1e-10:
                            continue
                        
                        best_col = remaining[local_idx]
                        keep = [c for c in remaining if c != best_col]
                        
                        # Check span membership
                        U_keep = current_U[:, keep]
                        V_keep = current_V[:, keep]
                        W_keep = current_W[:, keep]
                        
                        try:
                            U_coeffs, _, _, _ = np.linalg.lstsq(U_keep, current_U[:, best_col], rcond=1e-10)
                            U_res = np.linalg.norm(U_keep @ U_coeffs - current_U[:, best_col])
                            V_coeffs, _, _, _ = np.linalg.lstsq(V_keep, current_V[:, best_col], rcond=1e-10)
                            V_res = np.linalg.norm(V_keep @ V_coeffs - current_V[:, best_col])
                            W_coeffs, _, _, _ = np.linalg.lstsq(W_keep, current_W[:, best_col], rcond=1e-10)
                            W_res = np.linalg.norm(W_keep @ W_coeffs - current_W[:, best_col])
                        except Exception:
                            continue
                        
                        if U_res < 1e-8 and V_res < 1e-8 and W_res < 1e-8:
                            eliminated.add(best_col)
                            found_elimination = True
                            break
                    
                    if found_elimination:
                        break
                
                if not found_elimination:
                    break
            
            if eliminated and len(eliminated) > 0:
                keep = [c for c in range(rank) if c not in eliminated]
                new_rank = len(keep)
                final_U = current_U[:, keep]
                final_V = current_V[:, keep]
                final_W = current_W[:, keep]
                
                if verify_float(final_U, final_V, final_W, new_rank) and new_rank < best_rank:
                    best_rank = new_rank
                    best_result = {
                        "U": to_list(final_U),
                        "V": to_list(final_V),
                        "W": to_list(final_W)
                    }
        except Exception:
            pass
    
    return best_result

# EVOLVE-BLOCK-END
