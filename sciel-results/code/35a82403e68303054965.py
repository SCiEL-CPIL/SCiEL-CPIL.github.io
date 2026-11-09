import time
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time as _time
import numpy as np

def construct():
    _start = _time.time()
    _deadline = _start + 140
    n, m, p = DIMENSIONS  # all 6

    def construct_rank189():
        strassen_U = [
            [(0, 0, 1), (1, 1, 1)],
            [(1, 0, 1), (1, 1, 1)],
            [(0, 0, 1)],
            [(1, 1, 1)],
            [(0, 0, 1), (0, 1, 1)],
            [(1, 0, 1), (0, 0, -1)],
            [(0, 1, 1), (1, 1, -1)],
        ]
        strassen_V = [
            [(0, 0, 1), (1, 1, 1)],
            [(0, 0, 1)],
            [(0, 1, 1), (1, 1, -1)],
            [(1, 0, 1), (0, 0, -1)],
            [(1, 1, 1)],
            [(0, 0, 1), (0, 1, 1)],
            [(1, 0, 1), (1, 1, 1)],
        ]
        strassen_W = [
            [(0, 0, 1), (1, 1, 1)],
            [(1, 0, 1), (1, 1, -1)],
            [(0, 1, 1), (1, 1, 1)],
            [(0, 0, 1), (1, 0, 1)],
            [(0, 0, -1), (0, 1, 1)],
            [(1, 1, 1)],
            [(0, 0, 1)],
        ]
        rank = 189
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for bi in range(3):
            for bk in range(3):
                for bj in range(3):
                    for t in range(7):
                        for (li, lk, coeff) in strassen_U[t]:
                            gi = 2 * bi + li
                            gk = 2 * bk + lk
                            U[gi * m + gk][col] += coeff
                        for (lk, lj, coeff) in strassen_V[t]:
                            gk = 2 * bk + lk
                            gj = 2 * bj + lj
                            V[gk * p + gj][col] += coeff
                        for (li, lj, coeff) in strassen_W[t]:
                            gi = 2 * bi + li
                            gj = 2 * bj + lj
                            W[gj * n + gi][col] += coeff
                        col += 1
        return {"U": U, "V": V, "W": W, "rank": rank}

    result = construct_rank189()
    best_rank = 189
    best_result = result

    # Try to find a lower-rank decomposition via local search
    # Strategy: start from rank-189, try to zero out columns and fix
    if _time.time() < _deadline - 10:
        U_np = np.array(result["U"], dtype=float)
        V_np = np.array(result["V"], dtype=float)
        W_np = np.array(result["W"], dtype=float)
        
        # Verify the decomposition is correct
        # T[i,k,j] = sum_r U[i,k,r] * V[k,j,r] * W[j,i,r]
        # For matrix mult: C[i,j] = sum_k A[i,k]*B[k,j]
        # So T[i,k,j] should be 1 if this is a "valid" term, 0 otherwise
        # Actually the tensor is: for each (i,k,j), the coefficient is 1
        # The decomposition: sum_r U[i,k,r]*V[k,j,r]*W[j,i,r] = delta[i,k,j] for matmul tensor
        
        # Let's verify on a random instance
        np.random.seed(42)
        A = np.random.randint(-10, 10, size=(n, m))
        B = np.random.randint(-10, 10, size=(m, p))
        C_expected = A @ B
        
        # Compute C from decomposition
        # C[i,j] = sum_k sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
        C_computed = np.zeros((n, p))
        for r in range(best_rank):
            u_col = U_np[:, r]
            v_col = V_np[:, r]
            w_col = W_np[:, r]
            # Reshape
            u_mat = u_col.reshape(n, m)
            v_mat = v_col.reshape(m, p)
            w_mat = w_col.reshape(p, n)
            C_computed += u_mat @ v_mat * w_mat.T
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            # Decomposition is wrong, just return rank 189
            return {"U": best_result["U"], "V": best_result["V"], "W": best_result["W"]}
        
        # Try to reduce rank by setting some columns to zero
        # and checking if the decomposition still holds
        # Greedy: try to remove each column and see if it can be compensated
        # Simpler: try random subsets of columns to keep
        
        rng = np.random.RandomState(123)
        best_rank_found = best_rank
        best_U = U_np.copy()
        best_V = V_np.copy()
        best_W = W_np.copy()
        
        # Try to find a rank-r decomposition for r < 189
        # Use a simple approach: randomly zero out columns and try to fix
        # This is very slow, so let's try a more structured approach
        
        # Alternative: use the fact that many columns may be redundant
        # Try greedy column elimination
        for trial in range(50):
            if _time.time() > _deadline - 5:
                break
            
            # Try removing one column at a time
            removed = False
            for r_idx in range(best_rank_found - 1, -1, -1):
                if _time.time() > _deadline - 5:
                    break
                # Temporarily set column r_idx to zero
                old_U = best_U[:, r_idx].copy()
                old_V = best_V[:, r_idx].copy()
                old_W = best_W[:, r_idx].copy()
                best_U[:, r_idx] = 0
                best_V[:, r_idx] = 0
                best_W[:, r_idx] = 0
                
                # Check if decomposition still holds
                C_check = np.zeros((n, p))
                for r in range(best_rank_found):
                    if r == r_idx:
                        continue
                    u_mat = best_U[:, r].reshape(n, m)
                    v_mat = best_V[:, r].reshape(m, p)
                    w_mat = best_W[:, r].reshape(p, n)
                    C_check += u_mat @ v_mat * w_mat.T
                
                if np.allclose(C_check, C_expected, atol=1e-9):
                    # Column was redundant, keep it removed
                    best_rank_found -= 1
                    # Remove the column from arrays
                    mask = np.ones(best_rank_found + 1, dtype=bool)
                    mask[r_idx] = False
                    best_U = best_U[:, mask]
                    best_V = best_V[:, mask]
                    best_W = best_W[:, mask]
                    removed = True
                    break
                else:
                    # Restore
                    best_U[:, r_idx] = old_U
                    best_V[:, r_idx] = old_V
                    best_W[:, r_idx] = old_W
            
            if not removed:
                break
        
        if best_rank_found < best_rank:
            best_rank = best_rank_found
            best_result = {
                "U": best_U.tolist(),
                "V": best_V.tolist(),
                "W": best_W.tolist(),
                "rank": best_rank
            }
    
    if _time.time() > _deadline:
        return {"U": best_result["U"], "V": best_result["V"], "W": best_result["W"]}
        
    return {"U": best_result["U"], "V": best_result["V"], "W": best_result["W"]}

# EVOLVE-BLOCK-END
