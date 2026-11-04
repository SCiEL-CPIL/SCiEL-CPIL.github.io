import time
import sys
import numpy as np
from collections import defaultdict

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 4, 6
    nm, mp, pn = n * m, m * p, p * n  # 8, 24, 12

    def verify(U_mat, V_mat, W_mat, tol=1e-8):
        r = U_mat.shape[1]
        # Check all 1s
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = 0.0
                    for r_idx in range(r):
                        s += U_mat[a, r_idx] * V_mat[b, r_idx] * W_mat[c, r_idx]
                    if abs(s - 1.0) > tol:
                        return False
        
        # Check all 0s
        valid_set = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    valid_set.add((i*m+j, j*p+k, k*n+i))
                    
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    if (a,b,c) in valid_set:
                        continue
                    s = 0.0
                    for r_idx in range(r):
                        s += U_mat[a, r_idx] * V_mat[b, r_idx] * W_mat[c, r_idx]
                    if abs(s) > tol:
                        return False
        return True

    # 1. Construct Rank 42 (Strassen blocks)
    rank42 = 42
    U42 = np.zeros((nm, rank42), dtype=float)
    V42 = np.zeros((mp, rank42), dtype=float)
    W42 = np.zeros((pn, rank42), dtype=float)

    def add_strassen_block(i_start, j_start, k_start, offset):
        a00 = i_start * m + j_start
        a01 = i_start * m + j_start + 1
        a10 = (i_start + 1) * m + j_start
        a11 = (i_start + 1) * m + j_start + 1
        
        b00 = j_start * p + k_start
        b01 = j_start * p + k_start + 1
        b10 = (j_start + 1) * p + k_start
        b11 = (j_start + 1) * p + k_start + 1
        
        w00 = k_start * n + i_start
        w01 = (k_start + 1) * n + i_start
        w10 = k_start * n + i_start + 1
        w11 = (k_start + 1) * n + i_start + 1
        
        U42[a00, offset] += 1.0; U42[a11, offset] += 1.0
        V42[b00, offset] += 1.0; V42[b11, offset] += 1.0
        W42[w00, offset] += 1.0; W42[w11, offset] += 1.0

        U42[a10, offset+1] += 1.0; U42[a11, offset+1] += 1.0
        V42[b00, offset+1] += 1.0
        W42[w10, offset+1] += 1.0; W42[w11, offset+1] -= 1.0

        U42[a00, offset+2] += 1.0
        V42[b01, offset+2] += 1.0; V42[b11, offset+2] -= 1.0
        W42[w01, offset+2] += 1.0; W42[w11, offset+2] += 1.0

        U42[a11, offset+3] += 1.0
        V42[b10, offset+3] += 1.0; V42[b00, offset+3] -= 1.0
        W42[w00, offset+3] += 1.0; W42[w10, offset+3] += 1.0

        U42[a00, offset+4] += 1.0; U42[a01, offset+4] += 1.0
        V42[b11, offset+4] += 1.0
        W42[w00, offset+4] -= 1.0; W42[w01, offset+4] += 1.0

        U42[a10, offset+5] += 1.0; U42[a00, offset+5] -= 1.0
        V42[b00, offset+5] += 1.0; V42[b01, offset+5] += 1.0
        W42[w11, offset+5] += 1.0

        U42[a01, offset+6] += 1.0; U42[a11, offset+6] -= 1.0
        V42[b10, offset+6] += 1.0; V42[b11, offset+6] += 1.0
        W42[w00, offset+6] += 1.0

    offset = 0
    for i_start in range(0, n, 2):
        for j_start in range(0, m, 2):
            for k_start in range(0, p, 2):
                add_strassen_block(i_start, j_start, k_start, offset)
                offset += 7

    best_U, best_V, best_W = U42.copy(), V42.copy(), W42.copy()
    best_rank = 42

    if not verify(best_U, best_V, best_W):
        # Fallback to schoolbook rank 48
        rank0 = n * m * p
        U = np.zeros((nm, rank0), dtype=float)
        V = np.zeros((mp, rank0), dtype=float)
        W = np.zeros((pn, rank0), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    col = i * (m * p) + j * p + k
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Systematic merging: group columns by (U, V) signature
    # Two columns can be merged if U_i = s * U_j and V_i = t * V_j
    # Then term_j = U_j V_j W_j = (1/s)(1/t) U_i V_i W_j
    # So we can add (1/s)(1/t) W_j to W_i

    def merge_columns(U_mat, V_mat, W_mat):
        """Merge all possible columns systematically."""
        current_U = U_mat.copy()
        current_V = V_mat.copy()
        current_W = W_mat.copy()
        r = current_U.shape[1]
        
        # Create signatures for each column
        # A column is identified by its U and V vectors
        # We'll use a dictionary mapping (U_tuple, V_tuple) -> list of column indices
        # But we need to handle proportional vectors
        
        # Better approach: iterate and try to merge each column into an existing one
        keep = [True] * r
        
        for i in range(r):
            if not keep[i]:
                continue
            for j in range(i + 1, r):
                if not keep[j]:
                    continue
                
                # Check if U_i and U_j are proportional
                nz_i = np.nonzero(current_U[:, i])[0]
                nz_j = np.nonzero(current_U[:, j])[0]
                
                if len(nz_i) == 0 and len(nz_j) == 0:
                    s = 1.0
                elif len(nz_i) == 0 or len(nz_j) == 0:
                    continue
                elif not np.array_equal(nz_i, nz_j):
                    continue
                else:
                    ratios = current_U[nz_i, i] / current_U[nz_i, j]
                    if np.std(ratios) > 1e-10:
                        continue
                    s = ratios[0]
                    if abs(s) < 1e-10:
                        continue
                
                # Check if V_i and V_j are proportional
                nz_v_i = np.nonzero(current_V[:, i])[0]
                nz_v_j = np.nonzero(current_V[:, j])[0]
                
                if len(nz_v_i) == 0 and len(nz_v_j) == 0:
                    t = 1.0
                elif len(nz_v_i) == 0 or len(nz_v_j) == 0:
                    continue
                elif not np.array_equal(nz_v_i, nz_v_j):
                    continue
                else:
                    ratios_v = current_V[nz_v_i, i] / current_V[nz_v_i, j]
                    if np.std(ratios_v) > 1e-10:
                        continue
                    t = ratios_v[0]
                    if abs(t) < 1e-10:
                        continue
                
                # Merge j into i
                scale = 1.0 / (s * t)
                current_W[:, i] += scale * current_W[:, j]
                keep[j] = False
        
        # Build result
        kept_indices = [i for i in range(r) if keep[i]]
        return current_U[:, kept_indices], current_V[:, kept_indices], current_W[:, kept_indices], len(kept_indices)

    # Apply merging iteratively until no more merges possible
    changed = True
    while changed:
        if time.time() > deadline:
            break
        new_U, new_V, new_W, new_rank = merge_columns(best_U, best_V, best_W)
        if new_rank < best_rank:
            if verify(new_U, new_V, new_W):
                best_U, best_V, best_W = new_U, new_V, new_W
                best_rank = new_rank
                changed = True
            else:
                changed = False
        else:
            changed = False

    # New: Try to reduce rank by finding linear dependencies in the W columns
    # If W_k = alpha * W_i + beta * W_j, we might be able to eliminate a column
    # by adjusting the U and V matrices accordingly
    if time.time() < deadline - 10:
        # Check for W column linear dependencies
        r = best_U.shape[1]
        W_mat = best_W
        U_mat = best_U
        V_mat = best_V
        
        # Find if any W column is a linear combination of others
        # For each column k, check if it's in the span of all other columns
        for k in range(r):
            if time.time() > deadline:
                break
            
            # Get all other W columns
            other_cols = list(range(r))
            other_cols.remove(k)
            
            # Check if W[:, k] is in the span of W[:, other_cols]
            # Use least squares to check
            W_others = W_mat[:, other_cols]
            W_k = W_mat[:, k]
            
            if W_k.size == 0:
                continue
                
            # Solve W_others @ coeffs = W_k
            # If the residual is small, then W_k is in the span
            try:
                coeffs, residuals, rank_w, sv = np.linalg.lstsq(W_others, W_k, rcond=1e-10)
            except:
                continue
            
            if len(residuals) > 0 and residuals[0] < 1e-8:
                # W_k is in the span of other columns
                # We can try to eliminate column k by expressing it as a combination
                # This requires adjusting U and V, which is complex
                # For now, just note this as a potential improvement
                pass

    if best_rank < 42 and verify(best_U, best_V, best_W):
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # For now, return the best we have
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

if __name__ == "__main__":
    result = construct()
    print("Constructed", file=sys.stderr)
