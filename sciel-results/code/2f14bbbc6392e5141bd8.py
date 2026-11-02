import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    t_start = time.time()
    deadline = t_start + 130  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Build the rank-28 Strassen decomposition
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    strassen_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),       # M1
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),              # M2
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),              # M3
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),              # M4
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),              # M5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),              # M6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),              # M7
    ]
    
    def map_u_a(side, local_idx):
        if side == 'L':
            return [0, 1, 4, 5][local_idx]
        else:
            return [2, 3, 6, 7][local_idx]
            
    def map_v_b(block, local_idx):
        r = local_idx // 2
        c = local_idx % 2
        if block == 'TL':
            return r * 4 + c
        elif block == 'TR':
            return r * 4 + (2 + c)
        elif block == 'BL':
            return (2 + r) * 4 + c
        elif block == 'BR':
            return (2 + r) * 4 + (2 + c)
            
    def map_w_c(side, local_idx):
        r = local_idx // 2
        c = local_idx % 2
        if side == 'L':
            k = c
        else:
            k = 2 + c
        return k * 2 + r

    def apply_strassen(a_side, b_block, c_side, start_col):
        col = start_col
        for u_local, v_local, w_local in strassen_terms:
            for idx, coeff in u_local:
                U[map_u_a(a_side, idx), col] += coeff
            for idx, coeff in v_local:
                V[map_v_b(b_block, idx), col] += coeff
            for idx, coeff in w_local:
                W[map_w_c(c_side, idx), col] += coeff
            col += 1
        return col

    col = apply_strassen('L', 'TL', 'L', 0)
    col = apply_strassen('R', 'BL', 'L', col)
    col = apply_strassen('L', 'TR', 'R', col)
    col = apply_strassen('R', 'BR', 'R', col)
    
    # Verify rank-28 is valid
    def verify_decomp(U, V, W):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        A = np.arange(n * m).reshape(n, m) + 1.0
        B = np.arange(m * p).reshape(m, p) + 2.0
        C = A @ B
        Cf = np.zeros((n, p))
        for ii in range(n):
            for kk in range(p):
                acc = 0.0
                for cc in range(U.shape[1]):
                    for jj in range(m):
                        acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
                Cf[ii, kk] = acc
        return np.allclose(C, Cf, atol=1e-10)
    
    if not verify_decomp(U, V, W):
        # Fallback: return as-is (should not happen)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Strategy: Try to find a rank-27 decomposition by checking if any column of W
    # is a linear combination of the others, which would allow merging two bilinear terms.
    # For each pair (i, j), check if W[:, i] can be expressed as alpha * W[:, j] + beta * (sum of others)
    # More directly: try to find if removing one column and adjusting others preserves the tensor.
    
    # A more systematic approach: for each column k, try to express W[:, k] as a linear combination
    # of the other 27 columns. If possible, we can absorb the corresponding U and V columns.
    
    best_rank = 28
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # For each column k to eliminate, solve for coefficients in the other columns
    for k in range(rank):
        if time.time() > deadline - 5:
            break
        # Try to express W[:, k] as linear combination of W[:, :k] + W[:, k+1:]
        other_cols = list(range(k)) + list(range(k + 1, rank))
        W_others = W[:, other_cols]  # shape (8, 27)
        W_target = W[:, k]  # shape (8,)
        
        # Solve: W_others @ coeffs = W_target (least squares)
        try:
            coeffs, residuals, rank_l, sv = np.linalg.lstsq(W_others, W_target, rcond=None)
        except:
            continue
        
        # Check if the solution is exact
        W_reconstructed = W_others @ coeffs
        if np.allclose(W_reconstructed, W_target, atol=1e-10):
            # Found a linear dependency! Now try to merge the bilinear terms.
            # The k-th term is: sum_i U[i,k] * sum_j V[j,k] * W[k*2+i, k]
            # We want to replace it with: sum_l coeffs[l] * (sum_i U[i, other_cols[l]] * sum_j V[j, other_cols[l]] * W[k*2+i, other_cols[l]])
            # This requires that U[:, k] = sum_l coeffs[l] * U[:, other_cols[l]] and similarly for V.
            
            U_others = U[:, other_cols]
            V_others = V[:, other_cols]
            U_target = U[:, k]
            V_target = V[:, k]
            
            # Check if U[:, k] is also a linear combination with the same coefficients
            U_check = U_others @ coeffs
            V_check = V_others @ coeffs
            
            if np.allclose(U_check, U_target, atol=1e-10) and np.allclose(V_check, V_target, atol=1e-10):
                # Perfect! We can eliminate column k.
                # New matrices have 27 columns
                new_U = U[:, other_cols].copy()
                new_V = V[:, other_cols].copy()
                new_W = W[:, other_cols].copy()
                
                # Verify
                if verify_decomp(new_U, new_V, new_W):
                    best_rank = 27
                    best_U, best_V, best_W = new_U, new_V, new_W
                    break
    
    if best_rank == 27:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
