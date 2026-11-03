import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def _time_ok(deadline):
    return time.time() < deadline

def _verify_decomposition(U, V, W, n, m, p):
    """Verify the tensor decomposition using random test matrices."""
    rank = len(U[0]) if U else 0
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # C[i,j] = sum_r sum_k U[i*m+k][r] * V[k*p+j][r] * W[j*n+i][r]
    
    import random
    random.seed(12345)
    A = np.array([[random.randint(-3, 3) for _ in range(m)] for _ in range(n)], dtype=float)
    B = np.array([[random.randint(-3, 3) for _ in range(p)] for _ in range(m)], dtype=float)
    C = A @ B
    
    # Compute C_decomp using the decomposition
    C_decomp = np.zeros((n, p))
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    for r in range(rank):
        # U[:, r] has shape (n*m,), reshape to (n, m)
        Ur = U_arr[:, r].reshape(n, m)
        # V[:, r] has shape (m*p,), reshape to (m, p)
        Vr = V_arr[:, r].reshape(m, p)
        # W[:, r] has shape (p*n,), W[j*n+i][r] = W_arr[j*n+i, r]
        # So W_r[i, j] = W_arr[j*n + i, r]
        Wr = W_arr[:, r].reshape(p, n).T  # shape (n, p)
        
        C_decomp += Ur @ Vr * Wr
    
    if not np.allclose(C, C_decomp, atol=1e-9):
        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0
    
    # Strassen components for 2x2x2
    # local_a: 0=A11, 1=A12, 2=A21, 3=A22
    # local_b: 0=B11, 1=B12, 2=B21, 3=B22
    # local_w: 0=C11, 1=C12, 2=C21, 3=C22
    strassen = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),   # M1
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),           # M2
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),           # M3
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),           # M4
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),           # M5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),           # M6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),           # M7
    ]
    
    # Blocks: (a_rows, a_cols, b_rows, b_cols)
    blocks = [
        ([0, 1], [0, 1], [0, 1], [0, 1]),
        ([0, 1], [0, 1], [0, 1], [2, 3]),
        ([0, 1], [0, 1], [0, 1], [4, 5]),
        ([0, 1], [2, 3], [2, 3], [0, 1]),
        ([0, 1], [2, 3], [2, 3], [2, 3]),
        ([0, 1], [2, 3], [2, 3], [4, 5]),
    ]
    
    rank = len(blocks) * len(strassen)  # 6 * 7 = 42
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for a_rows, a_cols, b_rows, b_cols in blocks:
        if not _time_ok(deadline):
            break
        
        a_map = {
            0: a_rows[0] * m + a_cols[0],
            1: a_rows[0] * m + a_cols[1],
            2: a_rows[1] * m + a_cols[0],
            3: a_rows[1] * m + a_cols[1]
        }
        b_map = {
            0: b_rows[0] * p + b_cols[0],
            1: b_rows[0] * p + b_cols[1],
            2: b_rows[1] * p + b_cols[0],
            3: b_rows[1] * p + b_cols[1]
        }
        w_map = {
            0: b_cols[0] * n + a_rows[0],
            1: b_cols[1] * n + a_rows[0],
            2: b_cols[0] * n + a_rows[1],
            3: b_cols[1] * n + a_rows[1]
        }
        
        for u_coeffs, v_coeffs, w_coeffs in strassen:
            if not _time_ok(deadline):
                break
            for local_idx, coeff in u_coeffs:
                U[a_map[local_idx], col] += coeff
            for local_idx, coeff in v_coeffs:
                V[b_map[local_idx], col] += coeff
            for local_idx, coeff in w_coeffs:
                W[w_map[local_idx], col] += coeff
            col += 1
    
    # Try to reduce rank by finding linear dependencies among the rank-1 terms
    # The matrix multiplication tensor T is defined by T[i,j,k] = delta_{i+k, j}
    # where i in [0,n), j in [0,m), k in [0,p)
    # A rank-1 term is u ⊗ v ⊗ w where (u⊗v⊗w)[i,j,k] = u[i] * v[j] * w[k]
    # The sum of rank-1 terms must equal T
    
    # Verify the decomposition
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    if not _time_ok(deadline):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    is_valid = _verify_decomposition(U_list, V_list, W_list, n, m, p)
    
    # Try to reduce rank by checking for proportional rank-1 terms
    if is_valid and not _time_ok(deadline - 5):
        # Quick check for proportional terms
        rank = len(U_list[0])
        dim = n * m * p
        terms = np.zeros((rank, dim))
        for r in range(rank):
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        terms[r, (i*m+k)*p + j] = U_list[i*m+k][r] * V_list[k*p+j][r] * W_list[j*n+i][r]
        
        # Check for proportional pairs
        removed = [False] * rank
        new_terms = []
        new_U = []
        new_V = []
        new_W = []
        
        for r in range(rank):
            if removed[r]:
                continue
            # Check if this term is proportional to any already-kept term
            is_dup = False
            for kept_r in range(len(new_terms)):
                if np.allclose(terms[r], new_terms[kept_r], atol=1e-10):
                    is_dup = True
                    break
                # Check if proportional (same direction, different scale)
                # Find non-zero entry
                nz = np.where(np.abs(terms[r]) > 1e-10)[0]
                if len(nz) > 0:
                    idx = nz[0]
                    if np.abs(new_terms[kept_r][idx]) > 1e-10:
                        ratio = terms[r][idx] / new_terms[kept_r][idx]
                        if np.allclose(terms[r], ratio * new_terms[kept_r], atol=1e-10):
                            # They're proportional. If ratio is 1, they're identical.
                            # We can't easily merge unless ratio allows it.
                            # For now, skip.
                            is_dup = True
                            break
            
            if not is_dup:
                new_terms.append(terms[r])
                new_U.append([U_list[i][r] for i in range(n*m)])
                new_V.append([V_list[i][r] for i in range(m*p)])
                new_W.append([W_list[i][r] for i in range(p*n)])
        
        if len(new_terms) < rank:
            # We found duplicates, use the reduced set
            U_list = [new_U[r] for r in range(len(new_U))]
            V_list = [new_V[r] for r in range(len(new_V))]
            W_list = [new_W[r] for r in range(len(new_W))]
            # Re-verify
            if not _verify_decomposition(U_list, V_list, W_list, n, m, p):
                # Revert if verification fails
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
