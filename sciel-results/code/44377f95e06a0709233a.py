"""
Matrix multiplication 2x4x7 via tensor decomposition.
Attempts a targeted search for a rank-49 decomposition by optimizing the
last 3 columns (2x4x3 sub-problem) to rank 21, combined with the
known rank-28 decomposition for the first 4 columns.

Score = seed / rank. seed = 56 (schoolbook).
Rank 50 -> 1.12, Rank 49 -> 1.1429.
"""

import time
import numpy as np

# ---------------------------------------------------------------------------
# Deadline guard
# ---------------------------------------------------------------------------
_EVAL_DEADLINE = 150.0
_START = time.time()
TIME_BUDGET = _EVAL_DEADLINE - (time.time() - _START) - 5.0

def _time_left():
    return _EVAL_DEADLINE - 5.0 - time.time()

# ---------------------------------------------------------------------------
# Build the multiplication tensor T in [r,i,j] layout
# r = k*n + i, i is row of A, j = row of B / col of B (flattened j*p+k)
# ---------------------------------------------------------------------------
def _build_tensor(n, m, p):
    r_dim = p * n
    i_dim = n
    j_dim = m * p
    T = np.zeros((r_dim, i_dim, j_dim))
    for k in range(p):
        for i in range(n):
            for j in range(m):
                r = k * n + i
                jj = j * p + k
                T[r, i, jj] = 1.0
    return T

# ---------------------------------------------------------------------------
# Known rank-50 construction
# ---------------------------------------------------------------------------
def _construct_rank50(n=2, m=4, p=7):
    rank = 50
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    col = [0]

    def add_strassen(i0, j0, k0):
        c = col[0]
        # M1
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        c += 1
        # M2
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] -= 1
        c += 1
        # M3
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] -= 1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        c += 1
        # M4
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] -= 1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        c += 1
        # M5
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] -= 1
        W[(k0+1)*n + i0][c] += 1
        c += 1
        # M6
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] -= 1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        c += 1
        # M7
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] -= 1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        c += 1
        col[0] = c

    # 2x4x4 block: four 2x2x2 Strassen blocks
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    
    # 2x4x2 block: two 2x2x2 Strassen blocks (for k=4,5)
    add_strassen(0, 0, 4)
    add_strassen(0, 2, 4)
    
    # 2x4x1 block: direct terms (for k=6)
    for i in range(n):
        for j in range(m):
            U[i*m + j][col[0]] = 1.0
            V[j*p + 6][col[0]] = 1.0
            W[6*n + i][col[0]] = 1.0
            col[0] += 1

    assert col[0] == rank, f"Rank mismatch: used {col[0]}, expected {rank}"
    return {"U": U, "V": V, "W": W}

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def _verify(U, V, W, n, m, p, tol=1e-10):
    r_dim = p * n
    i_dim = n
    j_dim = m * p
    T = _build_tensor(n, m, p)
    T_mat = T.reshape(r_dim, i_dim * j_dim)
    
    rank = len(W[0])
    Ur = np.array(U).T   # (rank, i_dim)
    Vr = np.array(V).T   # (rank, j_dim)
    Wr = np.array(W).T   # (rank, r_dim)
    
    cand = np.zeros_like(T_mat)
    for t in range(rank):
        wt = Wr[t, :]
        ut = Ur[t, :]
        vt = Vr[t, :]
        outer = ut[:, None] * vt[None, :]
        cand += np.outer(wt, outer.ravel())
        
    max_err = np.max(np.abs(cand - T_mat))
    return max_err < tol

# ---------------------------------------------------------------------------
# Targeted Search for Rank 49
# ---------------------------------------------------------------------------
def _search_rank49(n=2, m=4, p=7, time_budget=30.0):
    """
    Focuses on the last 3 columns (k=4,5,6).
    Base: 14 terms (Strassen) + 8 terms (Direct) = 22 terms.
    Goal: Reduce to 21 terms.
    """
    if _time_left() < 5:
        return None

    # Get base rank 50
    base = _construct_rank50(n, m, p)
    U_base = np.array(base["U"])
    V_base = np.array(base["V"])
    W_base = np.array(base["W"])
    
    # The first 28 terms are for k=0..3. They are independent of the last 3 cols.
    # We only modify terms 28..49.
    # We want to replace these 22 terms with 21 terms.
    
    # Extract the sub-tensor for k=4,5,6
    # r indices: k*n+i. For k=4, r=8,9; k=5, r=10,11; k=6, r=11,12? No.
    # r = k*n + i. n=2.
    # k=4: r=8,9
    # k=5: r=10,11
    # k=6: r=12,13
    # i indices: 0,1
    # j indices: j*p+k.
    # k=4: j=0..3 -> 0*7+4=4, 1*7+4=11, 2*7+4=18, 3*7+4=25
    # k=5: j=0..3 -> 5, 12, 19, 26
    # k=6: j=0..3 -> 6, 13, 20, 27
    
    # We will search for a rank 21 decomposition of the sub-tensor T_sub.
    # T_sub has dimensions (6, 2, 12) effectively?
    # r_sub: 6 entries.
    # i_sub: 2 entries.
    # j_sub: 12 entries (4 for each k).
    
    # Build T_sub
    r_sub_indices = [8, 9, 10, 11, 12, 13]
    i_sub_indices = [0, 1]
    j_sub_indices = [4, 11, 18, 25, 5, 12, 19, 26, 6, 13, 20, 27]
    
    T_full = _build_tensor(n, m, p)
    T_sub = T_full[r_sub_indices][:, i_sub_indices][:, j_sub_indices]
    T_sub_mat = T_sub.reshape(6, 2 * 12)
    
    # Initial 22 terms for this sub-tensor
    U_sub_base = U_base[:, 28:50] # (8, 22)
    V_sub_base = V_base[:, 28:50] # (28, 22) -> Need to map to j_sub
    # V indices in base are global j*p+k.
    # We need to extract the values corresponding to j_sub_indices.
    V_sub_base_mapped = np.zeros((12, 22))
    for local_j, global_j in enumerate(j_sub_indices):
        V_sub_base_mapped[local_j, :] = V_base[global_j, 28:50]
        
    W_sub_base = W_base[:, 28:50] # (14, 22) -> Need to map to r_sub
    W_sub_base_mapped = np.zeros((6, 22))
    for local_r, global_r in enumerate(r_sub_indices):
        W_sub_base_mapped[local_r, :] = W_base[global_r, 28:50]
        
    # Verify base sub-decomposition
    def verify_sub(U_s, V_s, W_s, T_s_mat, tol=1e-10):
        rank = U_s.shape[1]
        cand = np.zeros_like(T_s_mat)
        for t in range(rank):
            wt = W_s[:, t]
            ut = U_s[:, t]
            vt = V_s[:, t]
            outer = ut[:, None] * vt[None, :]
            cand += np.outer(wt, outer.ravel())
        return np.max(np.abs(cand - T_s_mat)) < tol

    if not verify_sub(U_sub_base, V_sub_base_mapped, W_sub_base_mapped, T_sub_mat):
        # If base is wrong, return None
        return None

    # Search for rank 21
    # Strategy: Try to merge two of the 22 terms into one.
    # The 8 direct terms (last 8) are candidates for merging with the 14 Strassen terms.
    
    allowed_vals = list(set([float(v) for v in range(-8, 9)] + [float(v+0.5) for v in range(-8, 8)] + [float(-(v+0.5)) for v in range(-8, 8)]))
    av = np.array(allowed_vals)
    rng = np.random.RandomState(42)
    
    target_rank = 21
    base_rank = 22
    
    # Try merging each pair (t1, t2)
    for t1 in range(base_rank):
        if _time_left() < 5:
            break
        for t2 in range(t1 + 1, base_rank):
            if _time_left() < 5:
                break
            
            # Remove t1, t2
            mask = [x for x in range(base_rank) if x != t1 and x != t2]
            U_rem = U_sub_base[:, mask]
            V_rem = V_sub_base_mapped[:, mask]
            W_rem = W_sub_base_mapped[:, mask]
            
            # The residual to be covered by the new term
            residual = T_sub_mat.copy()
            for t in range(target_rank):
                wt = W_rem[:, t]
                ut = U_rem[:, t]
                vt = V_rem[:, t]
                outer = ut[:, None] * vt[None, :]
                residual -= np.outer(wt, outer.ravel())
                
            # Try to find a rank-1 term (u, v, w) such that outer(w, outer(u,v).flat) == residual
            # residual is (6, 24).
            # w is (6,), u is (2,), v is (12,)
            
            # SVD approach
            U_svd, s, Vt_svd = np.linalg.svd(residual, full_matrices=False)
            if s[0] < 1e-10:
                continue # Residual is zero? Unlikely.
            
            # The best rank-1 approximation is u_new * v_new * w_new
            # residual ~= w_new * (u_new * v_new).flat
            # Let M = residual. M[r, i*12+j] = w[r] * u[i] * v[j]
            # This is a tensor of rank 1.
            # We can try to solve for u, v, w given the structure.
            
            # Random search for the new term
            found = False
            for _ in range(50):
                if _time_left() < 5:
                    break
                nzu = rng.randint(1, 3)
                nzv = rng.randint(1, 6)
                nzw = rng.randint(1, 6)
                
                pos_u = rng.randint(0, 2, size=nzu)
                pos_v = rng.randint(0, 12, size=nzv)
                pos_w = rng.randint(0, 6, size=nzw)
                
                u_new = np.zeros(2)
                v_new = np.zeros(12)
                w_new = np.zeros(6)
                
                u_new[pos_u] = av[rng.randint(0, len(av), size=nzu)]
                v_new[pos_v] = av[rng.randint(0, len(av), size=nzv)]
                w_new[pos_w] = av[rng.randint(0, len(av), size=nzw)]
                
                term = np.outer(w_new, np.outer(u_new, v_new).flatten())
                if np.max(np.abs(term - residual)) < 1e-10:
                    found = True
                    break
            
            if found:
                # Construct full rank 49 solution
                U_full = np.zeros((8, 49))
                V_full = np.zeros((28, 49))
                W_full = np.zeros((14, 49))
                
                # Copy first 28 terms
                U_full[:, :28] = U_base[:, :28]
                V_full[:, :28] = V_base[:, :28]
                W_full[:, :28] = W_base[:, :28]
                
                # Copy remaining 20 terms from sub-problem
                U_full[:, 28:48] = U_rem
                # Map V_rem back to global
                for local_j, global_j in enumerate(j_sub_indices):
                    V_full[global_j, 28:48] = V_rem[local_j, :]
                # Map W_rem back to global
                for local_r, global_r in enumerate(r_sub_indices):
                    W_full[global_r, 28:48] = W_rem[local_r, :]
                    
                # Add new term
                U_full[:, 48] = u_new
                for local_j, global_j in enumerate(j_sub_indices):
                    V_full[global_j, 48] = v_new[local_j]
                for local_r, global_r in enumerate(r_sub_indices):
                    W_full[global_r, 48] = w_new[local_r]
                    
                # Verify full solution
                if _verify(U_full.tolist(), V_full.tolist(), W_full.tolist(), n, m, p):
                    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
                    
    return None

# ---------------------------------------------------------------------------
# Main construct
# ---------------------------------------------------------------------------
def construct():
    n, m, p = 2, 4, 7
    
    # Known rank-50 solution
    fallback = _construct_rank50(n, m, p)
    
    # Try targeted search for rank 49
    if _time_left() > 10:
        result = _search_rank49(n, m, p, min(_time_left(), 40.0))
        if result is not None:
            return result
            
    return fallback
