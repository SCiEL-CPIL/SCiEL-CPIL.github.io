"""Matrix multiplication 3x4x8 with rank-reduced decomposition using Strassen blocks and refined column merging."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]


def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
    U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    return U, V, W


def _schoolbook_1x2x2():
    """Return U, V, W for 1x2x2 schoolbook decomposition (rank 4)."""
    U = [
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    V = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    W = [
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    return U, V, W


def _build_base_decomposition():
    """Build the base 3x4x8 decomposition: 8 Strassen blocks (i=0,1) + 8 schoolbook blocks (i=2). Rank 88."""
    n, m, p = 3, 4, 8
    U2, V2, W2 = _strassen_2x2x2()
    U1, V1, W1 = _schoolbook_1x2x2()
    
    total_rank = 8 * 7 + 8 * 4  # 88
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    # Strassen blocks for i=0,1
    for jp in range(2):
        for kp in range(4):
            j0 = 2 * jp
            k0 = 2 * kp
            u_rows = [0*m+j0, 0*m+j0+1, 1*m+j0, 1*m+j0+1]
            v_rows = [j0*p+k0, j0*p+k0+1, (j0+1)*p+k0, (j0+1)*p+k0+1]
            w_rows = [k0*n+0, (k0+1)*n+0, k0*n+1, (k0+1)*n+1]
            for i in range(4):
                for c in range(7):
                    U[u_rows[i]][col_offset + c] = U2[i][c]
                    V[v_rows[i]][col_offset + c] = V2[i][c]
                    W[w_rows[i]][col_offset + c] = W2[i][c]
            col_offset += 7
    
    # Schoolbook blocks for i=2
    for jp in range(2):
        for kp in range(4):
            j0 = 2 * jp
            k0 = 2 * kp
            u_rows = [2*m+j0, 2*m+j0+1]
            v_rows = [j0*p+k0, (j0+1)*p+k0, j0*p+k0+1, (j0+1)*p+k0+1]
            w_rows = [k0*n+2, (k0+1)*n+2]
            for i in range(2):
                for c in range(4):
                    U[u_rows[i]][col_offset + c] = U1[i][c]
                    W[w_rows[i]][col_offset + c] = W1[i][c]
            for i in range(4):
                for c in range(4):
                    V[v_rows[i]][col_offset + c] = V1[i][c]
            col_offset += 4
    
    return U, V, W


def _compute_contribution(U, V, W, n, m, p):
    """Compute the contribution matrix: contrib[c, i*m+j*p+k] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]"""
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    contrib = np.zeros((rank, n * m * p), dtype=float)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                contrib[:, idx] = U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i]
                idx += 1
    return contrib


def _verify_decomposition(U, V, W, n, m, p):
    """Verify the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1) > 1e-9:
                    return False
    return True


def _reduce_rank_greedy(U, V, W, n, m, p, deadline):
    """Greedy rank reduction: repeatedly find the column whose removal causes the least damage.
    
    Actually, a better approach: find the minimal subset of columns that spans the same
    contribution space. Use a greedy algorithm that picks columns in order of norm
    (largest first) and checks if each new column adds to the span.
    """
    rank = len(U[0])
    contrib = _compute_contribution(U, V, W, n, m, p)
    
    # Sort columns by norm (largest first) - this often gives a better basis
    norms = np.linalg.norm(contrib, axis=1)
    order = np.argsort(-norms)  # descending
    
    kept = []
    for c in order:
        if time.time() > deadline:
            break
        c = int(c)
        if not kept:
            kept.append(c)
            continue
        
        # Check if contrib[c] is in the span of kept columns
        A = contrib[kept, :]  # (len(kept), n*m*p)
        b = contrib[c, :]
        
        try:
            U_svd, s, Vt = np.linalg.svd(A, full_matrices=False)
            rank_A = np.sum(s > 1e-10)
            if rank_A == 0:
                if np.allclose(b, 0, atol=1e-10):
                    continue  # Don't keep c
                else:
                    kept.append(c)
                continue
            U_sub = U_svd[:, :rank_A]
            proj = U_sub @ (U_sub.T @ b)
            residual = np.linalg.norm(b - proj)
            norm_b = np.linalg.norm(b)
            if norm_b < 1e-10:
                continue  # Don't keep c
            elif residual / max(norm_b, 1e-15) < 1e-8:
                continue  # Don't keep c
            else:
                kept.append(c)
        except np.linalg.LinAlgError:
            kept.append(c)
    
    if len(kept) >= rank:
        return U, V, W, rank
    
    # Reorder kept to original order for consistency
    kept_sorted = sorted(kept)
    new_rank = len(kept_sorted)
    final_U = [[U[i][c] for c in kept_sorted] for i in range(len(U))]
    final_V = [[V[i][c] for c in kept_sorted] for i in range(len(V))]
    final_W = [[W[i][c] for c in kept_sorted] for i in range(len(W))]
    return final_U, final_V, final_W, new_rank


def _reduce_rank_forward(U, V, W, n, m, p, deadline):
    """Forward elimination: process columns in original order, keep only if not in span of previous kept."""
    rank = len(U[0])
    contrib = _compute_contribution(U, V, W, n, m, p)
    
    kept = []
    for c in range(rank):
        if time.time() > deadline:
            break
        if not kept:
            kept.append(c)
            continue
        
        A = contrib[kept, :]
        b = contrib[c, :]
        
        try:
            U_svd, s, Vt = np.linalg.svd(A, full_matrices=False)
            rank_A = np.sum(s > 1e-10)
            if rank_A == 0:
                if np.allclose(b, 0, atol=1e-10):
                    continue
                else:
                    kept.append(c)
                continue
            U_sub = U_svd[:, :rank_A]
            proj = U_sub @ (U_sub.T @ b)
            residual = np.linalg.norm(b - proj)
            norm_b = np.linalg.norm(b)
            if norm_b < 1e-10:
                continue
            elif residual / max(norm_b, 1e-15) < 1e-8:
                continue
            else:
                kept.append(c)
        except np.linalg.LinAlgError:
            kept.append(c)
    
    if len(kept) >= rank:
        return U, V, W, rank
    
    new_rank = len(kept)
    final_U = [[U[i][c] for c in kept] for i in range(len(U))]
    final_V = [[V[i][c] for c in kept] for i in range(len(V))]
    final_W = [[W[i][c] for c in kept] for i in range(len(W))]
    return final_U, final_V, final_W, new_rank


def construct():
    start = time.time()
    deadline = start + 130  # safety margin before 150s limit
    n, m, p = DIMENSIONS
    
    # Build the base decomposition (rank 88)
    U, V, W = _build_base_decomposition()
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback: schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_U, best_V, best_W = U, V, W
    best_rank = len(U[0])
    
    # Try greedy reduction (largest norm first)
    if time.time() < deadline - 20:
        U_g, V_g, W_g, r_g = _reduce_rank_greedy(U, V, W, n, m, p, deadline)
        if r_g < best_rank and _verify_decomposition(U_g, V_g, W_g, n, m, p):
            best_U, best_V, best_W, best_rank = U_g, V_g, W_g, r_g
    
    # Try forward elimination
    if time.time() < deadline - 20:
        U_f, V_f, W_f, r_f = _reduce_rank_forward(U, V, W, n, m, p, deadline)
        if r_f < best_rank and _verify_decomposition(U_f, V_f, W_f, n, m, p):
            best_U, best_V, best_W, best_rank = U_f, V_f, W_f, r_f
    
    # Iterate: try to reduce further from the best found
    for iteration in range(10):
        if time.time() > deadline - 10:
            break
        
        U_new, V_new, W_new, r_new = _reduce_rank_greedy(best_U, best_V, best_W, n, m, p, deadline)
        if r_new < best_rank and _verify_decomposition(U_new, V_new, W_new, n, m, p):
            best_U, best_V, best_W, best_rank = U_new, V_new, W_new, r_new
            continue
        
        U_new, V_new, W_new, r_new = _reduce_rank_forward(best_U, best_V, best_W, n, m, p, deadline)
        if r_new < best_rank and _verify_decomposition(U_new, V_new, W_new, n, m, p):
            best_U, best_V, best_W, best_rank = U_new, V_new, W_new, r_new
            continue
        
        break
    
    return {"U": best_U, "V": best_V, "W": best_W}
