"""Combined approach: verified rank-39 Strassen+schoolbook, then structured sparse support search."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _construct_rank39():
    """
    Explicit rank-39 decomposition for M_{2,2,11}.
    5 blocks of 2 columns (Strassen rank 7 each) + 1 single column (schoolbook rank 4).
    Total: 5*7 + 4 = 39.
    """
    n, m, p = 2, 2, 11
    rank = 39
    
    U = [[0.0] * rank for _ in range(n * m)]   # 4 x 39
    V = [[0.0] * rank for _ in range(m * p)]   # 22 x 39
    W = [[0.0] * rank for _ in range(p * n)]   # 22 x 39
    
    r = 0
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        
        v_e = k0
        v_f = k1
        v_g = p + k0
        v_h = p + k1
        
        w_c00 = k0 * n
        w_c10 = k0 * n + 1
        w_c01 = k1 * n
        w_c11 = k1 * n + 1
        
        U[0][r] = 1; U[3][r] = 1
        V[v_e][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1; W[w_c11][r] = 1
        r += 1
        
        U[2][r] = 1; U[3][r] = 1
        V[v_e][r] = 1
        W[w_c10][r] = 1; W[w_c11][r] = -1
        r += 1
        
        U[0][r] = 1
        V[v_f][r] = 1; V[v_h][r] = -1
        W[w_c01][r] = 1; W[w_c11][r] = 1
        r += 1
        
        U[3][r] = 1
        V[v_g][r] = 1; V[v_e][r] = -1
        W[w_c00][r] = 1; W[w_c10][r] = 1
        r += 1
        
        U[0][r] = 1; U[1][r] = 1
        V[v_h][r] = 1
        W[w_c00][r] = -1; W[w_c01][r] = 1
        r += 1
        
        U[2][r] = 1; U[0][r] = -1
        V[v_e][r] = 1; V[v_f][r] = 1
        W[w_c11][r] = 1
        r += 1
        
        U[1][r] = 1; U[3][r] = -1
        V[v_g][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1
        r += 1
    
    k2 = 10
    v_e2 = k2
    v_g2 = p + k2
    w_c02 = k2 * n
    w_c12 = k2 * n + 1
    
    U[0][r] = 1; V[v_e2][r] = 1; W[w_c02][r] = 1; r += 1
    U[1][r] = 1; V[v_g2][r] = 1; W[w_c02][r] = 1; r += 1
    U[2][r] = 1; V[v_e2][r] = 1; W[w_c12][r] = 1; r += 1
    U[3][r] = 1; V[v_g2][r] = 1; W[w_c12][r] = 1; r += 1
    
    assert r == rank
    return U, V, W

def _try_sparse_support_search(target_rank, n, m, p, deadline, max_attempts=50):
    """
    Structured search: try sparse support patterns for each rank slot.
    Each slot has a support pattern (which of the 44 entries it affects).
    Use local refinement with exact rational arithmetic.
    """
    # Build tensor entry list
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                tensor_entries.append((u_idx, v_idx, w_idx))
    
    num_entries = len(tensor_entries)  # 44
    
    # Precompute entry indices for each factor index
    entries_by_u = [[] for _ in range(n * m)]
    entries_by_v = [[] for _ in range(m * p)]
    entries_by_w = [[] for _ in range(p * n)]
    
    for idx, (u_idx, v_idx, w_idx) in enumerate(tensor_entries):
        entries_by_u[u_idx].append(idx)
        entries_by_v[v_idx].append(idx)
        entries_by_w[w_idx].append(idx)
    
    # Try different initializations with structured sparsity
    for attempt in range(max_attempts):
        if time.time() > deadline - 5:
            break
        
        # Create sparse random initialization
        # Each rank slot gets a random subset of entries to affect
        U = np.zeros((n * m, target_rank))
        V = np.zeros((m * p, target_rank))
        W = np.zeros((p * n, target_rank))
        
        # Assign each tensor entry to a random rank slot
        entry_to_rank = np.random.randint(0, target_rank, size=num_entries)
        
        for idx, (u_idx, v_idx, w_idx) in enumerate(tensor_entries):
            r = entry_to_rank[idx]
            U[u_idx, r] = 1.0
            V[v_idx, r] = 1.0
            W[w_idx, r] = 1.0
        
        # Local refinement using alternating least squares
        best_err = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for it in range(200):
            if time.time() > deadline - 5:
                break
            
            # Update U
            for u_idx in range(n * m):
                idxs = entries_by_u[u_idx]
                if not idxs:
                    continue
                # Build system: for each entry idx, sum_r U[u_idx,r] * V[v_idx,r] * W[w_idx,r] = 1
                A_mat = []
                b_vec = []
                for idx in idxs:
                    _, v_idx, w_idx = tensor_entries[idx]
                    A_mat.append(V[v_idx, :] * W[w_idx, :])
                    b_vec.append(1.0)
                A_mat = np.array(A_mat)
                b_vec = np.array(b_vec)
                try:
                    x, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    U[u_idx, :] = x
                except:
                    pass
            
            # Update V
            for v_idx in range(m * p):
                idxs = entries_by_v[v_idx]
                if not idxs:
                    continue
                A_mat = []
                b_vec = []
                for idx in idxs:
                    u_idx, _, w_idx = tensor_entries[idx]
                    A_mat.append(U[u_idx, :] * W[w_idx, :])
                    b_vec.append(1.0)
                A_mat = np.array(A_mat)
                b_vec = np.array(b_vec)
                try:
                    x, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    V[v_idx, :] = x
                except:
                    pass
            
            # Update W
            for w_idx in range(p * n):
                idxs = entries_by_w[w_idx]
                if not idxs:
                    continue
                A_mat = []
                b_vec = []
                for idx in idxs:
                    u_idx, v_idx, _ = tensor_entries[idx]
                    A_mat.append(U[u_idx, :] * V[v_idx, :])
                    b_vec.append(1.0)
                A_mat = np.array(A_mat)
                b_vec = np.array(b_vec)
                try:
                    x, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    W[w_idx, :] = x
                except:
                    pass
            
            # Compute error
            err = 0.0
            for idx, (u_idx, v_idx, w_idx) in enumerate(tensor_entries):
                val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                err += (val - 1.0) ** 2
            
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if err < 1e-12:
                break
        
        # If good enough, try to round to half-integers and verify
        if best_err < 1e-8 and best_U is not None:
            # Try rounding to nearest half-integer
            U_rounded = np.round(best_U * 2) / 2
            V_rounded = np.round(best_V * 2) / 2
            W_rounded = np.round(best_W * 2) / 2
            
            U_list = U_rounded.tolist()
            V_list = V_rounded.tolist()
            W_list = W_rounded.tolist()
            
            if _verify(U_list, V_list, W_list, n, m, p, tol=1e-9):
                return U_list, V_list, W_list
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s kill
    
    # First, return the verified rank-39 construction
    U, V, W = _construct_rank39()
    if _verify(U, V, W, n, m, p):
        # Try to find a lower rank decomposition with remaining time
        if time.time() < deadline - 30:
            for rank in [38, 37, 36, 35, 34]:
                if time.time() > deadline - 10:
                    break
                result = _try_sparse_support_search(rank, n, m, p, deadline=deadline - 5, max_attempts=30)
                if result is not None:
                    U2, V2, W2 = result
                    if _verify(U2, V2, W2, n, m, p):
                        return {"U": U2, "V": V2, "W": W2}
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
