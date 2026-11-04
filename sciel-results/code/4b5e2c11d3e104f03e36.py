"""Matrix multiplication 3x4x8 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
from itertools import product, combinations

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build true tensor
    T_true = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_true[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Reconstruct
    T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    return np.allclose(T_recon, T_true, atol=1e-10)

def _strassen_2x2x2():
    """
    Return the Strassen decomposition for 2x2x2 matrix multiplication.
    Returns (U, V, W) each of shape (4, 7) where:
    U[row, col] corresponds to input A indices (row = i*2+j)
    V[row, col] corresponds to input B indices (row = j*2+k)
    W[row, col] corresponds to output C indices (row = k*2+i)
    """
    # Strassen's 7 products:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    #
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    
    # Index mapping:
    # A indices: a11=0, a12=1, a21=2, a22=3  (row = i*2+j, so 0=0,1=1,2=2,3=3)
    # B indices: b11=0, b12=1, b21=2, b22=3  (row = j*2+k, so 0=0,1=1,2=2,3=3)
    # C indices: c11=0, c12=1, c21=2, c22=3  (row = k*2+i, so 0=0,1=1,2=2,3=3)
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (a11+a22)(b11+b22)
    U[0, 0] += 1  # a11
    U[3, 0] += 1  # a22
    V[0, 0] += 1  # b11
    V[3, 0] += 1  # b22
    
    # M2 = (a21+a22)b11
    U[2, 1] += 1  # a21
    U[3, 1] += 1  # a22
    V[0, 1] += 1  # b11
    
    # M3 = a11(b12-b22)
    U[0, 2] += 1  # a11
    V[1, 2] += 1  # b12
    V[3, 2] -= 1  # b22
    
    # M4 = a22(b21-b11)
    U[3, 3] += 1  # a22
    V[2, 3] += 1  # b21
    V[0, 3] -= 1  # b11
    
    # M5 = (a11+a12)b22
    U[0, 4] += 1  # a11
    U[1, 4] += 1  # a12
    V[3, 4] += 1  # b22
    
    # M6 = (a21-a11)(b11+b12)
    U[2, 5] += 1  # a21
    U[0, 5] -= 1  # a11
    V[0, 5] += 1  # b11
    V[1, 5] += 1  # b12
    
    # M7 = (a12-a22)(b21+b22)
    U[1, 6] += 1  # a12
    U[3, 6] -= 1  # a22
    V[2, 6] += 1  # b21
    V[3, 6] += 1  # b22
    
    # Output combinations:
    # c11 = M1 + M4 - M5 + M7  -> C index 0
    W[0, 0] += 1
    W[0, 3] += 1
    W[0, 4] -= 1
    W[0, 6] += 1
    
    # c12 = M3 + M5  -> C index 1
    W[1, 2] += 1
    W[1, 4] += 1
    
    # c21 = M2 + M4  -> C index 2
    W[2, 1] += 1
    W[2, 3] += 1
    
    # c22 = M1 - M2 + M3 + M6  -> C index 3
    W[3, 0] += 1
    W[3, 1] -= 1
    W[3, 2] += 1
    W[3, 5] += 1
    
    return U, V, W

def _try_strassen_packing(n, m, p):
    """
    Try to find disjoint 2x2x2 blocks in the tensor and apply Strassen to each.
    Return (U, V, W) with the decomposition, or None if no packing found.
    
    A 2x2x2 block is defined by choosing 2 i-values, 2 j-values, 2 k-values.
    The block covers all (i,j,k) tuples within those sets.
    Blocks must be disjoint (no shared (i,j,k) tuple).
    """
    # Get Strassen decomposition
    S_U, S_V, S_W = _strassen_2x2x2()
    
    # All possible 2x2x2 blocks
    i_sets = list(combinations(range(n), 2))
    j_sets = list(combinations(range(m), 2))
    k_sets = list(combinations(range(p), 2))
    
    # For each block, determine which (i,j,k) tuples it covers
    def block_triples(i_set, j_set, k_set):
        return frozenset(product(i_set, j_set, k_set))
    
    # Find all blocks
    all_blocks = []
    for i_set in i_sets:
        for j_set in j_sets:
            for k_set in k_sets:
                triples = block_triples(i_set, j_set, k_set)
                all_blocks.append((i_set, j_set, k_set, triples))
    
    # Find disjoint sets of blocks using greedy/backtracking
    best_blocks = []
    
    def find_disjoint(blocks, used_triples, current, start_idx):
        nonlocal best_blocks
        if len(current) > len(best_blocks):
            best_blocks = list(current)
        
        if start_idx >= len(blocks):
            return
        
        for idx in range(start_idx, len(blocks)):
            i_set, j_set, k_set, triples = blocks[idx]
            if not (triples & used_triples):
                current.append((i_set, j_set, k_set))
                find_disjoint(blocks, used_triples | triples, current, idx + 1)
                current.pop()
    
    find_disjoint(all_blocks, frozenset(), [], 0)
    
    if not best_blocks:
        return None
    
    # Build the decomposition using Strassen for each block + schoolbook for rest
    num_blocks = len(best_blocks)
    block_rank = 7 * num_blocks
    remaining_triples = set(product(range(n), range(m), range(p)))
    
    for i_set, j_set, k_set in best_blocks:
        for i in i_set:
            for j in j_set:
                for k in k_set:
                    remaining_triples.discard((i, j, k))
    
    rem_rank = len(remaining_triples)
    total_rank = block_rank + rem_rank
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Add Strassen blocks
    for blk_idx, (i_set, j_set, k_set) in enumerate(best_blocks):
        i0, i1 = i_set
        j0, j1 = j_set
        k0, k1 = k_set
        
        col_offset = blk_idx * 7
        
        for local_i, global_i in enumerate([i0, i1]):
            for local_j, global_j in enumerate([j0, j1]):
                a_idx = global_i * m + global_j
                for r in range(7):
                    U[a_idx, col_offset + r] = S_U[local_i * 2 + local_j, r]
        
        for local_j, global_j in enumerate([j0, j1]):
            for local_k, global_k in enumerate([k0, k1]):
                b_idx = global_j * p + global_k
                for r in range(7):
                    V[b_idx, col_offset + r] = S_V[local_j * 2 + local_k, r]
        
        for local_k, global_k in enumerate([k0, k1]):
            for local_i, global_i in enumerate([i0, i1]):
                c_idx = global_k * n + global_i
                for r in range(7):
                    W[c_idx, col_offset + r] = S_W[local_k * 2 + local_i, r]
    
    # Add schoolbook for remaining triples
    col = block_rank
    for (i, j, k) in sorted(remaining_triples):
        U[i * m + j, col] = 1
        V[j * p + k, col] = 1
        W[k * n + i, col] = 1
        col += 1
    
    # Verify
    if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        return U.tolist(), V.tolist(), W.tolist()
    
    return None

def _try_structured_decomposition(n, m, p):
    """
    Try a structured decomposition that exploits the tensor structure.
    For 3x4x8, we can try to use the fact that p=8 is large.
    """
    # Strategy: For each (i,j), the output entries are at positions k*n+i for k=0..p-1
    # We can group by i and j to find patterns
    
    # Try a rank that's less than schoolbook (n*m*p = 96)
    # Use the Strassen-based packing first
    result = _try_strassen_packing(n, m, p)
    if result is not None:
        return result
    
    return None

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition."""
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

def _try_alss(n, m, p, target_rank, max_iters=200, seed=42, deadline=None):
    """
    Try ALS (Alternating Least Squares) to find a rank-target_rank decomposition.
    Returns (U, V, W) if found, else None.
    """
    T_true = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_true[i*m + j, j*p + k, k*n + i] = 1.0
    
    T_flat = T_true.reshape(n*m, m*p * p*n)
    
    rng = np.random.RandomState(seed)
    
    for trial in range(3):
        U = rng.randn(n*m, target_rank) * 0.1
        V = rng.randn(m*p, target_rank) * 0.1
        W = rng.randn(p*n, target_rank) * 0.1
        
        for it in range(max_iters):
            if deadline is not None and time.time() > deadline - 5:
                break
            
            # Update U
            # M_u[b*p*n + c, r] = V[b, r] * W[c, r]
            # We want U @ M_u.T = T_flat
            M_u = np.einsum('br,cr->bcr', V, W).reshape(m*p * p*n, target_rank)
            try:
                U_new, _, _, _ = np.linalg.lstsq(M_u.T, T_flat.T, rcond=None)
                U = U_new.T
            except:
                pass
            
            # Update V
            M_v = np.einsum('ar,cr->acr', U, W).reshape(n*m * p*n, target_rank)
            T_v = T_true.transpose(1, 0, 2).reshape(m*p, n*m * p*n)
            try:
                V_new, _, _, _ = np.linalg.lstsq(M_v.T, T_v.T, rcond=None)
                V = V_new.T
            except:
                pass
            
            # Update W
            M_w = np.einsum('ar,br->abr', U, V).reshape(n*m * m*p, target_rank)
            T_w = T_true.transpose(2, 0, 1).reshape(p*n, n*m * m*p)
            try:
                W_new, _, _, _ = np.linalg.lstsq(M_w.T, T_w.T, rcond=None)
                W = W_new.T
            except:
                pass
            
            if it % 10 == 0:
                T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.max(np.abs(T_recon - T_true))
                if err < 1e-8:
                    # Try to round to half-integers
                    for scale in [1, 2, 4, 8, 16]:
                        U_r = np.round(U * scale) / scale
                        V_r = np.round(V * scale) / scale
                        W_r = np.round(W * scale) / scale
                        if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
                            return U_r.tolist(), V_r.tolist(), W_r.tolist()
                    # Try integer rounding
                    U_r = np.round(U)
                    V_r = np.round(V)
                    W_r = np.round(W)
                    if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
                        return U_r.tolist(), V_r.tolist(), W_r.tolist()
                    break
        
        # Final check
        for scale in [1, 2, 4]:
            U_r = np.round(U * scale) / scale
            V_r = np.round(V * scale) / scale
            W_r = np.round(W * scale) / scale
            if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
                return U_r.tolist(), V_r.tolist(), W_r.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # First try Strassen packing
    result = _try_structured_decomposition(n, m, p)
    if result is not None:
        U, V, W = result
        return {"U": U, "V": V, "W": W}
    
    # Try ALS with various ranks below schoolbook
    schoolbook_rank = n * m * p  # 96
    
    # Try ranks in decreasing order of promise
    candidates = [72, 80, 84, 88, 90, 92, 94]
    
    for r in candidates:
        if time.time() > deadline - 30:
            break
        result = _try_alss(n, m, p, r, max_iters=100, seed=42, deadline=deadline)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
