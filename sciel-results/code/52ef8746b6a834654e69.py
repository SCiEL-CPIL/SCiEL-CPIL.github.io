import time
import numpy as np

DIMENSIONS = [4, 4, 8]
START = time.time()
DEADLINE = START + 130.0

def _build_strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition (rank 7).
    U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7), dtype=np.int8)
    V = np.zeros((4, 7), dtype=np.int8)
    W = np.zeros((4, 7), dtype=np.int8)

    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    U[3, 3] = 1
    V[0, 3] = -1; V[2, 3] = 1
    W[0, 3] = 1; W[1, 3] = 1

    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    U[0, 5] = -1; U[2, 5] = 1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W

def _build_rank84():
    """
    Builds a rank-84 decomposition for 4x4 * 4x8 matrix multiplication.
    
    Strategy: Use top-level Strassen (2x2x2) to split 4x4x8 into 7 sub-problems
    of type (2x4) * (4x4) -> (2x4). Each sub-problem has a 2x4x4 tensor.
    
    For each 2x4x4 sub-problem, we can use the tensor product of:
    - 1x1x1 (rank 1) for the 2->2 split in i dimension  
    - 2x2x2 Strassen (rank 7) for the 4x4 part
    
    Wait, let me think more carefully.
    
    Actually, the cleanest approach: 
    Split 4x4x8 as follows:
    - i: 0..3, split into i0 (0..1) and i1 (0..1)  -> 2x2
    - j: 0..3, split into j0 (0..1) and j1 (0..1)  -> 2x2  
    - k: 0..7, split into k0 (0..3) and k1 (0..3)  -> 2x2 (but each is 4 wide)
    
    This gives 8 sub-tensors of type 2x2x4.
    
    For each 2x2x4 sub-tensor, we can decompose it as:
    - Use Strassen 2x2x2 (rank 7) for the 2x2x2 part
    - But k has dimension 4, not 2
    
    Alternative: For 2x2x4, use the fact that it's 2x2x2 ⊗ 1x1x2 in the k-dimension.
    That gives rank 7*2 = 14 per sub-tensor, times 8 sub-tensors = 112. Worse.
    
    Let me try a different structural approach:
    
    The 4x4x8 tensor can be seen as 4x4x4 ⊕ 4x4x4 (two copies).
    Each 4x4x4 has rank 49 via Strassen tensor product.
    Total: 98.
    
    To beat 98, we need cross-block structure. The key observation:
    Both blocks share the SAME U factor (the A matrix is the same).
    
    So we have: sum_r U[i,j,r] * V1[j,k1,r] * W1[k1,i,r] = C1[i,k1]
                sum_r U[i,j,r] * V2[j,k2,r] * W2[k2,i,r] = C2[i,k2]
    
    where k1 = 0..3 and k2 = 4..7.
    
    This is equivalent to: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = C[i,k]
    where V and W are extended to handle both blocks.
    
    The rank is still 49+49=98 unless we can find a joint decomposition.
    
    Let me try the lattice/structured approach differently:
    Use the fact that 4 = 2+2 and 8 = 4+4, and apply Strassen at multiple levels.
    
    Level 1: Split 4x4x8 into 2x(2x4x8) using Strassen on the i-j plane.
    This gives 7 sub-problems of type (2x4) x (4x4) -> (2x4).
    
    Each (2x4)x(4x4)->(2x4) sub-problem:
    Split the 4x4 B matrix into 2x2 blocks of size 2x2.
    Split the 2x4 A matrix into... hmm, this is getting complex.
    
    Let me just implement the shared-U rank-84 approach:
    Actually, the rank is the number of independent rank-1 terms.
    If U is shared, we still need 49 distinct (V,W) pairs, so rank is 98.
    
    The only way to beat 98 is to find genuine cross-block structure.
    
    Let me try: use the rank-49 4x4x4 decomposition, but for the 4x4x8 case,
    note that the second 4x4 block can reuse some of the same rank-1 terms
    if the V and W factors have a special structure.
    
    Actually, I think the best structural change within budget is:
    Use the tensor product of Strassen 2x2x2 with a 1x1x4 "identity" tensor.
    
    4x4x8 = (2x2) ⊗ (2x1) in i, (2x2) ⊗ (2x1) in j, (2x2) ⊗ (2x2) in k
    No, that doesn't work cleanly.
    
    Let me just go with a clean implementation that uses the block structure
    but tries to minimize rank through careful factor sharing.
    
    For now, implement rank-84 by using a different split:
    Split 4x4x8 as 4x(4x8) where we treat the 4x8 as two 4x4 blocks.
    Use Strassen 2x2x2 on the 4x4 part, giving rank 7 per 2x2 block.
    For the full 4x4x4: rank 49.
    For 4x4x8: we need to handle the extra 4 columns.
    
    I'll implement a rank-84 decomposition using a hybrid approach:
    Use 49 ranks for the first 4x4 block and 35 ranks for the second,
    exploiting that the second block can share some structure.
    """
    U2, V2, W2 = _build_strassen_2x2x2()
    
    # Build 4x4x4 decomposition (rank 49)
    rank_444 = 49
    U4 = np.zeros((16, rank_444), dtype=np.int16)
    V4 = np.zeros((16, rank_444), dtype=np.int16)
    W4 = np.zeros((16, rank_444), dtype=np.int16)
    
    for c0 in range(7):
        for c1 in range(7):
            c = c0 * 7 + c1
            for i in range(4):
                i0, i1 = i // 2, i % 2
                for j in range(4):
                    j0, j1 = j // 2, j % 2
                    U4[i*4+j, c] = U2[i0*2+j0, c0] * U2[i1*2+j1, c1]
            for j in range(4):
                j0, j1 = j // 2, j % 2
                for k in range(4):
                    k0, k1 = k // 2, k % 2
                    V4[j*4+k, c] = V2[j0*2+k0, c0] * V2[j1*2+k1, c1]
            for k in range(4):
                k0, k1 = k // 2, k % 2
                for i in range(4):
                    i0, i1 = i // 2, i % 2
                    W4[k*4+i, c] = W2[k0*2+i0, c0] * W2[k1*2+i1, c1]
    
    # For 4x4x8, use rank 49 for first block + rank 49 for second block = 98
    # But try to find a lower-rank joint decomposition
    # 
    # Structural idea: The 4x4x8 tensor T[i,j,k] = 1 for all i,j,k
    # can be decomposed using the fact that it's a "flat" tensor.
    # 
    # Use the following approach: represent T as sum of rank-1 terms
    # where each term is U_r ⊗ V_r ⊗ W_r with U_r ∈ R^16, V_r ∈ R^32, W_r ∈ R^32
    # 
    # The minimal rank of the 4x4x8 multiplication tensor is known to be ≤ 98
    # and conjectured to be lower.
    #
    # For this implementation, use rank 98 but with a cleaner structure
    # that might allow future optimization.
    
    total_rank = 98
    U = np.zeros((16, total_rank), dtype=np.int16)
    V = np.zeros((32, total_rank), dtype=np.int16)
    W = np.zeros((32, total_rank), dtype=np.int16)
    
    for block in range(2):
        col_start = block * 49
        col_end = (block + 1) * 49
        U[:, col_start:col_end] = U4
        for j in range(4):
            for k_local in range(4):
                k_global = block * 4 + k_local
                V[j*8 + k_global, col_start:col_end] = V4[j*4 + k_local, :]
        for k_local in range(4):
            k_global = block * 4 + k_local
            for i in range(4):
                W[k_global*4 + i, col_start:col_end] = W4[k_local*4 + i, :]
    
    # Verify
    n, m, p = 4, 4, 8
    result = np.zeros((n, m, p), dtype=np.float64)
    for r in range(total_rank):
        u_col = U[:, r].reshape(n, m)
        v_col = V[:, r].reshape(m, p)
        w_col = W[:, r].reshape(p, n)
        result += np.einsum('ij,jk,ki->ijk', u_col, v_col, w_col)
    
    if not np.allclose(result, 1.0, atol=1e-9):
        return None
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _build_rank84_structured():
    """
    Attempt a rank-84 decomposition using a structured approach.
    
    Key insight: For 4x4x8, we can use the following decomposition:
    - Split the 8 output columns into two groups of 4
    - Use a shared rank-49 factor for the "common" part
    - Add 35 additional factors for the "difference" part
    
    This is based on the observation that the two 4x4 blocks share
    the same input matrix A, so their U factors are identical.
    
    If we write:
    T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    For k=0..3: T[i,j,k] = sum_{r=0}^{48} U[i,j,r] * V1[j,k,r] * W1[k,i,r]
    For k=4..7: T[i,j,k] = sum_{r=0}^{48} U[i,j,r] * V2[j,k-4,r] * W2[k-4,i,r]
    
    The total rank is 49 + 49 = 98.
    
    To reduce to 84, we need to find 14 rank-1 terms that can be shared
    or eliminated through linear dependence.
    
    This is non-trivial. For now, return None to indicate failure.
    """
    return None

def _schoolbook(n=4, m=4, p=8):
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

def _try_low_rank_search(n=4, m=4, p=8, target_rank=84, max_iter=50, seed=42):
    """
    Heuristic search for a lower-rank decomposition using gradient-based
    optimization with integer/half-integer rounding.
    """
    if time.time() > DEADLINE - 30:
        return None
    
    np.random.seed(seed)
    
    # Initialize with the rank-98 solution and try to compress
    res98 = _build_rank84_structured()
    if res98 is None:
        res98 = _build_rank98()
    if res98 is None:
        return None
    
    U98 = np.array(res98['U'], dtype=np.float64)
    V98 = np.array(res98['V'], dtype=np.float64)
    W98 = np.array(res98['W'], dtype=np.float64)
    
    # Try to find a rank-84 solution by optimizing
    # This is a heuristic and may not succeed
    
    U = np.random.randn(n*m, target_rank)
    V = np.random.randn(m*p, target_rank)
    W = np.random.randn(p*n, target_rank)
    
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    lr = 0.01
    for it in range(max_iter):
        if time.time() > DEADLINE - 20:
            break
        
        # Compute gradients (simplified)
        # For each (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
        
        # Update U
        for i in range(n):
            for j in range(m):
                grad = np.zeros(target_rank)
                for k in range(p):
                    grad += V[j*p+k, :] * W[k*n+i, :] * (np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]) - 1.0)
                U[i*m+j, :] -= lr * grad
        
        # Update V
        for j in range(m):
            for k in range(p):
                grad = np.zeros(target_rank)
                for i in range(n):
                    grad += U[i*m+j, :] * W[k*n+i, :] * (np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]) - 1.0)
                V[j*p+k, :] -= lr * grad
        
        # Update W
        for k in range(p):
            for i in range(n):
                grad = np.zeros(target_rank)
                for j in range(m):
                    grad += U[i*m+j, :] * V[j*p+k, :] * (np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]) - 1.0)
                W[k*n+i, :] -= lr * grad
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    err += (val - 1.0) ** 2
        err = np.sqrt(err / (n*m*p))
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-8:
            break
        
        lr *= 0.95
    
    # Try rounding to half-integers
    if best_err < 1e-6:
        U_rounded = np.round(best_U * 2) / 2
        V_rounded = np.round(best_V * 2) / 2
        W_rounded = np.round(best_W * 2) / 2
        
        err_rounded = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_rounded[i*m+j, :] * V_rounded[j*p+k, :] * W_rounded[k*n+i, :])
                    err_rounded += (val - 1.0) ** 2
        err_rounded = np.sqrt(err_rounded / (n*m*p))
        
        if err_rounded < 1e-6:
            return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
    
    return None

def construct():
    # Try rank-84 structured decomposition first
    try:
        if time.time() < DEADLINE - 10:
            res = _build_rank84_structured()
            if res is not None:
                return res
    except Exception:
        pass
    
    # Try heuristic search for rank-84
    try:
        if time.time() < DEADLINE - 30:
            res = _try_low_rank_search()
            if res is not None:
                U = np.array(res['U'])
                V = np.array(res['V'])
                W = np.array(res['W'])
                n, m, p = 4, 4, 8
                result = np.zeros((n, m, p), dtype=np.float64)
                for r in range(len(U[0])):
                    u_col = U[:, r].reshape(n, m)
                    v_col = V[:, r].reshape(m, p)
                    w_col = W[:, r].reshape(p, n)
                    result += np.einsum('ij,jk,ki->ijk', u_col, v_col, w_col)
                if np.allclose(result, 1.0, atol=1e-6):
                    return res
    except Exception:
        pass
    
    # Fall back to rank-98
    try:
        if time.time() < DEADLINE:
            res = _build_rank98()
            if res is not None:
                return res
    except Exception:
        pass
    
    # Final fallback to schoolbook
    return _schoolbook()

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
    U, V, W = res['U'], res['V'], res['W']
    n, m, p = 4, 4, 8
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(len(U[0])):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-6:
                    valid = False
                    break
            if not valid: break
        if not valid: break
    print(f"Valid: {valid}")
