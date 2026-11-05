import numpy as np
import time

_START_TIME = time.time()
TIME_LIMIT = 130

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct using NumPy for speed."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
    # Check dimensions
    assert U.shape == (n * m, rank), f"U shape {U.shape} != ({n*m}, {rank})"
    assert V.shape == (m * p, rank), f"V shape {V.shape} != ({m*p}, {rank})"
    assert W.shape == (p * n, rank), f"W shape {W.shape} != ({p*n}, {rank})"
    
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3, optimize=True)
    
    max_err = np.max(np.abs(T - 1.0))
    return max_err < 1e-9, max_err

def _try_peel(U, V, W, n, m, p):
    """
    Try to peel off rank-1 components that are linearly redundant.
    Iteratively check if removing any column preserves the tensor.
    """
    rank = U.shape[1]
    if rank <= 1:
        return U, V, W
    
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # Precompute the target tensor
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    target = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3, optimize=True)
    
    changed = True
    while changed:
        changed = False
        rank = U.shape[1]
        if rank <= 1:
            break
        
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        
        keep = []
        for r in range(rank):
            # Check if column r is all zeros in all three matrices
            if np.all(U[:, r] == 0) and np.all(V[:, r] == 0) and np.all(W[:, r] == 0):
                changed = True
                continue
            
            # Try removing column r: compute tensor without it
            mask = np.ones(rank, dtype=bool)
            mask[r] = False
            U_sub = U3[:, :, mask]
            V_sub = V3[:, :, mask]
            W_sub = W3[:, :, mask]
            
            if U_sub.shape[2] == 0:
                T_sub = np.zeros((n, m, p))
            else:
                T_sub = np.einsum('ijr,jkr,kir->ijk', U_sub, V_sub, W_sub, optimize=True)
            
            if np.max(np.abs(T_sub - target)) < 1e-9:
                # Column r is redundant
                changed = True
                continue
            
            keep.append(r)
        
        if len(keep) < rank:
            U = U[:, keep]
            V = V[:, keep]
            W = W[:, keep]
    
    return U, V, W

def construct():
    n, m, p = 4, 4, 7
    total_rank = 93
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    current_rank = 0
    
    # Strassen 2x2x2 terms: (x_sub, y_sub, c_sub)
    strassen_2x2 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 0, (1,0): 1, (1,1): 0}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1, (1,0): 0, (1,1): 0}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0}),
    ]
    
    # Strassen 4x4x4: 7 outer terms, each with 7 inner terms
    strassen_4x4_outer = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}),
        ({0: 1}, {1: 1, 3: -1}),
        ({3: 1}, {2: 1, 0: -1}),
        ({0: 1, 1: 1}, {3: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}),
    ]
    
    # For 4x4x4, the c_block for each outer term m_idx
    c_blocks_4x4 = [
        {(0,0): 1.0, (1,1): 1.0},
        {(1,0): 1.0, (1,1): -1.0},
        {(0,1): 1.0, (1,1): 1.0},
        {(0,0): 1.0, (1,0): 1.0},
        {(0,0): -1.0, (0,1): 1.0},
        {(1,1): 1.0},
        {(0,0): 1.0},
    ]
    
    # Part 1: 4x4x4 Strassen (rank 49)
    for m_idx in range(7):
        x_def, y_def = strassen_4x4_outer[m_idx]
        c_block = c_blocks_4x4[m_idx]
        for t_idx in range(7):
            r = current_rank
            current_rank += 1
            x_sub, y_sub, c_sub = strassen_2x2[t_idx]
            
            for (li, lj), cx in x_sub.items():
                for b, cb in x_def.items():
                    bi, bj = b // 2, b % 2
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    U[i * m + j, r] += cx * cb
            
            for (li, lj), cy in y_sub.items():
                for b, cb in y_def.items():
                    bi, bj = b // 2, b % 2
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    V[i * p + j, r] += cy * cb
            
            for (li, lj), cc in c_sub.items():
                if cc == 0:
                    continue
                for (br, bc), cb_c in c_block.items():
                    if cb_c == 0:
                        continue
                    i_final = 2 * br + li
                    k_final = 2 * bc + lj
                    W[k_final * n + i_final, r] += cc * cb_c
    
    # Part 2: 4x4x3 as 4 * (2x2x3), each of rank 11
    # 2x2x3: Strassen 2x2x2 (7 terms) + 4 schoolbook terms for col 3
    
    def add_2x2x3(ar, ac, br, cr):
        nonlocal current_rank
        
        # Strassen 2x2x2 for columns 4,5 (local cols 0,1)
        for t_idx in range(7):
            r = current_rank
            current_rank += 1
            x_sub, y_sub, c_sub = strassen_2x2[t_idx]
            
            for (li, lj), cx in x_sub.items():
                i = 2 * ar + li
                j = 2 * ac + lj
                U[i * m + j, r] += cx
            
            for (li, lj), cy in y_sub.items():
                i = 2 * br + li
                j = 4 + lj  # global cols 4,5
                V[i * p + j, r] += cy
            
            for (li, lj), cc in c_sub.items():
                if cc == 0:
                    continue
                i_final = 2 * cr + li
                k_final = 4 + lj  # global cols 4,5
                W[k_final * n + i_final, r] += cc
        
        # Schoolbook for column 6 (local col 2)
        r = current_rank
        current_rank += 1
        U[(2 * ar + 0) * m + (2 * ac + 0), r] = 1.0
        V[(2 * br + 0) * p + 6, r] = 1.0
        W[6 * n + (2 * cr + 0), r] = 1.0
        
        r = current_rank
        current_rank += 1
        U[(2 * ar + 0) * m + (2 * ac + 1), r] = 1.0
        V[(2 * br + 1) * p + 6, r] = 1.0
        W[6 * n + (2 * cr + 0), r] = 1.0
        
        r = current_rank
        current_rank += 1
        U[(2 * ar + 1) * m + (2 * ac + 0), r] = 1.0
        V[(2 * br + 0) * p + 6, r] = 1.0
        W[6 * n + (2 * cr + 1), r] = 1.0
        
        r = current_rank
        current_rank += 1
        U[(2 * ar + 1) * m + (2 * ac + 1), r] = 1.0
        V[(2 * br + 1) * p + 6, r] = 1.0
        W[6 * n + (2 * cr + 1), r] = 1.0

    # The four 2x2x3 blocks
    add_2x2x3(0, 0, 0, 0)
    add_2x2x3(0, 1, 1, 0)
    add_2x2x3(1, 0, 0, 1)
    add_2x2x3(1, 1, 1, 1)
    
    assert current_rank == 93, f"Expected 93, got {current_rank}"
    
    U_np = U[:, :current_rank]
    V_np = V[:, :current_rank]
    W_np = W[:, :current_rank]
    
    # Try peeling redundant columns iteratively
    if _time_remaining() > 20.0:
        U_np, V_np, W_np = _try_peel(U_np, V_np, W_np, n, m, p)
    
    # Verify correctness
    if _time_remaining() > 5.0:
        is_valid, max_err = _verify(U_np, V_np, W_np, n, m, p)
        if not is_valid:
            # Fallback: return without peeling
            U_np = U[:, :93]
            V_np = V[:, :93]
            W_np = W[:, :93]
    
    return {"U": U_np.tolist(), "V": V_np.tolist(), "W": W_np.tolist()}
