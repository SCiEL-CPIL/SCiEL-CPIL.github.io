import numpy as np
import time
from fractions import Fraction

_START_TIME = time.time()
TIME_LIMIT = 140

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def _verify_fast(U, V, W, n, m, p):
    """Fast numerical verification."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Use einsum for speed
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # We need T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
    
    # Reshape U to (n, m, rank)
    U3 = U.reshape(n, m, rank)
    # Reshape V to (m, p, rank)
    V3 = V.reshape(m, p, rank)
    # Reshape W to (p, n, rank)
    W3 = W.reshape(p, n, rank)
    
    # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # This is a 3-way sum
    T = np.einsum('ijr,jkr,kip->ijk', U3, V3, W3)
    
    # Check if T equals identity in the appropriate sense
    # Actually, for matrix multiplication, T should be 1 when the indices match
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] should be 1 for all i,j,k (it's the bilinear form)
    # Wait, no. The tensor T represents the multiplication map.
    # T[i,j,k] = 1 means C[i,k] gets contribution A[i,j]*B[j,k]
    # So T should be all ones? No, that's not right either.
    
    # Let me reconsider. The standard definition is:
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form, T is a trilinear form such that:
    # T(A,B,C) = sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] * C[k,i]  (with trace)
    # For exact decomposition, we need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1  for all i,j,k
    # This is because C[k,i] = sum_j A[i,j]*B[j,k], and the trace convention
    # W[k*n+i, r] corresponds to the (k,i) entry of C.
    
    # So yes, T should be all 1s.
    expected = np.ones((n, m, p))
    max_err = np.max(np.abs(T - expected))
    
    return max_err < 1e-8, max_err

def construct():
    n, m, p = 4, 4, 7
    
    # Use rank 93 decomposition
    total_rank = 93
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    current_rank = 0
    
    # Strassen 2x2x2 terms
    strassen_2x2 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 0, (1,0): 1, (1,1): 0}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1, (1,0): 0, (1,1): 0}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0}),
    ]
    
    # Strassen 4x4x4 outer terms
    strassen_4x4_outer = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}),
        ({0: 1}, {1: 1, 3: -1}),
        ({3: 1}, {2: 1, 0: -1}),
        ({0: 1, 1: 1}, {3: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}),
    ]
    
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
    def add_2x2x3(ar, ac, br, cr):
        nonlocal current_rank
        
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
                j = 4 + lj
                V[i * p + j, r] += cy
            
            for (li, lj), cc in c_sub.items():
                if cc == 0:
                    continue
                i_final = 2 * cr + li
                k_final = 4 + lj
                W[k_final * n + i_final, r] += cc
        
        r = current_rank; current_rank += 1
        U[(2*ar)*m + (2*ac), r] = 1.0
        V[(2*br)*p + 6, r] = 1.0
        W[6*n + (2*cr), r] = 1.0
        
        r = current_rank; current_rank += 1
        U[(2*ar)*m + (2*ac+1), r] = 1.0
        V[(2*br+1)*p + 6, r] = 1.0
        W[6*n + (2*cr), r] = 1.0
        
        r = current_rank; current_rank += 1
        U[(2*ar+1)*m + (2*ac), r] = 1.0
        V[(2*br)*p + 6, r] = 1.0
        W[6*n + (2*cr+1), r] = 1.0
        
        r = current_rank; current_rank += 1
        U[(2*ar+1)*m + (2*ac+1), r] = 1.0
        V[(2*br+1)*p + 6, r] = 1.0
        W[6*n + (2*cr+1), r] = 1.0
    
    add_2x2x3(0, 0, 0, 0)
    add_2x2x3(0, 1, 1, 0)
    add_2x2x3(1, 0, 0, 1)
    add_2x2x3(1, 1, 1, 1)
    
    assert current_rank == 93, f"Expected 93, got {current_rank}"
    
    U_np = U[:, :current_rank].copy()
    V_np = V[:, :current_rank].copy()
    W_np = W[:, :current_rank].copy()
    
    # Try to peel zero columns
    keep_mask = []
    for r in range(current_rank):
        if np.any(U_np[:, r] != 0) or np.any(V_np[:, r] != 0) or np.any(W_np[:, r] != 0):
            keep_mask.append(r)
    
    if len(keep_mask) < current_rank:
        U_np = U_np[:, keep_mask]
        V_np = V_np[:, keep_mask]
        W_np = W_np[:, keep_mask]
    
    U_final = U_np.tolist()
    V_final = V_np.tolist()
    W_final = W_np.tolist()
    
    return {"U": U_final, "V": V_final, "W": W_final}
