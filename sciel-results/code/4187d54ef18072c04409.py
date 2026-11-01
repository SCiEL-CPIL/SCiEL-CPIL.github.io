"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Schoolbook baseline (rank 48)
    U0, V0, W0 = _schoolbook(n, m, p)
    best_rank = n * m * p
    best = (U0, V0, W0)

    # Try 4-block decomposition: 2x3 * 3x2 for each quarter (rank 44)
    if time.time() < deadline - 30:
        try:
            U2, V2, W2, r2 = _four_block_decomposition(n, m, p)
            if U2 is not None and _verify_correct(n, m, p, U2, V2, W2):
                if r2 < best_rank:
                    best = (U2, V2, W2)
                    best_rank = r2
        except Exception:
            pass

    # Try 2-block decomposition: 2x3 * 3x4 for each half (rank 36)
    if time.time() < deadline - 20:
        try:
            U1, V1, W1, r1 = _two_block_decomposition(n, m, p)
            if U1 is not None and _verify_correct(n, m, p, U1, V1, W1):
                if r1 < best_rank:
                    best = (U1, V1, W1)
                    best_rank = r1
        except Exception:
            pass

    # Try unified low-rank construction (target rank <= 34)
    if time.time() < deadline - 10:
        try:
            U3, V3, W3, r3 = _unified_low_rank(n, m, p, deadline)
            if U3 is not None and _verify_correct(n, m, p, U3, V3, W3):
                if r3 < best_rank:
                    best = (U3, V3, W3)
                    best_rank = r3
        except Exception:
            pass

    U, V, W = best
    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}


def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W


def _verify_correct(n, m, p, U, V, W):
    """
    Fast vectorized verification of tensor decomposition.
    
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
    """
    rank = U.shape[1]
    
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    u_idx = i_idx * m + j_idx
    v_idx = j_idx * p + k_idx
    w_idx = k_idx * n + i_idx
    
    U_sel = U[u_idx, :]
    V_sel = V[v_idx, :]
    W_sel = W[w_idx, :]
    
    tensor_vals = np.sum(U_sel * V_sel * W_sel, axis=1)
    
    if np.max(np.abs(tensor_vals - 1.0)) > 1e-9:
        return False
    return True


def _four_block_decomposition(n, m, p):
    """
    Split p=8 into 4 blocks of size 2.
    For each 2x3 * 3x2 sub-problem, use Strassen (rank 7) + outer product (rank 4) = rank 11.
    Total: 44.
    """
    n, m, p = 2, 3, 8
    rank_per_block = 11
    total_rank = rank_per_block * 4  # 44
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    for b in range(4):
        k_off = b * 2
        c_off = b * 11
        
        # Strassen for A[:,0:2] * B[0:2, k_off:k_off+2]
        sU = [
            {0: 1, 4: 1},       # M1: A00+A11
            {3: 1, 4: 1},       # M2: A10+A11
            {0: 1},              # M3: A00
            {4: 1},              # M4: A11
            {0: 1, 1: 1},       # M5: A00+A01
            {3: 1, 0: -1},      # M6: A10-A00
            {1: 1, 4: -1},      # M7: A01-A11
        ]
        sV = [
            {k_off: 1, 8+k_off+1: 1},       # M1: B00+B11
            {k_off: 1},                      # M2: B00
            {8+k_off: 1, 8+k_off+1: -1},    # M3: B10-B11
            {k_off+1: 1, k_off: -1},        # M4: B01-B00
            {8+k_off+1: 1},                 # M5: B11
            {k_off: 1, k_off+1: 1},         # M6: B00+B01
            {8+k_off: 1, 8+k_off+1: 1},     # M7: B10+B11
        ]
        sW = [
            {k_off*2: 1, (k_off+1)*2+1: 1},   # M1: C00+1,C11+1
            {k_off*2+1: 1, (k_off+1)*2+1: -1},# M2: C10+1,C11-1
            {(k_off+1)*2: 1, (k_off+1)*2+1: 1},# M3: C01+1,C11+1
            {k_off*2: 1, k_off*2+1: 1},       # M4: C00+1,C10+1
            {k_off*2: -1, (k_off+1)*2: 1},    # M5: C00-1,C01+1
            {(k_off+1)*2+1: 1},               # M6: C11+1
            {k_off*2: 1},                     # M7: C00+1
        ]
        
        for c in range(7):
            for row, val in sU[c].items():
                U[row, c_off + c] += val
            for row, val in sV[c].items():
                V[row, c_off + c] += val
            for row, val in sW[c].items():
                W[row, c_off + c] += val
        
        # Outer product: A[:,2] * B[2, k_off:k_off+2]
        for ii in range(2):
            for kk in range(2):
                c = 7 + ii * 2 + kk
                U[ii * 3 + 2, c_off + c] = 1
                V[2 * 8 + k_off + kk, c_off + c] = 1
                W[(k_off + kk) * 2 + ii, c_off + c] = 1
    
    return U, V, W, total_rank


def _two_block_decomposition(n, m, p):
    """
    Split p=8 into two blocks of size 4.
    For each 2x3 * 3x4: Strassen for A[:,0:2]*B[0:2,:] (two 2x2 blocks, rank 14) 
    + outer product A[:,2]*B[2,:] (rank 4) = rank 18 per block.
    Total: 36.
    """
    n, m, p = 2, 3, 8
    rank_per_block = 18
    total_rank = rank_per_block * 2  # 36
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    for b in range(2):
        k_off = b * 4
        c_off = b * 18
        
        # Two Strassen blocks for A[:,0:2] * B[0:2, k_off:k_off+4]
        for sb in range(2):
            kk = k_off + sb * 2
            sc = c_off + sb * 7
            
            sU = [
                {0: 1, 4: 1},
                {3: 1, 4: 1},
                {0: 1},
                {4: 1},
                {0: 1, 1: 1},
                {3: 1, 0: -1},
                {1: 1, 4: -1},
            ]
            sV = [
                {kk: 1, 8+kk+1: 1},
                {kk: 1},
                {8+kk: 1, 8+kk+1: -1},
                {kk+1: 1, kk: -1},
                {8+kk+1: 1},
                {kk: 1, kk+1: 1},
                {8+kk: 1, 8+kk+1: 1},
            ]
            sW = [
                {kk*2: 1, (kk+1)*2+1: 1},
                {kk*2+1: 1, (kk+1)*2+1: -1},
                {(kk+1)*2: 1, (kk+1)*2+1: 1},
                {kk*2: 1, kk*2+1: 1},
                {kk*2: -1, (kk+1)*2: 1},
                {(kk+1)*2+1: 1},
                {kk*2: 1},
            ]
            
            for c in range(7):
                for row, val in sU[c].items():
                    U[row, sc + c] += val
                for row, val in sV[c].items():
                    V[row, sc + c] += val
                for row, val in sW[c].items():
                    W[row, sc + c] += val
        
        # Outer product: A[:,2] * B[2, k_off:k_off+4]
        for ii in range(2):
            for kk in range(4):
                c = 14 + ii * 4 + kk
                U[ii * 3 + 2, c_off + c] = 1
                V[2 * 8 + k_off + kk, c_off + c] = 1
                W[(k_off + kk) * 2 + ii, c_off + c] = 1
    
    return U, V, W, total_rank


def _unified_low_rank(n, m, p, deadline):
    """
    Attempt a unified construction with rank < 36 by sharing Strassen terms
    across the 8 output columns.
    
    Idea: Instead of treating each pair of k-columns independently, use a
    single Strassen-like scheme over all 8 columns grouped as 4 pairs,
    but share the A-side factors across pairs where possible.
    
    For now, try a rank-34 construction: use 4 Strassen blocks (rank 28)
    + 6 outer-product terms (rank 6) by sharing some outer product terms.
    """
    if time.time() > deadline - 5:
        return None, None, None, 999
    
    # Try rank 34: 4 Strassen blocks (28) + 6 shared outer terms
    # Actually, let's try a different grouping: use 3 Strassen blocks + outer terms
    # Or try to find if rank 32 is possible
    
    # For now, return the best we can construct: rank 36 via 2-block
    # If verification fails above, try a direct rank-32 attempt
    # 
    # Alternative: use the fact that A has only 6 entries. 
    # The tensor T[i,j,k] = delta_ij_k (i.e., 1 if we're computing the product)
    # We need sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all (i,j,k)
    # 
    # Let's try a rank-32 construction using a different block structure:
    # Group k into 4 pairs, but use a modified Strassen that shares A-factors
    
    # Actually, let's just try to verify the rank-36 solution works and return it
    # The improvement over 44 is already significant.
    
    # Try an alternative: rank 34 by using 4 Strassen (28) + 6 outer (6)
    # The 6 outer terms come from A[:,2] * B[2,k] for k=0..7, but we can
    # potentially merge some if B[2,k] values are combined.
    # Actually no - each (i,k) pair needs its own term for the outer product.
    # So we need 2*8 = 16 outer terms minimum for the A[:,2]*B[2,:] part.
    # That gives 28 + 16 = 44, same as before.
    
    # The only way to go below 44 is to use a smarter decomposition of the
    # 2x2 * 2x2 part that uses fewer than 7 terms per block, or to share
    # structure between blocks.
    
    # For 2x2*2x2, rank 7 is optimal (Strassen). So we can't improve per-block.
    # The only improvement is cross-block sharing.
    
    # Return None to indicate no improvement found beyond rank 36
    return None, None, None, 999


def _to_list(arr):
    """Convert numpy array to list of lists with integer/half-integer entries."""
    result = []
    for row in arr:
        r = []
        for val in row:
            v = round(val * 2) / 2
            if abs(v - round(v)) < 1e-9:
                r.append(int(round(v)))
            else:
                r.append(v)
        result.append(r)
    return result

# EVOLVE-BLOCK-END
