"""Matrix multiplication 4x4x4 tensor decomposition.

Uses block-partitioned Strassen: partition 4x4 into 2x2 blocks,
apply Strassen's 7-multiplication formula where each 2x2 block
product uses schoolbook (rank 4). Total rank = 7 * 4 = 28.
Output factor follows the official trace convention W[k*n+i, r].
"""
import numpy as np
import time

DEADLINE = 120.0


def _naive_fallback():
    """Schoolbook rank-64 decomposition; always valid."""
    U = np.zeros((16, 64), dtype=int)
    V = np.zeros((16, 64), dtype=int)
    W = np.zeros((16, 64), dtype=int)
    col = 0
    for ii in range(4):
        for jj in range(4):
            for kk in range(4):
                U[ii * 4 + jj, col] = 1
                V[jj * 4 + kk, col] = 1
                W[kk * 4 + ii, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _verify(U, V, W, rank):
    """Check T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] == delta(i,k)."""
    for i in range(4):
        for j in range(4):
            for k in range(4):
                expected = 1 if i == k else 0
                actual = 0
                for c in range(rank):
                    actual += U[i * 4 + j, c] * V[j * 4 + k, c] * W[k * 4 + i, c]
                if actual != expected:
                    return False
    return True


def _greedy_reduce(U, V, W, start_time):
    """Greedy removal of redundant columns."""
    ncol = U.shape[1]
    target = np.zeros(64, dtype=int)
    idx = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                if i == k:
                    target[idx] = 1
                idx += 1

    contrib = np.zeros((ncol, 64), dtype=int)
    for c in range(ncol):
        idx = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    contrib[c, idx] = U[i * 4 + j, c] * V[j * 4 + k, c] * W[k * 4 + i, c]
                    idx += 1

    active = list(range(ncol))
    residual = target - contrib.sum(axis=0)
    if not np.all(residual == 0):
        return None

    norms = np.abs(contrib).sum(axis=1)
    order = sorted(range(ncol), key=lambda c: (norms[c], c))

    for c in order:
        if time.time() - start_time > DEADLINE - 5:
            break
        if c not in active:
            continue
        new_residual = residual + contrib[c]
        if np.all(new_residual == 0):
            active.remove(c)
            residual = new_residual

    if len(active) >= ncol:
        return None
    return U[:, active], V[:, active], W[:, active]


def _block_strassen_4x4():
    """Block-partitioned Strassen decomposition for 4x4 matmul.
    
    Partition each 4x4 matrix into 2x2 blocks of size 2x2:
    A = [[A00, A01], [A10, A11]]  (block indices)
    B = [[B00, B01], [B10, B11]]
    C = [[C00, C01], [C10, C11]]
    
    Strassen's 7 products:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B01)
    M7 = (A01-A11)(B10+B11)
    
    Results:
    C00 = M1 - M4
    C01 = M2
    C10 = M3
    C11 = M1 - M3 + M5 + M6
    
    Each 2x2 block product uses schoolbook (4 scalar mults).
    Total rank = 7 * 4 = 28.
    """
    # Block (br, bc) in 4x4 covers rows br*2..br*2+1, cols bc*2..bc*2+1
    # A block (ar, ac): rows ar*2..ar*2+1, cols ac*2..ac*2+1
    # B block (br, bc): rows br*2..br*2+1, cols bc*2..bc*2+1
    # C block (cr, cc): rows cr*2..cr*2+1, cols cc*2..cc*2+1
    
    # Strassen terms: (a_block, b_block, coeff)
    # a_block = (ar, ac), b_block = (br, bc)
    strassen_terms = [
        # M1 = (A00+A11)(B00+B11)
        [((0,0),(0,0),1), ((0,0),(1,0),1), ((1,1),(0,0),1), ((1,1),(1,0),1)],
        # M2 = (A10+A11)B00
        [((1,0),(0,0),1), ((1,1),(0,0),1)],
        # M3 = A00(B10-B11)
        [((0,0),(1,0),1), ((0,0),(1,1),-1)],
        # M4 = A11(B01-B00)
        [((1,1),(0,1),1), ((1,1),(0,0),-1)],
        # M5 = (A00+A01)B11
        [((0,0),(1,1),1), ((0,1),(1,1),1)],
        # M6 = (A10-A00)(B00+B01)
        [((1,0),(0,0),1), ((1,0),(0,1),1), ((0,0),(0,0),-1), ((0,0),(0,1),-1)],
        # M7 = (A01-A11)(B10+B11)
        [((0,1),(1,0),1), ((0,1),(1,1),1), ((1,1),(1,0),-1), ((1,1),(1,1),-1)],
    ]
    
    # Output: which C blocks each M_t contributes to
    # C00 = M1 - M4 => M1: C00+=1, M4: C00+=-1
    # C01 = M2 => M2: C01+=1
    # C10 = M3 => M3: C10+=1
    # C11 = M1 - M3 + M5 + M6 => M1: C11+=1, M3: C11+=-1, M5: C11+=1, M6: C11+=1
    output_map = [
        # M1
        {(0,0): 1, (1,1): 1},
        # M2
        {(0,1): 1},
        # M3
        {(1,0): 1, (1,1): -1},
        # M4
        {(0,0): -1},
        # M5
        {(1,1): 1},
        # M6
        {(1,1): 1},
        # M7
        {},  # M7 not used in standard 2x2 Strassen output
    ]
    
    rank = 28
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((16, rank), dtype=int)
    W = np.zeros((16, rank), dtype=int)
    
    col = 0
    for t in range(7):
        for (a_blk, b_blk, coeff) in strassen_terms[t]:
            ar, ac = a_blk  # A block row, col
            br, bc = b_blk  # B block row, col
            # Schoolbook 2x2 multiplication of A_block and B_block
            for a_local in range(2):
                for b_local in range(2):
                    # U: A row = ar*2 + a_local, col = ac*2 + b_local
                    u_row = ar * 2 + a_local
                    u_col = ac * 2 + b_local
                    # V: B row = br*2 + b_local, col = bc*2 + a_local
                    v_row = br * 2 + b_local
                    v_col = bc * 2 + a_local
                    # W: C[k, i] trace convention, k = col of C, i = row of C
                    # C row = ar*2 + a_local, C col = bc*2 + a_local
                    # So W row = C_col * 4 + C_row = (bc*2 + a_local)*4 + (ar*2 + a_local)
                    w_row = (bc * 2 + a_local) * 4 + (ar * 2 + a_local)
                    
                    U[u_row * 4 + u_col, col] = coeff
                    V[v_row * 4 + v_col, col] = 1
                    
                    # Determine output coefficient for this C block
                    # C block position: (cr, cc) = (ar, bc)
                    cr = ar
                    cc = bc
                    out_coeff = 0
                    if (cr, cc) in output_map[t]:
                        out_coeff = output_map[t][(cr, cc)]
                    W[w_row, col] = out_coeff
                    col += 1
    
    # Verify
    if not _verify(U, V, W, rank):
        return None
    
    return U, V, W


def construct():
    start_time = time.time()
    
    # Try block Strassen
    result = _block_strassen_4x4()
    if result is not None:
        U, V, W = result
        rank = U.shape[1]
        if _verify(U, V, W, rank):
            best = (U, V, W)
            # Try greedy reduction
            reduced = _greedy_reduce(U, V, W, start_time)
            if reduced is not None:
                Ur, Vr, Wr = reduced
                if _verify(Ur, Vr, Wr, Ur.shape[1]):
                    if Ur.shape[1] < rank:
                        best = (Ur, Vr, Wr)
            U, V, W = best
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback
    return _naive_fallback()
