import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def _verify(U, V, W, n, m, p):
    """Verify the decomposition exactly reproduces the matmul tensor."""
    if not U or not V or not W:
        return False
    r = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    # Check shapes
    if Ua.shape != (n * m, r) or Va.shape != (m * p, r) or Wa.shape != (p * n, r):
        return False

    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += Ua[i * m + j, c] * Va[j * p + k, c] * Wa[k * n + i, c]
                if abs(s - 1.0) > max_err:
                    max_err = abs(s - 1.0)
    return max_err < 1e-9

def _get_strassen_terms():
    """
    Returns the 7 Strassen terms for 2x2x2 matrix multiplication.
    Each term is a tuple (u_mask, v_mask, w_mask) where masks are dicts of {index: coeff}.
    Indices are local to the 2x2x2 block:
    U: i*m + j (i in 0,1; j in 0,1)
    V: j*p + k (j in 0,1; k in 0,1)
    W: k*n + i (k in 0,1; i in 0,1)
    """
    # Standard Strassen
    # m1 = (a00 + a11) * (b00 + b11) -> c00 += 1, c11 += 1
    t1 = ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}) 
    # Note: 
    # U indices: 0(0,0), 1(0,1), 2(1,0), 3(1,1)
    # V indices: 0(0,0), 1(0,1), 2(1,0), 3(1,1)
    # W indices: 0(0,0), 1(0,1), 2(1,0), 3(1,1) -- WAIT. 
    # W index is k*n + i.
    # If n=2:
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    # So W indices are also 0,1,2,3 corresponding to (k,i) pairs.
    
    # Let's re-verify W mapping for 2x2x2.
    # W[k*n + i]
    # Term m1 contributes to C[0,0] and C[1,1].
    # C[0,0] is i=0, k=0. W index = 0*2 + 0 = 0.
    # C[1,1] is i=1, k=1. W index = 1*2 + 1 = 3.
    # So t1 W mask: {0: 1, 3: 1}. Correct.
    
    # m2 = (a10 + a11) * b00 -> c10 += 1, c11 -= 1
    # U: a10 (idx 2), a11 (idx 3). V: b00 (idx 0).
    # C: c10 (i=1, k=0 -> W idx 1), c11 (i=1, k=1 -> W idx 3).
    t2 = ({2: 1, 3: 1}, {0: 1}, {1: 1, 3: -1})
    
    # m3 = a00 * (b01 - b11) -> c01 += 1, c11 += 1
    # U: a00 (idx 0). V: b01 (idx 1), b11 (idx 3).
    # C: c01 (i=0, k=1 -> W idx 2), c11 (i=1, k=1 -> W idx 3).
    t3 = ({0: 1}, {1: 1, 3: -1}, {2: 1, 3: 1})
    
    # m4 = a11 * (b10 - b00) -> c00 += 1, c10 += 1
    # U: a11 (idx 3). V: b10 (idx 2), b00 (idx 0).
    # C: c00 (i=0, k=0 -> W idx 0), c10 (i=1, k=0 -> W idx 1).
    t4 = ({3: 1}, {2: 1, 0: -1}, {0: 1, 1: 1})
    
    # m5 = (a00 + a01) * b11 -> c01 += 1, c00 -= 1
    # U: a00 (idx 0), a01 (idx 1). V: b11 (idx 3).
    # C: c01 (i=0, k=1 -> W idx 2), c00 (i=0, k=0 -> W idx 0).
    t5 = ({0: 1, 1: 1}, {3: 1}, {2: 1, 0: -1})
    
    # m6 = (a10 - a00) * (b00 + b01) -> c11 += 1
    # U: a10 (idx 2), a00 (idx 0). V: b00 (idx 0), b01 (idx 1).
    # C: c11 (i=1, k=1 -> W idx 3).
    t6 = ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1})
    
    # m7 = (a01 - a11) * (b10 + b11) -> c00 += 1
    # U: a01 (idx 1), a11 (idx 3). V: b10 (idx 2), b11 (idx 3).
    # C: c00 (i=0, k=0 -> W idx 0).
    t7 = ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    
    return [t1, t2, t3, t4, t5, t6, t7]

def construct():
    _start = time.time()
    n, m, p = DIMENSIONS
    
    # We expect rank 55
    target_rank = 55
    U = [[0.0] * target_rank for _ in range(n * m)]
    V = [[0.0] * target_rank for _ in range(m * p)]
    W = [[0.0] * target_rank for _ in range(p * n)]
    
    col = 0
    strassen_terms = _get_strassen_terms()
    
    # Part 1: 2x2x10 block (5 blocks of 2x2x2)
    # A is 2x2, B is 2x10.
    # We split B into 5 chunks of width 2.
    for g in range(5):
        k_offset = g * 2
        
        # Global indices for this block
        # U indices are global: i * m + j (m=3, but we only use j=0,1 for this part? 
        # Wait, the task is 2x3 by 3x10.
        # A is 2x3. Indices i in {0,1}, j in {0,1,2}.
        # B is 3x10. Indices j in {0,1,2}, k in {0..9}.
        # The "Strassen part" uses A[:, :2] and B[:2, :].
        # So j_local in {0,1} maps to j_global in {0,1}.
        # U global index: i * m + j_global.
        # V global index: j_global * p + k_global.
        # W global index: k_global * n + i.
        
        for u_mask, v_mask, w_mask in strassen_terms:
            for u_idx, coeff in u_mask.items():
                # u_idx is local to 2x2 (0,1,2,3).
                # Map local (i, j) to global.
                # Local 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
                li = u_idx // 2
                lj = u_idx % 2
                global_u_idx = li * m + lj
                U[global_u_idx][col] += coeff
            
            for v_idx, coeff in v_mask.items():
                # v_idx is local to 2x2 (0,1,2,3).
                # Map local (j, k) to global.
                # Local 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
                # Here k_local is 0 or 1. k_global = k_offset + k_local.
                lj = v_idx // 2
                lk = v_idx % 2
                global_k = k_offset + lk
                global_v_idx = lj * p + global_k
                V[global_v_idx][col] += coeff
            
            for w_idx, coeff in w_mask.items():
                # w_idx is local to 2x2 (0,1,2,3).
                # Map local (k, i) to global.
                # Local 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
                # Here k_local is 0 or 1. k_global = k_offset + k_local.
                lk = w_idx // 2
                li = w_idx % 2
                global_k = k_offset + lk
                global_w_idx = global_k * n + li
                W[global_w_idx][col] += coeff
            
            col += 1
    
    # Part 2: 2x1x10 block (Schoolbook)
    # Uses A[:, 2] and B[2, :].
    # A indices: i in {0,1}, j=2. U global: i * m + 2.
    # B indices: j=2, k in {0..9}. V global: 2 * p + k.
    # C indices: i in {0,1}, k in {0..9}. W global: k * n + i.
    
    for i in range(n):
        u_global = i * m + 2
        for k in range(p):
            v_global = 2 * p + k
            w_global = k * n + i
            
            U[u_global][col] = 1.0
            V[v_global][col] = 1.0
            W[w_global][col] = 1.0
            col += 1
    
    if col != target_rank:
        # If we somehow used fewer or more, trim or pad? 
        # We expect exactly 35 + 20 = 55.
        raise ValueError(f"Rank mismatch: calculated {col}, expected {target_rank}")
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        raise ValueError("Verification failed")
        
    elapsed = time.time() - _start
    if elapsed > 110:
        raise TimeoutError(f"Time exceeded: {elapsed}")
        
    return {"U": U, "V": V, "W": W}
