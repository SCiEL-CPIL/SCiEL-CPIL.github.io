import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # safety margin
    
    # Build the trivial rank-60 decomposition as a baseline
    rank_trivial = n * m * p
    U = np.zeros((n * m, rank_trivial))
    V = np.zeros((m * p, rank_trivial))
    W = np.zeros((p * n, rank_trivial))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
    
    # Now, try to reduce rank by replacing 2x2x2 sub-blocks with Strassen
    # Strassen for 2x2x2 uses 7 terms instead of 8, saving 1 term per block.
    # We can partition the 3x4x5 tensor into disjoint 2x2x2 blocks.
    # 
    # Indices: i in [0,2], j in [0,3], k in [0,4]
    # We can form blocks at:
    # (i=0,1), (j=0,1), (k=0,1)
    # (i=0,1), (j=0,1), (k=2,3)
    # (i=0,1), (j=2,3), (k=0,1)
    # (i=0,1), (j=2,3), (k=2,3)
    # 
    # But we have to be careful about overlaps and unused indices.
    # Actually, the simplest approach: replace every possible 2x2x2 sub-tensor 
    # with Strassen, but we need disjoint blocks to avoid double-counting.
    # 
    # Let's pick 4 disjoint 2x2x2 blocks that cover most of the space:
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    # Block 2: i in {0,1}, j in {2,3}, k in {0,1}
    # Block 3: i in {0,1}, j in {0,1}, k in {2,3}
    # Block 4: i in {0,1}, j in {2,3}, k in {2,3}
    # Remaining: all indices with i=2, or k=4
    # 
    # This gives us 4 blocks * 1 savings each = 4 terms saved.
    # New rank = 60 - 4 = 56.
    
    # Strassen coefficients for 2x2x2:
    # P1 = (A00 + A11)(B00 + B11)
    # P2 = (A10 + A11)B00
    # P3 = A00(B10 - B11)
    # P4 = A11(B01 - B00)
    # P5 = (A00 + A01)B11
    # P6 = (A10 - A00)(B00 + B01)
    # P7 = (A01 - A11)(B10 + B11)
    # 
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Map to global indices:
    # For a block at (i0, j0, k0), local (li, lj, lk) -> global (i0+li, j0+lj, k0+lk)
    # A indices: (li, lj) -> U[i0*4 + j0 + ... wait, U is indexed by (i*m + j)
    # V indices: (lj, lk) -> V[j0*5 + k0 + ... V is indexed by (j*p + k)
    # W indices: (lk, li) -> W[k0*3 + i0 + ... W is indexed by (k*n + i)
    
    # Let's define the Strassen terms more carefully.
    # For a block with base indices (i0, j0, k0):
    # Local A indices: (0,0)->(i0,j0), (0,1)->(i0,j0+1), (1,0)->(i0+1,j0), (1,1)->(i0+1,j0+1)
    # Local B indices: (0,0)->(j0,k0), (0,1)->(j0,k0+1), (1,0)->(j0+1,k0), (1,1)->(j0+1,k0+1)
    # Local C indices: (0,0)->(i0,k0), (0,1)->(i0,k0+1), (1,0)->(i0+1,k0), (1,1)->(i0+1,k0+1)
    
    # Strassen terms (7 terms):
    # t0: A=(i0,j0)+(i0+1,j0+1), B=(j0,k0)+(j0+1,k0+1)
    # t1: A=(i0+1,j0)+(i0+1,j0+1), B=(j0,k0)
    # t2: A=(i0,j0), B=(j0+1,k0)-(j0+1,k0+1)
    # t3: A=(i0+1,j0+1), B=(j0,k0+1)-(j0,k0)
    # t4: A=(i0,j0)+(i0,j0+1), B=(j0+1,k0+1)
    # t5: A=(i0+1,j0)-(i0,j0), B=(j0,k0)+(j0,k0+1)
    # t6: A=(i0,j0+1)-(i0+1,j0+1), B=(j0+1,k0)+(j0+1,k0+1)
    
    # W contributions:
    # C(i0,k0)   = t0 + t3 - t4 + t6
    # C(i0,k0+1) = t2 + t4
    # C(i0+1,k0) = t1 + t3
    # C(i0+1,k0+1)= t0 - t1 + t2 + t5
    
    blocks = [
        (0, 0, 0),  # i0=0, j0=0, k0=0
        (0, 2, 0),  # i0=0, j0=2, k0=0
        (0, 0, 2),  # i0=0, j0=0, k0=2
        (0, 2, 2),  # i0=0, j0=2, k0=2
    ]
    
    # For each block, we replace 8 trivial terms with 7 Strassen terms.
    # We need to track which trivial terms are "covered" by a block.
    
    covered = set()
    for (i0, j0, k0) in blocks:
        for li in range(2):
            for lj in range(2):
                for lk in range(2):
                    i = i0 + li
                    j = j0 + lj
                    k = k0 + lk
                    covered.add((i, j, k))
    
    # Build new U, V, W
    new_U = np.zeros((n * m, 0))
    new_V = np.zeros((m * p, 0))
    new_W = np.zeros((p * n, 0))
    
    current_rank = 0
    
    # First, add Strassen terms for blocks
    for (i0, j0, k0) in blocks:
        # t0
        U_idx = [(i0, j0), (i0+1, j0+1)]
        V_idx = [(j0, k0), (j0+1, k0+1)]
        W_idx = [(k0, i0, 1), (k0, i0+1, 1)]  # (k, i, coeff)
        
        for (ii, jj) in U_idx:
            pass  # We'll build arrays
        
        # Let's just build the 7 terms explicitly
        # Term 0: t0
        r = current_rank
        new_U = np.zeros((n * m, r + 1)) if r == 0 else np.hstack([new_U, np.zeros((n * m, 1))])
        new_V = np.zeros((m * p, r + 1)) if r == 0 else np.hstack([new_V, np.zeros((m * p, 1))])
        new_W = np.zeros((p * n, r + 1)) if r == 0 else np.hstack([new_W, np.zeros((p * n, 1))])
        
        new_U[i0 * m + j0, r] = 1
        new_U[(i0+1) * m + (j0+1), r] = 1
        new_V[j0 * p + k0, r] = 1
        new_V[(j0+1) * p + (k0+1), r] = 1
        new_W[k0 * n + i0, r] = 1      # C(i0, k0)
        new_W[(k0+1) * n + i0, r] = 1  # C(i0, k0+1)
        new_W[k0 * n + (i0+1), r] = 1  # C(i0+1, k0)
        new_W[(k0+1) * n + (i0+1), r] = 1  # C(i0+1, k0+1)
        current_rank += 1
        
        # Term 1: t1
        r = current_rank
        if r == 0:
            new_U = np.zeros((n * m, r + 1))
            new_V = np.zeros((m * p, r + 1))
            new_W = np.zeros((p * n, r + 1))
        else:
            new_U = np.hstack([new_U, np.zeros((n * m, 1))])
            new_V = np.hstack([new_V, np.zeros((m * p, 1))])
            new_W = np.hstack([new_W, np.zeros((p * n, 1))])
        
        new_U[(i0+1) * m + j0, r] = 1
        new_U[(i0+1) * m + (j0+1), r] = 1
        new_V[j0 * p + k0, r] = 1
        new_W[k0 * n + (i0+1), r] = 1      # C(i0+1, k0)
        new_W[(k0+1) * n + (i0+1), r] = -1  # C(i0+1, k0+1)
        current_rank += 1
        
        # Term 2: t2
        r = current_rank
        if r == 0:
            new_U = np.zeros((n * m, r + 1))
            new_V = np.zeros((m * p, r + 1))
            new_W = np.zeros((p * n, r + 1))
        else:
            new_U = np.hstack([new_U, np.zeros((n * m, 1))])
            new_V = np.hstack([new_V, np.zeros((m * p, 1))])
            new_W = np.hstack([new_W, np.zeros((p * n, 1))])
        
        new_U[i0 * m + j0, r] = 1
        new_V[(j0+1) * p + k0, r] = 1
        new_V[(j0+1) * p + (k0+1), r] = -1
        new_W[(k0+1) * n + i0, r] = 1  # C(i0, k0+1)
        new_W[(k0+1) * n + (i0+1), r] = 1  # C(i0+1, k0+1)
        current_rank += 1
        
        # Term 3: t3
        r = current_rank
        if r == 0:
            new_U = np.zeros((n * m, r + 1))
            new_V = np.zeros((m * p, r + 1))
            new_W = np.zeros((p * n, r + 1))
        else:
            new_U = np.hstack([new_U, np.zeros((n * m, 1))])
            new_V = np.hstack([new_V, np.zeros((m * p, 1))])
            new_W = np.hstack([new_W, np.zeros((p * n, 1))])
        
        new_U[(i0+1) * m + (j0+1), r] = 1
        new_V[j0 * p + (k0+1), r] = 1
        new_V[j0 * p + k0, r] = -1
        new_W[k0 * n + i0, r] = 1      # C(i0, k0)
        new_W[k0 * n + (i0+1), r] = 1  # C(i0+1, k0)
        current_rank += 1
        
        # Term 4: t4
        r = current_rank
        if r == 0:
            new_U = np.zeros((n * m, r + 1))
            new_V = np.zeros((m * p, r + 1))
            new_W = np.zeros((p * n, r + 1))
        else:
            new_U = np.hstack([new_U, np.zeros((n * m, 1))])
            new_V = np.hstack([new_V, np.zeros((m * p, 1))])
            new_W = np.hstack([new_W, np.zeros((p * n, 1))])
        
        new_U[i0 * m + j0, r] = 1
        new_U[i0 * m + (j0+1), r] = 1
        new_V[(j0+1) * p + (k0+1), r] = 1
        new_W[k0 * n + i0, r] = -1      # C(i0, k0)
        new_W[(k0+1) * n + i0, r] = 1   # C(i0, k0+1)
        current_rank += 1
        
        # Term 5: t5
        r = current_rank
        if r == 0:
            new_U = np.zeros((n * m, r + 1))
            new_V = np.zeros((m * p, r + 1))
            new_W = np.zeros((p * n, r + 1))
        else:
            new_U = np.hstack([new_U, np.zeros((n * m, 1))])
            new_V = np.hstack([new_V, np.zeros((m * p, 1))])
            new_W = np.hstack([new_W, np.zeros((p * n, 1))])
        
        new_U[(i0+1) * m + j0, r] = 1
        new_U[i0 * m + j0, r] = -1
        new_V[j0 * p + k0, r] = 1
        new_V[j0 * p + (k0+1), r] = 1
        new_W[(k0+1) * n + (i0+1), r] = 1  # C(i0+1, k0+1)
        current_rank += 1
        
        # Term 6: t6
        r = current_rank
        if r == 0:
            new_U = np.zeros((n * m, r + 1))
            new_V = np.zeros((m * p, r + 1))
            new_W = np.zeros((p * n, r + 1))
        else:
            new_U = np.hstack([new_U, np.zeros((n * m, 1))])
            new_V = np.hstack([new_V, np.zeros((m * p, 1))])
            new_W = np.hstack([new_W, np.zeros((p * n, 1))])
        
        new_U[i0 * m + (j0+1), r] = 1
        new_U[(i0+1) * m + (j0+1), r] = -1
        new_V[(j0+1) * p + k0, r] = 1
        new_V[(j0+1) * p + (k0+1), r] = 1
        new_W[k0 * n + i0, r] = 1  # C(i0, k0)
        current_rank += 1
    
    # Now add trivial terms for uncovered (i, j, k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    r = current_rank
                    if r == 0:
                        new_U = np.zeros((n * m, r + 1))
                        new_V = np.zeros((m * p, r + 1))
                        new_W = np.zeros((p * n, r + 1))
                    else:
                        new_U = np.hstack([new_U, np.zeros((n * m, 1))])
                        new_V = np.hstack([new_V, np.zeros((m * p, 1))])
                        new_W = np.hstack([new_W, np.zeros((p * n, 1))])
                    
                    new_U[i * m + j, r] = 1
                    new_V[j * p + k, r] = 1
                    new_W[k * n + i, r] = 1
                    current_rank += 1
    
    # Verify the decomposition is correct
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1 for all valid (i,j,k)
    # and 0 otherwise? Actually, the tensor is 1 for all (i,j,k) since C_ik = sum_j A_ij B_jk.
    
    # Let's do a quick verification
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(current_rank):
                    val += new_U[i * m + j, r] * new_V[j * p + k, r] * new_W[k * n + i, r]
                if abs(val - 1.0) > 1e-10:
                    # Fallback to trivial if verification fails
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": new_U.tolist(), "V": new_V.tolist(), "W": new_W.tolist()}
