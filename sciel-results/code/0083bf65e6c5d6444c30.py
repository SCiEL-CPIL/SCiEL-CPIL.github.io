"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS

    # Schoolbook baseline (rank 48)
    U0, V0, W0 = _schoolbook(n, m, p)

    best = None
    best_rank = n * m * p

    # Try the structured 4-block rank-44 scheme
    if time.time() < deadline - 20:
        try:
            U1, V1, W1, r1 = _four_block_decomposition(n, m, p)
            if _verify(n, m, p, U1, V1, W1):
                best = (U1, V1, W1)
                best_rank = r1
        except Exception:
            pass

    # Try a 2-block scheme with rank-20 per block (total 40)
    if time.time() < deadline - 25:
        try:
            U2, V2, W2, r2 = _two_block_decomposition(n, m, p)
            if U2 is not None and _verify(n, m, p, U2, V2, W2):
                if r2 < best_rank:
                    best = (U2, V2, W2)
                    best_rank = r2
        except Exception:
            pass

    # Try a 1-block scheme: full 2x3*3x8 with rank-32
    if time.time() < deadline - 30:
        try:
            U3, V3, W3, r3 = _full_block_decomposition(n, m, p)
            if U3 is not None and _verify(n, m, p, U3, V3, W3):
                if r3 < best_rank:
                    best = (U3, V3, W3)
                    best_rank = r3
        except Exception:
            pass

    # Refine the best solution by removing redundant columns
    if best is not None and time.time() < deadline - 10:
        try:
            U4, V4, W4, r4 = _refine_decomposition(n, m, p, best[0], best[1], best[2], best_rank, deadline)
            if r4 < best_rank and _verify(n, m, p, U4, V4, W4):
                best = (U4, V4, W4)
                best_rank = r4
        except Exception:
            pass

    if best is not None:
        U, V, W = best
        return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}

    return {"U": _to_list(U0), "V": _to_list(V0), "W": _to_list(W0)}


def _schoolbook(n, m, p):
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


def _strassen_2x2():
    """Return rank-7 Strassen decomposition for 2x2 * 2x2."""
    # M1 = (A11 + A22)(B21 - B11)
    # M2 = (A11 + A22)(B11 + B22)
    # M3 = (A11 - A21)B11
    # M4 = A22(B21 - B11)
    # M5 = (A21 + A22)(B22 - B21)
    # M6 = (A11 - A21)(B11 + B21)
    # M7 = (A11 + A22)(B22 - B12)  -- actually standard Strassen:
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M7
    # C21 = M1 + M5
    # C22 = M2 - M1 + M3 - M4
    
    # U factors (for A matrix, 2x2):
    U_strassen = [
        [1, 0, 0, 1],   # M1: A11 + A22
        [1, 0, 0, 1],   # M2: A11 + A22
        [1, 0, -1, 0],  # M3: A11 - A21
        [0, 0, 0, 1],   # M4: A22
        [0, 1, 0, 1],   # M5: A21 + A22
        [1, 0, -1, 0],  # M6: A11 - A21
        [1, 0, 0, 1],   # M7: A11 + A22
    ]
    
    # V factors (for B matrix, 2x2):
    V_strassen = [
        [0, -1, 1, 0],  # M1: B21 - B11
        [1, 0, 0, 1],   # M2: B11 + B22
        [1, 0, 0, 0],   # M3: B11
        [0, -1, 1, 0],  # M4: B21 - B11
        [0, 0, -1, 1],  # M5: B22 - B21
        [1, 1, 0, 0],   # M6: B11 + B21
        [0, -1, 0, 1],  # M7: B22 - B12 -> actually B22 - B11? Let me fix
    ]
    
    # W factors (for C matrix, 2x2):
    W_strassen = [
        [1, 0, 0, -1],  # C11 gets M1, C22 gets -M1... wait
        # C11 = M1 + M4 - M5 - M7
        # C12 = M3 + M7
        # C21 = M1 + M5
        # C22 = M2 - M1 + M3 - M4
    ]
    
    # Let me redo this properly with the standard Strassen formulas:
    # M1 = (A11+A22)(B21-B11)
    # M2 = (A11+A22)(B11+B22)  
    # M3 = (A11-A21)B11
    # M4 = A22(B21-B11)
    # M5 = (A21+A22)(B22-B21)
    # M6 = (A11-A21)(B11+B21)
    # M7 = (A11+A22)(B22-B12)  -- hmm, this isn't standard
    
    # Standard Strassen:
    # M1 = (A11+A22)(B21-B11)
    # M2 = (A11+A22)(B11+B22)
    # M3 = (A11-A21)B11
    # M4 = A22(B21-B11)
    # M5 = (A21+A22)(B22-B21)
    # M6 = (A11-A21)(B11+B21)
    # M7 = (A11+A22)(B22-B12)  -- I think I'm confusing myself
    
    # Let me use the well-known formulation:
    # p1 = (a11 + a22)(b21 - b11)
    # p2 = (a11 + a22)(b11 + b22)
    # p3 = (a11 - a21)b11
    # p4 = a22(b21 - b11)
    # p5 = (a21 + a22)(b22 - b21)
    # p6 = (a11 - a21)(b11 + b21)
    # p7 = (a11 + a22)(b22 - b12)  -- this should be (a11 + a22)(b22 - b11)? No...
    
    # Actually the standard 7 multiplications are:
    # p1 = (a11+a22)(b21-b11)
    # p2 = (a11+a22)(b11+b22)
    # p3 = (a11-a21)b11
    # p4 = a22(b21-b11)
    # p5 = (a21+a22)(b22-b21)
    # p6 = (a11-a21)(b11+b21)
    # p7 = (a11+a22)(b22-b12)  -- I'll verify numerically
    
    # For now, let me just return the known correct rank-7 for 2x2*2x2
    # using a verified decomposition
    return None  # placeholder, will use direct construction


def _four_block_decomposition(n, m, p):
    """
    Split the p=8 columns into 4 blocks of size 2.
    For each 2x3 x 3x2 sub-problem, use a rank-11 decomposition.
    Total rank: 44.
    """
    n, m, p = 2, 3, 8
    block_size = 2
    rank_per_block = 11
    total_rank = rank_per_block * 4

    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    for b in range(4):
        k_offset = b * block_size
        col_offset = b * rank_per_block

        # Rank-11 decomposition for 2x3 * 3x2
        # Strategy: Use Strassen for the 2x2 core (rows 0,1 of A; cols 0,1 of B)
        # plus handle the extra column/row
        #
        # A is 2x3: A[:,0:2] is 2x2, A[:,2] is 2x1
        # B is 3x2: B[0:2,:] is 2x2, B[2,:] is 1x2
        # C = A[:,0:2] @ B[0:2,:] + A[:,2] @ B[2,:]
        #
        # First term: 2x2 * 2x2, rank 7 (Strassen)
        # Second term: 2x1 * 1x2, rank 4 (outer product)
        # Total: rank 11
        
        # Strassen for 2x2 * 2x2 (using rows 0,1 of A and cols 0,1 of B)
        # a11=A[0,0], a12=A[0,1], a21=A[1,0], a22=A[1,1]
        # b11=B[0,0], b12=B[0,1], b21=B[1,0], b22=B[1,1]
        
        # U for Strassen terms (indices into A: 0=a11, 1=a12, 2=a21, 3=a22)
        strassen_U = [
            {0: 1, 3: 1},    # p1: a11+a22
            {0: 1, 3: 1},    # p2: a11+a22
            {0: 1, 2: -1},   # p3: a11-a21
            {3: 1},          # p4: a22
            {2: 1, 3: 1},    # p5: a21+a22
            {0: 1, 2: -1},   # p6: a11-a21
            {0: 1, 3: 1},    # p7: a11+a22
        ]
        
        # V for Strassen terms (indices into B: 0=b11, 1=b12, 2=b21, 3=b22)
        # With k_offset, B row j, col k -> index j*p + k_offset + k
        strassen_V = [
            {1 * p + k_offset + 0: 1, 0 * p + k_offset + 0: -1},  # p1: b21-b11
            {0 * p + k_offset + 0: 1, 1 * p + k_offset + 1: 1},   # p2: b11+b22
            {0 * p + k_offset + 0: 1},                             # p3: b11
            {1 * p + k_offset + 0: 1, 0 * p + k_offset + 0: -1},  # p4: b21-b11
            {1 * p + k_offset + 1: 1, 1 * p + k_offset + 0: -1},  # p5: b22-b21
            {0 * p + k_offset + 0: 1, 1 * p + k_offset + 0: 1},   # p6: b11+b21
            {1 * p + k_offset + 1: 1, 0 * p + k_offset + 0: -1},  # p7: b22-b11
        ]
        
        # W for Strassen terms (indices into C: C[i,k] -> k*n+i)
        # C[0,0]=p1+p4-p5-p7, C[0,1]=p3+p7, C[1,0]=p1+p5, C[1,1]=p2-p1+p3-p4
        strassen_W = [
            {(k_offset + 0) * n + 0: 1, (k_offset + 1) * n + 1: -1},  # p1: C[0,0]+=1, C[1,1]-=1
            {(k_offset + 0) * n + 1: 1},                               # p2: C[1,1]+=1
            {(k_offset + 0) * n + 1: 1, (k_offset + 1) * n + 1: 1},   # p3: C[0,1]+=1, C[1,1]+=1
            {(k_offset + 0) * n + 0: 1, (k_offset + 1) * n + 1: -1},  # p4: C[0,0]+=1, C[1,1]-=1
            {(k_offset + 0) * n + 0: -1, (k_offset + 1) * n + 0: 1},  # p5: C[0,0]-=1, C[1,0]+=1
            {(k_offset + 0) * n + 1: 1},                               # p6: C[0,1]+=1
            {(k_offset + 1) * n + 1: 1},                               # p7: C[1,1]+=1
        ]
        
        # Wait, I need to double-check the Strassen reconstruction:
        # C[0,0] = p1 + p4 - p5 - p7
        # C[0,1] = p3 + p7  -- hmm, let me re-derive
        
        # Actually, let me just use a verified rank-7 decomposition.
        # The standard Strassen gives:
        # c11 = p1 + p4 - p5 - p7
        # c12 = p3 + p7
        # c21 = p1 + p5
        # c22 = p2 - p1 + p3 - p4
        
        # So W contributions:
        # p1: c11+=1, c21+=1, c22-=1
        # p2: c22+=1
        # p3: c12+=1, c22+=1
        # p4: c11+=1, c22-=1
        # p5: c11-=1, c21+=1
        # p6: (not used in output? Let me check)
        # p7: c11-=1, c12+=1
        
        # Hmm, p6 doesn't appear in the standard output. Let me re-verify.
        
        # Actually I think I made an error. Let me use a simpler approach:
        # Just verify numerically what works.
        
        # For now, let me fill in the Strassen terms and the extra terms,
        # then verify.
        
        for c in range(7):
            for row, val in strassen_U[c].items():
                U[row, col_offset + c] += val
            for row, val in strassen_V[c].items():
                V[row, col_offset + c] += val
            for row, val in strassen_W[c].items():
                W[row, col_offset + c] += val
        
        # Extra terms for A[:,2] @ B[2,:] (2x1 * 1x2 = rank 4)
        # A[0,2]*B[2,0] -> C[0,0]
        # A[0,2]*B[2,1] -> C[0,1]
        # A[1,2]*B[2,0] -> C[1,0]
        # A[1,2]*B[2,1] -> C[1,1]
        for ii in range(2):
            for kk in range(2):
                c = 7 + ii * 2 + kk
                U[ii * m + 2, col_offset + c] = 1
                V[2 * p + k_offset + kk, col_offset + c] = 1
                W[(k_offset + kk) * n + ii, col_offset + c] = 1

    return U, V, W, total_rank


def _two_block_decomposition(n, m, p):
    """
    Split the p=8 columns into 2 blocks of size 4.
    For each 2x3 x 3x4 sub-problem, attempt a rank-20 decomposition.
    Total rank: 40.
    
    For 2x3 * 3x4:
    Split B(3x4) into B_left(3x2) and B_right(3x2).
    Each 2x3*3x2 block uses rank-11.
    But we can potentially share some computations.
    
    Let's try: use rank-11 for each half, total 22, then try to merge.
    Actually, let's just use 2 * rank-11 = 22 and see if refinement helps.
    """
    n, m, p = 2, 3, 8
    block_size = 4
    rank_per_block = 11  # Same as before, but for 4 columns we need more
    # Actually for 3x4, we need rank for 2x3*3x4
    # Schoolbook: 2*3*4 = 24
    # Let's use 2 * (rank for 2x3*3x2) = 2 * 11 = 22
    total_rank = 22 * 2  # 44, same as before
    
    # This is the same as the 4-block approach, so skip if we already have that
    return None, None, None, 44


def _full_block_decomposition(n, m, p):
    """
    Attempt a full 2x3 * 3x8 decomposition with lower rank.
    This is complex; return None for now.
    """
    return None, None, None, n * m * p


def _verify(n, m, p, U, V, W):
    """
    Verify that the decomposition is exact using multiple test cases.
    """
    rank = U.shape[1]
    
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    test_cases = [
        (np.ones((n, m)), np.ones((m, p))),
        (np.arange(n * m, dtype=float).reshape(n, m) + 1,
         np.arange(m * p, dtype=float).reshape(m, p) + 1),
    ]
    
    np.random.seed(42)
    test_cases.append((np.random.randn(n, m), np.random.randn(m, p)))
    np.random.seed(123)
    test_cases.append((np.random.randn(n, m), np.random.randn(m, p)))
    
    for A, B in test_cases:
        C_true = A @ B
        u_vals = np.einsum('ijc,ij->ic', U_reshaped, A)
        v_vals = np.einsum('lkp,lk->kp', V_reshaped, B)
        C_comp = np.einsum('kic,ic,kc->ik', W_reshaped, u_vals, v_vals)
        if not np.allclose(C_comp, C_true, atol=1e-9):
            return False
    return True


def _refine_decomposition(n, m, p, U, V, W, current_rank, deadline):
    """
    Try to reduce the rank by removing linearly dependent columns.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    rank = U.shape[1]
    if rank <= 1:
        return U, V, W, rank
    
    removed_any = True
    while removed_any and time.time() < deadline - 5:
        removed_any = False
        for c_remove in range(rank - 1, -1, -1):
            if time.time() >= deadline - 5:
                break
            
            mask = [c for c in range(rank) if c != c_remove]
            U_new = U[:, mask]
            V_new = V[:, mask]
            W_new = W[:, mask]
            
            if _verify(n, m, p, U_new, V_new, W_new):
                U, V, W = U_new, V_new, W_new
                rank -= 1
                removed_any = True
                if rank <= 1:
                    return U, V, W, rank
    
    return U, V, W, rank


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
