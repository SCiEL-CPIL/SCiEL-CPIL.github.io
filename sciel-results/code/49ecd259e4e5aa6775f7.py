"""Combined approach: Strassen 4x4x4 (rank 49) + 4x4x3 via 2x2x3 blocks (rank 44) = rank 93."""
import time
import random
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START

def _exact_check(U, V, W, n, m, p):
    """Exact verification using Fraction arithmetic."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    uc = Fraction(str(U[i*m+j][c])) if isinstance(U[i*m+j][c], (int, float)) else U[i*m+j][c]
                    vc = Fraction(str(V[j*p+k][c])) if isinstance(V[j*p+k][c], (int, float)) else V[j*p+k][c]
                    wc = Fraction(str(W[k*n+i][c])) if isinstance(W[k*n+i][c], (int, float)) else W[k*n+i][c]
                    s += uc * vc * wc
                if s != 1:
                    return False
    return True


def _fast_check(U, V, W, n, m, p, tol=1e-9):
    """Fast approximate check using numpy."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(s - 1.0) > tol:
                    return False
    return True


def _make_schoolbook(n, m, p):
    """Trivial rank-n*m*p decomposition."""
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W


def _build_rank_93_decomposition():
    """Build the rank-93 decomposition: Strassen 4x4x4 (49) + 4x4x3 via 2x2x3 blocks (44).
    
    The 4x4x3 part is decomposed into four 2x2x3 blocks, each of rank 11:
    - 7 terms from Strassen 2x2x2 for the first 2 columns
    - 4 schoolbook terms for the 3rd column
    Total: 49 + 4*11 = 49 + 44 = 93
    """
    n, m, p = 4, 4, 7
    total_rank = 93
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    current_rank = 0
    
    # Strassen 2x2x2 terms: (x_sub, y_sub, c_sub) where each is a dict mapping (row,col) -> value
    strassen_2x2 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    # Strassen 4x4x4 outer structure
    # For 4x4x4, we use the Kronecker product of 2x2x2 Strassen
    # The 4x4 matrix is viewed as 2x2 blocks of 2x2 matrices
    # Outer term t (0-6) uses strassen_2x2[t] at the block level
    
    # Part 1: 4x4x4 Strassen (rank 49)
    # We decompose 4x4x4 as 7 outer terms, each being a Kronecker product
    # of two 2x2x2 Strassen terms
    
    # The 4x4x4 multiplication tensor can be written as sum over a,b of 7x7 = 49 terms
    # Term (a,b): U = kron(U_a, U_b), V = kron(V_a, V_b), W = kron(W_a, W_b)
    # where U_a, V_a, W_a are the 2x2x2 Strassen terms
    
    for a in range(7):
        for b in range(7):
            r = current_rank
            current_rank += 1
            x_a, y_a, c_a = strassen_2x2[a]
            x_b, y_b, c_b = strassen_2x2[b]
            
            # U: 4x4 matrix, index (i,j) where i=row of A, j=col of A
            # i = 2*bi + li, j = 2*bj + lj
            for (li_a, lj_a), cx_a in x_a.items():
                for (li_b, lj_b), cx_b in x_b.items():
                    i = 2 * li_a + li_b
                    j = 2 * lj_a + lj_b
                    U[i * m + j, r] += cx_a * cx_b
            
            # V: 4x7 matrix, but for the 4x4x4 part only first 4 cols
            # j = row of B (which is col of A), k = col of B
            # j = 2*bi + li, k = 2*bk + lk
            for (li_a, lk_a), cy_a in y_a.items():
                for (li_b, lk_b), cy_b in y_b.items():
                    j = 2 * li_a + li_b
                    k = 2 * lk_a + lk_b
                    V[j * p + k, r] += cy_a * cy_b
            
            # W: 4x4 matrix, index (k,i) where k=row of C, i=col of C
            # k = 2*bk + lk, i = 2*bi + li
            for (lk_a, li_a), cc_a in c_a.items():
                for (lk_b, li_b), cc_b in c_b.items():
                    k = 2 * lk_a + lk_b
                    i = 2 * li_a + li_b
                    W[k * n + i, r] += cc_a * cc_b
    
    # Part 2: 4x4x3 as 4 * (2x2x3), each of rank 11
    # The 4x4x3 multiplication: A(4x4) * B(4x3) = C(4x3)
    # Split into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # B = [[B00, B01, B02], [B10, B11, B12]] where B0* is 2x3, B1* is 2x3
    # C = [[C00, C01, C02], [C10, C11, C12]] where C0* is 2x3, C1* is 2x3
    #
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C02 = A00*B02 + A01*B12
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # C12 = A10*B02 + A11*B12
    #
    # For each 2x2x3 block (A_block * B_block -> C_block):
    # Use Strassen 2x2x2 for cols 0,1 (7 terms) + schoolbook for col 2 (4 terms) = 11 terms
    #
    # But we need to handle the cross terms: A00*B00, A01*B10, etc.
    # Each of these is a 2x2 * 2x3 = 2x3 multiplication
    #
    # Actually, let's think differently. The 4x4x3 can be seen as:
    # For each output row block (top or bottom) and each input row block:
    #
    # Top output rows (0,1):
    #   C[0:2, 0:2] = A[0:2, 0:2]*B[0:2, 0:2] + A[0:2, 2:4]*B[2:4, 0:2]
    #   C[0:2, 2]   = A[0:2, 0:2]*B[0:2, 2]   + A[0:2, 2:4]*B[2:4, 2]
    #   C[2:4, 0:2] = A[2:4, 0:2]*B[0:2, 0:2] + A[2:4, 2:4]*B[2:4, 0:2]
    #   C[2:4, 2]   = A[2:4, 0:2]*B[0:2, 2]   + A[2:4, 2:4]*B[2:4, 2]
    #
    # This is getting complex. Let me use the approach from the inspiration:
    # 4 blocks of 2x2x3, each using Strassen for 2 cols + schoolbook for 1 col
    
    def add_2x2x3(ar, ac, br, cr):
        """Add a 2x2x3 decomposition block.
        
        ar: row block index of A (0 or 1)
        ac: col block index of A (0 or 1)  
        br: row block index of B (0 or 1)
        cr: row block index of C (0 or 1)
        
        This computes A[2*ar:2*ar+2, 2*ac:2*ac+2] * B[2*br:2*br+2, 4:7] -> C[2*cr:2*cr+2, 4:7]
        """
        nonlocal current_rank
        
        # Local indices: A_local[li, lj], B_local[li, lk], C_local[li, lk]
        # Global: A[2*ar+li, 2*ac+lj], B[2*br+li, 4+lk], C[2*cr+li, 4+lk]
        
        # Strassen 2x2x2 for columns 4,5 (local cols 0,1)
        for t_idx in range(7):
            r = current_rank
            current_rank += 1
            x_sub, y_sub, c_sub = strassen_2x2[t_idx]
            
            for (li, lj), cx in x_sub.items():
                i = 2 * ar + li
                j = 2 * ac + lj
                U[i * m + j, r] += cx
            
            for (li, lk), cy in y_sub.items():
                i = 2 * br + li
                j = 4 + lk  # global cols 4,5
                V[i * p + j, r] += cy
            
            for (li, lk), cc in c_sub.items():
                if cc == 0:
                    continue
                i_final = 2 * cr + li
                k_final = 4 + lk  # global cols 4,5
                W[k_final * n + i_final, r] += cc
        
        # Schoolbook for column 6 (local col 2)
        # C[li, 2] = sum_lj A[li, lj] * B[lj, 2]
        for li in range(2):
            for lj in range(2):
                r = current_rank
                current_rank += 1
                U[(2 * ar + li) * m + (2 * ac + lj), r] = 1.0
                V[(2 * br + lj) * p + 6, r] = 1.0
                W[6 * n + (2 * cr + li), r] = 1.0
    
    # The four 2x2x3 blocks for the 4x4x3 part:
    # Top C rows (cr=0): A top rows (ar=0) * B both row blocks
    # Bottom C rows (cr=1): A bottom rows (ar=1) * B both row blocks
    # For each, we need A col blocks (ac=0,1) * B row blocks (br=0,1)
    
    # C[0:2, 4:7] = A[0:2, 0:2]*B[0:2, 4:7] + A[0:2, 2:4]*B[2:4, 4:7]
    add_2x2x3(0, 0, 0, 0)  # A[0:2,0:2] * B[0:2,4:7] -> C[0:2,4:7]
    add_2x2x3(0, 1, 1, 0)  # A[0:2,2:4] * B[2:4,4:7] -> C[0:2,4:7]
    
    # C[2:4, 4:7] = A[2:4, 0:2]*B[0:2, 4:7] + A[2:4, 2:4]*B[2:4, 4:7]
    add_2x2x3(1, 0, 0, 1)  # A[2:4,0:2] * B[0:2,4:7] -> C[2:4,4:7]
    add_2x2x3(1, 1, 1, 1)  # A[2:4,2:4] * B[2:4,4:7] -> C[2:4,4:7]
    
    assert current_rank == 93, f"Expected 93, got {current_rank}"
    
    U = U[:, :current_rank].tolist()
    V = V[:, :current_rank].tolist()
    W = W[:, :current_rank].tolist()
    
    return U, V, W


def _greedy_peel(U, V, W, n, m, p, deadline):
    """Greedy tensor peeling: try to remove redundant terms."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    keep = [True] * rank
    best_rank = rank
    
    T_target = np.ones((n, m, p))
    
    def compute_residual(keep_mask):
        res = np.zeros((n, m, p))
        for c in range(rank):
            if not keep_mask[c]:
                continue
            for i in range(n):
                for j in range(m):
                    uc = U_arr[i*m+j, c]
                    if uc == 0:
                        continue
                    for k in range(p):
                        vc = V_arr[j*p+k, c]
                        if vc == 0:
                            continue
                        wc = W_arr[k*n+i, c]
                        if wc == 0:
                            continue
                        res[i,j,k] += uc * vc * wc
        return res
    
    contributions = []
    for c in range(rank):
        contrib = 0.0
        for i in range(n):
            for j in range(m):
                uc = U_arr[i*m+j, c]
                if uc == 0: continue
                for k in range(p):
                    vc = V_arr[j*p+k, c]
                    if vc == 0: continue
                    wc = W_arr[k*n+i, c]
                    if wc == 0: continue
                    contrib += abs(uc * vc * wc)
        contributions.append(contrib)
    
    order = sorted(range(rank), key=lambda c: contributions[c])
    
    for idx in order:
        if time.time() > deadline - 40:
            break
        if not keep[idx]:
            continue
        keep[idx] = False
        residual = compute_residual(keep)
        if np.allclose(residual, T_target, atol=1e-9):
            best_rank = sum(keep)
        else:
            keep[idx] = True
    
    if best_rank < rank:
        keep_indices = [c for c in range(rank) if keep[c]]
        new_U = U_arr[:, keep_indices].tolist()
        new_V = V_arr[:, keep_indices].tolist()
        new_W = W_arr[:, keep_indices].tolist()
        return new_U, new_V, new_W, best_rank
    return U, V, W, rank


def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    n, m, p = DIMENSIONS
    
    # Build the rank-93 decomposition
    U, V, W = _build_rank_93_decomposition()
    
    # Verify with fast check
    if not _fast_check(U, V, W, n, m, p):
        # Fall back to schoolbook
        U, V, W = _make_schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
    
    # Try greedy peeling to see if we can reduce rank
    U, V, W, r = _greedy_peel(U, V, W, n, m, p, deadline)
    
    # Final verification with exact arithmetic
    if not _exact_check(U, V, W, n, m, p):
        # Fall back to schoolbook
        U, V, W = _make_schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
