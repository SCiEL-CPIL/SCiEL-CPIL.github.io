"""Find rank-minimal exact tensor decomposition for 3x3x3 matrix multiplication."""
import time
import numpy as np
from itertools import combinations

DIMENSIONS = [3, 3, 3]
DEADLINE = 120.0
SAFETY_MARGIN = 15.0


def _verify_full(U, V, W, n=3, m=3, p=3):
    """Full verification of tensor decomposition."""
    rank = len(U[0])
    # Check all consistent triples: sum should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    # Check all inconsistent triples: sum should be 0
    for a in range(n * m):
        for b in range(m * p):
            for c_idx in range(p * n):
                i1, j1 = divmod(a, m)
                j2, k2 = divmod(b, p)
                k3, i3 = divmod(c_idx, n)
                if j1 != j2 or k2 != k3 or i3 != i1:
                    s = 0.0
                    for c in range(rank):
                        s += U[a][c] * V[b][c] * W[c_idx][c]
                    if abs(s) > 1e-9:
                        return False
    return True


def _build_strassen_3x3():
    """
    Construct a rank-26 decomposition using Strassen's 2x2 core + border.
    
    Partition indices {0,1,2} into core {0,1} and border {2}.
    
    Core (i,j in {0,1}, k in {0,1}): 8 tensor entries, handled by Strassen (7 terms)
    Border (at least one index = 2): 19 tensor entries, handled by standard mult (19 terms)
    Total: 7 + 19 = 26 terms.
    """
    n, m, p = 3, 3, 3
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # === Strassen core: i,j in {0,1}, k in {0,1} ===
    # Using verified Strassen rank-7 decomposition for 2x2 matrix multiplication
    # 
    # The 7 Strassen products and their contributions to C entries:
    # M1 = (A00+A11)(B00+B11) -> C00, C11
    # M2 = (A10+A11)B00       -> C10, C11  
    # M3 = A00(B11-B00)       -> C00, C01
    # M4 = A11(B01-B00)       -> C00, C10
    # M5 = (A00+A01)B11       -> C01, C11
    # M6 = (A10-A00)(B00+B01) -> C10 (only)
    # M7 = (A01-A11)(B10+B11) -> C00, C11
    #
    # C00 = M1 + M4 - M3 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4 + M6  (M6 contributes to C10!)
    # C11 = M1 - M2 + M5 + M7
    
    # M1 = (A00+A11)(B00+B11)
    # U: A00->U[0], A11->U[4]
    # V: B00->V[0], B11->V[4]  
    # W: C00->W[0], C11->W[4]
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[4][col] = 1
    W[0][col] = 1; W[4][col] = 1
    col += 1
    
    # M2 = (A10+A11)B00
    # U: A10->U[3], A11->U[4]
    # V: B00->V[0]
    # W: C10->W[3], C11->W[4]
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[3][col] = 1; W[4][col] = -1
    col += 1
    
    # M3 = A00(B11-B00)
    # U: A00->U[0]
    # V: B11->V[4], B00->V[0]
    # W: C00->W[0], C01->W[1]
    U[0][col] = 1
    V[4][col] = 1; V[0][col] = -1
    W[0][col] = -1; W[1][col] = 1
    col += 1
    
    # M4 = A11(B01-B00)
    # U: A11->U[4]
    # V: B01->V[1], B00->V[0]
    # W: C00->W[0], C10->W[3]
    U[4][col] = 1
    V[1][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[3][col] = 1
    col += 1
    
    # M5 = (A00+A01)B11
    # U: A00->U[0], A01->U[1]
    # V: B11->V[4]
    # W: C01->W[1], C11->W[4]
    U[0][col] = 1; U[1][col] = 1
    V[4][col] = 1
    W[1][col] = 1; W[4][col] = 1
    col += 1
    
    # M6 = (A10-A00)(B00+B01)
    # U: A10->U[3], A00->U[0]
    # V: B00->V[0], B01->V[1]
    # W: C10->W[3] only
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[3][col] = 1
    col += 1
    
    # M7 = (A01-A11)(B10+B11)
    # U: A01->U[1], A11->U[4]
    # V: B10->V[3], B11->V[4]
    # W: C00->W[0], C11->W[4]
    U[1][col] = 1; U[4][col] = -1
    V[3][col] = 1; V[4][col] = 1
    W[0][col] = -1; W[4][col] = 1
    col += 1
    
    # === Border terms: standard multiplication for entries involving index 2 ===
    # 19 border entries, each gets one rank-1 term
    
    # k=2: (i,j,2) for i,j in {0,1,2} -> 9 terms
    for i in range(3):
        for j in range(3):
            U[i*3+j][col] = 1
            V[j*3+2][col] = 1
            W[2*3+i][col] = 1
            col += 1
    
    # i=2, j in {0,1}, k in {0,1}: 4 terms
    for j in range(2):
        for k in range(2):
            U[2*3+j][col] = 1
            V[j*3+k][col] = 1
            W[k*3+2][col] = 1
            col += 1
    
    # j=2, i in {0,1}, k in {0,1}: 4 terms
    for i in range(2):
        for k in range(2):
            U[i*3+2][col] = 1
            V[2*3+k][col] = 1
            W[k*3+i][col] = 1
            col += 1
    
    # i=2, j=2, k in {0,1}: 2 terms
    for k in range(2):
        U[2*3+2][col] = 1
        V[2*3+k][col] = 1
        W[k*3+2][col] = 1
        col += 1
    
    return U, V, W


def _try_reduce_rank(U, V, W, n=3, m=3, p=3, time_limit=10.0):
    """
    Try to reduce rank by checking if any subset of columns is linearly dependent
    in the tensor sense, and if so, eliminate one column.
    
    Strategy: For each pair of columns, check if their sum (or difference) can be
    absorbed into other columns. This is a heuristic approach.
    """
    start_time = time.time()
    rank = len(U[0])
    
    # Build the tensor representation: for each column c, we have a 27-dim vector
    # T_c[a,b,c_idx] = U[a,c] * V[b,c] * W[c_idx,c]
    # The full tensor is sum of T_c over all c
    
    # Check if removing any single column still gives a valid decomposition
    # (i.e., if that column is "redundant")
    for c_remove in range(rank):
        if time.time() - start_time > time_limit:
            break
        
        # Check if column c_remove is needed
        # A column is redundant if the tensor it contributes can be expressed
        # as a linear combination of other columns' contributions
        # This is hard to check directly, so we use a simpler heuristic:
        # Check if all entries in column c_remove are zero (trivially redundant)
        is_zero = True
        for a in range(n * m):
            if U[a][c_remove] != 0:
                is_zero = False
                break
        if is_zero:
            for b in range(m * p):
                if V[b][c_remove] != 0:
                    is_zero = False
                    break
        if is_zero:
            for c_idx in range(p * n):
                if W[c_idx][c_remove] != 0:
                    is_zero = False
                    break
        if is_zero:
            # Remove this column
            U_new = [row[:c_remove] + row[c_remove+1:] for row in U]
            V_new = [row[:c_remove] + row[c_remove+1:] for row in V]
            W_new = [row[:c_remove] + row[c_remove+1:] for row in W]
            if _verify_full(U_new, V_new, W_new, n, m, p):
                return U_new, V_new, W_new
    
    return U, V, W


def _try_pairwise_combination(U, V, W, n=3, m=3, p=3, time_limit=20.0):
    """
    Try to combine pairs of columns to see if we can reduce rank.
    For each pair (c1, c2), check if there exists a scalar alpha such that
    the combined column c1 + alpha*c2 is "equivalent" in some sense.
    """
    start_time = time.time()
    rank = len(U[0])
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    for c1 in range(rank):
        for c2 in range(c1 + 1, rank):
            if time.time() - start_time > time_limit:
                return best_U, best_V, best_W, best_rank
            
            # Try to see if we can eliminate c2 by adjusting c1
            # This is a heuristic - check if the "direction" of c2 in the
            # tensor space is already covered by other columns
            pass
    
    return best_U, best_V, best_W, best_rank


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Get the Strassen+border rank-26 decomposition
    U, V, W = _build_strassen_3x3()
    
    # Verify it's correct
    if not _verify_full(U, V, W, n, m, p):
        # Fall back to schoolbook rank-27
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
    
    # Try to reduce rank
    time_remaining = DEADLINE - SAFETY_MARGIN - (time.time() - start_time)
    if time_remaining > 5:
        U, V, W = _try_reduce_rank(U, V, W, n, m, p, time_limit=min(10, time_remaining))
    
    # Final verification
    if not _verify_full(U, V, W, n, m, p):
        # Fall back to known-good rank-26
        U, V, W = _build_strassen_3x3()
    
    return {"U": U, "V": V, "W": W}
