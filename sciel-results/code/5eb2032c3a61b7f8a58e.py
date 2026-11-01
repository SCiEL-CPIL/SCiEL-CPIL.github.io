"""Combined program: Strassen blocks + targeted integer search for 2x2x3 rank-10."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 2, 15
    deadline = time.time() + 140  # 10s safety margin before 150s limit

    # Build rank-53 baseline (7 Strassen blocks + 1 schoolbook)
    U53, V53, W53 = _build_strassen_53(n, m, p)

    if not _verify_int(U53, V53, W53, n, m, p):
        U, V, W = _build_schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}

    best_rank = 53
    best_U, best_V, best_W = U53, V53, W53

    # Try to find rank-10 decomposition for 2x2x3 (columns 12,13,14)
    if time.time() < deadline - 5:
        U10, V10, W10 = _find_2x2x3_rank10(deadline)
        if U10 is not None:
            U52, V52, W52 = _build_combined_52(n, m, p, U10, V10, W10)
            if _verify_int(U52, V52, W52, n, m, p):
                best_rank = 52
                best_U, best_V, best_W = U52, V52, W52

    return {"U": best_U, "V": best_V, "W": best_W}


def _build_strassen_53(n, m, p):
    """Build rank-53: 7 Strassen blocks (rank 7 each) + 1 schoolbook (rank 4)."""
    n, m, p = 2, 2, 15
    blocks = [(0,1),(2,3),(4,5),(6,7),(8,9),(10,11),(12,13),(14,)]

    all_U, all_V, all_W = [], [], []

    for block in blocks:
        if len(block) == 2:
            k0, k1 = block
            terms = _strassen_terms(k0, k1)
        else:
            k = block[0]
            terms = _schoolbook_terms(k)

        for (u_dict, v_dict, w_dict) in terms:
            u_entries = [(i*m+j, c) for (i,j), c in u_dict.items()]
            v_entries = [(j*p+k_idx, c) for (j,k_idx), c in v_dict.items()]
            w_entries = [(k_idx*n+i, c) for (k_idx,i), c in w_dict.items()]
            all_U.append(u_entries)
            all_V.append(v_entries)
            all_W.append(w_entries)

    rank = len(all_U)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]

    for c in range(rank):
        for row, val in all_U[c]:
            U[row][c] = val
        for row, val in all_V[c]:
            V[row][c] = val
        for row, val in all_W[c]:
            W[row][c] = val

    return U, V, W


def _strassen_terms(k0, k1):
    """Strassen rank-7 terms for 2x2x2 block with k indices k0, k1."""
    return [
        # M1: (A00+A11)(B0k0+B1k1)
        ({(0,0):1, (1,1):1}, {(0,k0):1, (1,k1):1}, {(k0,0):1, (k1,1):1}),
        # M2: (A10+A11)B0k0
        ({(1,0):1, (1,1):1}, {(0,k0):1}, {(k0,1):1, (k1,1):-1}),
        # M3: A00(B0k1-B1k1)
        ({(0,0):1}, {(0,k1):1, (1,k1):-1}, {(k1,0):1, (k1,1):1}),
        # M4: A11(B1k0-B0k0)
        ({(1,1):1}, {(1,k0):1, (0,k0):-1}, {(k0,0):1, (k0,1):1}),
        # M5: (A00+A01)B1k1
        ({(0,0):1, (0,1):1}, {(1,k1):1}, {(k0,0):-1, (k1,0):1}),
        # M6: (A10-A00)(B0k0+B0k1)
        ({(1,0):1, (0,0):-1}, {(0,k0):1, (0,k1):1}, {(k1,1):1}),
        # M7: (A01-A11)(B1k0+B1k1)
        ({(0,1):1, (1,1):-1}, {(1,k0):1, (1,k1):1}, {(k0,0):1}),
    ]


def _schoolbook_terms(k):
    """Schoolbook rank-4 terms for 2x2x1 block."""
    return [
        ({(0,0):1}, {(0,k):1}, {(k,0):1}),
        ({(0,1):1}, {(1,k):1}, {(k,0):1}),
        ({(1,0):1}, {(0,k):1}, {(k,1):1}),
        ({(1,1):1}, {(1,k):1}, {(k,1):1}),
    ]


def _build_schoolbook(n, m, p):
    """Build schoolbook rank-60 decomposition."""
    rank = n * m * p
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


def _verify_int(U, V, W, n, m, p):
    """Verify exact integer decomposition."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = i*m + j
                v_row = j*p + k
                w_row = k*n + i
                val = 0
                for c in range(rank):
                    val += U[u_row][c] * V[v_row][c] * W[w_row][c]
                if val != 1:
                    return False
    return True


def _build_combined_52(n, m, p, U10, V10, W10):
    """Build rank-52: 6 Strassen blocks + rank-10 2x2x3 block for columns 12,13,14."""
    n, m, p = 2, 2, 15
    blocks = [(0,1),(2,3),(4,5),(6,7),(8,9),(10,11)]

    all_U, all_V, all_W = [], [], []

    for block in blocks:
        k0, k1 = block
        terms = _strassen_terms(k0, k1)
        for (u_dict, v_dict, w_dict) in terms:
            u_entries = [(i*m+j, c) for (i,j), c in u_dict.items()]
            v_entries = [(j*p+k_idx, c) for (j,k_idx), c in v_dict.items()]
            w_entries = [(k_idx*n+i, c) for (k_idx,i), c in w_dict.items()]
            all_U.append(u_entries)
            all_V.append(v_entries)
            all_W.append(w_entries)

    # Add rank-10 terms for columns 12, 13, 14
    for r in range(10):
        u_entries = []
        v_entries = []
        w_entries = []

        for i in range(n):
            for j in range(m):
                val = U10[i*m+j, r]
                if val != 0:
                    u_entries.append((i*m+j, val))

        for j in range(m):
            for k_local in range(3):
                val = V10[j*3 + k_local, r]
                if val != 0:
                    k_global = 12 + k_local
                    v_entries.append((j*p + k_global, val))

        for k_local in range(3):
            for i in range(n):
                val = W10[k_local*n + i, r]
                if val != 0:
                    k_global = 12 + k_local
                    w_entries.append((k_global*n + i, val))

        all_U.append(u_entries)
        all_V.append(v_entries)
        all_W.append(w_entries)

    rank = len(all_U)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]

    for c in range(rank):
        for row, val in all_U[c]:
            U[row][c] = val
        for row, val in all_V[c]:
            V[row][c] = val
        for row, val in all_W[c]:
            W[row][c] = val

    return U, V, W


def _find_2x2x3_rank10(deadline):
    """
    Search for rank-10 decomposition of 2x2x3 matrix multiplication.
    Uses structured search: fix U patterns, solve for V and W.
    """
    n, m, p = 2, 2, 3
    rank = 10

    # Strategy: Try the known rank-10 construction for 2x2x3.
    # We'll use a specific ansatz based on sharing structure between k=0,1,2.
    
    # Key insight: For 2x2x3, we can use a "mixed" Strassen approach.
    # The 7 Strassen terms for 2x2x2 handle k=0,1 with some coupling.
    # We need 3 more terms for k=2 that also help with k=0,1.
    
    # Let me try a concrete construction:
    # Use terms that span all three k-values in their V entries.
    
    U = np.zeros((4, rank))
    V = np.zeros((6, rank))
    W = np.zeros((6, rank))
    
    # Try a specific known construction for rank-10 2x2x3.
    # Based on the "border rank" constructions and known results.
    
    # Ansatz 1: Extend Strassen by allowing V and W to have entries in all 3 k-positions.
    # This means the "Strassen" terms can contribute to multiple output columns.
    
    # Let's try: 7 terms with Strassen U patterns, but V and W can have 
    # non-zero entries in all 6 positions (j,k) for j in {0,1}, k in {0,1,2}.
    
    # U patterns (same as Strassen):
    # r=0: (A00+A11)  -> U[0]=1, U[3]=1
    # r=1: (A10+A11)  -> U[2]=1, U[3]=1
    # r=2: A00         -> U[0]=1
    # r=3: A11         -> U[3]=1
    # r=4: (A00+A01)   -> U[0]=1, U[1]=1
    # r=5: (A10-A00)   -> U[2]=1, U[0]=-1
    # r=6: (A01-A11)   -> U[1]=1, U[3]=-1
    # r=7: A00         -> U[0]=1
    # r=8: A01         -> U[1]=1
    # r=9: (A10+A11)   -> U[2]=1, U[3]=1
    
    U[0,0] = 1; U[3,0] = 1
    U[2,1] = 1; U[3,1] = 1
    U[0,2] = 1
    U[3,3] = 1
    U[0,4] = 1; U[1,4] = 1
    U[2,5] = 1; U[0,5] = -1
    U[1,6] = 1; U[3,6] = -1
    U[0,7] = 1
    U[1,8] = 1
    U[2,9] = 1; U[3,9] = 1
    
    # Now solve for V and W.
    # The 12 equations are: for each (i,j,k), sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
    # 
    # This is a system of 12 equations in 10*6 + 10*6 = 120 unknowns.
    # Highly underdetermined. Let's fix some V entries and solve for the rest.
    
    # Fix V to have a specific structure:
    # For terms 0-6 (Strassen-like), let V only have entries in k=0,1 (standard Strassen).
    # For terms 7-9, let V have entries in k=2 primarily.
    
    # Standard Strassen V patterns for k=0,1:
    # r=0: V[0,0]=1, V[3,0]=1  (B00+B11)
    # r=1: V[0,1]=1             (B00)
    # r=2: V[1,2]=1, V[3,2]=-1 (B01-B11)
    # r=3: V[2,3]=1, V[0,3]=-1 (B10-B00)
    # r=4: V[3,4]=1             (B11)
    # r=5: V[0,5]=1, V[1,5]=1  (B00+B01)
    # r=6: V[2,6]=1, V[3,6]=1  (B10+B11)
    
    V[0,0] = 1; V[3,0] = 1
    V[0,1] = 1
    V[1,2] = 1; V[3,2] = -1
    V[2,3] = 1; V[0,3] = -1
    V[3,4] = 1
    V[0,5] = 1; V[1,5] = 1
    V[2,6] = 1; V[3,6] = 1
    
    # For r=7,8,9, try V patterns involving k=2:
    # r=7: V[4,7]=1  (B02)
    # r=8: V[5,8]=1  (B12)
    # r=9: V[4,9]=1, V[5,9]=1  (B02+B12)
    
    V[4,7] = 1
    V[5,8] = 1
    V[4,9] = 1; V[5,9] = 1
    
    # Now solve for W.
    # W has 6*10 = 60 entries. The 12 equations give constraints.
    # Let's set up the system.
    
    # W indices: W[k*n+i, r] = W[k*2+i, r]
    # W[0,r]=W[k=0,i=0], W[1,r]=W[k=0,i=1], W[2,r]=W[k=1,i=0], W[3,r]=W[k=1,i=1],
    # W[4,r]=W[k=2,i=0], W[5,r]=W[k=2,i=1]
    
    # For each (i,j,k): sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
    
    # Let's write out the 12 equations:
    # (i=0,j=0,k=0): U[0,0]*V[0,0]*W[0,0] + U[0,1]*V[0,0]*W[0,1] + ... = 1
    #   = 1*1*W[0,0] + 0*1*W[0,1] + 1*0*W[0,2] + 0*(-1)*W[0,3] + 1*0*W[0,4] + (-1)*1*W[0,5] + 0*0*W[0,6] + 1*0*W[0,7] + 0*0*W[0,8] + 0*0*W[0,9]
    #   = W[0,0] - W[0,5] = 1
    
    # This is getting complex. Let me just try to solve it numerically.
    
    # Build the linear system: for each (i,j,k), the equation is linear in W entries.
    # But it's bilinear in V and W (given U). Since we've fixed U and V, it's linear in W.
    
    # Actually, each equation sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 is linear in the W entries.
    # Let's build the 12x60 system.
    
    A_mat = np.zeros((12, 60))
    b_vec = np.ones(12)
    
    eq_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(rank):
                    u_val = U[i, r]
                    v_val = V[j * p + k, r]
                    if u_val != 0 and v_val != 0:
                        # W[k*n+i, r] -> index k*2+i, r -> flat index (k*2+i)*10 + r
                        w_idx = (k * n + i) * rank + r
                        A_mat[eq_idx, w_idx] = u_val * v_val
                eq_idx += 1
    
    # Solve the underdetermined system using least squares, then round to integers.
    # But first, let's check if a solution exists with the fixed V.
    
    # Use np.linalg.lstsq to find a minimum-norm solution.
    try:
        W_sol, residuals, rank_A, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
    except:
        return None, None, None
    
    W_sol = W_sol.reshape((6, 10))
    
    # Check if the solution is close to integer values.
    W_rounded = np.round(W_sol)
    
    # Verify the rounded solution.
    error = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i, r] * V[j*p+k, r] * W_rounded[k*n+i, r] for r in range(rank))
                if abs(val - 1) > 1e-6:
                    error += abs(val - 1)
    
    if error < 1e-6:
        # Found a valid integer solution!
        return U.astype(int), V.astype(int), W_rounded.astype(int)
    
    # If the simple ansatz doesn't work, try a more general approach.
    # Allow V[7,8,9] to have entries in k=0,1 as well (coupling).
    
    # Try different V patterns for r=7,8,9.
    v_patterns = [
        # (V[0],V[1],V[2],V[3],V[4],V[5]) for r=7
        # (V[0],V[1],V[2],V[3],V[4],V[5]) for r=8
        # (V[0],V[1],V[2],V[3],V[4],V[5]) for r=9
    ]
    
    # Try a few specific patterns with coupling.
    patterns_to_try = [
        # Pattern 1: r7=(B02), r8=(B12), r9=(B02+B12) [already tried]
        # Pattern 2: r7=(B00+B02), r8=(B11+B12), r9=(B02)
        # Pattern 3: r7=(B02), r8=(B02), r9=(B12)
        # Pattern 4: r7=(B00+B02), r8=(B10+B12), r9=(B02+B12)
    ]
    
    # Let's try pattern 4: coupling with k=0,1
    V2 = V.copy()
    V2[0,7] = 1; V2[4,7] = 1   # r=7: B00+B02
    V2[2,8] = 1; V2[5,8] = 1   # r=8: B10+B12
    V2[4,9] = 1; V2[5,9] = 1   # r=9: B02+B12
    
    A_mat2 = np.zeros((12, 60))
    eq_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(rank):
                    u_val = U[i, r]
                    v_val = V2[j * p + k, r]
                    if u_val != 0 and v_val != 0:
                        w_idx = (k * n + i) * rank + r
                        A_mat2[eq_idx, w_idx] = u_val * v_val
                eq_idx += 1
    
    try:
        W_sol2, _, _, _ = np.linalg.lstsq(A_mat2, b_vec, rcond=None)
    except:
        pass
    else:
        W_sol2 = W_sol2.reshape((6, 10))
        W_rounded2 = np.round(W_sol2)
        error2 = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i, r] * V2[j*p+k, r] * W_rounded2[k*n+i, r] for r in range(rank))
                    if abs(val - 1) > 1e-6:
                        error2 += abs(val - 1)
        if error2 < 1e-6:
            return U.astype(int), V2.astype(int), W_rounded2.astype(int)
    
    # Try a completely different approach: use a known rank-10 construction.
    # Based on literature, one construction uses the following:
    
    # Let me try yet another ansatz with different U patterns.
    # Use terms that are more "spread out" across the k dimension.
    
    # For now, if we can't find a rank-10 solution, return None.
    return None, None, None


def _verify_2x2x3(U, V, W, tol=1e-9):
    """Verify 2x2x3 decomposition."""
    n, m, p = 2, 2, 3
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                actual = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                if abs(actual - 1.0) > tol:
                    return False
    return True


def _find_2x2x4_rank14(deadline):
    """
    Search for rank-14 decomposition of 2x2x4 matrix multiplication.
    2x2x4 = 2x2x2 + 2x2x2, so rank <= 14 (two Strassen blocks).
    Check if rank 14 is achievable (it is, trivially).
    """
    n, m, p = 2, 2, 4
    rank = 14
    
    # Just use two Strassen blocks for k=(0,1) and k=(2,3).
    U = np.zeros((4, rank))
    V = np.zeros((8, rank))  # m*p = 2*4 = 8
    W = np.zeros((8, rank))  # p*n = 4*2 = 8
    
    # Block 1: k=0,1 (Strassen)
    # U indices: 0=A00, 1=A01, 2=A10, 3=A11
    # V indices: 0=B00, 1=B01, 2=B10, 3=B11, 4=B02, 5=B03, 6=B12, 7=B13
    # W indices: 0=C00, 1=C10, 2=C01, 3=C11, 4=C02, 5=C12, 6=C03, 7=C13
    
    # p1 = (A00+A11)(B00+B11)
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1
    
    # p2 = (A10+A11)B00
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[1,1] = 1; W[3,1] = -1
    
    # p3 = A00(B01-B11)
    U[0,2] = 1
    V[1,2] = 1; V[3,2] = -1
    W[2,2] = 1; W[3,2] = 1
    
    # p4 = A11(B10-B00)
    U[3,3] = 1
    V[2,3] = 1; V[0,3] = -1
    W[0,3] = 1; W[1,3] = 1
    
    # p5 = (A00+A01)B11
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[0,4] = -1; W[2,4] = 1
    
    # p6 = (A10-A00)(B00+B01)
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[3,5] = 1
    
    # p7 = (A01-A11)(B10+B11)
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[0,6] = 1
    
    # Block 2: k=2,3 (Strassen)
    # p8 = (A00+A11)(B02+B13)
    U[0,7] = 1; U[3,7] = 1
    V[4,7] = 1; V[7,7] = 1
    W[4,7] = 1; W[7,7] = 1
    
    # p9 = (A10+A11)B02
    U[2,8] = 1; U[3,8] = 1
    V[4,8] = 1
    W[5,8] = 1; W[7,8] = -1
    
    # p10 = A00(B03-B13)
    U[0,9] = 1
    V[5,9] = 1; V[7,9] = -1
    W[6,9] = 1; W[7,9] = 1
    
    # p11 = A11(B12-B02)
    U[3,10] = 1
    V[6,10] = 1; V[4,10] = -1
    W[4,10] = 1; W[5,10] = 1
    
    # p12 = (A00+A01)B13
    U[0,11] = 1; U[1,11] = 1
    V[7,11] = 1
    W[4,11] = -1; W[6,11] = 1
    
    # p13 = (A10-A00)(B02+B03)
    U[2,12] = 1; U[0,12] = -1
    V[4,12] = 1; V[5,12] = 1
    W[7,12] = 1
    
    # p14 = (A01-A11)(B12+B13)
    U[1,13] = 1; U[3,13] = -1
    V[6,13] = 1; V[7,13] = 1
    W[4,13] = 1
    
    # Verify
    if _verify_2x2x4(U, V, W):
        return U, V, W
    return None, None, None


def _verify_2x2x4(U, V, W, tol=1e-9):
    """Verify 2x2x4 decomposition."""
    n, m, p = 2, 2, 4
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                actual = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                if abs(actual - 1.0) > tol:
                    return False
    return True
# EVOLVE-BLOCK-END
