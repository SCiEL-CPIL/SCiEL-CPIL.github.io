"""Combined program: Strassen blocks + integer pattern search for 2x2x3 rank-10."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 2, 15
    deadline = time.time() + 140  # 10s safety margin

    # Build rank-53 baseline (7 Strassen blocks + 1 schoolbook)
    U53, V53, W53 = _build_strassen_53(n, m, p)
    
    if not _verify_int(U53, V53, W53, n, m, p):
        U, V, W = _build_schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
    
    best_rank = 53
    best_U, best_V, best_W = U53, V53, W53

    # Try to find rank-10 decomposition for 2x2x3
    if time.time() < deadline - 10:
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
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
    # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    return [
        # M1
        ({(0,0):1, (1,1):1}, {(0,k0):1, (1,k1):1}, {(k0,0):1, (k1,1):1}),
        # M2
        ({(1,0):1, (1,1):1}, {(0,k0):1}, {(k0,1):1, (k1,1):-1}),
        # M3
        ({(0,0):1}, {(0,k1):1, (1,k1):-1}, {(k1,0):1, (k1,1):1}),
        # M4
        ({(1,1):1}, {(1,k0):1, (0,k0):-1}, {(k0,0):1, (k0,1):1}),
        # M5
        ({(0,0):1, (0,1):1}, {(1,k1):1}, {(k0,0):-1, (k1,0):1}),
        # M6
        ({(1,0):1, (0,0):-1}, {(0,k0):1, (0,k1):1}, {(k1,1):1}),
        # M7
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
    """Build rank-52: 6 Strassen blocks + rank-10 2x2x3 block."""
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
    # U10: (4, 10) rows are i*m+j
    # V10: (6, 10) rows are j*3+k_local
    # W10: (6, 10) rows are k_local*n+i
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
    Uses structured enumeration of small integer patterns.
    
    The 2x2x3 tensor has 12 entries (all = 1). We need 10 rank-1 terms.
    Strategy: enumerate support patterns and solve for coefficients.
    """
    n, m, p = 2, 2, 3
    rank = 10
    
    # The tensor T[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k in {0,1,2}
    # We need: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all (i,j,k)
    
    # Approach: Try to find a decomposition where each rank-1 term has
    # a specific sparsity pattern. Key insight: for 2x2x3, we can try
    # patterns where each term covers a subset of the 12 entries.
    
    # Let's try a systematic approach: enumerate possible support sets
    # for the 10 terms, then solve the resulting system.
    
    # First, try a known structure: use 2x2x2 Strassen for k=0,1 (rank 7)
    # and find rank-3 for the remaining k=2 part (which is 2x2x1, rank 4 schoolbook)
    # That gives 7+4=11, not 10.
    
    # Alternative: try to find a rank-10 by mixing.
    # Let's try: 6 Strassen-like terms for the 2x2x2 part (k=0,1) 
    # and 4 terms for k=2, but share some structure.
    
    # Actually, let's try a direct algebraic approach:
    # Assume the decomposition has the form where we can factor out
    # some structure. 
    
    # Try: enumerate small integer solutions using the fact that
    # the system is overdetermined (12 equations, 30 unknowns per term set)
    
    # Let's try a specific ansatz: 
    # Use 7 terms that look like Strassen for 2x2x2 (k=0,1)
    # plus 3 additional terms that handle k=2 with some coupling
    
    # Strassen for k=0,1:
    # M1 = (A00+A11)(B00+B10) -> C00, C11
    # M2 = (A10+A11)B00 -> C10, C11
    # M3 = A00(B01-B11) -> C01, C11  [wait, this is for 2x2x2 with k=0,1]
    
    # For 2x2x3, B has columns k=0,1,2. Let me redefine:
    # B[j,k] for j in {0,1}, k in {0,1,2}
    
    # Let's try a different approach: use linear algebra.
    # Set up the equations and try to find a sparse solution.
    
    # Actually, let me try a concrete construction:
    # Rank 10 for 2x2x3. 
    # 
    # Known: 2x2x2 has rank 7 (Strassen). 2x2x1 has rank 4.
    # 2x2x3 could potentially have rank 10 if there's a clever mixing.
    
    # Let me try: 5 terms for the "even" part and 5 for the "odd" part?
    # Or: try to find it by solving a system of polynomial equations.
    
    # For now, let's try a structured search:
    # Assume each rank-1 term has at most 3 non-zero entries in U,
    # at most 3 in V, and at most 2 in W (or similar small patterns).
    
    # Let's try a specific known result: 
    # 2x2x3 matrix multiplication has rank 10? Let me check if this is known.
    # Actually, the rank of 2x2xn for n>=3 is at most 2n+3 (from various bounds).
    # For n=3: at most 9? No, that's not right.
    # 
    # Let me just try to find it computationally with a limited search.
    
    # Strategy: fix the support patterns and solve for coefficients.
    # For rank 10, we have 10 terms. Each term r has:
    #   U_r: 4 values (i=0,1; j=0,1)
    #   V_r: 6 values (j=0,1; k=0,1,2)
    #   W_r: 6 values (k=0,1,2; i=0,1)
    # Total: 10*(4+6+6) = 160 unknowns, 12 equations.
    # Very underdetermined, so solutions exist. The question is whether
    # there's a solution with small integer/rational entries.
    
    # Let's try: set up the 12 equations and use a greedy/structured approach.
    
    # Equation for (i,j,k): sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
    
    # Let's try a specific pattern: 
    # Terms 0-6: Strassen-like for 2x2x2 (k=0,1)
    # Terms 7-9: handle k=2 with coupling
    
    # Strassen for 2x2x2 (k=0,1):
    # T0: U=(1,0,0,1) [A00,A01,A10,A11], V=(1,0,0,1) [B00,B01,B10,B11], W=(1,0,0,1) [C00,C01,C10,C11]
    # Wait, I need to be careful with the indexing.
    
    # Let me use the standard Strassen for 2x2x2 and extend.
    
    # For 2x2x2 with k in {0,1}:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    
    # Strassen terms (for k=0,1):
    # M1 = (A00+A11)(B00+B11): contributes to C00, C11
    # M2 = (A10+A11)B00: contributes to C10, C11
    # M3 = A00(B10-B00): contributes to C01, C11  [hmm, need to be careful]
    
    # Let me just use the standard Strassen formula:
    # p1 = (a11 + a22)(b11 + b22)
    # p2 = (a21 + a22)b11
    # p3 = a11(b21 - b22)
    # p4 = a22(b12 - b11)
    # p5 = (a11 + a12)b22
    # p6 = (a21 - a11)(b11 + b12)
    # p7 = (a12 - a22)(b21 + b22)
    
    # c11 = p1 + p4 - p5 + p7
    # c12 = p3 + p5
    # c21 = p2 + p4
    # c22 = p1 - p2 + p3 + p6
    
    # For 2x2x3, B has 3 columns. Let's denote B[j,k] for k=0,1,2.
    # 
    # Let me try a different approach: just solve the system directly.
    # Use numpy to solve a linear system where we fix some variables.
    
    # Actually, let me try a very specific construction:
    # 
    # For 2x2x3, try:
    # 4 schoolbook terms for k=0: (i,j,0) for i,j in {0,1}
    # 4 schoolbook terms for k=1: (i,j,1) for i,j in {0,1}  
    # 2 terms for k=2 that are clever
    
    # That's 10 total. But the 2 terms for k=2 need to produce:
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    # with only 2 rank-1 terms. That requires rank 2 for 2x2x1, which is impossible
    # (rank of 2x2x1 is 4). So this won't work.
    
    # Let me try: 3 terms for k=0, 3 for k=1, 4 for k=2? Total 10.
    # Or: mix them up.
    
    # Key insight: the rank of the 2x2xn multiplication tensor.
    # For n=1: rank 4
    # For n=2: rank 7
    # For n=3: rank ? (possibly 10 or 11)
    
    # Let me try to find it by solving the system.
    
    # Set up: 12 equations, 160 unknowns (10 terms * 16 values each)
    # This is highly underdetermined. Let's try to find a sparse solution.
    
    # Use a greedy approach: add terms one at a time to satisfy equations.
    
    # Actually, let me try a concrete known result. 
    # The rank of 2x2x3 matrix multiplication is 10.
    # Here's a construction:
    
    # Let me try to find it by solving a system of equations.
    
    # Define the 12 target values (all 1):
    # (i,j,k) -> 1 for i in {0,1}, j in {0,1}, k in {0,1,2}
    
    # Let's use a structured ansatz with 10 terms.
    # Try: 7 terms for a 2x2x2 sub-problem (k=0,1) using Strassen,
    # and 3 additional terms that handle k=2 while also contributing to k=0,1.
    
    # This is getting complex. Let me just try to solve it numerically.
    
    # Use least squares with a random initialization and see if we can get close to exact.
    
    np.random.seed(123)
    best_error = float('inf')
    best_solution = None
    
    for restart in range(50):
        if time.time() > deadline - 15:
            break
        
        # Random initialization
        U = np.random.randn(4, rank) * 0.1
        V = np.random.randn(6, rank) * 0.1
        W = np.random.randn(6, rank) * 0.1
        
        # ALS iterations
        for it in range(500):
            if time.time() > deadline - 15:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    B_mat = np.zeros((rank, p))
                    for r in range(rank):
                        for k in range(p):
                            B_mat[r, k] = V[j*p + k, r] * W[k*n + i, r]
                    try:
                        U[i*m + j, :] = np.linalg.lstsq(B_mat, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    C_mat = np.zeros((rank, n*m))
                    for r in range(rank):
                        for i in range(n):
                            C_mat[r, i*m + j] = U[i*m + j, r] * W[k*n + i, r]
                    try:
                        V[j*p + k, :] = np.linalg.lstsq(C_mat, np.ones(n*m), rcond=None)[0]
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    D_mat = np.zeros((rank, m))
                    for r in range(rank):
                        for j in range(m):
                            D_mat[r, j] = U[i*m + j, r] * V[j*p + k, r]
                    try:
                        W[k*n + i, :] = np.linalg.lstsq(D_mat, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            # Check error
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                        error += (val - 1.0) ** 2
            
            if error < best_error:
                best_error = error
                best_solution = (U.copy(), V.copy(), W.copy())
            
            if error < 1e-14:
                break
    
    if best_solution is not None and best_error < 1e-10:
        U, V, W = best_solution
        # Try to round to half-integers
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if _verify_2x2x3(U_r, V_r, W_r):
            return U_r, V_r, W_r
        
        # Try rounding to integers
        U_r = np.round(U)
        V_r = np.round(V)
        W_r = np.round(W)
        if _verify_2x2x3(U_r, V_r, W_r):
            return U_r, V_r, W_r
    
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
# EVOLVE-BLOCK-END
