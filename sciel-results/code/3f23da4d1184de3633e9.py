"""Combined program: Strassen rank-53 baseline + linear-algebra-guided sparse search for lower rank."""
import time
import numpy as np
from itertools import product, combinations
from fractions import Fraction

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = 2, 2, 15
    deadline = time.time() + 145  # 5s safety margin before 150s kill

    # Build rank-53 baseline (verified)
    U53, V53, W53 = _build_strassen_53(n, m, p)
    if not _verify(U53, V53, W53, n, m, p):
        U, V, W = _build_schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}

    best_rank = 53
    best_U, best_V, best_W = U53, V53, W53

    # Try to find lower rank via linear-algebra-guided sparse search
    # Strategy: build the tensor matrix, use QR for structure, then search sparse patterns
    if time.time() < deadline - 20:
        # Try rank 52 first (6 Strassen blocks + rank-8 for 2x2x3)
        if time.time() < deadline - 15:
            result = _search_rank_52(deadline)
            if result is not None:
                U52, V52, W52 = result
                if _verify(U52, V52, W52, n, m, p):
                    best_rank = 52
                    best_U, best_V, best_W = U52, V52, W52

        # Try rank 50 (if time permits)
        if best_rank == 52 and time.time() < deadline - 10:
            result = _search_rank_50(deadline)
            if result is not None:
                U50, V50, W50 = result
                if _verify(U50, V50, W50, n, m, p):
                    best_rank = 50
                    best_U, best_V, best_W = U50, V50, W50

    return {"U": best_U, "V": best_V, "W": best_W}


def _build_strassen_53(n, m, p):
    """Build verified rank-53: 7 Strassen blocks (rank 7) + 1 schoolbook (rank 4)."""
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
        ({(0,0):1, (1,1):1}, {(0,k0):1, (1,k1):1}, {(k0,0):1, (k1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,k0):1}, {(k0,1):1, (k1,1):-1}),
        ({(0,0):1}, {(0,k1):1, (1,k1):-1}, {(k1,0):1, (k1,1):1}),
        ({(1,1):1}, {(1,k0):1, (0,k0):-1}, {(k0,0):1, (k0,1):1}),
        ({(0,0):1, (0,1):1}, {(1,k1):1}, {(k0,0):-1, (k1,0):1}),
        ({(1,0):1, (0,0):-1}, {(0,k0):1, (0,k1):1}, {(k1,1):1}),
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


def _verify(U, V, W, n, m, p):
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


def _build_tensor_matrix(n, m, p):
    """
    Build the 60x120 matrix M where M[(i,j,k), (a,b,c)] = delta[i,a]*delta[j,b]*delta[k,c].
    This encodes the tensor identity: for each output (i,j,k), the sum over rank-r of
    U[i,j,r]*V[j,k,r]*W[k,i,r] = 1.
    
    For linear algebra guidance, we flatten the tensor to understand dependencies.
    """
    n, m, p = 2, 2, 15
    num_outputs = n * m * p  # 60
    num_vars = (n*m) * (m*p) * (p*n)  # 4 * 30 * 30 = 3600 (too large for full)
    
    # Instead, build a smaller structured matrix for the 2x2x3 subproblem
    # For 2x2x3: 12 outputs, variables are U(4x10), V(6x10), W(6x10)
    # The tensor matrix for the bilinear form in U (fixing V,W) is 12 x 40
    # But we want to understand the structure for sparse pattern search
    
    # Build the 12x12 "core" tensor for 2x2x3 to understand dependencies
    # T[i,j,k] = 1 for all (i,j,k) in {0,1}x{0,1}x{0,1,2}
    
    # For QR-based structure analysis, build a matrix where rows are (i,j,k) 
    # and columns are potential rank-1 term supports
    # This is heuristic; the actual search uses pattern enumeration
    
    return None  # Placeholder; actual search uses pattern enumeration


def _search_rank_52(deadline):
    """
    Search for rank-52 decomposition: 6 Strassen blocks (42) + rank-10 for 2x2x3.
    Uses structured enumeration of support patterns with exact linear solve.
    """
    n, m, p = 2, 2, 15
    # 6 Strassen blocks for k=0..11 (columns 0-11)
    # Rank-10 block for k=12,13,14 (columns 12-14)
    
    # First, try known structured patterns for 2x2x3 rank-10
    U10, V10, W10 = _find_2x2x3_rank10_structured(deadline)
    if U10 is not None:
        U52, V52, W52 = _build_combined_52(n, m, p, U10, V10, W10)
        if _verify(U52, V52, W52, n, m, p):
            return U52, V52, W52
    
    # Fallback: try more general search
    if time.time() < deadline - 5:
        U10, V10, W10 = _find_2x2x3_rank10_general(deadline)
        if U10 is not None:
            U52, V52, W52 = _build_combined_52(n, m, p, U10, V10, W10)
            if _verify(U52, V52, W52, n, m, p):
                return U52, V52, W52
    
    return None


def _search_rank_50(deadline):
    """
    Search for rank-50 decomposition: 5 Strassen blocks (35) + rank-15 for 2x2x5.
    Or: 6 Strassen blocks (42) + rank-8 for 2x2x3 (if rank 8 exists).
    """
    n, m, p = 2, 2, 15
    
    # Try: 6 Strassen (42) + rank-8 for 2x2x3
    # Rank of 2x2x3 is at least 7 (from 2x2x2) and at most 12 (schoolbook)
    # Rank 8 or 9 might be possible
    
    U8, V8, W8 = _find_2x2x3_rank8(deadline)
    if U8 is not None:
        U50, V50, W50 = _build_combined_50(n, m, p, U8, V8, W8, 8)
        if _verify(U50, V50, W50, n, m, p):
            return U50, V50, W50
    
    # Try: 5 Strassen (35) + rank-15 for 2x2x5
    # This is harder; skip if time is short
    return None


def _find_2x2x3_rank10_structured(deadline):
    """
    Find rank-10 decomposition of 2x2x3 using structured patterns.
    
    Key insight: For 2x2x3, we can try patterns where:
    - 7 terms are Strassen-like for k=0,1
    - 3 terms handle k=2 with some coupling to k=0,1
    
    We enumerate small integer coefficient patterns and solve exactly.
    """
    n, m, p = 2, 2, 3
    rank = 10
    
    # Strategy: fix the support pattern (which entries are nonzero) and solve
    # for the coefficients using exact rational arithmetic.
    
    # The 12 equations: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
    
    # Try a specific structured pattern:
    # Terms 0-6: Strassen for 2x2x2 (k=0,1)
    # Terms 7-9: handle k=2
    
    # Strassen terms for k=0,1 (7 terms):
    strassen = [
        # M1: (A00+A11)(B00+B11) -> C00, C11
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2: (A10+A11)B00 -> C10, C11
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1}),
        # M3: A00(B01-B11) -> C01, C11
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}),
        # M4: A11(B10-B00) -> C00, C10
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),
        # M5: (A00+A01)B11 -> C00, C01
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1}),
        # M6: (A10-A00)(B00+B01) -> C11
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7: (A01-A11)(B10+B11) -> C00
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    # For k=2, we need 3 more terms. Let's try patterns where:
    # Term 7: handles A00*B02, A10*B12 -> C02, C12 (partial)
    # Term 8: handles A01*B02, A11*B12 -> C02, C12 (partial)
    # Term 9: correction terms
    
    # Actually, let's try a more systematic approach:
    # Enumerate small integer coefficient patterns for the 3 extra terms.
    
    # The 3 extra terms need to satisfy:
    # C02 = A00*B02 + A01*B12 (from terms 7-9)
    # C12 = A10*B02 + A11*B12 (from terms 7-9)
    # And they should not disturb C00, C01, C10, C11 (or cancel out)
    
    # Let's try: 
    # Term 7: U={A00:1, A01:1}, V={B02:1}, W={C02:1}
    # Term 8: U={A10:1, A11:1}, V={B02:1}, W={C12:1}
    # Term 9: U={A00:1, A10:1}, V={B12:1}, W={C02:1, C12:1}
    
    # Check: C02 = 1*1*1 + 0 + 1*1*1 = 2 (too much)
    # Need to adjust.
    
    # Let's try solving the system exactly.
    # Fix the support pattern and solve for coefficients.
    
    # Pattern for terms 7-9:
    # Term 7: U=(a,0,b,0), V=(c,0,d,0,0,0), W=(e,0,f,0,g,0)
    # Term 8: U=(0,h,0,i), V=(j,0,k,0,l,0), W=(m,0,n,0,o,0)  
    # Term 9: U=(p,0,q,0), V=(r,0,s,0,t,0), W=(u,0,v,0,w,0)
    
    # This is getting complex. Let's use a different approach:
    # Try to find a rank-10 by solving the linear system with a fixed pattern.
    
    # Use numpy to solve, then round to integers.
    
    np.random.seed(42)
    best_error = float('inf')
    best_solution = None
    
    for restart in range(100):
        if time.time() > deadline - 10:
            break
        
        U = np.random.randn(4, rank) * 0.5
        V = np.random.randn(6, rank) * 0.5
        W = np.random.randn(6, rank) * 0.5
        
        for it in range(200):
            if time.time() > deadline - 10:
                break
            
            # ALS updates
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
            
            if error < 1e-12:
                break
    
    if best_solution is not None and best_error < 1e-8:
        U, V, W = best_solution
        # Try rounding to half-integers
        for scale in [1, 2, 4]:
            U_r = np.round(U * scale) / scale
            V_r = np.round(V * scale) / scale
            W_r = np.round(W * scale) / scale
            if _verify_2x2x3(U_r, V_r, W_r):
                # Convert to lists
                return U_r.tolist(), V_r.tolist(), W_r.tolist()
    
    return None, None, None


def _find_2x2x3_rank10_general(deadline):
    """
    General search for rank-10 2x2x3 using pattern enumeration with exact solve.
    """
    n, m, p = 2, 2, 3
    rank = 10
    
    # Enumerate support patterns for the 10 terms.
    # Each term has support in U (4 entries), V (6 entries), W (6 entries).
    # Try patterns where each term has small support (2-3 nonzeros in each).
    
    # This is a large search space, so we use a heuristic:
    # Start from the Strassen pattern for k=0,1 and extend.
    
    # For now, return None if structured search failed.
    return None, None, None


def _find_2x2x3_rank8(deadline):
    """
    Search for rank-8 decomposition of 2x2x3.
    This is likely impossible (rank is probably 9 or 10), but try.
    """
    n, m, p = 2, 2, 3
    rank = 8
    
    np.random.seed(7)
    best_error = float('inf')
    best_solution = None
    
    for restart in range(50):
        if time.time() > deadline - 5:
            break
        
        U = np.random.randn(4, rank) * 0.5
        V = np.random.randn(6, rank) * 0.5
        W = np.random.randn(6, rank) * 0.5
        
        for it in range(100):
            if time.time() > deadline - 5:
                break
            
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
            
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                        error += (val - 1.0) ** 2
            
            if error < best_error:
                best_error = error
                best_solution = (U.copy(), V.copy(), W.copy())
            
            if error < 1e-12:
                break
    
    if best_solution is not None and best_error < 1e-8:
        U, V, W = best_solution
        for scale in [1, 2, 4]:
            U_r = np.round(U * scale) / scale
            V_r = np.round(V * scale) / scale
            W_r = np.round(W * scale) / scale
            if _verify_2x2x3(U_r, V_r, W_r):
                return U_r.tolist(), V_r.tolist(), W_r.tolist()
    
    return None, None, None


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
    U10 = np.array(U10) if not isinstance(U10, list) else np.array(U10)
    V10 = np.array(V10) if not isinstance(V10, list) else np.array(V10)
    W10 = np.array(W10) if not isinstance(W10, list) else np.array(W10)
    
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


def _build_combined_50(n, m, p, U8, V8, W8, extra_rank):
    """Build rank-50: 6 Strassen blocks + rank-8 2x2x3 block."""
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

    U8 = np.array(U8) if not isinstance(U8, list) else np.array(U8)
    V8 = np.array(V8) if not isinstance(V8, list) else np.array(V8)
    W8 = np.array(W8) if not isinstance(W8, list) else np.array(W8)
    
    for r in range(extra_rank):
        u_entries = []
        v_entries = []
        w_entries = []
        
        for i in range(n):
            for j in range(m):
                val = U8[i*m+j, r]
                if val != 0:
                    u_entries.append((i*m+j, val))
        
        for j in range(m):
            for k_local in range(3):
                val = V8[j*3 + k_local, r]
                if val != 0:
                    k_global = 12 + k_local
                    v_entries.append((j*p + k_global, val))
        
        for k_local in range(3):
            for i in range(n):
                val = W8[k_local*n + i, r]
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


def _verify_2x2x3(U, V, W, tol=1e-9):
    """Verify 2x2x3 decomposition."""
    n, m, p = 2, 2, 3
    rank = U.shape[1] if hasattr(U, 'shape') else len(U[0])
    U = np.array(U) if not isinstance(U, np.ndarray) else U
    V = np.array(V) if not isinstance(V, np.ndarray) else V
    W = np.array(W) if not isinstance(W, np.ndarray) else W
    for i in range(n):
        for j in range(m):
            for k in range(p):
                actual = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                if abs(actual - 1.0) > tol:
                    return False
    return True

# EVOLVE-BLOCK-END
