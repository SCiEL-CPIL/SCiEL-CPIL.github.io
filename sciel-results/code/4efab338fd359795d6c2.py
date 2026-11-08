import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    start_time = time.time()
    
    # The matrix multiplication tensor T has entries T[i*m+j, j*p+k, k*n+i] = 1
    # for i in [0,n), j in [0,m), k in [0,p), and 0 otherwise.
    #
    # Structural approach: Direct construction exploiting the rectangular shape.
    # The 2x4x6 tensor can be decomposed by viewing it as two 1x4x6 tensors
    # (one for i=0, one for i=1) with shared structure.
    #
    # For i=0: T[j, j*6+k, k] = 1  (j in 0..3, k in 0..5)
    # For i=1: T[4+j, j*6+k, 2+k] = 1  (j in 0..3, k in 0..5)
    #
    # We'll try a rank-12 construction using the following idea:
    # Group the 24 terms (j,k pairs) into groups and use linear combinations.
    #
    # Alternative: Use the fact that M_{2,4,6} embeds into M_{3,3,3} (rank <= 23)
    # but we need a better bound.
    #
    # Let's try a direct rank-13 construction by carefully designing rank-1 terms.
    #
    # Actually, let's use a different structural decomposition:
    # 2x4x6 = (1x2x3) x (2x2x2) gives 2*7=14 (current)
    # But we can try: view 4 as 2+2 and 6 as 3+3, giving a block structure.
    #
    # Block decomposition:
    # A is 2x4 = [A_left | A_right] where each is 2x2
    # B is 4x6 = [B_top; B_bottom] where each is 2x6
    # C = A*B = [A_left*B_top + A_right*B_bottom] (2x6)
    #
    # This gives C = A_left*B_top + A_right*B_bottom
    # Each 2x2*2x6 multiplication has rank <= 7*2 = 14 (using Strassen x 1x2x3)
    # But the sum doesn't directly help since we need the sum.
    #
    # Better structural idea: Use the "pencil" structure.
    # The tensor is a sum over j of terms involving A[:,j] * B[j,:]
    # This is a rank-m decomposition if we could do each column/row separately,
    # but that's rank 4*6=24.
    #
    # Let me try a computational approach: search for a rank-13 or rank-12
    # decomposition using the specific structure.
    
    # First, build the expected tensor
    T_expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_expected[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try rank 12 first, then 13, then fall back to 14
    for target_rank in [12, 13, 14]:
        if time.time() - start_time > 120:
            break
        result = _try_rank(n, m, p, target_rank, T_expected, start_time)
        if result is not None:
            return result
    
    # Fallback: Kronecker product (rank 14)
    result = _kronecker_solution(n, m, p, T_expected)
    if result is not None:
        return result
    
    return _schoolbook_solution(n, m, p)

def _try_rank(n, m, p, target_rank, T_expected, start_time):
    """Try to find a decomposition of given rank using structured search."""
    # Use a structured approach: initialize with Kronecker-like structure
    # and try to merge/eliminate terms.
    
    # For now, use the Kronecker product as base and try to reduce
    U_base, V_base, W_base = _kronecker_matrices(n, m, p)
    
    if U_base is None:
        return None
    
    # Try to find a lower-rank subset or linear combination
    # This is a structural change: instead of random search, use
    # the algebraic structure to find dependencies.
    
    # Check if rank-12 is achievable by finding 2 linearly dependent
    # rank-1 terms that can be merged.
    # Build the full tensor from base
    T_from_base = np.einsum('ar,br,cr->abc', U_base, V_base, W_base)
    
    if not np.allclose(T_from_base, T_expected, atol=1e-10):
        return None
    
    # Try to find a rank-reduction by checking if any two columns
    # in the decomposition are linearly dependent in a useful way.
    # This is complex; for now, return the base if it works at target_rank >= 14
    if target_rank >= 14:
        rank = 14
        U_out = [[float(U_base[a, r]) for r in range(rank)] for a in range(n * m)]
        V_out = [[float(V_base[b, r]) for r in range(rank)] for b in range(m * p)]
        W_out = [[float(W_base[c, r]) for r in range(rank)] for c in range(p * n)]
        return {"U": U_out, "V": V_out, "W": W_out}
    
    return None

def _kronecker_matrices(n, m, p):
    """Build the Kronecker product decomposition matrices (rank 14)."""
    # Strassen 2x2x2 (rank 7)
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    W_s[0, 0] = 1; W_s[3, 0] = 1
    
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[0, 1] = 1
    W_s[2, 1] = 1; W_s[3, 1] = 1
    
    U_s[0, 2] = 1
    V_s[2, 2] = 1; V_s[3, 2] = -1
    W_s[1, 2] = 1; W_s[3, 2] = 1
    
    U_s[3, 3] = 1
    V_s[1, 3] = 1; V_s[0, 3] = -1
    W_s[0, 3] = 1; W_s[2, 3] = 1
    
    U_s[0, 4] = 1; U_s[1, 4] = 1
    V_s[3, 4] = 1
    W_s[0, 4] = -1; W_s[1, 4] = 1
    
    U_s[2, 5] = 1; U_s[0, 5] = -1
    V_s[0, 5] = 1; V_s[1, 5] = 1
    W_s[3, 5] = 1
    
    U_s[1, 6] = 1; U_s[3, 6] = -1
    V_s[2, 6] = 1; V_s[3, 6] = 1
    W_s[0, 6] = 1
    
    # 1x2x3 decomposition (rank 2)
    U_123 = np.zeros((2, 2))
    V_123 = np.zeros((6, 2))
    W_123 = np.zeros((3, 2))
    
    U_123[0, 0] = 1
    V_123[0, 0] = 1
    W_123[0, 0] = 1
    
    U_123[1, 1] = 1
    V_123[4, 1] = 1
    W_123[1, 1] = 1
    
    # Combine
    total_rank = 14
    U_full = np.zeros((n*m, total_rank))
    V_full = np.zeros((m*p, total_rank))
    W_full = np.zeros((p*n, total_rank))
    
    for r1 in range(7):
        for r2 in range(2):
            r = r1 * 2 + r2
            for i1 in range(2):
                for j1 in range(2):
                    us_val = U_s[i1*2 + j1, r1]
                    if us_val == 0:
                        continue
                    for i2 in range(1):
                        for j2 in range(2):
                            u123_val = U_123[i2*2 + j2, r2]
                            if u123_val == 0:
                                continue
                            i = i1 + i2
                            j = 2*j1 + j2
                            a = i * 4 + j
                            U_full[a, r] += us_val * u123_val
            for j1 in range(2):
                for k1 in range(2):
                    vs_val = V_s[j1*2 + k1, r1]
                    if vs_val == 0:
                        continue
                    for j2 in range(2):
                        for k2 in range(3):
                            v123_val = V_123[j2*3 + k2, r2]
                            if v123_val == 0:
                                continue
                            j = 2*j1 + j2
                            k = 3*k1 + k2
                            b = j * 6 + k
                            V_full[b, r] += vs_val * v123_val
            for k1 in range(2):
                for i1 in range(2):
                    ws_val = W_s[k1*2 + i1, r1]
                    if ws_val == 0:
                        continue
                    for k2 in range(3):
                        for i2 in range(1):
                            w123_val = W_123[k2 + i2, r2]
                            if w123_val == 0:
                                continue
                            k = 3*k1 + k2
                            i = i1 + i2
                            c = k * 2 + i
                            W_full[c, r] += ws_val * w123_val
    
    return U_full, V_full, W_full

def _kronecker_solution(n, m, p, T_expected):
    """Return the Kronecker product solution if valid."""
    U_full, V_full, W_full = _kronecker_matrices(n, m, p)
    if U_full is None:
        return None
    
    T_comp = np.einsum('ar,br,cr->abc', U_full, V_full, W_full)
    if not np.allclose(T_comp, T_expected, atol=1e-10):
        return None
    
    rank = 14
    U_out = [[float(U_full[a, r]) for r in range(rank)] for a in range(n * m)]
    V_out = [[float(V_full[b, r]) for r in range(rank)] for b in range(m * p)]
    W_out = [[float(W_full[c, r]) for r in range(rank)] for c in range(p * n)]
    return {"U": U_out, "V": V_out, "W": W_out}

def _schoolbook_solution(n, m, p):
    """Fallback schoolbook solution with rank = n*m*p"""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1.0
                V[j*p + k][col] = 1.0
                W[k*n + i][col] = 1.0
                col += 1
    
    return {"U": U, "V": V, "W": W}
