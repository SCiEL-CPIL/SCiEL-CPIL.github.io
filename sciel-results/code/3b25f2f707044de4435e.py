"""Matrix multiplication tensor decomposition for 2x3x8 with structural decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]
_START = time.time()
_DEADLINE = _START + 140.0

def _check_time():
    return time.time() > _DEADLINE - 15

def _validate_exact(U, V, W):
    """Check if factor matrices exactly decompose the matmul tensor using Fraction arithmetic."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = Fraction(0)
                for c in range(rank):
                    val += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                if val != 1:
                    return False
    return True

def _validate_numeric(U, V, W, tol=1e-9):
    """Quick numeric check."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.sum(U_arr[u_idx] * V_arr[v_idx] * W_arr[w_idx])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _strassen_2x2x2():
    """Return Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) where:
    U indexed by (i,j) for A, V indexed by (j,k) for B, W indexed by (k,i) for C.
    sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k.
    """
    # Strassen's 7 products for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # U: A[i,j] coefficients for each product
    # A00=U[0], A01=U[1], A10=U[2], A11=U[3]
    U = [
        [1, 0, 0, 0, 1, 0, 0],  # A00: M1, M5
        [0, 1, 0, 0, 1, 0, -1], # A01: M3, M5, M7
        [0, 0, 1, 0, 0, 1, 0],  # A10: M2, M6
        [1, 0, 1, 1, 0, 0, -1], # A11: M1, M2, M4, M7
    ]
    # V: B[j,k] coefficients
    # B00=V[0], B01=V[1], B10=V[2], B11=V[3]
    V = [
        [1, 0, 0, 0, 0, 1, 0],  # B00: M1, M2, M6
        [1, 1, 0, -1, 0, 1, 0], # B01: M1, M3, M4, M6
        [0, 0, 1, 1, 0, 0, 1],  # B10: M4, M5, M7
        [1, -1, 0, -1, 1, 0, 1], # B11: M1, M3, M4, M5, M7
    ]
    # W: C[k,i] coefficients
    # C00=W[0], C01=W[1], C10=W[2], C11=W[3]
    W = [
        [1, 0, 0, 0, -1, 0, 1],  # C00: M1, M4, M5, M7
        [0, 1, 0, 0, 1, 0, 0],   # C01: M3, M5
        [0, 0, 1, 0, 0, 1, 0],   # C10: M2, M4, M6
        [1, 0, 0, 1, -1, 0, 0],  # C11: M1, M2, M3, M6
    ]
    return U, V, W

def _strassen_2x3x2():
    """Decompose 2x3x2 matmul using Strassen on 2x2 blocks + 1x2 block.
    Split m=3 into m1=2 (Strassen) + m2=1 (direct).
    A is 2x3 = [A_2x2 | A_2x1], B is 3x2 = [B_2x2; B_1x2]
    C = A_2x2 * B_2x2 + A_2x1 * B_1x2
    Strassen for 2x2x2: rank 7
    Direct for 2x1x2: rank 2 (C[i,k] = A[i,2]*B[2,k])
    Total rank: 7 + 2 = 9
    """
    U_s, V_s, W_s = _strassen_2x2x2()
    
    # U: 2x3 matrix, 6 entries. Rows: (i,j) for i=0,1 j=0,1,2
    # V: 3x2 matrix, 6 entries. Rows: (j,k) for j=0,1,2 k=0,1
    # W: 2x2 matrix, 4 entries. Rows: (k,i) for k=0,1 i=0,1
    
    # Strassen part handles j=0,1 (the 2x2 block)
    # Direct part handles j=2 (the 2x1 * 1x2 part)
    
    rank = 9
    U = [[0]*rank for _ in range(6)]  # 2*3
    V = [[0]*rank for _ in range(6)]  # 3*2
    W = [[0]*rank for _ in range(4)]  # 2*2
    
    # Strassen part (columns 0-6): handles A[0:2, 0:2] * B[0:2, 0:2]
    # U rows for i=0,1; j=0,1 -> U[i*3+j]
    for i in range(2):
        for j in range(2):
            for c in range(7):
                U[i*3+j][c] = U_s[i*2+j][c]
    for j in range(2):
        for k in range(2):
            for c in range(7):
                V[j*2+k][c] = V_s[j*2+k][c]
    for k in range(2):
        for i in range(2):
            for c in range(7):
                W[k*2+i][c] = W_s[k*2+i][c]
    
    # Direct part (columns 7-8): handles A[i,2] * B[2,k]
    # For i=0,1 and k=0,1: C[i,k] += A[i,2]*B[2,k]
    # Two products: (i=0,k=0) and (i=1,k=1)? No, we need all 4 combinations.
    # Actually rank 2 for 2x1x2: C[i,k] = A[i,2]*B[2,k]
    # This is rank 4 in general... unless we use the structure.
    # Wait: 2x1x2 has rank 4 (schoolbook). Let me reconsider.
    # Actually for 2x1x2: C[i,k] = A[i,0]*B[0,k], this is rank 4.
    # So total would be 7+4=11.
    
    # Let me redo: use rank 4 for the 2x1x2 part
    rank = 11
    U = [[0]*rank for _ in range(6)]
    V = [[0]*rank for _ in range(6)]
    W = [[0]*rank for _ in range(4)]
    
    # Strassen part (columns 0-6)
    for i in range(2):
        for j in range(2):
            for c in range(7):
                U[i*3+j][c] = U_s[i*2+j][c]
    for j in range(2):
        for k in range(2):
            for c in range(7):
                V[j*2+k][c] = V_s[j*2+k][c]
    for k in range(2):
        for i in range(2):
            for c in range(7):
                W[k*2+i][c] = W_s[k*2+i][c]
    
    # Direct 2x1x2 part (columns 7-10)
    # A[i,2] * B[2,k] for i=0,1 and k=0,1
    # 4 products: (i,k) = (0,0), (0,1), (1,0), (1,1)
    for idx, (i, k) in enumerate([(0,0), (0,1), (1,0), (1,1)]):
        col = 7 + idx
        U[i*3+2][col] = 1  # A[i,2]
        V[2*2+k][col] = 1  # B[2,k]
        W[k*2+i][col] = 1  # C[k,i]
    
    return U, V, W

def _build_2x3x8_structural():
    """Build 2x3x8 decomposition by splitting p=8 into 4+4 and using 2x3x4 blocks.
    Or split into 2+2+2+2 and use 2x3x2 blocks.
    
    Strategy: Split p=8 into two p=4 blocks.
    C[:, 0:4] = A * B[:, 0:4]  (2x3 * 3x4)
    C[:, 4:8] = A * B[:, 4:8]  (2x3 * 3x4)
    
    For each 2x3x4, split m=3 into 2+1:
    2x3x4 = 2x2x4 + 2x1x4
    2x2x4: use Strassen on 2x2 with p=4 -> rank 7*2=14? No.
    
    Better: 2x2x4 can be done with rank 2*rank(2x2x2) = 2*7=14 by splitting p=4 into 2+2.
    Or: 2x2x4 has known rank. Let me use the direct approach.
    
    Actually, let's use: 2x3x8 = 2*(2x3x4) where each 2x3x4 uses:
    Split m=3 into 1+2:
    2x1x4: rank 4*2=8 (schoolbook for 2x1x4)
    2x2x4: rank 2*7=14 (Strassen on 2x2x2, doubled for p=4)
    Total per 2x3x4: 8+14=22, total for 2x3x8: 44.
    
    Even better: 2x2x4 can potentially be done with rank < 14.
    But let's start with 44 and try to improve.
    """
    n, m, p = 2, 3, 8
    
    # Split p=8 into 4+4
    # For each half, split m=3 into 1+2
    # 2x1x4: schoolbook rank 8
    # 2x2x4: Strassen-based rank 14 (7 per 2x2x2, two blocks)
    # Total: 2*(8+14) = 44
    
    rank = 44
    U = [[0]*rank for _ in range(n*m)]  # 6 x 44
    V = [[0]*rank for _ in range(m*p)]  # 24 x 44
    W = [[0]*rank for _ in range(p*n)]  # 16 x 44
    
    col = 0
    for half in range(2):  # two halves of p
        p_offset = half * 4
        # 2x1x4 part: A[i,0] * B[0, p_offset+k]
        for i in range(2):
            for k in range(4):
                U[i*m + 0][col] = 1
                V[0*p + p_offset + k][col] = 1
                W[(p_offset + k)*n + i][col] = 1
                col += 1
        # 2x2x4 part: Strassen on 2x2x2, doubled
        U_s, V_s, W_s = _strassen_2x2x2()
        for sub in range(2):  # two sub-blocks of p=2
            p_sub_offset = p_offset + sub * 2
            for c in range(7):
                for i in range(2):
                    for j in range(1, 3):  # j=1,2 (the 2x2 part)
                        U[i*m + j][col + c] = U_s[i*2 + (j-1)][c]
                for j in range(1, 3):
                    for k in range(2):
                        V[j*p + p_sub_offset + k][col + c] = V_s[(j-1)*2 + k][c]
                for k in range(2):
                    for i in range(2):
                        W[(p_sub_offset + k)*n + i][col + c] = W_s[k*2 + i][c]
            col += 7
    
    return U, V, W

def _als_step(U, V, W, n, m, p, target_r):
    """One ALS iteration, vectorized where possible."""
    U_new = U.copy()
    V_new = V.copy()
    W_new = W.copy()
    
    for i in range(n):
        for j in range(m):
            u_idx = i * m + j
            A_sys = np.zeros((p, target_r))
            for k in range(p):
                A_sys[k] = V[j*p+k] * W[k*n+i]
            try:
                sol, _, _, _ = np.linalg.lstsq(A_sys, np.ones(p), rcond=None)
                U_new[u_idx] = sol
            except:
                pass
    
    for j in range(m):
        for k in range(p):
            v_idx = j * p + k
            A_sys = np.zeros((n, target_r))
            for i in range(n):
                A_sys[i] = U_new[i*m+j] * W[k*n+i]
            try:
                sol, _, _, _ = np.linalg.lstsq(A_sys, np.ones(n), rcond=None)
                V_new[v_idx] = sol
            except:
                pass
    
    for k in range(p):
        for i in range(n):
            w_idx = k * n + i
            A_sys = np.zeros((m, target_r))
            for j in range(m):
                A_sys[j] = U_new[i*m+j] * V_new[j*p+k]
            try:
                sol, _, _, _ = np.linalg.lstsq(A_sys, np.ones(m), rcond=None)
                W_new[w_idx] = sol
            except:
                pass
    
    return U_new, V_new, W_new

def _compute_error(U, V, W, n, m, p):
    """Compute max error of the decomposition."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.sum(U_arr[u_idx] * V_arr[v_idx] * W_arr[w_idx])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def _recover_rationals(U, V, W, max_den=100):
    """Try to recover exact rationals from numerical solution."""
    def to_frac(x):
        if abs(x - round(x)) < 1e-10:
            return int(round(x))
        for den in range(2, max_den + 1):
            num = x * den
            if abs(num - round(num)) < 1e-10:
                return Fraction(int(round(num)), den)
        return Fraction(x).limit_denominator(max_den)
    
    U_fr = [[to_frac(float(x)) for x in row] for row in U]
    V_fr = [[to_frac(float(x)) for x in row] for row in V]
    W_fr = [[to_frac(float(x)) for x in row] for row in W]
    return U_fr, V_fr, W_fr

def _search_direct(n, m, p, target_r, max_trials=5, max_iters=200):
    """Search for rank target_r decomposition of n x m x p matmul tensor."""
    best_err = float('inf')
    best_sol = None
    
    for trial in range(max_trials):
        if _check_time():
            break
        
        rng = np.random.default_rng(42 + trial * 137 + target_r)
        
        U = rng.standard_normal((n*m, target_r)) * 0.5
        V = rng.standard_normal((m*p, target_r)) * 0.5
        W = rng.standard_normal((p*n, target_r)) * 0.5
        
        U /= np.linalg.norm(U, axis=0, keepdims=True) + 1e-10
        V /= np.linalg.norm(V, axis=0, keepdims=True) + 1e-10
        W /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-10
        
        prev_err = float('inf')
        for it in range(max_iters):
            if _check_time():
                break
            
            U, V, W = _als_step(U, V, W, n, m, p, target_r)
            
            err = _compute_error(U, V, W, n, m, p)
            
            if err < best_err:
                best_err = err
                best_sol = (U.copy(), V.copy(), W.copy())
            
            if err < 1e-10:
                break
            
            if abs(prev_err - err) < 1e-14:
                break
            prev_err = err
    
    if best_err < 1e-8:
        return best_sol, best_err
    return None, best_err

def construct():
    """Find rank-minimal exact tensor decomposition for 2x3x8 matrix multiplication."""
    n, m, p = DIMENSIONS
    
    # Build schoolbook solution as fallback
    schoolbook_rank = n * m * p  # 48
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = schoolbook_rank
    
    # Strategy 1: Structural decomposition (split m=3 into 1+2, use Strassen)
    # This gives rank 44
    if not _check_time():
        U_struct, V_struct, W_struct = _build_2x3x8_structural()
        # Validate
        if _validate_numeric(U_struct, V_struct, W_struct, tol=1e-6):
            # Try to recover exact rationals
            U_fr, V_fr, W_fr = _recover_rationals(U_struct, V_struct, W_struct)
            if _validate_exact(U_fr, V_fr, W_fr):
                best_rank = 44
                best_result = {"U": U_fr, "V": V_fr, "W": W_fr}
            else:
                # The structural solution should be exact with integers
                # Let's verify it's exact as-is
                U_int = [[int(x) for x in row] for row in U_struct]
                V_int = [[int(x) for x in row] for row in V_struct]
                W_int = [[int(x) for x in row] for row in W_struct]
                if _validate_exact(U_int, V_int, W_int):
                    best_rank = 44
                    best_result = {"U": U_int, "V": V_int, "W": W_int}
    
    # Strategy 2: Try to improve below 44 using ALS
    candidate_ranks = [42, 40, 38, 36, 34, 32, 30, 28, 26, 24]
    
    for target_r in candidate_ranks:
        if _check_time():
            break
        if target_r >= best_rank:
            continue
        
        sol, err = _search_direct(n, m, p, target_r, max_trials=3, max_iters=100)
        if sol is not None:
            U_sol, V_sol, W_sol = sol
            U_fr, V_fr, W_fr = _recover_rationals(U_sol, V_sol, W_sol)
            if _validate_exact(U_fr, V_fr, W_fr):
                if target_r < best_rank:
                    best_rank = target_r
                    best_result = {"U": U_fr, "V": V_fr, "W": W_fr}
                break
    
    return best_result
