"""Search for a low-rank exact tensor decomposition of (2,2,8) matrix multiplication."""
import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 2, 8]
N, M, P = DIMENSIONS
NM = N * M   # 4
MP = M * P   # 16
PN = P * N   # 16
SCHOOBOOK_RANK = N * M * P  # 32


def verify(U, V, W, n=N, m=M, p=P):
    """Verify that U, V, W form an exact decomposition. Returns True/False."""
    nm, mp, pn = n*m, m*p, p*n
    rank = len(U[0]) if U else 0
    if len(U) != nm or len(V) != mp or len(W) != pn:
        return False
    for row in U:
        if len(row) != rank:
            return False
    for row in V:
        if len(row) != rank:
            return False
    for row in W:
        if len(row) != rank:
            return False
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid.add((i*m+j, j*p+k, k*n+i))
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                s = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                expected = 1 if (a, b, c) in valid else 0
                if s != expected:
                    return False
    return True


def verify_fast(U, V, W, n=N, m=M, p=P):
    """Faster verification using the structure."""
    nm, mp, pn = n*m, m*p, p*n
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                if s != 1:
                    return False
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid.add((i*m+j, j*p+k, k*n+i))
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                if (a, b, c) not in valid:
                    s = 0
                    for r in range(rank):
                        s += U[a][r] * V[b][r] * W[c][r]
                    if s != 0:
                        return False
    return True


def schoolbook():
    """Return the schoolbook rank-32 decomposition."""
    n, m, p = N, M, P
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
    return U, V, W


def strassen_base():
    """
    Build a rank-28 decomposition using Strassen on the 2x2 core,
    extended to the 2x8 case by tiling the p-dimension.
    
    The 2x2*2x2 Strassen decomposition has rank 7. For 2x2*2x8, we can
    view B as four 2x2 blocks. The key insight from the inspiration:
    share the linear combinations of A across blocks of B.
    
    Strassen's 7 products for C = A*B where A is 2x2, B is 2x2:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B01-B11)
    M4 = A11(B10-B00)
    M5 = (A00+A10)B11
    M6 = (A11-A00)(B00+B10)
    M7 = (A01-A11)(B01+B11)
    
    C00 = M1+M4-M5+M7
    C01 = M3+M5
    C10 = M2+M4
    C11 = M1-M2+M3+M6
    
    For 2x2 * 2x8: B is 2x8, so B has 4 blocks of 2x2 along k.
    A is 2x2. For each block of B (2x2), we need 7 Strassen products.
    But the A-side combinations are the same for all blocks!
    
    So:
    - The U factor encodes the 7 A-side linear combinations (same for all 4 k-blocks)
    - The V factor encodes the 7 B-side linear combinations per block
    - The W factor encodes the output reconstruction
    
    Actually, let's think about this more carefully.
    
    For each k-block (k0 in {0,1,2,3}), B has entries B[0, 2*k0], B[0, 2*k0+1], B[1, 2*k0], B[1, 2*k0+1].
    
    The 7 Strassen products for block k0:
    M1_k0 = (A00+A11)(B0_{2k0}+B1_{2k0+1})  -- wait, B00 of the block is B[0, 2k0], B01 is B[0, 2k0+1], B10 is B[1, 2k0], B11 is B[1, 2k0+1]
    
    M1 = (A00+A11)(B00+B11) where B00=B[0,2k0], B11=B[1,2k0+1]
    M2 = (A10+A11)B00 = (A10+A11)B[0,2k0]
    M3 = A00(B01-B11) = A00(B[0,2k0+1]-B[1,2k0+1])
    M4 = A11(B10-B00) = A11(B[1,2k0]-B[0,2k0])
    M5 = (A00+A10)B11 = (A00+A10)B[1,2k0+1]
    M6 = (A11-A00)(B00+B10) = (A11-A00)(B[0,2k0]+B[1,2k0])
    M7 = (A01-A11)(B01+B11) = (A01-A11)(B[0,2k0+1]+B[1,2k0+1])
    
    Output:
    C[0, 2k0] = M1+M4-M5+M7
    C[0, 2k0+1] = M3+M5
    C[1, 2k0] = M2+M4
    C[1, 2k0+1] = M1-M2+M3+M6
    
    Now in the tensor decomposition framework:
    U has shape (4, rank) where row index is i*m+j:
    row 0: i=0,j=0 -> A00
    row 1: i=0,j=1 -> A01
    row 2: i=1,j=0 -> A10
    row 3: i=1,j=1 -> A11
    
    V has shape (16, rank) where row index is j*p+k:
    rows 0-7: j=0, k=0..7 -> B[0,0]..B[0,7]
    rows 8-15: j=1, k=0..7 -> B[1,0]..B[1,7]
    
    W has shape (16, rank) where row index is k*n+i:
    rows 0-1: k=0, i=0,1 -> C[0,0], C[1,0]
    rows 2-3: k=1, i=0,1 -> C[0,1], C[1,1]
    ...
    rows 2k0, 2k0+1: k=2k0, i=0,1 -> C[0,2k0], C[1,2k0]
    rows 2k0+2, 2k0+3: k=2k0+1, i=0,1 -> C[0,2k0+1], C[1,2k0+1]
    
    For each k-block k0, we have 7 rank terms. Total rank = 4*7 = 28.
    
    For term (k0, s) where s in 0..6 (the 7 Strassen products):
    The rank index r = k0*7 + s
    
    U[0, r] (A00): contributes to M1, M3, M5, M6
       M1: coeff 1, M3: coeff 1, M5: coeff 1, M6: coeff -1
    U[1, r] (A01): contributes to M7
       M7: coeff 1
    U[2, r] (A10): contributes to M2, M5
       M2: coeff 1, M5: coeff 1
    U[3, r] (A11): contributes to M1, M2, M4, M6, M7
       M1: coeff 1, M2: coeff 1, M4: coeff 1, M6: coeff 1, M7: coeff -1
    
    Wait, let me re-derive. For a given k0 and Strassen product s:
    
    s=0 (M1): U coeffs: A00->1, A11->1; V coeffs: B[0,2k0]->1, B[1,2k0+1]->1
    s=1 (M2): U coeffs: A10->1, A11->1; V coeffs: B[0,2k0]->1
    s=2 (M3): U coeffs: A00->1; V coeffs: B[0,2k0+1]->1, B[1,2k0+1]->-1
    s=3 (M4): U coeffs: A11->1; V coeffs: B[1,2k0]->1, B[0,2k0]->-1
    s=4 (M5): U coeffs: A00->1, A10->1; V coeffs: B[1,2k0+1]->1
    s=5 (M6): U coeffs: A11->1, A00->-1; V coeffs: B[0,2k0]->1, B[1,2k0]->1
    s=6 (M7): U coeffs: A01->1, A11->-1; V coeffs: B[0,2k0+1]->1, B[1,2k0+1]->1
    
    W reconstruction:
    C[0, 2k0] = M1+M4-M5+M7 -> W[2k0*2+0, r] for r in {k0*7+0, k0*7+3, k0*7+4, k0*7+6}
       r=k0*7+0: +1, r=k0*7+3: +1, r=k0*7+4: -1, r=k0*7+6: +1
    C[1, 2k0] = M2+M4 -> W[2k0*2+1, r] for r in {k0*7+1, k0*7+3}
       r=k0*7+1: +1, r=k0*7+3: +1
    C[0, 2k0+1] = M3+M5 -> W[(2k0+1)*2+0, r] for r in {k0*7+2, k0*7+4}
       r=k0*7+2: +1, r=k0*7+4: +1
    C[1, 2k0+1] = M1-M2+M3+M6 -> W[(2k0+1)*2+1, r] for r in {k0*7+0, k0*7+1, k0*7+2, k0*7+5}
       r=k0*7+0: +1, r=k0*7+1: -1, r=k0*7+2: +1, r=k0*7+5: +1
    """
    rank = 28
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    
    for k0 in range(4):
        for s in range(7):
            r = k0 * 7 + s
            # U coefficients (A-side)
            if s == 0:  # M1
                U[0][r] = 1  # A00
                U[3][r] = 1  # A11
            elif s == 1:  # M2
                U[2][r] = 1  # A10
                U[3][r] = 1  # A11
            elif s == 2:  # M3
                U[0][r] = 1  # A00
            elif s == 3:  # M4
                U[3][r] = 1  # A11
            elif s == 4:  # M5
                U[0][r] = 1  # A00
                U[2][r] = 1  # A10
            elif s == 5:  # M6
                U[3][r] = 1  # A11
                U[0][r] = -1  # A00
            elif s == 6:  # M7
                U[1][r] = 1  # A01
                U[3][r] = -1  # A11
            
            # V coefficients (B-side)
            b0_2k0 = 0 * P + 2 * k0      # B[0, 2k0]
            b0_2k01 = 0 * P + (2 * k0 + 1)  # B[0, 2k0+1]
            b1_2k0 = 1 * P + 2 * k0      # B[1, 2k0]
            b1_2k01 = 1 * P + (2 * k0 + 1)  # B[1, 2k0+1]
            
            if s == 0:  # M1
                V[b0_2k0][r] = 1
                V[b1_2k01][r] = 1
            elif s == 1:  # M2
                V[b0_2k0][r] = 1
            elif s == 2:  # M3
                V[b0_2k01][r] = 1
                V[b1_2k01][r] = -1
            elif s == 3:  # M4
                V[b1_2k0][r] = 1
                V[b0_2k0][r] = -1
            elif s == 4:  # M5
                V[b1_2k01][r] = 1
            elif s == 5:  # M6
                V[b0_2k0][r] = 1
                V[b1_2k0][r] = 1
            elif s == 6:  # M7
                V[b0_2k01][r] = 1
                V[b1_2k01][r] = 1
            
            # W coefficients (C-side)
            c_0_2k0 = (2 * k0) * N + 0    # C[0, 2k0]
            c_1_2k0 = (2 * k0) * N + 1    # C[1, 2k0]
            c_0_2k01 = (2 * k0 + 1) * N + 0  # C[0, 2k0+1]
            c_1_2k01 = (2 * k0 + 1) * N + 1  # C[1, 2k0+1]
            
            if s == 0:  # M1
                W[c_0_2k0][r] = 1
                W[c_1_2k01][r] = 1
            elif s == 1:  # M2
                W[c_1_2k0][r] = 1
                W[c_1_2k01][r] = -1
            elif s == 2:  # M3
                W[c_0_2k01][r] = 1
                W[c_1_2k01][r] = 1
            elif s == 3:  # M4
                W[c_0_2k0][r] = 1
                W[c_1_2k0][r] = 1
            elif s == 4:  # M5
                W[c_0_2k0][r] = -1
                W[c_0_2k01][r] = 1
            elif s == 5:  # M6
                W[c_1_2k01][r] = 1
            elif s == 6:  # M7
                W[c_0_2k0][r] = 1
    
    return U, V, W


def _build_tensor():
    """Build the full 4x16x16 tensor T as a numpy array."""
    T = np.zeros((NM, MP, PN), dtype=np.int32)
    for i in range(N):
        for j in range(M):
            for k in range(P):
                a = i * M + j
                b = j * P + k
                c = k * N + i
                T[a, b, c] = 1
    return T


def _float_to_frac_list(mat_float):
    """Convert a float matrix to a list of lists of Fractions."""
    rows, cols = mat_float.shape
    result = []
    for i in range(rows):
        row = []
        for j in range(cols):
            v = mat_float[i, j]
            fv = round(v * 2) / 2
            row.append(Fraction(fv).limit_denominator(2))
        result.append(row)
    return result


def _exact_residual_check(U, V, W):
    """Check if the decomposition is exact using Fraction arithmetic."""
    rank = len(U[0])
    for i in range(N):
        for j in range(M):
            for k in range(P):
                a = i * M + j
                b = j * P + k
                c = k * N + i
                s = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                if s != 1:
                    return False
    valid = set()
    for i in range(N):
        for j in range(M):
            for k in range(P):
                valid.add((i*M+j, j*P+k, k*N+i))
    for a in range(NM):
        for b in range(MP):
            for c in range(PN):
                if (a, b, c) not in valid:
                    s = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                    if s != 0:
                        return False
    return True


def _local_search_float(Uf, Vf, Wf, T_float, deadline, max_iters=80):
    """Local search using float64 matrices with incremental residual updates."""
    rank = Uf.shape[1]
    Uf = Uf.copy()
    Vf = Vf.copy()
    Wf = Wf.copy()
    
    recon = np.einsum('ar,br,cr->abc', Uf, Vf, Wf)
    residual = T_float - recon
    best_norm = float(np.sum(residual**2))
    
    if best_norm < 1e-10:
        return Uf, Vf, Wf
    
    vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        improved = False
        
        for a in range(NM):
            for r in range(rank):
                old_val = Uf[a, r]
                for new_val in vals:
                    if abs(new_val - old_val) < 1e-12:
                        continue
                    delta = new_val - old_val
                    res_new = residual + delta * np.outer(Vf[:, r], Wf[:, r])
                    norm_new = float(np.sum(res_new**2))
                    if norm_new < best_norm - 1e-10:
                        Uf[a, r] = new_val
                        residual = res_new
                        best_norm = norm_new
                        improved = True
                        break
        
        for b in range(MP):
            for r in range(rank):
                old_val = Vf[b, r]
                for new_val in vals:
                    if abs(new_val - old_val) < 1e-12:
                        continue
                    delta = new_val - old_val
                    res_new = residual + delta * np.outer(Uf[:, r], Wf[:, r])
                    norm_new = float(np.sum(res_new**2))
                    if norm_new < best_norm - 1e-10:
                        Vf[b, r] = new_val
                        residual = res_new
                        best_norm = norm_new
                        improved = True
                        break
        
        for c in range(PN):
            for r in range(rank):
                old_val = Wf[c, r]
                for new_val in vals:
                    if abs(new_val - old_val) < 1e-12:
                        continue
                    delta = new_val - old_val
                    res_new = residual + delta * np.outer(Uf[:, r], Vf[:, r])
                    norm_new = float(np.sum(res_new**2))
                    if norm_new < best_norm - 1e-10:
                        Wf[c, r] = new_val
                        residual = res_new
                        best_norm = norm_new
                        improved = True
                        break
        
        if not improved:
            break
    
    return Uf, Vf, Wf


def _try_improve_from_strassen(deadline):
    """Try to improve the Strassen-based rank-28 decomposition to lower rank."""
    T = _build_tensor()
    T_float = T.astype(np.float64)
    
    U0, V0, W0 = strassen_base()
    
    # Verify the Strassen decomposition
    if not verify_fast(U0, V0, W0):
        return None
    
    # Try to reduce rank by removing columns that are linearly dependent or zero
    # Convert to float for manipulation
    Uf = np.array(U0, dtype=np.float64)
    Vf = np.array(V0, dtype=np.float64)
    Wf = np.array(W0, dtype=np.float64)
    
    # Try local search to see if we can find a better solution
    Uf_new, Vf_new, Wf_new = _local_search_float(Uf, Vf, Wf, T_float, deadline, max_iters=40)
    
    recon = np.einsum('ar,br,cr->abc', Uf_new, Vf_new, Wf_new)
    residual = T_float - recon
    norm = float(np.sum(residual**2))
    
    if norm < 1e-10:
        U_frac = _float_to_frac_list(Uf_new)
        V_frac = _float_to_frac_list(Vf_new)
        W_frac = _float_to_frac_list(Wf_new)
        if _exact_residual_check(U_frac, V_frac, W_frac):
            return (U_frac, V_frac, W_frac)
    
    return (U0, V0, W0)


def search_deadline():
    """Main search with time limit."""
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s kill
    
    best = None
    best_rank = SCHOOBOOK_RANK
    
    # Start with schoolbook
    U0, V0, W0 = schoolbook()
    if verify_fast(U0, V0, W0):
        best = (U0, V0, W0)
        best_rank = SCHOOBOOK_RANK
    
    # Try Strassen-based decomposition (rank 28)
    if time.time() < deadline:
        strassen_result = _try_improve_from_strassen(deadline)
        if strassen_result is not None:
            U_s, V_s, W_s = strassen_result
            rank_s = len(U_s[0])
            if rank_s < best_rank:
                best = strassen_result
                best_rank = rank_s
    
    # Try random searches for lower ranks
    rng = random.Random(42)
    T = _build_tensor()
    T_float = T.astype(np.float64)
    
    for target_rank in range(16, best_rank):
        if time.time() > deadline:
            break
        
        remaining = deadline - time.time()
        if remaining < 5:
            break
        
        if remaining > 60:
            n_trials = 300 if target_rank <= 18 else 200 if target_rank <= 22 else 100
        elif remaining > 30:
            n_trials = 100 if target_rank <= 18 else 60 if target_rank <= 22 else 30
        else:
            n_trials = 30 if target_rank <= 18 else 15 if target_rank <= 22 else 8
        
        for trial in range(n_trials):
            if time.time() > deadline:
                break
            
            seed = rng.randint(0, 2**31 - 1)
            np_rng = np.random.default_rng(seed)
            vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
            probs = np.array([0.3, 0.15, 0.2, 0.15, 0.2])
            
            Uf = np_rng.choice(vals, size=(NM, target_rank), p=probs)
            Vf = np_rng.choice(vals, size=(MP, target_rank), p=probs)
            Wf = np_rng.choice(vals, size=(PN, target_rank), p=probs)
            
            recon = np.einsum('ar,br,cr->abc', Uf, Vf, Wf)
            residual = T_float - recon
            norm = float(np.sum(residual**2))
            
            if norm < 1e-10:
                U_frac = _float_to_frac_list(Uf)
                V_frac = _float_to_frac_list(Vf)
                W_frac = _float_to_frac_list(Wf)
                if _exact_residual_check(U_frac, V_frac, W_frac):
                    best = (U_frac, V_frac, W_frac)
                    best_rank = target_rank
                    break
            
            if time.time() > deadline:
                break
            
            if norm < 100:
                Uf_new, Vf_new, Wf_new = _local_search_float(Uf, Vf, Wf, T_float, deadline, max_iters=50)
                
                recon_new = np.einsum('ar,br,cr->abc', Uf_new, Vf_new, Wf_new)
                residual_new = T_float - recon_new
                norm_new = float(np.sum(residual_new**2))
                
                if norm_new < 1e-10:
                    U_frac = _float_to_frac_list(Uf_new)
                    V_frac = _float_to_frac_list(Vf_new)
                    W_frac = _float_to_frac_list(Wf_new)
                    if _exact_residual_check(U_frac, V_frac, W_frac):
                        best = (U_frac, V_frac, W_frac)
                        best_rank = target_rank
                        break
        
        if best_rank <= target_rank:
            if best_rank < 16:
                break
    
    if best is not None:
        U, V, W = best
        return {"U": U, "V": V, "W": W}
    else:
        U, V, W = schoolbook()
        return {"U": U, "V": V, "W": W}


def construct():
    """Entry point: return the best decomposition found."""
    try:
        return search_deadline()
    except Exception:
        # Fallback to schoolbook
        U, V, W = schoolbook()
        return {"U": U, "V": V, "W": W}
