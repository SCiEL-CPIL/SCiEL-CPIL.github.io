"""Matrix multiplication 3x5x5 rank search with deadline guard."""
import time
import sys
import numpy as np

DIMENSIONS = [3, 5, 5]

# Hard wall-clock deadline (evaluator kills at 150s; keep safety margin)
DEADLINE = 145.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def _time_left():
    return DEADLINE - _elapsed()

def _schoolbook():
    """Schoolbook decomposition of rank n*m*p = 75."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    """Return Strassen rank-7 decomposition for 2x2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) where:
    U[i,j,c] * V[j,k,c] * W[k,i,c] summed over c gives C[i,k] for A*B.
    Convention: U indexed by (i,j), V by (j,k), W by (k,i).
    """
    # Strassen's 7 products for 2x2x2
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # U: 4 entries (i,j) for i,j in {0,1}, 7 columns
    # V: 4 entries (j,k) for j,k in {0,1}, 7 columns  
    # W: 4 entries (k,i) for k,i in {0,1}, 7 columns
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Index: U[i*2+j], V[j*2+k], W[k*2+i]
    
    # M1 = (A00+A11)*(B00+B11)
    U[0, 0] = 1  # A00
    U[3, 0] = 1  # A11
    V[0, 0] = 1  # B00
    V[3, 0] = 1  # B11
    
    # M2 = (A10+A11)*B00
    U[2, 1] = 1  # A10
    U[3, 1] = 1  # A11
    V[0, 1] = 1  # B00
    
    # M3 = A00*(B01-B11)
    U[0, 2] = 1  # A00
    V[1, 2] = 1  # B01
    V[3, 2] = -1 # B11
    
    # M4 = A11*(B10-B00)
    U[3, 3] = 1  # A11
    V[2, 3] = 1  # B10
    V[0, 3] = -1 # B00
    
    # M5 = (A00+A01)*B11
    U[0, 4] = 1  # A00
    U[1, 4] = 1  # A01
    V[3, 4] = 1  # B11
    
    # M6 = (A10-A00)*(B00+B01)
    U[2, 5] = 1  # A10
    U[0, 5] = -1 # A00
    V[0, 5] = 1  # B00
    V[1, 5] = 1  # B01
    
    # M7 = (A01-A11)*(B10+B11)
    U[1, 6] = 1  # A01
    U[3, 6] = -1 # A11
    V[2, 6] = 1  # B10
    V[3, 6] = 1  # B11
    
    # W: contribution to C
    # C00 = M1+M4-M5+M7
    W[0, 0] = 1  # M1 -> C00
    W[0, 3] = 1  # M4 -> C00
    W[0, 4] = -1 # M5 -> C00
    W[0, 6] = 1  # M7 -> C00
    
    # C01 = M3+M5
    W[1, 2] = 1  # M3 -> C01
    W[1, 4] = 1  # M5 -> C01
    
    # C10 = M2+M4
    W[2, 1] = 1  # M2 -> C10
    W[2, 3] = 1  # M4 -> C10
    
    # C11 = M1-M2+M3+M6
    W[3, 0] = 1  # M1 -> C11
    W[3, 1] = -1 # M2 -> C11
    W[3, 2] = 1  # M3 -> C11
    W[3, 5] = 1  # M6 -> C11
    
    return U, V, W

def _block_decomposition(n, m, p):
    """Try various block partitions and use Strassen where possible.
    Returns best decomposition found.
    """
    best_result = None
    best_rank = n * m * p
    
    # Try different block sizes
    # For 3x5x5, try partitions of n=3, m=5, p=5
    # Possible block sizes: 1, 2, 3 for n; 1, 2, 3, 5 for m; 1, 2, 3, 5 for p
    
    n_blocks = []
    for a in range(1, n+1):
        if n - a >= 1:
            n_blocks.append((a, n-a))
        if a < n:
            n_blocks.append((a, n-a))
    
    m_blocks = []
    for a in range(1, m+1):
        if m - a >= 1:
            m_blocks.append((a, m-a))
    
    p_blocks = []
    for a in range(1, p+1):
        if p - a >= 1:
            p_blocks.append((a, p-a))
    
    # Also try single block (schoolbook)
    all_n_parts = [(n,)]
    all_m_parts = [(m,)]
    all_p_parts = [(p,)]
    
    for part in n_blocks:
        all_n_parts.append(part)
    for part in m_blocks:
        all_m_parts.append(part)
    for part in p_blocks:
        all_p_parts.append(part)
    
    for n_part in all_n_parts:
        for m_part in all_m_parts:
            for p_part in all_p_parts:
                if _time_left() < 5.0:
                    break
                
                result = _build_block_decomp(n, m, p, n_part, m_part, p_part)
                if result is not None:
                    rank = len(result["U"][0])
                    if rank < best_rank:
                        best_rank = rank
                        best_result = result
    
    return best_result if best_result is not None else _schoolbook(), best_rank

def _build_block_decomp(n, m, p, n_part, m_part, p_part):
    """Build a block decomposition using Strassen for 2x2x2 blocks and schoolbook otherwise."""
    U_list = []
    V_list = []
    W_list = []
    
    n_offsets = []
    offset = 0
    for size in n_part:
        n_offsets.append(offset)
        offset += size
    
    m_offsets = []
    offset = 0
    for size in m_part:
        m_offsets.append(offset)
        offset += size
    
    p_offsets = []
    offset = 0
    for size in p_part:
        p_offsets.append(offset)
        offset += size
    
    for ni, ns in enumerate(n_part):
        for mi, ms in enumerate(m_part):
            for pi, ps in enumerate(p_part):
                if ns == 2 and ms == 2 and ps == 2:
                    # Use Strassen
                    U_s, V_s, W_s = _strassen_2x2x2()
                    for c in range(7):
                        U_row = []
                        V_row = []
                        W_row = []
                        for i in range(ns):
                            for j in range(ms):
                                U_row.append(U_s[i*2+j, c])
                            # Pad to full n*m
                        # Actually, let's build full-size matrices
                        pass
                    
                    # Build full-size contributions
                    U_full = np.zeros((n*m, 7))
                    V_full = np.zeros((m*p, 7))
                    W_full = np.zeros((p*n, 7))
                    
                    for i in range(2):
                        for j in range(2):
                            U_full[(n_offsets[ni]+i)*m + (m_offsets[mi]+j), :] = U_s[i*2+j, :]
                    
                    for j in range(2):
                        for k in range(2):
                            V_full[(m_offsets[mi]+j)*p + (p_offsets[pi]+k), :] = V_s[j*2+k, :]
                    
                    for k in range(2):
                        for i in range(2):
                            W_full[(p_offsets[pi]+k)*n + (n_offsets[ni]+i), :] = W_s[k*2+i, :]
                    
                    U_list.append(U_full)
                    V_list.append(V_full)
                    W_list.append(W_full)
                else:
                    # Schoolbook for this block
                    rank = ns * ms * ps
                    U_full = np.zeros((n*m, rank))
                    V_full = np.zeros((m*p, rank))
                    W_full = np.zeros((p*n, rank))
                    
                    c = 0
                    for i in range(ns):
                        for j in range(ms):
                            for k in range(ps):
                                U_full[(n_offsets[ni]+i)*m + (m_offsets[mi]+j), c] = 1
                                V_full[(m_offsets[mi]+j)*p + (p_offsets[pi]+k), c] = 1
                                W_full[(p_offsets[pi]+k)*n + (n_offsets[ni]+i), c] = 1
                                c += 1
                    
                    U_list.append(U_full)
                    V_list.append(V_full)
                    W_list.append(W_full)
    
    if not U_list:
        return None
    
    U = np.hstack(U_list)
    V = np.hstack(V_list)
    W = np.hstack(W_list)
    
    # Verify
    Ua = U.reshape(n, m, -1)
    Va = V.reshape(m, p, -1)
    Wa = W.reshape(p, n, -1)
    T = np.einsum('ijc,jkc,kic->ijk', Ua, Va, Wa)
    
    if np.max(np.abs(T - 1.0)) < 1e-9:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None


def _verify(U, V, W, tol=1e-6):
    """Check that sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] == 1 for all i,j,k."""
    import numpy as np
    n, m, p = DIMENSIONS
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    Ua = Ua.reshape(n, m, rank)
    Va = Va.reshape(m, p, rank)
    Wa = Wa.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ua, Va, Wa)
    return np.max(np.abs(T - 1.0)) < tol


def _als_one_step(U, V, W, n, m, p, rank):
    """One step of alternating least squares."""
    import numpy as np
    U_new = U.copy()
    V_new = V.copy()
    W_new = W.copy()
    
    for i in range(n):
        for j in range(m):
            A_sys = np.zeros((p, rank))
            for k in range(p):
                A_sys[k, :] = V[j*p+k, :] * W[k*n+i, :]
            try:
                U_new[i*m+j, :] = np.linalg.lstsq(A_sys, np.ones(p), rcond=None)[0]
            except:
                pass
    
    for j in range(m):
        for k in range(p):
            A_sys = np.zeros((n, rank))
            for i in range(n):
                A_sys[i, :] = U_new[i*m+j, :] * W[k*n+i, :]
            try:
                V_new[j*p+k, :] = np.linalg.lstsq(A_sys, np.ones(n), rcond=None)[0]
            except:
                pass
    
    for k in range(p):
        for i in range(n):
            A_sys = np.zeros((m, rank))
            for j in range(m):
                A_sys[j, :] = U_new[i*m+j, :] * V_new[j*p+k, :]
            try:
                W_new[k*n+i, :] = np.linalg.lstsq(A_sys, np.ones(m), rcond=None)[0]
            except:
                pass
    
    return U_new, V_new, W_new


def _compute_error(U, V, W, n, m, p, rank):
    """Compute the max absolute error of the tensor decomposition."""
    import numpy as np
    Ua = np.array(U, dtype=float).reshape(n, m, rank)
    Va = np.array(V, dtype=float).reshape(m, p, rank)
    Wa = np.array(W, dtype=float).reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ua, Va, Wa)
    return np.max(np.abs(T - 1.0))


def _try_als(U0, V0, W0, target_rank, max_iters=30, tol=1e-7):
    """Try ALS to find a rank-`target_rank` decomposition."""
    import numpy as np
    n, m, p = DIMENSIONS
    orig_rank = len(U0[0])
    
    Ua = np.array(U0, dtype=float)
    Va = np.array(V0, dtype=float)
    Wa = np.array(W0, dtype=float)
    
    # Try multiple random projections
    for attempt in range(3):
        if _time_left() < 2.0:
            return None
        
        if attempt == 0:
            # Use SVD-based projection for better initial guess
            try:
                # Create a combined tensor and project
                U_flat = Ua.reshape(-1, orig_rank)
                V_flat = Va.reshape(-1, orig_rank)
                W_flat = Wa.reshape(-1, orig_rank)
                
                # Use the first few columns as basis
                if orig_rank >= target_rank:
                    U = Ua[:, :target_rank]
                    V = Va[:, :target_rank]
                    W = Wa[:, :target_rank]
                else:
                    P = np.random.randn(orig_rank, target_rank)
                    U = Ua @ P
                    V = Va @ P
                    W = Wa @ P
            except:
                P = np.random.randn(orig_rank, target_rank)
                U = Ua @ P
                V = Va @ P
                W = Wa @ P
        else:
            P = np.random.randn(orig_rank, target_rank)
            U = Ua @ P
            V = Va @ P
            W = Wa @ P
        
        for it in range(max_iters):
            if _time_left() < 1.0:
                break
            
            U, V, W = _als_one_step(U, V, W, n, m, p, target_rank)
            
            # Project to half-integers
            U_half = np.round(U * 2) / 2.0
            V_half = np.round(V * 2) / 2.0
            W_half = np.round(W * 2) / 2.0
            
            err = _compute_error(U_half, V_half, W_half, n, m, p, target_rank)
            
            if err < tol:
                return {"U": U_half.tolist(), "V": V_half.tolist(), "W": W_half.tolist()}
    
    return None


def _random_search(n, m, p, target_rank, max_trials=5, tol=1e-7):
    """Try random initializations for ALS."""
    import numpy as np
    
    for trial in range(max_trials):
        if _time_left() < 5.0:
            break
        
        # Use structured random init: sparse with some structure
        U = np.zeros((n * m, target_rank))
        V = np.zeros((m * p, target_rank))
        W = np.zeros((p * n, target_rank))
        
        for c in range(target_rank):
            # Randomly assign entries
            idx_u = np.random.choice(n * m, size=min(3, n * m), replace=False)
            idx_v = np.random.choice(m * p, size=min(3, m * p), replace=False)
            idx_w = np.random.choice(p * n, size=min(3, p * n), replace=False)
            
            U[idx_u, c] = np.random.choice([-1, 1], size=len(idx_u))
            V[idx_v, c] = np.random.choice([-1, 1], size=len(idx_v))
            W[idx_w, c] = np.random.choice([-1, 1], size=len(idx_w))
        
        for it in range(25):
            if _time_left() < 2.0:
                break
            
            U, V, W = _als_one_step(U, V, W, n, m, p, target_rank)
            
            U_half = np.round(U * 2) / 2.0
            V_half = np.round(V * 2) / 2.0
            W_half = np.round(W * 2) / 2.0
            
            err = _compute_error(U_half, V_half, W_half, n, m, p, target_rank)
            
            if err < tol:
                return {"U": U_half.tolist(), "V": V_half.tolist(), "W": W_half.tolist()}
    
    return None


def _try_strassen_block(n, m, p, target_rank, max_iters=20, tol=1e-7):
    """Try a structured decomposition inspired by Strassen's algorithm."""
    import numpy as np
    
    # Get the block decomposition as starting point
    block_result, block_rank = _block_decomposition(n, m, p)
    
    if block_rank >= target_rank:
        # Can't reduce below target with this approach
        return None
    
    U = np.array(block_result["U"], dtype=float)
    V = np.array(block_result["V"], dtype=float)
    W = np.array(block_result["W"], dtype=float)
    
    # Project down to target_rank
    P = np.random.randn(block_rank, target_rank)
    U = U @ P
    V = V @ P
    W = W @ P
    
    for it in range(max_iters):
        if _time_left() < 1.0:
            return None
        
        U, V, W = _als_one_step(U, V, W, n, m, p, target_rank)
        
        U_half = np.round(U * 2) / 2.0
        V_half = np.round(V * 2) / 2.0
        W_half = np.round(W * 2) / 2.0
        
        err = _compute_error(U_half, V_half, W_half, n, m, p, target_rank)
        
        if err < tol:
            return {"U": U_half.tolist(), "V": V_half.tolist(), "W": W_half.tolist()}
    
    return None


def construct():
    if _time_left() < 5.0:
        return _schoolbook()

    np.random.seed(42)
    n, m, p = DIMENSIONS
    
    # First try block decomposition with Strassen
    block_result, block_rank = _block_decomposition(n, m, p)
    
    best = block_result
    best_rank = block_rank
    
    if _time_left() < 10.0:
        if _verify(best["U"], best["V"], best["W"], tol=1e-9):
            return best
        return _schoolbook()
    
    # Try to further optimize with ALS from the block decomposition
    U0 = np.array(best["U"], dtype=float)
    V0 = np.array(best["V"], dtype=float)
    W0 = np.array(best["W"], dtype=float)
    
    # Try various target ranks below the block rank
    for target_rank in range(best_rank - 1, max(30, best_rank - 20), -1):
        if _time_left() < 10.0:
            break
        
        # Try ALS from current best
        result = _try_als(best["U"], best["V"], best["W"], target_rank, max_iters=20)
        if result is not None:
            if _verify(result["U"], result["V"], result["W"], tol=1e-9):
                best = result
                best_rank = target_rank
                continue
        
        # Try random search
        if _time_left() > 15.0:
            result = _random_search(n, m, p, target_rank, max_trials=5)
            if result is not None:
                if _verify(result["U"], result["V"], result["W"], tol=1e-9):
                    best = result
                    best_rank = target_rank
                    continue
    
    # Ensure the returned result is valid and has integer/half-integer entries
    if _verify(best["U"], best["V"], best["W"], tol=1e-9):
        return best
    else:
        return _schoolbook()
